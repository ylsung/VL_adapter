#!/usr/bin/env python3
"""
Convert the original VL-Adapter `vlt5_dataset` zip into a HuggingFace-Hub-friendly
layout, streaming straight out of the zip so the archive is never fully extracted.

Original layout (inside VLadapter.zip)        HF repo layout produced here
---------------------------------------       -----------------------------------------
vlt5_dataset/<D>/clip_features/                features/<D>/data_clip_RN101_att/
    data_clip_RN101_att/<img_id>.h5                part-XXXXX.parquet   (id, features)
vlt5_dataset/nlvr/images/<img_id>.png          images/nlvr/part-XXXXX.parquet  (id, image)
vlt5_dataset/<everything else>                 annotations/<same relative path>

Each `.h5` holds exactly one group named after the image id with a single
(49, 2048) float16 dataset `features`, so the payload is stored losslessly in
parquet as a nested fixed-size list and rebuilt by `restore_datasets.py`.

Disk use is bounded: one shard (~300 MB) plus at most one annotation file is on
disk at a time when --upload is used.
"""
import argparse
import gc
import io
import json
import os
import sys
import time
import zipfile
from pathlib import Path

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = "vlt5_dataset/"
FEAT_SHAPE = (49, 2048)
FEAT_NUMEL = FEAT_SHAPE[0] * FEAT_SHAPE[1]


# --------------------------------------------------------------------------- #
# zip inventory
# --------------------------------------------------------------------------- #
def classify(zf):
    """Split the archive into feature / nlvr-image / plain-file groups.

    Entries keep central-directory order, which for this archive equals physical
    order, so reading them back is a sequential scan of the file.
    """
    feats, images, others = {}, [], []
    for info in zf.infolist():
        name = info.filename
        if name.endswith("/"):
            continue
        if not name.startswith(ROOT):
            raise SystemExit(f"unexpected entry outside {ROOT}: {name}")
        rel = name[len(ROOT):]
        if "/clip_features/" in rel and rel.endswith(".h5"):
            dataset, _, variant, _ = rel.split("/")
            feats.setdefault((dataset, variant), []).append(info)
        elif rel.startswith("nlvr/images/"):
            images.append(info)
        else:
            others.append(info)
    return feats, images, others


# --------------------------------------------------------------------------- #
# parquet writers
# --------------------------------------------------------------------------- #
def features_table(ids, buf, count):
    """(49, 2048) float16 arrays -> fixed_size_list<fixed_size_list<half>[2048]>[49]."""
    flat = pa.array(buf[: count * FEAT_NUMEL], type=pa.float16())
    inner = pa.FixedSizeListArray.from_arrays(flat, FEAT_SHAPE[1])
    outer = pa.FixedSizeListArray.from_arrays(inner, FEAT_SHAPE[0])
    return pa.table({"id": pa.array(ids, type=pa.string()), "features": outer})


def images_table(ids, blobs):
    """PNG bytes in the struct layout the HF `datasets` Image feature expects."""
    image = pa.StructArray.from_arrays(
        [pa.array(blobs, type=pa.binary()), pa.array(ids, type=pa.string())],
        names=["bytes", "path"],
    )
    return pa.table({"id": pa.array(ids, type=pa.string()), "image": image})


def write_shard(table, path, compression, level, meta):
    path.parent.mkdir(parents=True, exist_ok=True)
    schema = table.schema.with_metadata({k: str(v) for k, v in meta.items()})
    pq.write_table(
        table.replace_schema_metadata(schema.metadata),
        path,
        compression=compression,
        compression_level=level if compression == "zstd" else None,
    )
    return path.stat().st_size


# --------------------------------------------------------------------------- #
# upload / state
# --------------------------------------------------------------------------- #
class Sink:
    """Writes shards locally and, with --upload, pushes then deletes each one."""

    def __init__(self, out_dir, repo_id, upload, state_path, token=None, private=False):
        self.out_dir = Path(out_dir)
        self.repo_id = repo_id
        self.upload = upload
        self.state_path = Path(state_path)
        self.done = set()
        if self.state_path.exists():
            self.done = set(json.loads(self.state_path.read_text())["done"])
        self.api = None
        if upload:
            from huggingface_hub import HfApi

            self.api = HfApi(token=token)
            self.api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
            who = self.api.whoami()["name"]
            print(f"[hub] authenticated as {who}; target dataset {repo_id} "
                  f"({'private' if private else 'public'})", flush=True)

    def has(self, rel):
        return rel in self.done

    def mark(self, rel):
        self.done.add(rel)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"done": sorted(self.done)}))
        tmp.replace(self.state_path)

    def push(self, local_path, rel):
        if self.upload:
            for attempt in range(5):
                try:
                    self.api.upload_file(
                        path_or_fileobj=str(local_path),
                        path_in_repo=rel,
                        repo_id=self.repo_id,
                        repo_type="dataset",
                        commit_message=f"Add {rel}",
                    )
                    break
                except Exception as exc:  # noqa: BLE001 - retry any transport error
                    wait = 2 ** attempt * 5
                    print(f"    upload failed ({exc}); retry in {wait}s", flush=True)
                    time.sleep(wait)
            else:
                raise SystemExit(f"giving up uploading {rel}")
            local_path.unlink()
        self.mark(rel)


# --------------------------------------------------------------------------- #
# conversion passes
# --------------------------------------------------------------------------- #
def convert_features(zf, group, infos, sink, args, manifest):
    dataset, variant = group
    prefix = f"features/{dataset}/{variant}"
    total = len(infos)
    shards = (total + args.shard_rows - 1) // args.shard_rows
    print(f"[features] {dataset}/{variant}: {total} files -> {shards} shards", flush=True)

    buf = np.empty(args.shard_rows * FEAT_NUMEL, dtype=np.float16)
    for shard_idx in range(shards):
        rel = f"{prefix}/part-{shard_idx:05d}.parquet"
        chunk = infos[shard_idx * args.shard_rows : (shard_idx + 1) * args.shard_rows]
        manifest.extend(
            {"path": f"{dataset}/clip_features/{variant}/{i.filename.rsplit('/', 1)[1]}",
             "size": i.file_size, "crc": i.CRC, "shard": rel}
            for i in chunk
        )
        if sink.has(rel):
            print(f"  skip {rel} (done)", flush=True)
            continue

        ids = []
        t0 = time.time()
        for row, info in enumerate(chunk):
            raw = zf.read(info)
            stem = info.filename.rsplit("/", 1)[1][:-3]
            with h5py.File(io.BytesIO(raw), "r") as h5:
                keys = list(h5.keys())
                # restore_datasets.py names the rebuilt file after the group key,
                # so the two must agree or the tree would come back renamed.
                if keys != [stem]:
                    raise SystemExit(f"{info.filename}: expected single group {stem!r}, got {keys}")
                arr = h5[stem]["features"]
                if arr.shape != FEAT_SHAPE or arr.dtype != np.float16:
                    raise SystemExit(f"unexpected array {arr.shape}/{arr.dtype} in {info.filename}")
                arr.read_direct(buf[row * FEAT_NUMEL : (row + 1) * FEAT_NUMEL].reshape(FEAT_SHAPE))
            ids.append(stem)

        table = features_table(ids, buf, len(chunk))
        path = Path(args.out) / rel
        size = write_shard(table, path, args.compression, args.level,
                           {"source": f"{ROOT}{dataset}/clip_features/{variant}",
                            "shape": str(FEAT_SHAPE), "dtype": "float16"})
        del table
        gc.collect()
        print(f"  {rel}: {len(chunk)} rows, {size/1e6:.1f} MB, {time.time()-t0:.1f}s", flush=True)
        sink.push(path, rel)


def convert_images(zf, infos, sink, args, manifest):
    print(f"[images] nlvr: {len(infos)} png", flush=True)
    shard_idx, nbytes = 0, 0

    def flush(shard_idx, batch, blobs):
        rel = f"images/nlvr/part-{shard_idx:05d}.parquet"
        if sink.has(rel):
            print(f"  skip {rel} (done)", flush=True)
            return
        table = images_table(batch, blobs)
        path = Path(args.out) / rel
        size = write_shard(table, path, "zstd", 1, {"source": f"{ROOT}nlvr/images"})
        print(f"  {rel}: {len(batch)} images, {size/1e6:.1f} MB", flush=True)
        sink.push(path, rel)

    pending = []
    for info in infos:
        name = info.filename.rsplit("/", 1)[1]
        rel_shard = f"images/nlvr/part-{shard_idx:05d}.parquet"
        manifest.append({"path": f"nlvr/images/{name}", "size": info.file_size,
                         "crc": info.CRC, "shard": rel_shard})
        pending.append(info)
        nbytes += info.file_size
        if nbytes >= args.image_shard_bytes:
            if not sink.has(rel_shard):
                batch = [i.filename.rsplit("/", 1)[1][:-4] for i in pending]
                blobs = [zf.read(i) for i in pending]
                flush(shard_idx, batch, blobs)
                del blobs
                gc.collect()
            else:
                print(f"  skip {rel_shard} (done)", flush=True)
            pending, nbytes = [], 0
            shard_idx += 1
    if pending:
        rel_shard = f"images/nlvr/part-{shard_idx:05d}.parquet"
        if not sink.has(rel_shard):
            batch = [i.filename.rsplit("/", 1)[1][:-4] for i in pending]
            blobs = [zf.read(i) for i in pending]
            flush(shard_idx, batch, blobs)


def convert_others(zf, infos, sink, args, manifest):
    print(f"[annotations] {len(infos)} files", flush=True)
    for info in infos:
        orig = info.filename[len(ROOT):]
        rel = f"annotations/{orig}"
        manifest.append({"path": orig, "size": info.file_size, "crc": info.CRC, "shard": rel})
        if sink.has(rel):
            print(f"  skip {rel} (done)", flush=True)
            continue
        path = Path(args.out) / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(info) as src, open(path, "wb") as dst:
            while True:
                block = src.read(8 << 20)
                if not block:
                    break
                dst.write(block)
        print(f"  {rel}: {info.file_size/1e6:.1f} MB", flush=True)
        sink.push(path, rel)


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--zip", required=True, help="path to VLadapter.zip")
    p.add_argument("--out", required=True, help="staging directory for shards")
    p.add_argument("--repo-id", help="HF dataset repo, e.g. user/VL-Adapter-datasets")
    p.add_argument("--upload", action="store_true", help="upload each shard then delete it locally")
    p.add_argument("--token", default=None)
    p.add_argument("--private", action="store_true", help="create the dataset repo private")
    p.add_argument("--shard-rows", type=int, default=4096)
    p.add_argument("--image-shard-bytes", type=int, default=700 * 1024 * 1024)
    p.add_argument("--compression", default="zstd", choices=["zstd", "snappy", "none"])
    p.add_argument("--level", type=int, default=9)
    p.add_argument("--only", choices=["features", "images", "annotations"], action="append")
    p.add_argument("--limit-shards", type=int, default=0, help="testing: stop after N feature shards per group")
    p.add_argument("--state", default=None, help="resume state file (default <out>/.convert_state.json)")
    args = p.parse_args()

    if args.upload and not args.repo_id:
        p.error("--upload requires --repo-id")
    args.state = args.state or os.path.join(args.out, ".convert_state.json")
    only = set(args.only or ["features", "images", "annotations"])

    zf = zipfile.ZipFile(args.zip)
    feats, images, others = classify(zf)
    sink = Sink(args.out, args.repo_id, args.upload, args.state, args.token, args.private)
    manifest = []

    if "annotations" in only:
        convert_others(zf, others, sink, args, manifest)
    if "images" in only:
        convert_images(zf, images, sink, args, manifest)
    if "features" in only:
        for group in sorted(feats):
            infos = feats[group]
            if args.limit_shards:
                infos = infos[: args.limit_shards * args.shard_rows]
            convert_features(zf, group, infos, sink, args, manifest)

    man_path = Path(args.out) / "manifest.json.gz"
    if not args.limit_shards:
        import gzip

        man_path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(man_path, "wt") as fh:
            json.dump({"root": ROOT, "files": manifest}, fh)
        print(f"[manifest] {len(manifest)} entries -> {man_path}", flush=True)
        sink.push(man_path, "manifest.json.gz")
    print("done.", flush=True)


if __name__ == "__main__":
    sys.exit(main())
