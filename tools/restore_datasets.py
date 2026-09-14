#!/usr/bin/env python3
"""
Rebuild the original VL-Adapter `datasets/` tree from the HuggingFace dataset repo.

    python restore_datasets.py --out ./datasets

Downloads one shard at a time, expands it, then deletes the shard, so the extra
disk needed on top of the final tree stays under ~1 GB. Re-running resumes:
files that already exist with the recorded size are left alone.

Resulting layout (what VL-T5/src/*_clip_data.py expects):

    datasets/
      COCO/clip_features/data_clip_RN101_att/<img_id>.h5
      COCO/dataset_coco.json
      GQA/clip_features/data_clip_RN101_att/<img_id>.h5
      GQA/*.json, eval.py, readme.txt
      VG/clip_features/data_clip_RN101_att/<img_id>.h5
      VG/*.json, *_vocab.txt, readme.txt
      nlvr/clip_features/data_clip_RN101_att/<img_id>.h5
      nlvr/images/<img_id>.png
      nlvr/*.json
      lxmert/*.json
      vqa/*.json
"""
import argparse
import gzip
import json
import os
import shutil
import sys
import zlib
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import pyarrow.parquet as pq

REPO_ID = "ylsung/VL-Adapter-datasets"
FEAT_SHAPE = (49, 2048)
# Directories that exist in the original archive but hold no files, plus
# nlvr/images, whose photographs are not redistributed (see NLVR2_NOTE).
EMPTY_DIRS = ["paragraphs", "COCO/clip_features/data_clip_RN101_fc", "nlvr/images"]

NLVR2_NOTE = """
NOTE: datasets/nlvr/images/ is empty on purpose.

The NLVR2 authors do not own the copyright to the NLVR2 photographs and ask that
they not be shared publicly, so this dataset ships only the CLIP features derived
from them (datasets/nlvr/clip_features/), not the images themselves.

Every script under VL-T5/scripts/image/ reads clip_features and runs without the
images. You only need them for the end-to-end pixel training paths
(VL-T5/src/*_raw_data.py). To obtain them, follow the instructions at
    https://github.com/lil-lab/nlvr/tree/master/nlvr2
and place the files as datasets/nlvr/images/<split>-<id>-<n>-img<0|1>.png
"""


def fetch(repo_id, rel, local_root, revision, token):
    """Return a local path for `rel`, downloading from the Hub unless it is already local."""
    if local_root:
        return Path(local_root) / rel, False
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id=repo_id, filename=rel, repo_type="dataset",
                           revision=revision, token=token)
    return Path(path), True


def drop(path, downloaded, keep_cache):
    """Delete a downloaded blob (and its symlink) once it has been expanded."""
    if not downloaded or keep_cache:
        return
    try:
        real = path.resolve()
        path.unlink(missing_ok=True)
        real.unlink(missing_ok=True)
    except OSError:
        pass


def write_h5(path, key, array):
    with h5py.File(path, "w") as fh:
        fh.create_group(key).create_dataset("features", data=array, dtype="float16")


def restore_features(shard_path, dataset, variant, out_root):
    target = out_root / dataset / "clip_features" / variant
    target.mkdir(parents=True, exist_ok=True)
    table = pq.read_table(shard_path)
    ids = table.column("id").to_pylist()
    feats = table.column("features")
    written = 0
    for i, img_id in enumerate(ids):
        dest = target / f"{img_id}.h5"
        if dest.exists():
            continue
        arr = np.asarray(feats[i].as_py(), dtype=np.float16).reshape(FEAT_SHAPE)
        write_h5(dest, img_id, arr)
        written += 1
    return len(ids), written


def restore_images(shard_path, out_root):
    target = out_root / "nlvr" / "images"
    target.mkdir(parents=True, exist_ok=True)
    table = pq.read_table(shard_path)
    ids = table.column("id").to_pylist()
    blobs = []
    for chunk in table.column("image").chunks:
        blobs.extend(chunk.field("bytes").to_pylist())
    written = 0
    for img_id, blob in zip(ids, blobs):
        dest = target / f"{img_id}.png"
        if dest.exists() and dest.stat().st_size == len(blob):
            continue
        dest.write_bytes(blob)
        written += 1
    return len(ids), written


def verify(out_root, manifest, sample):
    """Check size for every restored file and crc32 for a random sample."""
    import random

    files = manifest["files"]
    missing, bad_size, bad_crc = [], [], []
    for rec in files:
        dest = out_root / rec["path"]
        if not dest.exists():
            missing.append(rec["path"])
            continue
        if rec["path"].endswith(".h5"):
            continue  # rebuilt by h5py; size/crc need not match byte-for-byte
        if dest.stat().st_size != rec["size"]:
            bad_size.append(rec["path"])
    checkable = [r for r in files if not r["path"].endswith(".h5")]
    for rec in random.sample(checkable, min(sample, len(checkable))):
        dest = out_root / rec["path"]
        if not dest.exists():
            continue
        crc = 0
        with open(dest, "rb") as fh:
            while True:
                block = fh.read(8 << 20)
                if not block:
                    break
                crc = zlib.crc32(block, crc)
        if crc != rec["crc"]:
            bad_crc.append(rec["path"])
    print(f"[verify] {len(files)} expected, missing={len(missing)}, "
          f"size-mismatch={len(bad_size)}, crc-mismatch={len(bad_crc)}")
    for label, items in (("missing", missing), ("size", bad_size), ("crc", bad_crc)):
        for item in items[:10]:
            print(f"  {label}: {item}")
    return not (missing or bad_size or bad_crc)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="./datasets", help="destination datasets/ directory")
    p.add_argument("--repo-id", default=REPO_ID)
    p.add_argument("--revision", default="main")
    p.add_argument("--token", default=None)
    p.add_argument("--local-root", default=None,
                   help="read shards from this local directory instead of the Hub")
    p.add_argument("--only", choices=["features", "images", "annotations"], action="append")
    p.add_argument("--datasets", action="append",
                   help="restore only these feature datasets (COCO, GQA, VG, nlvr)")
    p.add_argument("--keep-cache", action="store_true", help="do not delete downloaded shards")
    p.add_argument("--verify", type=int, default=25, help="crc-check this many non-h5 files (0 disables)")
    p.add_argument("--limit-shards", type=int, default=0,
                   help="smoke test: restore at most this many shards, then stop")
    args = p.parse_args()

    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    only = set(args.only or ["features", "images", "annotations"])

    man_path, man_dl = fetch(args.repo_id, "manifest.json.gz", args.local_root, args.revision, args.token)
    with gzip.open(man_path, "rt") as fh:
        manifest = json.load(fh)

    by_shard = defaultdict(list)
    for rec in manifest["files"]:
        by_shard[rec["shard"]].append(rec)

    todo = sorted(by_shard)
    if "annotations" not in only:
        todo = [s for s in todo if not s.startswith("annotations/")]
    if "images" not in only:
        todo = [s for s in todo if not s.startswith("images/")]
    if "features" not in only:
        todo = [s for s in todo if not s.startswith("features/")]
    if args.datasets:
        keep = set(args.datasets)
        todo = [s for s in todo
                if not s.startswith("features/") or s.split("/")[1] in keep]

    if args.limit_shards:
        todo = todo[: args.limit_shards]
        print(f"[restore] SMOKE TEST: limited to {len(todo)} shards")
    print(f"[restore] {len(todo)} shards -> {out_root}")
    for n, shard in enumerate(todo, 1):
        recs = by_shard[shard]
        # Cheap resume: skip a shard whose outputs are all present.
        if all((out_root / r["path"]).exists() for r in recs):
            print(f"  [{n}/{len(todo)}] {shard}: already restored")
            continue

        path, downloaded = fetch(args.repo_id, shard, args.local_root, args.revision, args.token)
        if shard.startswith("annotations/"):
            dest = out_root / recs[0]["path"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dest)
            print(f"  [{n}/{len(todo)}] {shard} -> {recs[0]['path']}")
        elif shard.startswith("images/"):
            total, written = restore_images(path, out_root)
            print(f"  [{n}/{len(todo)}] {shard}: {written}/{total} png written")
        else:
            _, dataset, variant, _ = shard.split("/")
            total, written = restore_features(path, dataset, variant, out_root)
            print(f"  [{n}/{len(todo)}] {shard}: {written}/{total} h5 written")
        drop(path, downloaded, args.keep_cache)

    for rel in EMPTY_DIRS:
        (out_root / rel).mkdir(parents=True, exist_ok=True)
    drop(man_path, man_dl, args.keep_cache)

    if args.verify and not args.datasets and not args.limit_shards \
            and only == {"features", "images", "annotations"}:
        ok = verify(out_root, manifest, args.verify)
        if not ok:
            return 1
    print(NLVR2_NOTE)
    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
