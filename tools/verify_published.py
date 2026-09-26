#!/usr/bin/env python3
"""
Verify the published HuggingFace dataset against the original archive.

Downloads a random sample of parquet shards from the Hub, expands them the same
way `restore_datasets.py` does, and compares every row against the corresponding
entry read straight out of `VLadapter.zip`. Feature arrays must match bit-for-bit
and annotation files byte-for-byte.

    python verify_published.py --zip VLadapter.zip --shards 6 --annotations 5

Each shard is deleted after checking, so disk use stays around one shard.
"""
import argparse
import io
import random
import sys
import zipfile
from pathlib import Path

import h5py
import numpy as np
import pyarrow.parquet as pq

ROOT = "vlt5_dataset/"
FEAT_SHAPE = (49, 2048)


def drop(path):
    try:
        real = Path(path).resolve()
        Path(path).unlink(missing_ok=True)
        real.unlink(missing_ok=True)
    except OSError:
        pass


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--zip", required=True, help="the original VLadapter.zip")
    p.add_argument("--repo-id", default="ylsung/VL-Adapter-datasets")
    p.add_argument("--shards", type=int, default=6, help="random feature shards to check")
    p.add_argument("--annotations", type=int, default=5, help="random annotation files to check")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    from huggingface_hub import HfApi, hf_hub_download

    rng = random.Random(args.seed)
    api = HfApi()
    files = api.list_repo_files(args.repo_id, repo_type="dataset")
    shards = sorted(f for f in files if f.startswith("features/") and f.endswith(".parquet"))
    anns = sorted(f for f in files if f.startswith("annotations/"))
    print(f"[hub] {len(shards)} feature shards, {len(anns)} annotation files")

    zf = zipfile.ZipFile(args.zip)
    by_name = {i.filename: i for i in zf.infolist()}

    rows = mismatches = missing = 0

    for shard in rng.sample(shards, min(args.shards, len(shards))):
        _, dataset, variant, _ = shard.split("/")
        local = hf_hub_download(args.repo_id, shard, repo_type="dataset")
        table = pq.read_table(local)
        ids = table.column("id").to_pylist()
        feats = table.column("features")
        bad = 0
        for i, img_id in enumerate(ids):
            orig_name = f"{ROOT}{dataset}/clip_features/{variant}/{img_id}.h5"
            info = by_name.get(orig_name)
            if info is None:
                missing += 1
                continue
            with h5py.File(io.BytesIO(zf.read(info)), "r") as fh:
                ref = fh[f"{img_id}/features"][...]
            got = np.asarray(feats[i].as_py(), dtype=np.float16).reshape(FEAT_SHAPE)
            if not (got.dtype == ref.dtype and np.array_equal(got, ref)):
                bad += 1
            rows += 1
        mismatches += bad
        print(f"  {shard}: {len(ids)} rows, {bad} mismatched")
        drop(local)

    ann_bad = 0
    for ann in rng.sample(anns, min(args.annotations, len(anns))):
        orig_name = ROOT + ann[len("annotations/"):]
        info = by_name.get(orig_name)
        if info is None:
            print(f"  {ann}: NOT IN ARCHIVE")
            missing += 1
            continue
        local = hf_hub_download(args.repo_id, ann, repo_type="dataset")
        same = Path(local).read_bytes() == zf.read(info)
        ann_bad += (not same)
        print(f"  {ann}: {info.file_size/1e6:.1f} MB, {'identical' if same else 'DIFFERS'}")
        drop(local)

    print()
    print(f"feature rows compared : {rows}")
    print(f"  bit-exact           : {rows - mismatches}")
    print(f"  mismatched          : {mismatches}")
    print(f"annotations compared  : {min(args.annotations, len(anns))}, {ann_bad} differing")
    print(f"missing from archive  : {missing}")
    ok = not (mismatches or ann_bad or missing)
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
