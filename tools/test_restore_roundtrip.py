#!/usr/bin/env python3
"""
End-to-end test of the publish/restore round trip, at a size you can actually run.

Builds a miniature mirror of the real dataset out of the original archive, pushes
it to a throwaway *private* Hub repo, restores it with the plain documented
command, and checks the result against the archive. Then deletes the temp repo.

    python tools/test_restore_roundtrip.py --zip VLadapter.zip --namespace ylsung

Why a temp repo rather than testing against the published dataset: the
completeness check inside restore_datasets.py only runs when --datasets,
--limit-shards and --only are all absent, i.e. only for the bare command, and
running that against the real repo means restoring ~132 GB. A small mirror
exercises the same code path in a couple of minutes.

Checks, in order:
  1. bare `restore_datasets.py --out ./datasets` succeeds and its own
     manifest verification passes
  2. every restored .h5 is bit-exact vs the archive, read with the same access
     pattern as VL-T5/src/*_clip_data.py  (f[f"{img_id}/features"][...])
  3. every restored annotation is byte-identical
  4. directory shape matches, including the empty dirs the archive carries
  5. a second run is idempotent (downloads and rewrites nothing)
  6. deleting files mid-tree and rerunning restores exactly those
  7. the download cache is left ~empty (shards deleted after expanding)

Exits non-zero on the first failure. Pass --keep-repo to leave the temp repo for
inspection.
"""
import argparse
import io
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import h5py
import numpy as np

ROOT = "vlt5_dataset/"
TOOLS = Path(__file__).resolve().parent
FAILURES = []


def check(label, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"  — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(label)
    return ok


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def build_mini(src_zip, dest_zip, per_dataset, ann_max_bytes):
    """A small archive shaped like the real one: features + annotations, no images."""
    src = zipfile.ZipFile(src_zip)
    feats, others, dirs = {}, [], []
    for info in src.infolist():
        name = info.filename
        if name.endswith("/"):
            dirs.append(info)
            continue
        rel = name[len(ROOT):]
        if "/clip_features/" in rel and rel.endswith(".h5"):
            feats.setdefault(rel.split("/")[0], []).append(info)
        elif rel.startswith("nlvr/images/"):
            continue  # the published dataset ships no images; mirror that
        elif info.file_size <= ann_max_bytes:
            others.append(info)
    picked = list(dirs) + others
    for group in feats.values():
        picked += group[:per_dataset]
    with zipfile.ZipFile(dest_zip, "w", zipfile.ZIP_DEFLATED) as out:
        for info in picked:
            if info.filename.endswith("/"):
                out.writestr(zipfile.ZipInfo(info.filename), b"")
            else:
                out.writestr(info, src.read(info))
    return dest_zip


def compare_tree(mini_zip, tree):
    """Bit-exactness of features, byte-exactness of everything else."""
    z = zipfile.ZipFile(mini_zip)
    n_h5 = n_other = bad = missing = 0
    for info in z.infolist():
        if info.filename.endswith("/"):
            continue
        rel = info.filename[len(ROOT):]
        dest = tree / rel
        if not dest.exists():
            missing += 1
            continue
        orig = z.read(info)
        if rel.endswith(".h5"):
            stem = Path(rel).stem
            with h5py.File(io.BytesIO(orig), "r") as fh:
                ref = fh[f"{stem}/features"][...]
            with h5py.File(dest, "r") as fh:          # the loaders' access pattern
                got = fh[f"{stem}/features"][...]
            if not (got.shape == ref.shape == (49, 2048)
                    and got.dtype == ref.dtype == np.float16
                    and np.array_equal(got, ref)):
                bad += 1
            n_h5 += 1
        else:
            if dest.read_bytes() != orig:
                bad += 1
            n_other += 1
    return n_h5, n_other, bad, missing


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--zip", required=True, help="the original VLadapter.zip")
    p.add_argument("--namespace", required=True, help="Hub user or org for the temp repo")
    p.add_argument("--repo-name", default="vladapter-restore-smoketest")
    p.add_argument("--per-dataset", type=int, default=200, help="h5 files per dataset")
    p.add_argument("--shard-rows", type=int, default=64)
    p.add_argument("--ann-max-bytes", type=int, default=2_500_000,
                   help="skip annotations larger than this, to keep the test quick")
    p.add_argument("--keep-repo", action="store_true")
    p.add_argument("--workdir", default=None)
    args = p.parse_args()

    from huggingface_hub import HfApi

    repo_id = f"{args.namespace}/{args.repo_name}"
    work = Path(args.workdir or tempfile.mkdtemp(prefix="vladapter-test-"))
    work.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    py = sys.executable
    print(f"workdir : {work}\ntemp repo: {repo_id} (private)\n")

    try:
        print("[1/7] building miniature archive")
        mini = build_mini(args.zip, work / "mini.zip", args.per_dataset, args.ann_max_bytes)
        print(f"       {mini.stat().st_size/1e6:.1f} MB")

        print("[2/7] converting to the Hub layout")
        staging = work / "hf"
        r = run([py, str(TOOLS / "convert_zip_to_hf.py"), "--zip", str(mini),
                 "--out", str(staging), "--shard-rows", str(args.shard_rows),
                 "--only", "annotations", "--only", "features"])
        if r.returncode:
            print(r.stdout[-2000:], r.stderr[-2000:])
            return check("convert", False, "see output above") or 1
        shards = list(staging.rglob("*.parquet"))
        print(f"       {len(shards)} shards")

        print("[3/7] pushing the mirror to a private temp repo")
        api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
        api.upload_folder(folder_path=str(staging), repo_id=repo_id, repo_type="dataset",
                          ignore_patterns=[".convert_state*", "*.tmp"],
                          commit_message="Miniature mirror for restore testing")

        print("[4/7] bare restore: restore_datasets.py --out ./datasets")
        tree = work / "datasets"
        r = run([py, str(TOOLS / "restore_datasets.py"), "--out", str(tree),
                 "--repo-id", repo_id])
        out = r.stdout + r.stderr
        check("bare command exits 0", r.returncode == 0)
        check("manifest verification ran and passed",
              "missing=0, size-mismatch=0, crc-mismatch=0" in out,
              next((l.strip() for l in out.splitlines() if "[verify]" in l), "no [verify] line"))

        print("[5/7] comparing the restored tree against the archive")
        n_h5, n_other, bad, missing = compare_tree(mini, tree)
        check("every .h5 bit-exact", bad == 0 and missing == 0,
              f"{n_h5} h5 + {n_other} annotations, {bad} mismatched, {missing} missing")

        for rel in ["paragraphs", "COCO/clip_features/data_clip_RN101_fc", "nlvr/images"]:
            check(f"empty dir preserved: {rel}", (tree / rel).is_dir())
        for ds in ["COCO", "GQA", "VG", "nlvr"]:
            d = tree / ds / "clip_features" / "data_clip_RN101_att"
            check(f"loader path exists: {ds}", d.is_dir() and any(d.glob("*.h5")),
                  f"{len(list(d.glob('*.h5')))} .h5")

        print("[6/7] second run must be idempotent")
        r2 = run([py, str(TOOLS / "restore_datasets.py"), "--out", str(tree),
                  "--repo-id", repo_id])
        o2 = r2.stdout + r2.stderr
        check("nothing rewritten on rerun", "h5 written" not in o2,
              f"{o2.count('already restored')} shards skipped")

        print("[7/7] resume after a simulated interruption")
        gqa = sorted((tree / "GQA/clip_features/data_clip_RN101_att").glob("*.h5"))
        removed = gqa[: args.shard_rows]
        for f in removed:
            f.unlink()
        (tree / "nlvr/test.json").unlink(missing_ok=True)
        r3 = run([py, str(TOOLS / "restore_datasets.py"), "--out", str(tree),
                  "--repo-id", repo_id])
        rewritten = sum(int(l.split(":")[-1].split("/")[0].strip())
                        for l in (r3.stdout + r3.stderr).splitlines() if "h5 written" in l)
        check("resume rewrote exactly the deleted files", rewritten == len(removed),
              f"deleted {len(removed)}, rewrote {rewritten}")
        check("deleted annotation restored", (tree / "nlvr/test.json").exists())

        cache = Path.home() / ".cache/huggingface/hub"
        size = sum(f.stat().st_size for f in cache.rglob("*") if f.is_file()) if cache.exists() else 0
        check("download cache left ~empty", size < 50e6, f"{size/1e6:.1f} MB")

    finally:
        if not args.keep_repo:
            try:
                api.delete_repo(repo_id, repo_type="dataset")
                print(f"\ncleaned up {repo_id}")
            except Exception as exc:  # noqa: BLE001
                print(f"\ncould not delete {repo_id}: {exc}")
        if not args.workdir:
            shutil.rmtree(work, ignore_errors=True)

    print()
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): " + "; ".join(FAILURES))
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
