#!/usr/bin/env python3
"""Verify a downloaded HuggingFace checkpoint against the Hub's published LFS SHA-256.

WHY THIS EXISTS
---------------
The only documented reproduction failure of the single-Spark Qwen3.8-Flash-Next
recipe was not a config problem, a driver problem, or a kernel problem. It was
two safetensors shards that were the *correct size* but the *wrong bytes*, left
behind by a stalled download.

A byte-corrupt shard is not a loud failure. It:

  * loads cleanly, with no error and no warning,
  * reports the correct tensor shapes and dtypes,
  * produces activations of correct magnitude,
  * and yields fluent, confident token salad

...that survives every configuration change you can think of — MoE backend,
attention backend, CUDA graph mode, speculative decoding, prefix caching. The
person who hit it spent a day bisecting driver and firmware versions before
checking the bytes.

Comparing file *sizes* against the Hub API does not catch this. `hf download`
resuming over a partial file does not always catch it either. The Hub publishes
`lfs.sha256` for every LFS object in the same API response most tools already
parse. Use it.

USAGE
-----
    python3 scripts/verify-weights.py \
        --repo RadixArk/Qwen3.8-Flash-Next-NVFP4 \
        --dir  /path/to/local/checkpoint

Exit code 0 = every published hash matched. Exit code 1 = at least one file is
missing or corrupt; the names are printed so you can re-fetch only those.

    --jobs N     parallel hashing workers (default 6; this is I/O bound)
    --quiet      only print the summary and any failures
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

CHUNK = 8 << 20  # 8 MiB reads: large enough to keep NVMe busy, small enough to stream


def published_hashes(repo: str) -> dict[str, str]:
    """Return {filename: sha256} for every LFS object the Hub publishes."""
    url = f"https://huggingface.co/api/models/{repo}?blobs=true"
    try:
        with urllib.request.urlopen(url, timeout=60) as fh:
            meta = json.load(fh)
    except urllib.error.HTTPError as exc:
        sys.exit(f"!! Hub API returned {exc.code} for {repo}")
    except urllib.error.URLError as exc:
        sys.exit(f"!! could not reach the Hub API: {exc.reason}")

    out: dict[str, str] = {}
    for sib in meta.get("siblings", []):
        lfs = sib.get("lfs") or {}
        # The Hub has used both keys over time; oid is sometimes "sha256:<hex>".
        sha = lfs.get("sha256") or lfs.get("oid") or ""
        sha = sha.split(":")[-1].strip().lower()
        if len(sha) == 64:
            out[sib["rfilename"]] = sha
    return out


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--repo", required=True, help="HuggingFace repo id")
    ap.add_argument("--dir", required=True, help="local checkpoint directory")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    dest = os.path.expanduser(args.dir)
    if not os.path.isdir(dest):
        sys.exit(f"!! not a directory: {dest}")

    want = published_hashes(args.repo)
    if not want:
        sys.exit(f"!! {args.repo} publishes no LFS hashes — nothing to verify")
    print(f"{args.repo}: {len(want)} LFS objects carry a published sha256")

    total = len(want)
    done = 0
    bad: list[tuple[str, str]] = []

    def check(item: tuple[str, str]) -> tuple[str, str]:
        name, sha = item
        path = os.path.join(dest, name)
        if not os.path.exists(path):
            return name, "MISSING"
        return name, ("OK" if sha256_file(path) == sha else "CORRUPT")

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for name, status in pool.map(check, sorted(want.items())):
            done += 1
            if status != "OK":
                bad.append((name, status))
                print(f"  !! {status}: {name}")
            elif not args.quiet and done % 50 == 0:
                print(f"  ... {done}/{total}")

    print(f"\nchecked {done} files — {len(bad)} bad")
    if bad:
        print("\nRe-fetch ONLY these files, then re-run this script:\n")
        for name, status in bad:
            print(f"  {status:8s} {name}")
        print(
            "\n  hf download %s --local-dir %s --include %s"
            % (args.repo, args.dir, " ".join(f"'{n}'" for n, _ in bad))
        )
        return 1

    print("ALL SHARDS VERIFIED — the checkpoint is byte-correct.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
