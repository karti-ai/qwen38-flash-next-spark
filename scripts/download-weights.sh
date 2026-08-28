#!/usr/bin/env bash
# Download the Qwen3.8-Flash-Next NVFP4 checkpoint, then verify it byte-for-byte.
#
# ~122 GiB over 419 files. Budget ~130 GiB of free disk and put it on NVMe:
# the n-gram table is served from this directory at runtime through the page
# cache, so its read latency is on the token path.
#
#   REPO=<hf repo>   default RadixArk/Qwen3.8-Flash-Next-NVFP4
#   DEST=<dir>       default ./weights/Qwen3.8-Flash-Next-NVFP4
#   WORKERS=<n>      default 6   parallel file downloads
#   RETRIES=<n>      default 5   whole-download retry attempts
#   SKIP_VERIFY=1    skip the checksum pass (NOT recommended — read the script)
set -uo pipefail

REPO="${REPO:-RadixArk/Qwen3.8-Flash-Next-NVFP4}"
DEST="${DEST:-$(pwd)/weights/Qwen3.8-Flash-Next-NVFP4}"
WORKERS="${WORKERS:-6}"
RETRIES="${RETRIES:-5}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

command -v hf >/dev/null 2>&1 || {
  echo "!! the 'hf' CLI is not on PATH. Install it with:  pip install -U huggingface_hub"
  exit 1
}

mkdir -p "$DEST"
avail_gb=$(df -PBG "$DEST" | awk 'NR==2 {gsub("G","",$4); print $4}')
if [ "${avail_gb:-0}" -lt 130 ]; then
  echo "!! only ${avail_gb} GiB free at $DEST — this checkpoint needs ~130 GiB."
  exit 1
fi

echo ">> downloading $REPO -> $DEST"
ok=0
for i in $(seq 1 "$RETRIES"); do
  echo "== attempt $i/$RETRIES $(date -Is)"
  if hf download "$REPO" --local-dir "$DEST" --max-workers "$WORKERS"; then
    ok=1
    break
  fi
  echo "-- attempt $i failed; retrying in 30s"
  sleep 30
done

if [ "$ok" != 1 ]; then
  echo "!! download did not complete after $RETRIES attempts"
  exit 1
fi

if [ "${SKIP_VERIFY:-0}" = 1 ]; then
  echo ">> SKIP_VERIFY=1 — skipping checksums. If the model emits fluent"
  echo "   nonsense later, come back and run scripts/verify-weights.py first."
  exit 0
fi

echo ">> verifying every shard against the Hub's published lfs.sha256"
echo "   (this is not optional paranoia — see the header of verify-weights.py)"
exec python3 "$HERE/verify-weights.py" --repo "$REPO" --dir "$DEST"
