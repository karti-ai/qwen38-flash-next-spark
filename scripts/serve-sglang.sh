#!/usr/bin/env bash
# Serve Qwen3.8-Flash-Next with SGLang on one GB10.
#
# ⚠️  STATUS: EXPERIMENTAL / NOT YET WORKING ON A SINGLE SPARK.
#     Read docs/SGLANG.md before running this. The blocker is not SGLang support
#     for the architecture — it is that SGLang has no equivalent of the vLLM PLE
#     mmap patch, so the 44 GiB n-gram table is resident and the model does not
#     fit beside a KV cache on 121.7 GiB. This script exists so the attempt is
#     reproducible and so the failure mode is recorded, not hidden.
#
# ⚠️  THE SILENT-CORRUPTION TRAP, from RadixArk's own qualification notes:
#     "Loaders that only upcast the FP8 bytes will serve wrong PLE embeddings
#      silently." The PLE table ships as F8_E4M3 shards plus ONE scalar
#     `weight_scale`. A loader must DEQUANTIZE with that scale. One that merely
#     widens FP8 -> BF16 produces a server that runs, answers fluently, and is
#     wrong. Run scripts/smoke-test.sh before believing anything.
#
#   WEIGHTS=<dir>   REQUIRED
#   IMAGE=...       SGLang image with qwen4_exp support (default lmsysorg/sglang:latest)
#   PORT=8000       host port
#   CTX=262144      context length
#   MEM_FRAC=0.90   --mem-fraction-static. NOTE: on SGLang this is a fraction of
#                   TOTAL memory reserved as a STATIC pool at boot, not of what
#                   is free. A too-small pool fails at BOOT with an error that
#                   names the wrong remedy.
set -euo pipefail

IMAGE="${IMAGE:-lmsysorg/sglang:latest}"
NAME="${NAME:-qwen38-flash-next-sglang}"
PORT="${PORT:-8000}"
CTX="${CTX:-262144}"
MEM_FRAC="${MEM_FRAC:-0.90}"
EXTRA="${EXTRA:-}"

if [ -z "${WEIGHTS:-}" ]; then
  echo "!! set WEIGHTS=/path/to/Qwen3.8-Flash-Next-NVFP4"
  exit 1
fi
WEIGHTS="$(cd "$WEIGHTS" && pwd)"

cat <<'WARN'
================================================================================
  This path is NOT known to work on a single 128 GB GB10 yet.
  Expect the n-gram table to be resident and the box to run out of memory.
  See docs/SGLANG.md for what would have to change. Proceeding anyway...
================================================================================
WARN

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" \
  --gpus all --ipc=host --shm-size 32g --network host \
  -v "$WEIGHTS:/model:ro" \
  "$IMAGE" \
  sglang serve --model-path /model \
    --served-model-name qwen3.8-flash-next \
    --host 0.0.0.0 --port "$PORT" \
    --attention-backend flashinfer \
    --context-length "$CTX" \
    --mem-fraction-static "$MEM_FRAC" \
    --trust-remote-code \
    $EXTRA

echo ">> $NAME starting on :$PORT"
echo ">> follow:  docker logs -f $NAME"
echo ">> then:    PORT=$PORT scripts/smoke-test.sh   # DO run this — see the header"
