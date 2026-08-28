#!/usr/bin/env bash
# Serve Qwen3.8-Flash-Next on ONE GB10 (DGX Spark / GX10) with vLLM.
#
#   WEIGHTS=<dir>   REQUIRED. Local checkpoint directory (see download-weights.sh).
#   NAME=...        container name          (default qwen38-flash-next)
#   IMAGE=...       image built from ./Dockerfile (default qwen38-flash-next-spark)
#   PORT=8000       host port
#   CTX=262144      max model len. Native is 262144; YARN=1 reaches ~500k.
#   SEQS=8          --max-num-seqs. READ THE WARNING BELOW BEFORE LOWERING THIS.
#   GPU_MEM=0.85    fraction of the unified pool for weights+KV.
#                   0.875 was OOM-killed on a 300k prefill with MTP. Keep margin.
#   MTP=2           speculative tokens from the model's own MTP head (0 = off).
#   KV_DTYPE=auto   KEEP auto (= bf16). The QSA layers REFUSE fp8 KV.
#   PREWARM=0       1 = stream the 48 GiB table once at boot to warm the page cache.
#   WORKERS=32      mmap gather threads.
#   YARN=0          1 = enable YaRN rope scaling past the native context.
#   EXTRA=          extra vllm flags, passed verbatim.
#
# ⚠️  SEQS IS A THROUGHPUT CLIFF, NOT A TUNING KNOB. A low --max-num-seqs silently
#     queues requests: saturation and a request cap are indistinguishable if you
#     look at tok/s alone. A value of 2 costs ~4x aggregate throughput at c=8.
#     To tell them apart, read  vllm:request_queue_time_seconds_sum  from /metrics.
#
# ⚠️  NO PREFIX CACHING (GB10 GDN kernel corruption) and no torch.compile
#     (Inductor int64 indexing on sm_121). Both are upstream GB10 bugs, not
#     preferences. Do not "optimise" them back on.
set -euo pipefail

NAME="${NAME:-qwen38-flash-next}"
IMAGE="${IMAGE:-qwen38-flash-next-spark}"
PORT="${PORT:-8000}"
CTX="${CTX:-262144}"
SEQS="${SEQS:-8}"
GPU_MEM="${GPU_MEM:-0.85}"
MTP="${MTP:-2}"
KV_DTYPE="${KV_DTYPE:-auto}"
PREWARM="${PREWARM:-0}"
WORKERS="${WORKERS:-32}"
YARN="${YARN:-0}"
EXTRA="${EXTRA:-}"

if [ -z "${WEIGHTS:-}" ]; then
  echo "!! set WEIGHTS=/path/to/Qwen3.8-Flash-Next-NVFP4 (see scripts/download-weights.sh)"
  exit 1
fi
WEIGHTS="$(cd "$WEIGHTS" && pwd)"
[ -f "$WEIGHTS/config.json" ] || { echo "!! no config.json in $WEIGHTS"; exit 1; }

# The PLE gather is a CPU op plus a pageable host->device copy, so it MUST run
# outside CUDA graphs. Declare it a splitting op and capture PIECEWISE (never FULL*).
SPLIT='["vllm::unified_attention_with_output","vllm::unified_mla_attention_with_output","vllm::mamba_mixer2","vllm::mamba_mixer","vllm::short_conv","vllm::qwen3_8_flash_next_ple_short_conv","vllm::qwen3_8_flash_next_qsa_with_output","vllm::linear_attention","vllm::qwen_gdn_attention_core","vllm::qwen_gdn_attention_core_fused_norm_packed","vllm::sparse_attn_indexer","vllm::ple_mmap_lookup"]'

OVR=()
ALLOW_LONG=0
if [ "$YARN" != 0 ]; then
  OVR=(--hf-overrides '{"text_config": {"rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10], "rope_type": "yarn", "rope_theta": 10000000, "partial_rotary_factor": 0.25, "factor": 4.0, "original_max_position_embeddings": 262144}}}')
  ALLOW_LONG=1
fi

# MTP + YaRN: dict hf_overrides do not propagate to the draft model, whose
# max_model_len then stays at the native value and vLLM aborts. Force it through
# the speculative config.
SPEC=()
if [ "$MTP" != 0 ]; then
  if [ "$YARN" != 0 ]; then
    SPEC=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP},\"max_model_len\":${CTX}}")
  else
    SPEC=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP}}")
  fi
fi

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --restart unless-stopped \
  --gpus all --ipc=host --shm-size 16g -p "${PORT}:8000" \
  -v "$WEIGHTS:/model:ro" \
  -e VLLM_PLE_MMAP=1 \
  -e VLLM_PLE_MMAP_WORKERS="$WORKERS" \
  -e VLLM_PLE_MMAP_PREWARM="$PREWARM" \
  -e VLLM_USE_FLASHINFER_SAMPLER=1 \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN="$ALLOW_LONG" \
  "$IMAGE" \
  /model --served-model-name qwen3.8-flash-next \
    --host 0.0.0.0 --port 8000 --load-format safetensors \
    --max-model-len "$CTX" --max-num-seqs "$SEQS" \
    --gpu-memory-utilization "$GPU_MEM" \
    --no-enable-prefix-caching --enable-chunked-prefill \
    --max-num-batched-tokens 8192 \
    -cc.cudagraph_mode=PIECEWISE "-cc.splitting_ops=$SPLIT" \
    --no-enable-flashinfer-autotune \
    --kv-cache-dtype "$KV_DTYPE" \
    "${OVR[@]}" $EXTRA \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 \
    "${SPEC[@]}"

echo ">> $NAME starting on :$PORT  (ctx $CTX · seqs $SEQS · mtp $MTP · yarn $YARN)"
echo ">> first boot reads ~76 GiB of weights, roughly 8 minutes."
echo ">> follow:  docker logs -f $NAME"
echo ">> ready when the log says 'Application startup complete', then:"
echo "     PORT=$PORT scripts/smoke-test.sh"
