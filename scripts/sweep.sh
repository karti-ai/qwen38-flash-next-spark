#!/usr/bin/env bash
# Run a ladder of serving configs back to back and record each one.
#
# Every config needs a full restart (~12 min: ~10 min to read 79 GiB of weights,
# then compile and graph capture), so a five-config ladder is about an hour.
# Run it unattended and read results/ afterwards rather than watching it.
#
#   WEIGHTS=<dir>  REQUIRED
#   OUT=results    where the JSON lands
#   CONC=1,4,8     concurrency levels passed to bench.py
#   CONFIGS=...    space-separated subset of the ladder below (default: all)
#
# Each entry is  name|env assignments  and is applied on top of serve-vllm.sh's
# defaults. The point of the ladder is to isolate ONE variable at a time; keep
# it that way when you add to it.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
OUT="${OUT:-$ROOT/results}"
CONC="${CONC:-1,4,8}"
PORT="${PORT:-8000}"
NAME="${NAME:-qwen38-flash-next}"
BOOT_TIMEOUT="${BOOT_TIMEOUT:-1800}"

[ -n "${WEIGHTS:-}" ] || { echo "!! set WEIGHTS=/path/to/checkpoint"; exit 1; }
mkdir -p "$OUT"

# name                     | environment for serve-vllm.sh
# ⚠️ GPU_MEM HAS A HARD FLOOR AT THE WEIGHT FRACTION. The pool bounded by
# --gpu-memory-utilization holds weights AND KV. Weights here are 79.42 GiB of
# 121.7 = 0.653, so anything at or below ~0.66 computes a NEGATIVE KV cache and
# dies at boot with "Available KV cache memory: -6.78 GiB". Do not go under 0.70.
#
# ⚠️ AND NOTE WHAT DOES *NOT* WORK: lowering --max-model-len does not give the
# page cache more room. max-model-len caps how long ONE request may be; the KV
# pool size is set by gpu-memory-utilization alone. If you are trying to leave
# RAM for the n-gram table to page through, GPU_MEM is the only knob that moves
# it. We learned this the expensive way — see docs/BENCHMARKS.md.
LADDER=(
  "gpu085-mtp2-nopin  | CTX=32768 GPU_MEM=0.85 MTP=2 CPUSET="
  "gpu085-mtp2-pin    | CTX=32768 GPU_MEM=0.85 MTP=2"
  "gpu072-mtp2-pin    | CTX=32768 GPU_MEM=0.72 MTP=2"
  "gpu072-mtp2-warm   | CTX=32768 GPU_MEM=0.72 MTP=2 PREWARM=1"
  "gpu072-mtp1-pin    | CTX=32768 GPU_MEM=0.72 MTP=1"
  "gpu072-mtp0-pin    | CTX=32768 GPU_MEM=0.72 MTP=0"
)

wait_healthy() {
  local deadline=$(( $(date +%s) + BOOT_TIMEOUT ))
  while [ "$(date +%s)" -lt "$deadline" ]; do
    if curl -sf --max-time 5 "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
      return 0
    fi
    # A dead container will never become healthy; fail fast instead of waiting out
    # the full timeout on a config that crashed at boot.
    if ! docker ps --format '{{.Names}}' | grep -q "^${NAME}$"; then
      echo "   !! container is gone"
      return 1
    fi
    sleep 20
  done
  echo "   !! timed out after ${BOOT_TIMEOUT}s"
  return 1
}

for entry in "${LADDER[@]}"; do
  name="$(echo "${entry%%|*}" | xargs)"
  env_str="$(echo "${entry#*|}" | xargs)"

  if [ -n "${CONFIGS:-}" ] && ! echo " $CONFIGS " | grep -q " $name "; then
    continue
  fi

  echo
  echo "================================================================"
  echo "  $name"
  echo "  $env_str"
  echo "================================================================"

  docker rm -f "$NAME" >/dev/null 2>&1 || true
  sleep 5
  # Drop the page cache between configs so each one is measured from the same
  # cold start. Without this, a config inherits the previous run's warm n-gram
  # pages and looks faster than it is.
  sync; sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null \
    || echo "   (no sudo: page cache not dropped, results are warm-biased)"

  started=$(date +%s)
  # shellcheck disable=SC2086
  if ! env WEIGHTS="$WEIGHTS" PORT="$PORT" NAME="$NAME" $env_str \
       "$HERE/serve-vllm.sh" >/dev/null 2>&1; then
    echo "   !! serve-vllm.sh failed to launch"
    continue
  fi

  if ! wait_healthy; then
    docker logs "$NAME" 2>&1 | tail -20 > "$OUT/$name.bootfail.log" || true
    echo "   !! boot failed; log in $OUT/$name.bootfail.log"
    continue
  fi
  boot=$(( $(date +%s) - started ))
  echo "   booted in ${boot}s"

  # Correctness before speed, every single time. A config that serves garbage
  # fast is not a faster config.
  if ! PORT="$PORT" "$HERE/smoke-test.sh" >"$OUT/$name.smoke.txt" 2>&1; then
    echo "   !! CANARIES FAILED — recording and skipping the benchmark"
    continue
  fi
  echo "   canaries pass"

  python3 "$HERE/bench.py" --base-url "http://localhost:${PORT}/v1" \
      --concurrency "$CONC" --max-tokens 256 --out "$OUT/$name.json" \
    | tee "$OUT/$name.txt"

  # Record what the OS thought was going on, since paging is the whole story here.
  {
    echo "config: $name"
    echo "env: $env_str"
    echo "boot_seconds: $boot"
    free -g | head -2
    pid=$(docker inspect -f '{{.State.Pid}}' "$NAME" 2>/dev/null)
    [ -n "$pid" ] && grep -E '^(VmRSS|VmSwap)' "/proc/$pid/status" 2>/dev/null
    [ -n "$pid" ] && awk '{print "majflt: " $12}' "/proc/$pid/stat" 2>/dev/null
    curl -s "http://localhost:${PORT}/metrics" 2>/dev/null \
      | grep -E '^vllm:spec_decode_num_(draft|accepted)_tokens_total' || true
  } > "$OUT/$name.sysinfo.txt"
done

echo
echo "=== ladder complete — results in $OUT ==="
grep -H . "$OUT"/*.txt 2>/dev/null | grep -E "^\S+: +[0-9]+ " || true
