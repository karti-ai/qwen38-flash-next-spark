#!/usr/bin/env bash
# Correctness canaries. Run these before believing any performance number.
#
#   PORT=8000  MODEL=qwen3.8-flash-next  BASE=http://localhost:$PORT/v1
#
# WHY CANARIES AND NOT "does it respond": a byte-corrupt shard, a wrong KV dtype,
# or a broken quant gate all produce a server that answers fluently and wrongly.
# Each check below has a single unguessable right answer.
set -uo pipefail

PORT="${PORT:-8000}"
MODEL="${MODEL:-qwen3.8-flash-next}"
BASE="${BASE:-http://localhost:${PORT}/v1}"
fail=0

ask() {  # ask <prompt> <max_tokens>
  curl -s --max-time 300 "$BASE/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "$(jq -nc --arg m "$MODEL" --arg p "$1" --argjson n "$2" \
          '{model:$m,messages:[{role:"user",content:$p}],max_tokens:$n,temperature:0}')" \
    | jq -r '.choices[0].message.content // "<no content>"'
}

check() {  # check <label> <expected-regex> <prompt> <max_tokens>
  local label="$1" want="$2" prompt="$3" n="$4" got
  got="$(ask "$prompt" "$n" | tr -d '\n' | sed 's/^ *//;s/ *$//')"
  if printf '%s' "$got" | grep -qiE "$want"; then
    printf '  [PASS] %-22s %s\n' "$label" "${got:0:70}"
  else
    printf '  [FAIL] %-22s want /%s/  got: %s\n' "$label" "$want" "${got:0:90}"
    fail=$((fail+1))
  fi
}

command -v jq >/dev/null || { echo "!! needs jq"; exit 1; }

echo ">> $BASE"
if ! curl -sf --max-time 10 "$BASE/models" >/dev/null; then
  echo "!! server not answering on $BASE — is it still loading? (~8 min first boot)"
  exit 1
fi
echo ">> serving: $(curl -s "$BASE/models" | jq -r '.data[].id' | paste -sd, -)"
echo

# Arithmetic the model cannot pattern-match its way through. A quantisation or
# KV-dtype regression shows up here first, as a near-miss rather than nonsense.
check "arithmetic"      '437'            'What is 19 multiplied by 23? Reply with only the number.' 24
check "arithmetic-2"    '2352|2,352'     'What is 48 multiplied by 49? Reply with only the number.' 24

# A trick question that a degraded model answers with the tempting wrong number.
check "reasoning-trap"  '\b9\b'          'A farmer has 17 sheep. All but 9 die. How many are left? Reply with only the number.' 24

# Instruction following under a tight budget: catches a reasoning block eating
# the whole token allowance and returning empty content.
check "short-budget"    '[A-Za-z]'       'Say the word: acknowledged' 16

# Code generation, checked for a real token rather than vibes.
check "code"            'def|return'     'Write an iterative Python function fib(n) returning the nth Fibonacci number. Code only.' 200

echo
if [ "$fail" -eq 0 ]; then
  echo "all canaries passed."
  echo "next:  python3 scripts/vision-test.py --base-url $BASE"
else
  echo "$fail canary/canaries FAILED."
  echo "Before touching config: verify the weights."
  echo "  python3 scripts/verify-weights.py --repo RadixArk/Qwen3.8-Flash-Next-NVFP4 --dir <weights>"
  exit 1
fi
