#!/usr/bin/env bash
# Correctness canaries. Run these before believing any performance number.
#
#   PORT=8000  MODEL=qwen3.8-flash-next  BASE=http://localhost:$PORT/v1
#   THINK=0    1 = leave the model's reasoning block enabled (see below)
#
# WHY CANARIES AND NOT "does it respond": a byte-corrupt shard, a wrong KV dtype,
# or a broken PLE quant gate all produce a server that answers fluently and
# wrongly. Each check below has a single unguessable right answer.
#
# ⚠️  THINKING IS ON BY DEFAULT AND WILL EAT A SMALL TOKEN BUDGET WHOLE.
#     With --reasoning-parser qwen3, the chain of thought goes to `reasoning`
#     and `content` comes back NULL if the budget runs out first. Measured on
#     this model: max_tokens=24 -> reasoning_tokens=24, content=null,
#     finish_reason="length". No error. A caller that reads only `content` sees
#     an empty string and concludes the model is broken when it is not — the
#     reasoning field held the correct answer the whole time.
#
#     So these canaries send chat_template_kwargs {"enable_thinking": false}.
#     Set THINK=1 to exercise the reasoning path instead, which needs a much
#     larger budget (~512 tokens for one arithmetic question).
set -uo pipefail

PORT="${PORT:-8000}"
MODEL="${MODEL:-qwen3.8-flash-next}"
BASE="${BASE:-http://localhost:${PORT}/v1}"
THINK="${THINK:-0}"
fail=0

if [ "$THINK" = 1 ]; then
  KW='{}'
  BUDGET_MULT=20
else
  KW='{"enable_thinking": false}'
  BUDGET_MULT=1
fi

# Emits: <finish_reason>\t<content>. Content of a null field is reported as the
# literal NULL_CONTENT so a check can never accidentally match emptiness.
ask() {  # ask <prompt> <max_tokens>
  local n=$(( $2 * BUDGET_MULT ))
  curl -s --max-time 600 "$BASE/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "$(jq -nc --arg m "$MODEL" --arg p "$1" --argjson n "$n" --argjson kw "$KW" \
          '{model:$m,messages:[{role:"user",content:$p}],max_tokens:$n,temperature:0,chat_template_kwargs:$kw}')" \
    | jq -r '.choices[0] | "\(.finish_reason)\t\(.message.content // "NULL_CONTENT")"'
}

check() {  # check <label> <expected-regex> <prompt> <max_tokens>
  local label="$1" want="$2" prompt="$3" n="$4" raw finish got
  raw="$(ask "$prompt" "$n")"
  finish="${raw%%$'\t'*}"
  got="$(printf '%s' "${raw#*$'\t'}" | tr -d '\n' | sed 's/^ *//;s/ *$//')"

  if [ "$got" = "NULL_CONTENT" ]; then
    printf '  [FAIL] %-18s content was NULL (finish_reason=%s)\n' "$label" "$finish"
    if [ "$finish" = "length" ]; then
      printf '         the reasoning block consumed the whole budget — this is NOT a\n'
      printf '         broken model. Raise max_tokens or set enable_thinking false.\n'
    fi
    fail=$((fail+1))
    return
  fi
  if printf '%s' "$got" | grep -qiE "$want"; then
    printf '  [PASS] %-18s %s\n' "$label" "${got:0:64}"
  else
    printf '  [FAIL] %-18s want /%s/  got: %s\n' "$label" "$want" "${got:0:80}"
    fail=$((fail+1))
  fi
}

command -v jq >/dev/null || { echo "!! needs jq"; exit 1; }

echo ">> $BASE   (thinking $([ "$THINK" = 1 ] && echo ON || echo OFF))"
if ! curl -sf --max-time 10 "$BASE/models" >/dev/null; then
  echo "!! server not answering on $BASE — still loading? (first boot is ~15-20 min)"
  exit 1
fi
echo ">> serving: $(curl -s "$BASE/models" | jq -r '.data[].id' | paste -sd, -)"
echo

# Arithmetic the model cannot pattern-match. A quantisation or KV-dtype
# regression shows up here first, as a near-miss rather than as nonsense.
check "arithmetic"      '437'         'What is 19 multiplied by 23? Reply with only the number.' 24
check "arithmetic-2"    '2352|2,352'  'What is 48 multiplied by 49? Reply with only the number.' 24

# A trick question a degraded model answers with the tempting wrong number (8).
check "reasoning-trap"  '\b9\b'       'A farmer has 17 sheep. All but 9 die. How many are left? Reply with only the number.' 32

# Instruction following.
check "instruction"     'acknowledged' 'Reply with exactly one word: acknowledged' 16

# Code generation, checked for real tokens rather than vibes.
check "code"            'def.*fib|return' 'Write an iterative Python function fib(n) returning the nth Fibonacci number. Code only.' 220

echo
if [ "$fail" -eq 0 ]; then
  echo "all canaries passed."
  echo "next:  python3 scripts/vision-test.py --base-url $BASE"
else
  echo "$fail canary/canaries FAILED."
  echo "If content was NULL, that is a budget problem, not corruption — see the header."
  echo "Otherwise verify the weights before touching config:"
  echo "  python3 scripts/verify-weights.py --repo RadixArk/Qwen3.8-Flash-Next-NVFP4 --dir <weights>"
  exit 1
fi
