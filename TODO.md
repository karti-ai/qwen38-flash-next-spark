# TODO

Open work on this repo, in the order it should be done.

## 1. Re-measure everything with the fixed harness — blocking

`bench.py` used to count SSE chunks, which undercounts a speculative server by
roughly its acceptance length (measured at **3.08x** on a different model). It
now reports `usage.completion_tokens`. **Every Flash-Next number currently in
[`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) predates that fix and is an
undercount.** They are marked in place rather than scaled, because a multiplied
estimate is not a measurement.

Until this is redone, the repo cannot honestly claim how fast this model is, and
the comparison against a dense 27B is unsupported in both directions.

```bash
WEIGHTS=... scripts/serve-vllm.sh
scripts/smoke-test.sh
python3 scripts/bench.py --base-url http://localhost:8000/v1 \
    --concurrency 1,4,8,16,32,48 --out results/vllm-mtp2-fixed.json
```

Watch the new `tok/chunk` column. On this model with MTP=2 it should read
around 2.2 (`1 + accepted/drafts`, from the spec-decode counters). If it reads
~1.0, the server did not return usage and you are looking at an undercount again.

Then re-run the config ladder, since every rung in it was chunk-counted:

```bash
WEIGHTS=... scripts/sweep.sh
```

The *relative* ordering of the three ladder rungs is still valid — they were all
measured the same wrong way — so the two refuted hypotheses (CPU pinning, page
cache headroom) do not need re-litigating. Only the absolute numbers do.

## 2. `VLLM_PLE_CPU_OFFLOAD` — the one untested lever

Every optimisation we tried changed the *memory around* the n-gram gather and
none of it mattered. CPU offload changes the gather itself: the table lives in
pinned host RAM instead of being paged from NVMe. It is how upstream reached
17.1 tok/s with **no speculation at all**.

Needs `--cap-add=SYS_PTRACE` (yama `ptrace_scope=1` blocks the sibling-process
CUDA IPC handoff — see [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md)) and
the PLE quant-gate change so the FP8 PLE is accepted on an NVFP4 body.

Note that it does not escape paging: 76.6 GiB resident + 31 GiB KV leaves ~14 GiB
for a 44 GiB table, and upstream still measured 16 major faults per token. The
question is whether it is *cheaper* paging, not no paging.

## 3. Check the drafter matches the target quantisation

Learned on the dense 27B and it applies here. A speculative drafter trained
against an **FP8** target underperforms badly on an **NVFP4** target, silently:
no error, no warning, just a low acceptance rate. Symptom was accept length
1.52–2.15 and accept rate 0.07–0.16; the matched drafter gave 3.03–3.32 and
0.29–0.33.

Flash-Next uses its own in-checkpoint MTP head rather than a separate drafter, so
this may not apply — but we measured **62% acceptance / accept len ~2.24** and
never asked whether that is what this checkpoint should achieve. Find out.

## 4. MTP sweep

`num_speculative_tokens` is 2. vLLM warns that >1 re-runs the same MTP layer and
can *lower* acceptance, so 1 may beat 2. Test 0/1/2/3 and report acceptance
alongside throughput, not throughput alone.

## 5. The SGLang path

See [`docs/SGLANG.md`](docs/SGLANG.md). The blocker is not architecture support:
SGLang keeps the n-gram table resident, so the model does not fit. Porting the
mmap gather into SGLang's embedding layer is the actual contribution. Mind the
silent FP8-upcast trap documented there.

## 6. Vision follow-ups

Vision works (see [`docs/VISION.md`](docs/VISION.md)). Two loose ends:

- The OCR miss is deterministic and **moves with render size** — 78px misreads
  one glyph, 110px gets that glyph right and mangles the tail. That points at a
  preprocessor resize/tiling artifact rather than a model weakness. Worth
  isolating with a size sweep.
- Video input is untested. `video_preprocessor_config.json` ships in the
  checkpoint and nobody has tried it on this path.

## 7. Smaller things

- Publish a rendered PNG of `assets/memory-budget.svg` for platforms that will
  not inline SVG.
- Add a `results/README.md` explaining which files predate the harness fix.
- The `sweep.sh` page-cache drop needs passwordless sudo; it currently warns and
  continues warm-biased. Document the requirement or find a rootless approach.
