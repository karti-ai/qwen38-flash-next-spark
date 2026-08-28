# Benchmarks

> **Provenance rule for this file:** every number is labelled with who measured
> it. Nothing here is ours until it says so. When our runs land, they go in
> `results/` as raw JSON from `scripts/bench.py`, and the tables below cite them.

## Our measurements

**None yet.** Bring-up is in progress. This section will carry:

- single-stream decode, MTP on and off
- the concurrency sweep (c=1,8,16,32,48) with queue-time deltas
- the vision verdict (see [VISION.md](VISION.md))
- a same-box comparison against Qwen3.8-27B on identical prompts

## Upstream measurements (not ours)

Single GB10, `RadixArk/Qwen3.8-Flash-Next-NVFP4`, PLE served off-GPU.

| source | config | single-stream | aggregate |
|---|---|---:|---:|
| [jschmied](https://github.com/jschmied/qwen38-flash-next-gb10) | vLLM, CPU offload, no speculation | 17.1 tok/s | **266.8 tok/s @ c=48** |
| [blazux](https://github.com/blazux/qwen3.8-Flash-DGX) | vLLM, mmap, MTP=2 | 25–28 tok/s | — |
| [0xBakeer](https://github.com/0xBakeer/qwen38-flash-next-spark) | llama.cpp, `-ot per_layer_token_embd=CPU` | ~22 tok/s | n/a (`--parallel 1`) |

jschmied's full concurrency trace, from `/proc` and `/metrics` counters only:

| c | aggregate tok/s | per stream | majflt/token | TTFT | queue |
|---:|---:|---:|---:|---:|---:|
| 1 | 17.1 | 17.1 | 16.0 | 0.22 | 0.00 |
| 8 | 87.5 | 10.9 | 7.0 | 0.53 | 0.00 |
| 16 | 131.6 | 8.2 | 9.6 | 0.83 | 0.00 |
| 32 | 212.0 | 6.6 | 4.3 | 1.19 | 0.00 |
| 48 | **266.8** | 5.6 | **3.6** | 1.60 | 0.01 |

**Read the `majflt/token` column.** Page-fault cost per token *falls 4.4×* from c=1 to c=48. Batched tokens share n-gram rows and the page cache holds the hot set, so the marginal token is far cheaper than the first. The paged table is an argument **for** running this model concurrent.

Prefill, same box: ~540 tok/s on llama.cpp IQ4_XS vs ~2,000–2,600 tok/s on vLLM NVFP4 — a 4–5× gap from vLLM's sparse-attention kernels.

## Model quality (vendor-reported)

Alibaba's own numbers, Flash-Next vs the dense Qwen3.8-27B. **There are no independent evaluations of Flash-Next** as of 2026-08-27 — it is days old. Treat these as directional.

| benchmark | Qwen3.8-27B | Flash-Next | Δ |
|---|---:|---:|---:|
| JobBench | 33.4 | **55.7** | +22.3 |
| deepSWE 1.1 | 42.2 | **58.7** | +16.5 |
| Agents' Last Exam | 42.9 | **51.2** | +8.3 |
| ERQA | 65.5 | **72.3** | +6.8 |
| NL2Repo | 42.3 | **48.1** | +5.8 |
| Humanity's Last Exam | 30.8 | **35.9** | +5.1 |
| CoWorkBench | 70.7 | **73.9** | +3.2 |
| AndroidWorld | 81.9 | **84.5** | +2.6 |
| RealWorldQA | 85.9 | **88.5** | +2.6 |
| GPQA-Diamond | 89.2 | **91.7** | +2.5 |
| IFBench | 79.5 | **81.3** | +1.8 |
| LiveCodeBench v6 | 90.3 | **91.9** | +1.6 |
| MathVision w/ Python | 94.6 | **95.7** | +1.1 |
| Vision2Web | 62.9 | **64.0** | +1.1 |
| CharXiv w/o tools | 83.7 | **84.6** | +0.9 |
| SWE-bench Pro | 61.7 | **62.5** | +0.8 |
| MathVision | 90.0 | **90.6** | +0.6 |
| CharXiv | 90.2 | **90.6** | +0.4 |

Flash-Next wins all 18 shared benchmarks, with 6B active parameters against a 27B dense model. The agentic cluster is where it separates.

**Caveat worth stating twice:** these do not overlap everywhere. Terminal-Bench 2.1 (73.0), OSWorld-Verified (84.3) and WebArena-Verified (64.8) exist only for the 27B and have no Flash-Next counterpart — untested, not won.

## Checkpoint quality (RadixArk, published with the weights)

| eval | result | reference band |
|---|---|---|
| GSM8K (full 1319, single-shot) | 97.27% | 97.12–97.50 (BF16) — inside |
| AIME26 (30×8) | pass@1 98.75%, majority@8 100% | ≥236/240 — inside |

Plus a byte-equality audit (1,562 tensors, 118 GB compared, all passed), a scale audit, and a full `verify_hf` LFS-hash pass. They also record a stop-rate miss (98.86% against a 99% line) rather than burying it.

## Methodology notes

- `scripts/bench.py` pins temperature 0 for determinism. RadixArk's recommended sampling is temperature 1.0 / top-p 0.95 / top-k 20 — do not read quality conclusions off benchmark runs.
- Always report `--max-num-seqs`. A low cap and a saturated box look identical in tok/s; the harness reads `vllm:request_queue_time_seconds_sum` to tell them apart.
- Quiesce other clients first. A "single-stream" number taken while agents are hitting the same port is a *share*, not a ceiling.
