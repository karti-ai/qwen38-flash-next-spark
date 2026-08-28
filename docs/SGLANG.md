# The SGLang path

**Status: open. Not working on a single Spark yet, and the reason is specific.**

## The irony

[RadixArk](https://huggingface.co/RadixArk), who produced the NVFP4 checkpoint everyone is using, **are the SGLang team**. Their own qualification notes name SGLang as the runtime:

> Runtime: SGLang with `qwen4_exp` support plus one of: (a) the fp8-resident PLE loader in `Qiaolin-Yu/sglang-qwen-next` PR #40 (uses the declared `ple_embedding_dtype`), or (b) a loader that dequantizes the FP8 PLE table with its scalar `weight_scale` at load time.

And yet the only recipe that actually serves this checkpoint on one GB10 today is **vLLM** — a runtime their notes do not mention, via a patch nobody upstream wrote for it.

Worth noting alongside that: the compatibility tables circulating claim `RadixArk/…-NVFP4` does *not* load on vLLM. It does, once `_get_ple_embedding_quant_method()` accepts `modelopt`/`modelopt_fp4` instead of only `Fp8Config`. That correction is [jschmied's](https://github.com/blazux/qwen3.8-Flash-DGX/issues/1).

## Why SGLang is worth the effort anyway

On the *other* Qwen checkpoint we run on this box class, a matched head-to-head (same checkpoint, same harness) put SGLang ahead everywhere that matters for agents:

| tok/s | dedicated c1 | c1 (1024→256) | c2 | c4 | c8 |
|---|---:|---:|---:|---:|---:|
| vLLM MTP K3 | **27.83** | 19.24 | 32.00 | 34.61 | 82.89 |
| SGLang EAGLE | 25.62 | **27.65** | **43.74** | **74.31** | **123.90** |

vLLM wins exactly one case — a single long 2048-token generation. SGLang wins short agent turns by ~44% and every concurrency level by 37–115%. Agents do short turns.

If that ordering carries to Flash-Next, the SGLang path is the one worth having.

## The actual blocker

It is **not** architecture support. It is memory.

The whole reason Flash-Next fits on one 121.7 GiB box is that the 44 GiB FP8 n-gram (PLE) table is served from NVMe through the page cache instead of being resident. That patch exists **for vLLM only**. SGLang's loader keeps the table resident, so:

```
body ~78 GiB  +  PLE ~44 GiB  =  ~122 GiB  >  121.7 GiB available
```

...before a single token of KV cache. Both of RadixArk's suggested loaders — the PR #40 fp8-resident path and the dequantize-at-load path — are *resident* strategies. They solve correctness, not capacity. They are aimed at multi-GPU hosts.

So making SGLang work on one Spark means porting the mmap idea across: replacing SGLang's n-gram embedding with a page-cache-backed gather, keeping the scalar `weight_scale` for dequantization, and keeping the gather outside any graph capture. That is the contribution this repo is going after.

## The trap that will bite whoever tries

From RadixArk's notes, emphasis ours:

> Loaders that only upcast the FP8 bytes will serve **wrong PLE embeddings silently**.

The table is F8_E4M3 shards plus **one scalar** `weight_scale`. A loader that widens FP8 → BF16 without applying that scale produces a server that boots, serves, and answers fluently — and is wrong. There is no error and no warning.

This is the same failure signature as the corrupt-shard problem: **the model does not break loudly, it degrades confidently.** It is also why [`scripts/smoke-test.sh`](../scripts/smoke-test.sh) checks unguessable arithmetic rather than liveness. Run it before you believe any SGLang result, including your own.

## Plan

1. Bring up the vLLM path, measure it, answer the vision question. *(in progress)*
2. Try SGLang unmodified and record exactly how it fails. `scripts/serve-sglang.sh` exists for this.
3. Port the PLE mmap gather to SGLang's embedding layer.
4. Re-run the same harness on both and publish the comparison — the numbers above, but for Flash-Next.

## Sampling

RadixArk's `generation_config` recommends **temperature 1.0, top-p 0.95, top-k 20**. Our benchmark harness pins temperature 0 for determinism, which is a deliberate departure — do not read quality conclusions off benchmark runs.
