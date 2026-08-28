# Troubleshooting

Ordered by how much time each one costs if you meet it cold.

---

## Fluent output that is confidently wrong

The signature failure of this model. Three distinct causes produce **the same
symptom**: a server that boots cleanly, reports sane shapes, streams tokens at
normal speed, and is wrong. None of them raise an error.

### 1. Byte-corrupt weight shards

The only documented reproduction failure of this recipe. Two safetensors files
were the correct size and the wrong bytes, left by a stalled download.

```
"The capital of France is"  ->  " andufteth,,allwaysas2.logasasas1.myas2 kkl2 IIl1inkl l ul l lllK"
```

It survived every configuration change tried: MoE backend (CUTLASS and MARLIN
corrupted *identically*), flashinfer autotune, prefix caching, MRV1 vs MRV2,
async scheduling, the triton GDN decode kernel, and every form of the PLE path.
Output was byte-identical across most of them — which is itself the tell: **if
changing the config does not change the garbage, the fault is upstream of the
config.**

A partial-content check is not enough either. The original diagnosis sampled row
0 of shard 0, landed inside the intact head of a corrupt file, matched the
official table at cosine 0.999635, and concluded the checkpoint was sound.

**Fix:** verify `lfs.sha256` for every file, not sizes, not samples.

```bash
python3 scripts/verify-weights.py --repo RadixArk/Qwen3.8-Flash-Next-NVFP4 --dir <weights>
```

### 2. A PLE loader that upcasts instead of dequantizing

From RadixArk's qualification notes: *"Loaders that only upcast the FP8 bytes
will serve wrong PLE embeddings silently."* The table is F8_E4M3 shards plus one
scalar `weight_scale`; that scale must be applied. Mostly a risk on the SGLang
path — see [SGLANG.md](SGLANG.md).

### 3. A shadowed weight scale under CPU offload

Under `VLLM_PLE_CPU_OFFLOAD`, the GPU-side process must not register
`weight`/`weight_scale`. `load_weights()` keeps only `_offload_weight_scale`, so
a registered-but-unloaded `weight_scale` shadows it in
`_get_embedding_weight_scale()` and the lookup dequantizes against an
uninitialised value. Fluent garbage, no error.

**In all three cases:** run [`scripts/smoke-test.sh`](../scripts/smoke-test.sh)
before believing any output or any benchmark. It checks unguessable arithmetic,
not liveness.

---

## `no module or parameter named 'ngram_embedding.weight_scale'`

vLLM's `_get_ple_embedding_quant_method()` in `ple_layer.py` accepts only
`Fp8Config`. This checkpoint pairs an **FP8 PLE** with an **NVFP4 body**, so
`quant_config` is `modelopt_fp4`, the PLE is rejected, the embedding is built
unquantized, and loading dies here.

**Fix:** accept `modelopt` / `modelopt_fp4` in that gate. This is also why the
circulating compatibility tables are wrong — **RadixArk NVFP4 does load on
vLLM.**

---

## `pidfd_getfd: Operation not permitted`, ~10 minutes into boot

```
RuntimeError: pidfd_getfd: Operation not permitted
  torch/multiprocessing/reductions.py:179 in rebuild_cuda_tensor
  vllm/v1/ple_offload/worker.py:482 in accept_registrations
```

Surfaces late — both workers load all 206 shards first — and presents as an
unhelpful `Engine core initialization failed. Failed core proc(s): {}`.

The cause is **not** Docker seccomp. It is `kernel.yama.ptrace_scope = 1`, the
default on Ubuntu and DGX OS. `pidfd_getfd` requires `PTRACE_MODE_ATTACH`, which
`ptrace_scope=1` restricts to *descendants*. The PLE offload worker and the GPU
worker are **siblings** — both children of the engine — so neither may attach,
and the CUDA-IPC tensor handoff is refused.

| deployment | fix |
|---|---|
| Docker | `--cap-add=SYS_PTRACE` |
| bare metal / systemd | `AmbientCapabilities=CAP_SYS_PTRACE` on the unit |
| bare metal / shell | usually already works (inherits your login's caps) |

`CapabilityBoundingSet` alone is **not** enough — it bounds what may be held, and
a `User=` service holds no effective capabilities without `AmbientCapabilities`.
`sysctl kernel.yama.ptrace_scope=0` also works but weakens ptrace machine-wide.

**Does not affect the mmap path**, which is single-process with no IPC handoff.

---

## Boot failure with fp8 KV cache

The QSA layers **refuse** fp8 KV. Keep `--kv-cache-dtype auto` (= bf16).

This is the opposite of most Qwen checkpoints — several declare
`kv_cache_quant_algo: FP8` and want `fp8_e4m3`. Do not copy the flag across
configs.

---

## Inductor int64 indexing assert on sm_121

The PLE gather is CPU work plus a pageable host→device copy and cannot live
inside a CUDA graph capture. Use `-cc.cudagraph_mode=PIECEWISE` (never `FULL*`)
with `vllm::ple_mmap_lookup` in `-cc.splitting_ops`, or `--enforce-eager`.
`torch.compile` stays off for the same reason.

---

## Throughput plateaus that are not saturation

`--max-num-seqs` silently queues requests. Aggregate tok/s goes flat and looks
exactly like a saturated box. A value of `2` costs ~4× aggregate throughput at
c=8; one measured sweep sat at ~33 tok/s and was nearly reported as a hardware
ceiling.

**They are indistinguishable in throughput alone.** Read
`vllm:request_queue_time_seconds_sum` from `/metrics` — it hit 142 s while tok/s
stayed flat. [`scripts/bench.py`](../scripts/bench.py) reports this delta at
every sweep point and warns you.

---

## Other GB10 constraints, upstream and not optional

- **Prefix caching off** — GB10 GDN kernel corruption.
- **1M context unreachable** — QSA rejects fp8 KV; bf16 needs ~30 GiB/request.
- **Decode without MTP** lags GGUF slightly, from a host↔device sync per step.
- **`--gpu-memory-utilization 0.875`** was OOM-killed on a 300k prefill with MTP.

---

## A single-stream number that is really a share

If anything else is hitting the same port — an agent, a gateway, a dashboard —
your "c=1" measurement is a fraction of the box, not its ceiling. One such
measurement read 18.6 tok/s client-side while the server reported 45–71 tok/s
aggregate at the same instant. Quiesce other clients before quoting c=1.
