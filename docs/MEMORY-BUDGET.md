# Memory budget on one 121.7 GiB GB10

## The arithmetic that makes this hard

A DGX Spark reports 128 GB and gives you **121.7 GiB of unified memory shared between host and GPU**. There is no separate VRAM to spill into — host and device draw from the same pool.

| component | params | size |
|---|---:|---:|
| main MoE body (512 experts, 10 routed + 1 shared) | 125B | ~78 GiB NVFP4 |
| n-gram embedding table (PLE) | 51B | ~44 GiB FP8 |
| MTP speculative head | 4B | small |
| **checkpoint on disk** | ~180B | **~122 GiB** |

122 > 121.7 before any KV cache. Resident serving is arithmetically impossible on one box.

## What the mmap patch changes

A token touches **16 rows × 160 bytes** of the n-gram table — about 2.5 KB out of 44 GiB. So the table does not need residency; it needs *low-latency random reads*, which is what NVMe plus the page cache provides.

| | resident | PLE served from NVMe |
|---|---:|---:|
| body | ~78 GiB | ~78 GiB |
| n-gram table | ~44 GiB | 0 (paged) |
| KV cache | 0 ✗ | ~31 GiB |
| **total resident** | **~122 GiB — dead** | **~107 GiB — serves** |

Measured by jschmied: **76.61 GiB resident + 30.99 GiB KV** with the PLE off-GPU, on a 121.7 GiB box.

## The counter-intuitive part

Paging cost per token **falls** as concurrency rises:

| concurrency | major faults / token |
|---:|---:|
| 1 | 16.0 |
| 8 | 7.0 |
| 16 | 9.6 |
| 32 | 4.3 |
| 48 | **3.6** |

Batched tokens share n-gram rows and the page cache retains the hot set, so the marginal token is ~4.4× cheaper than the first. The PLE offload worker never exceeds ~24% of one core across c=1..96 — the gather is a lookup touching a handful of rows, not a GEMM. **The paged table is a reason to run this model concurrent, not a caution against it.**

## Practical consequences

- **One big model at a time.** ~107 of 121.7 GiB leaves ~14 GiB. A small sidecar may fit; a second large model starves the KV cache. Plan for the box to be dedicated.
- **`--gpu-memory-utilization 0.85`.** 0.875 was OOM-killed on a 300k prefill with MTP enabled. Keep the margin.
- **1M context is not reachable.** The QSA layers reject fp8 KV, and bf16 alone needs ~30 GiB *per request*. 262k native; ~500k with YaRN (validated by needle-in-haystack at 414k).
- **Put the weights on NVMe.** That directory is read on the token path. Not a spinning disk, not a network mount.
- **~130 GiB free disk**, not 122 — leave room for the download's working set.
