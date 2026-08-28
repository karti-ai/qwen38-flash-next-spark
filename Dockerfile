# Qwen3.8-Flash-Next on a single DGX Spark / GB10, served by vLLM.
#
# The official Qwen3.8-Flash-Next vLLM image plus one patch: it serves the
# 51B-parameter n-gram ("PLE") table from disk via mmap instead of holding it
# resident in the 128 GB unified pool. That single change is what lets a ~180B
# (122 GiB NVFP4) checkpoint fit next to a real KV cache on one box.
#
#   docker build -t qwen38-flash-next-spark .
#
# The patch is vendored UNMODIFIED from blazux/qwen3.8-Flash-DGX (Apache-2.0)
# at commit d2854bf. See NOTICE. It is a no-op unless VLLM_PLE_MMAP=1 is set at
# runtime, so this image behaves exactly like upstream when the flag is off.
#
# Base image pinned by digest, not tag: the numbers in docs/BENCHMARKS.md do not
# transfer to a different build.
FROM vllm/vllm-openai:qwen38-flash-next@sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8

ARG SP=/usr/local/lib/python3.12/dist-packages
ARG PLE=${SP}/vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py

COPY third_party/qwen38-flash-dgx/vllm_ple_mmap.py ${SP}/vllm_ple_mmap.py

RUN cp ${PLE} ${PLE}.orig \
 && printf '\n\n# --- qwen38-flash-next-spark: serve the PLE n-gram table from disk (VLLM_PLE_MMAP=1) ---\nfrom vllm_ple_mmap import apply as _ple_mmap_apply\n_ple_mmap_apply(Qwen3_8FlashNextNGramEmbedding)\n' >> ${PLE} \
 && python3 -c "import ast; ast.parse(open('${PLE}').read()); print('ple_layer.py patched OK')"
