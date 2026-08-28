# Vision on the single-Spark path

**Status: unanswered. This is the question the repository exists to settle.**

## What we know from the checkpoint

`RadixArk/Qwen3.8-Flash-Next-NVFP4`'s own `config.json` and file list say the tower is there:

| signal | value |
|---|---|
| `pipeline_tag` | `image-text-to-text` |
| `language_model_only` | `false` |
| `image_token_id` | `248056` |
| quant `ignore` list | includes `model.visual.*` — the tower stays **out** of NVFP4, i.e. BF16 |
| files present | `preprocessor_config.json`, `video_preprocessor_config.json` |
| architecture | `Qwen4ExpForConditionalGeneration` |

So the weights are shipped and deliberately held at higher precision. That is necessary, not sufficient.

## What nobody has done

As of 2026-08-27, **no published single-GB10 write-up has passed this model an image.** Not the vLLM recipe, not the llama.cpp one, not the forum threads. Every reported result — every tok/s figure, every smoke test — is text-only end to end. Searching the recipe repos for `vision`, `image`, `visual`, `multimodal` or `limit-mm` returns nothing.

## Why it might not work

The mmap patch replaces the n-gram (PLE) embedding lookup with a page-cache-backed gather, and wraps it in a custom op (`vllm::ple_mmap_lookup`) that is deliberately opaque to `torch.compile` and excluded from CUDA graph capture.

Image tokens change the shape and composition of the sequence feeding that lookup. The gather has never been exercised with multimodal input. Plausible outcomes, in rough order of likelihood:

1. **It just works** — the PLE operates on token ids and does not care where they came from.
2. **HTTP 400 on image input** — the multimodal path is not wired in this image at all. Clean, loud, easy to diagnose.
3. **Shape error inside the gather** — image tokens break an assumption in the chunked gather. Loud, fixable.
4. **Silent degradation** — it answers about the image, plausibly, and wrongly. This is the dangerous one, and it is why the test below uses ground truth that cannot be guessed.

## How we test it

[`scripts/vision-test.py`](../scripts/vision-test.py). Three images generated locally — no downloads, no dataset, deterministic:

| case | image | ground truth | tests |
|---|---|---|---|
| `shapes` | 3 red circles + 1 blue square | "3 circles, blue square" | basic perception, counting |
| `text` | one rendered string | `SPARK-GB10-7412` | OCR through the tower |
| `chart` | 3 labelled bars, values 30/75/45 | `beta` | the CharXiv-style task it is benchmarked on |

The string and the bar values are arbitrary and appear nowhere in the prompt, so a model that has lost its tower cannot produce them by guessing. **A vision-blind model does not error — it hallucinates a plausible description.** Every check is against something unguessable for exactly that reason.

```bash
python3 scripts/vision-test.py --base-url http://localhost:8000/v1
python3 scripts/vision-test.py --keep-images ./out   # eyeball the inputs
```

## Result

_Pending first boot. The outcome — including a negative one — gets written here with the raw transcript in `results/`._

If it works, this is the first documented multimodal inference of Qwen3.8-Flash-Next on a single 128 GB box. If it does not, the failure mode is worth just as much: everyone currently assuming "the tower is in the checkpoint, so vision works" is assuming it.
