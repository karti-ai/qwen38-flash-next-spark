# Contributing

This repo is a working record of getting a ~180B model onto one 128 GB box. The
most valuable contributions are **measurements and failure modes**, not features.

## What is especially welcome

- **Reproductions**, positive or negative. If the recipe does not work on your
  GB10, that is a finding — open an issue with your driver version, image
  digest, launch flags, and the first 50 lines of the failure.
- **A vision result** on any single-Spark path. See [docs/VISION.md](docs/VISION.md).
- **The SGLang port.** See [docs/SGLANG.md](docs/SGLANG.md) — the blocker is
  the resident n-gram table, not architecture support.
- **Anything that turns a silent failure into a loud one.** This model has at
  least three distinct ways to be confidently wrong with no error.

## Ground rules for numbers

A performance claim in this repo must state:

1. the image digest and checkpoint revision,
2. `--max-num-seqs` and whether speculative decoding was on,
3. whether anything else was hitting the endpoint (a "single-stream" figure
   taken while other clients are live is a *share*, not a ceiling),
4. and for aggregate figures, the queue-time delta — otherwise a request cap is
   indistinguishable from saturation.

`scripts/bench.py` emits all of this. Prefer its JSON over hand-timed numbers,
and commit the raw file to `results/`.

Before reporting any result: run `scripts/verify-weights.py`, then
`scripts/smoke-test.sh`. A byte-corrupt shard produces fluent, plausible output
and normal throughput. We would rather have no number than a wrong one.

## Attribution

Prior work is credited in the README and [NOTICE](NOTICE). If you build on
someone's finding, name them. Vendored code stays unmodified with its licence
intact; if it needs changing, patch it at runtime instead and say so.

## Style

- Shell: `bash -n` and `shellcheck` clean, `set -euo pipefail`.
- Python: standard library only for anything in `scripts/` that must run on the
  Spark itself (`vision-test.py` needs pillow and says so).
- Comments explain **why**, and especially why an obvious-looking alternative is
  wrong. Someone will try to "fix" `--kv-cache-dtype auto` otherwise.
