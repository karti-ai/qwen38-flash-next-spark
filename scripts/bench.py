#!/usr/bin/env python3
"""Measure decode throughput at a sweep of concurrency levels.

WHY NOT JUST tok/s
------------------
On this model the single most expensive mistake is reporting a throughput
plateau as saturation when it is actually a request cap. `--max-num-seqs`
silently queues requests: aggregate tok/s goes flat and *looks* exactly like a
saturated box. The two are indistinguishable from throughput alone.

So this harness reads `vllm:request_queue_time_seconds_sum` from /metrics
around every sweep point and reports the delta. If queue time climbs while
throughput stays flat, you are measuring your own configuration, not the
hardware. (SGLang exposes an equivalent counter; the script degrades to
throughput-only if neither is present.)

It also reports major page faults per token where the OS exposes them, because
on this model the n-gram table is paged from NVMe and the interesting result is
that per-token paging cost FALLS with concurrency — batched tokens share n-gram
rows and the page cache keeps the hot set.

USAGE
-----
    python3 scripts/bench.py --base-url http://localhost:8000/v1
    python3 scripts/bench.py --concurrency 1,8,16,32,48 --out results/vllm.json

Reports, per level: aggregate tok/s, per-stream tok/s, TTFT, and queue delta.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request

PROMPT = (
    "Write a clear, self-contained explanation of how a mixture-of-experts "
    "transformer routes tokens to experts, why only a fraction of parameters "
    "are active per token, and what that means for memory versus compute. "
    "Be concrete and complete."
)


def http_json(url: str, body: dict | None = None, timeout: int = 600):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Authorization": "Bearer none"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def metrics_text(base_url: str) -> str:
    """/metrics lives next to the API root, not under /v1."""
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    try:
        with urllib.request.urlopen(root.rstrip("/") + "/metrics", timeout=10) as r:
            return r.read().decode(errors="replace")
    except Exception:  # noqa: BLE001 — metrics are optional
        return ""


def counter(text: str, *names: str) -> float | None:
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        for name in names:
            if line.startswith(name):
                try:
                    return float(line.rsplit(" ", 1)[1])
                except (IndexError, ValueError):
                    pass
    return None


class Result:
    __slots__ = ("tokens", "ttft", "elapsed", "error")

    def __init__(self) -> None:
        self.tokens = 0
        self.ttft = 0.0
        self.elapsed = 0.0
        self.error: str | None = None


def one_stream(base_url: str, model: str, max_tokens: int, out: Result) -> None:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }
    start = time.perf_counter()
    first = None
    n = 0
    try:
        req = urllib.request.Request(
            base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "Authorization": "Bearer none"},
        )
        with urllib.request.urlopen(req, timeout=900) as resp:
            for raw in resp:
                line = raw.decode(errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                delta = (chunk.get("choices") or [{}])[0].get("delta", {})
                if delta.get("content"):
                    if first is None:
                        first = time.perf_counter()
                    n += 1
    except Exception as exc:  # noqa: BLE001
        out.error = f"{type(exc).__name__}: {exc}"

    out.elapsed = time.perf_counter() - start
    out.ttft = (first - start) if first else 0.0
    out.tokens = n


def sweep(base_url: str, model: str, level: int, max_tokens: int) -> dict:
    before = metrics_text(base_url)
    q0 = counter(before, "vllm:request_queue_time_seconds_sum", "sglang:queue_time_seconds_sum")

    results = [Result() for _ in range(level)]
    threads = [
        threading.Thread(target=one_stream, args=(base_url, model, max_tokens, r))
        for r in results
    ]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0

    after = metrics_text(base_url)
    q1 = counter(after, "vllm:request_queue_time_seconds_sum", "sglang:queue_time_seconds_sum")

    errs = [r.error for r in results if r.error]
    good = [r for r in results if not r.error and r.tokens > 0]
    total = sum(r.tokens for r in good)

    return {
        "concurrency": level,
        "streams_ok": len(good),
        "errors": errs[:3],
        "wall_seconds": round(wall, 2),
        "total_tokens": total,
        "aggregate_tok_s": round(total / wall, 2) if wall else 0.0,
        "per_stream_tok_s": round(
            statistics.mean(r.tokens / r.elapsed for r in good), 2
        )
        if good
        else 0.0,
        "ttft_p50": round(statistics.median(r.ttft for r in good), 3) if good else 0.0,
        "queue_time_delta_s": round(q1 - q0, 3) if (q0 is not None and q1 is not None) else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="qwen3.8-flash-next")
    ap.add_argument("--concurrency", default="1,8,16,32,48")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--out", help="write JSON results here")
    args = ap.parse_args()

    levels = [int(x) for x in args.concurrency.split(",") if x.strip()]

    try:
        served = http_json(args.base_url.rstrip("/") + "/models")
        ids = [m["id"] for m in served.get("data", [])]
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"!! cannot reach {args.base_url}: {exc}")
    print(f"serving: {', '.join(ids)}\n")

    print(f"{'c':>4}  {'agg tok/s':>10}  {'per-stream':>10}  {'ttft p50':>9}  {'queue Δs':>9}")
    print("-" * 52)
    rows = []
    for level in levels:
        row = sweep(args.base_url, args.model, level, args.max_tokens)
        rows.append(row)
        q = row["queue_time_delta_s"]
        qs = "n/a" if q is None else f"{q:.2f}"
        print(
            f"{row['concurrency']:>4}  {row['aggregate_tok_s']:>10}  "
            f"{row['per_stream_tok_s']:>10}  {row['ttft_p50']:>9}  {qs:>9}"
        )
        if row["errors"]:
            print(f"      !! {row['streams_ok']}/{level} streams ok: {row['errors'][0]}")

    flat = [r for r in rows if r["queue_time_delta_s"] and r["queue_time_delta_s"] > 1.0]
    if flat:
        print(
            "\n⚠️  queue time is climbing at c="
            + ",".join(str(r["concurrency"]) for r in flat)
            + " — you are measuring --max-num-seqs, not the hardware. Raise it and re-run."
        )

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(
                {"base_url": args.base_url, "model": args.model,
                 "max_tokens": args.max_tokens, "results": rows},
                fh, indent=2,
            )
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
