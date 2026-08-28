#!/usr/bin/env python3
"""Does the vision tower actually work on the single-Spark path?

THE OPEN QUESTION
-----------------
Qwen3.8-Flash-Next is natively multimodal, and the NVFP4 checkpoint carries the
tower: `pipeline_tag: image-text-to-text`, `language_model_only: false`, a set
`image_token_id`, `model.visual.*` deliberately held OUT of the NVFP4 quant
(so it stays BF16), and both `preprocessor_config.json` and
`video_preprocessor_config.json` present.

But as of 2026-08-27 **no published single-GB10 recipe has ever passed this
model an image.** Both the vLLM and llama.cpp write-ups are text-only end to
end. The PLE mmap patch in particular has never been exercised with multimodal
input, and there is a specific reason to be suspicious: image tokens change the
shape of the n-gram lookup's input, and that lookup is the thing we replaced.

So this is a real experiment with a real chance of failing. It is the reason
this repository exists.

WHAT IT TESTS
-------------
Three images, generated locally so the test is self-contained and
deterministic — no downloads, no dataset, no network:

  1. shapes   — count and name coloured geometric shapes (basic perception)
  2. text     — read an exact rendered string (OCR path through the tower)
  3. chart    — read a value off a plotted bar (the CharXiv-style task the
                model is actually benchmarked on)

A model that has silently lost its vision tower does not error. It hallucinates
a plausible description. So every check here is against a *known* ground truth
that cannot be guessed from the prompt.

USAGE
-----
    python3 scripts/vision-test.py --base-url http://localhost:8000/v1
    python3 scripts/vision-test.py --keep-images ./out   # to eyeball them

Exit 0 = all checks passed. Exit 1 = at least one failed (details printed).
Requires: pillow. `pip install pillow`
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import urllib.error
import urllib.request

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    sys.exit("!! this test needs pillow:  pip install pillow")


# --- ground truth, fixed here so the model cannot infer it from the prompt ---
SECRET_STRING = "SPARK-GB10-7412"
BAR_VALUES = {"alpha": 30, "beta": 75, "gamma": 45}
TALLEST_BAR = "beta"


def _font(size: int):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def make_shapes() -> Image.Image:
    """Three red circles and one blue square on white."""
    img = Image.new("RGB", (640, 480), "white")
    d = ImageDraw.Draw(img)
    for cx, cy in ((120, 120), (320, 140), (500, 110)):
        d.ellipse([cx - 55, cy - 55, cx + 55, cy + 55], fill="red")
    d.rectangle([260, 280, 390, 410], fill="blue")
    return img


def make_text() -> Image.Image:
    """One high-contrast string, large, centred."""
    img = Image.new("RGB", (900, 300), "white")
    d = ImageDraw.Draw(img)
    d.text((60, 110), SECRET_STRING, fill="black", font=_font(78))
    return img


def make_chart() -> Image.Image:
    """A labelled bar chart with a clear tallest bar and a gridline scale."""
    w, h = 760, 520
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    base_y, left, bar_w, gap = 440, 110, 120, 90
    d.line([left - 40, base_y, w - 40, base_y], fill="black", width=3)
    d.line([left - 40, base_y, left - 40, 60], fill="black", width=3)
    for val in (25, 50, 75, 100):
        y = base_y - val * 3.4
        d.line([left - 46, y, w - 40, y], fill="#dddddd", width=1)
        d.text((left - 100, y - 12), str(val), fill="black", font=_font(22))
    for i, (name, val) in enumerate(BAR_VALUES.items()):
        x0 = left + i * (bar_w + gap)
        d.rectangle([x0, base_y - val * 3.4, x0 + bar_w, base_y], fill="#2b6cb0")
        d.text((x0 + 12, base_y + 14), name, fill="black", font=_font(28))
    return img


def data_uri(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def ask(base_url: str, model: str, img: Image.Image, prompt: str, max_tokens: int) -> str:
    body = json.dumps(
        {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_uri(img)}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }
    ).encode()
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "Authorization": "Bearer none"},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        out = json.load(resp)
    return (out["choices"][0]["message"].get("content") or "").strip()


CASES = [
    (
        "shapes",
        make_shapes,
        "How many red circles are in this image, and what colour is the square? "
        "Answer in the form: <count> circles, <colour> square.",
        # ground truth: 3 red circles, blue square
        lambda a: ("3" in a or "three" in a.lower()) and "blue" in a.lower(),
        "3 circles, blue square",
        64,
    ),
    (
        "text",
        make_text,
        "Read the text in this image and reply with ONLY that text, exactly as written.",
        lambda a: SECRET_STRING.lower() in a.lower().replace(" ", ""),
        SECRET_STRING,
        48,
    ),
    (
        "chart",
        make_chart,
        "In this bar chart, which bar is tallest? Reply with only its label.",
        lambda a: TALLEST_BAR in a.lower(),
        TALLEST_BAR,
        32,
    ),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="qwen3.8-flash-next")
    ap.add_argument("--keep-images", metavar="DIR", help="also save the PNGs here")
    args = ap.parse_args()

    if args.keep_images:
        import os

        os.makedirs(args.keep_images, exist_ok=True)

    print(f"vision check against {args.base_url} (model {args.model})\n")
    failures = 0
    for name, build, prompt, check, truth, max_tokens in CASES:
        img = build()
        if args.keep_images:
            img.save(f"{args.keep_images}/{name}.png")
        try:
            answer = ask(args.base_url, args.model, img, prompt, max_tokens)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:400]
            print(f"  [{name:6s}] HTTP {exc.code}\n           {detail}\n")
            failures += 1
            continue
        except Exception as exc:  # noqa: BLE001 — any transport failure is a failure
            print(f"  [{name:6s}] request failed: {exc}\n")
            failures += 1
            continue

        ok = check(answer)
        failures += 0 if ok else 1
        flat = answer.replace("\n", " ")[:140]
        print(f"  [{name:6s}] {'PASS' if ok else 'FAIL'}")
        print(f"           expected : {truth}")
        print(f"           got      : {flat}\n")

    if failures:
        print(f"{failures}/{len(CASES)} vision checks FAILED.")
        print(
            "A confident but wrong answer means the tower is loaded and degraded;\n"
            "an HTTP 400 about image input means it is not wired on this path at all."
        )
        return 1
    print(f"all {len(CASES)} vision checks PASSED — the tower survives this path.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
