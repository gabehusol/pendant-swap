# pendant-swap

**It turns "generate and hope" into "generate, measure, and auto-retry until it meets spec."**

A bring-your-own-key toolkit that swaps a jewelry pendant onto a model photo using an AI
image model, with deterministic prep and a measure-and-retry QA loop wrapped around
the generation step. Also works with zero AI as a real-pixel composite.

---

## Problem

Editing a product pendant onto a model photo with an AI image model is unreliable: the
pendant renders at the wrong size, the framing/zoom drifts, and fine details (like the
bail) get reinvented. Doing it by hand means generating, eyeballing the result, and
re-rolling until it looks right — slow and unmeasurable.

`pendant-swap` makes the process deterministic and verifiable. It computes real-world
scale, prepares precise inputs, calls the image model with your own API key, then
**measures the output against spec and automatically retries** if it fails.

---

## Architecture

```
            +------------------+
            |  Minimal Web UI  |   static HTML/JS, posts to /swap
            +---------+--------+
                      |
            +---------v--------+
            |  FastAPI backend |   thin wrapper; passes key through, never stores
            +---------+--------+
                      |
            +---------v--------+
            |   Core engine    |   pure Python library (no web, no globals)
            |  prep/composite/ |
            |  generate/qa/    |
            |  finish          |
            +------------------+
                      ^
            +---------+--------+
            |       CLI        |   calls the same engine functions
            +------------------+
```

The **engine** is pure functions with no global state. The **CLI** and **FastAPI** are
thin adapters over the engine. The **generate** step is behind a protocol so backends
are swappable (Gemini default; fal/OpenAI later).

---

## Install

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

Requires Python 3.10+.

### API key setup (bring-your-own)

`pendant-swap` never stores, logs, or commits your API key. You supply it:

- **CLI**: pass `--api-key sk-...` or set `GEMINI_API_KEY` in your environment.
- **UI**: enter it in the password field — sent only to your local backend, never persisted.
- **API**: pass it in the `X-API-Key` request header.

Copy `.env.example` to `.env` and fill in your key for CLI convenience:

```bash
cp .env.example .env
# Edit .env: GEMINI_API_KEY=your-key-here
```

`.env` is gitignored. Never commit it.

---

## Workflow — composite mode (no AI key required)

The composite path takes the real pendant pixels, removes the background, scales and
places them onto the model photo with pixel-perfect placement. No generation, no
hallucinated details.

```bash
# 1. Prepare: remove background, compute scale, build guide overlay
pendant-swap prep \
  --model samples/model.jpg \
  --pendant samples/pendant.jpg \
  --target-mm 21 --ref-px 130 --ref-mm 13 \
  --out output/

# 2. Composite: real-pixel swap
pendant-swap composite \
  --model samples/model.jpg \
  --pendant samples/pendant.jpg \
  --target-mm 21 --ref-px 130 --ref-mm 13 \
  --x 512 --y 320 --rotate 3 --top-crop 40 \
  --out output/
```

---

## Workflow — AI generate loop

The generate path sends the model photo, cutout, and placement guide to a Gemini image
model with a detailed prompt, then measures the result and retries with corrective hints
until QA passes or the retry cap is hit.

```bash
pendant-swap generate \
  --model samples/model.jpg \
  --pendant samples/pendant.jpg \
  --target-mm 21 --ref-px 130 --ref-mm 13 \
  --max-retries 4 \
  --api-key $GEMINI_API_KEY \
  --out output/
```

---

## Workflow — QA on an existing result

```bash
pendant-swap qa \
  --result output/result.jpg \
  --target-mm 21 --ref-px 130 --ref-mm 13 \
  --search 400 200 650 500 \
  --annotate
```

Example output:

```
[PASS] Pendant height (mm): 20.800 (target 21.000)
[PASS] Aspect ratio:         0.920 (target ≤0.200 deviation from square)
[PASS] Chain color:          gold (hue 38, sat 0.62)
Overall: PASSED (3/3 checks)
Annotated result saved: output/result_qa.jpg
```

*(This example will be filled in with real numbers after running on sample images.)*

---

## QA report fields

| Check | What it measures | Pass condition |
|---|---|---|
| `pendant_height_mm` | Segmented pendant height in mm | Within ±10% of `target_mm` |
| `aspect_ratio` | width/height of segmented bbox | Within ±0.20 of 1.0 (near-square) |
| `chain_color` | Mean hue/saturation of chain region | Warm gold hue (if chain_region supplied) |

---

## CLI reference

```
pendant-swap prep      --model M --pendant P --target-mm 21 --ref-px 130 --ref-mm 13 [--x --y --tolerance] --out DIR
pendant-swap composite --model M --pendant P --target-mm 21 --ref-px 130 --ref-mm 13 [--x --y --rotate --top-crop] --out DIR
pendant-swap generate  --model M --pendant P --target-mm 21 --ref-px 130 --ref-mm 13 [--max-retries 4] --out DIR
pendant-swap qa        --result R --target-mm 21 --ref-px 130 --ref-mm 13 --search X0 Y0 X1 Y1 [--annotate]
pendant-swap finish    --image I [--remove-watermark X0 Y0 X1 Y1] [--method inpaint|corner_patch] [--crop T R B L] [--upscale 2] --out DIR
```

Run `pendant-swap <command> --help` for full option list.

---

## API server

```bash
uvicorn api:app --reload
# OpenAPI docs at http://localhost:8000/docs
```

Endpoints: `POST /swap`, `POST /composite`, `POST /prep`, `POST /qa`, `POST /finish`.

---

## Default values

These defaults are calibrated from a real reference pendant:

| Parameter | Default | Meaning |
|---|---|---|
| `ref_px_height` | 130 | Reference pendant height in pixels |
| `ref_mm` | 13.0 | Reference pendant real-world height (mm) |
| `target_mm` | 21.0 | Target pendant real-world height (mm) |
| Derived `ppm` | 10.0 px/mm | Scale factor |
| Derived `target_px` | 210 px | Target pendant pixel height |
| `max_retries` | 4 | Generate-retry cap |
| `size_tol` | 0.10 | ±10% size tolerance |
| `aspect_tol` | 0.20 | ±0.20 aspect ratio tolerance |

---

## Design notes

The generation step is the unreliable middle of the pipeline. AI image models render
the pendant at the wrong size, drift on framing, and reinvent fine details like the
bail. This tool wraps that unreliable step in deterministic bookends: a pixel-precise
prep stage (scale math, background removal, placement guide) and a measuring QA gate
that rejects bad results and feeds corrective hints back into the next attempt.

The composite path needs no AI at all — it pastes the real pendant pixels with
transparent alpha. This is the highest-fidelity option and the right starting point
before introducing generation.

The `upscale` utility (Lanczos) is a stopgap. A dedicated upscaler model (e.g. Real-ESRGAN)
will produce noticeably better results and is the recommended upgrade for production use.

---

## Roadmap / future work

- **Auto-detect hang point** — detect chain endpoint / pendant anchor from the model photo
- **Batch / catalog processing** — accept a product feed CSV, produce swapped images in bulk
- **Additional jewelry types** — rings and earrings with their own QA metrics (finger clearance, lobe position)
- **Additional generation backends** — fal.ai, OpenAI image API, pluggable via the `ImageEditor` protocol
- **Real upscaler** — integrate Real-ESRGAN or similar instead of Lanczos
- **Polished UI** — a proper React/Next.js UI; the `web/` page is functional-only

---

## Security

- No API key is ever hardcoded, stored, logged, or committed.
- The server never writes keys to disk or returns them in responses.
- `.env` and `samples/` are gitignored.
- Auth failures from the generation API surface as clear user-facing errors, never raw tracebacks.
