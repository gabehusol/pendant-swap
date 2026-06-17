# DESIGN.md — `pendant-swap`

A bring-your-own-key toolkit that swaps a jewelry pendant onto a model photo using an AI image model, with deterministic prep and a measure-and-retry QA loop wrapped around the generation step.

This document is the build spec. Implement it as written. Where a detail is left open, prefer the simplest reliable option and note the choice in the README. Build and test on real sample images before considering the task done.

---

## 1. Problem statement

Editing a product pendant onto a model photo with an AI image model is unreliable: the pendant renders at the wrong size, the framing/zoom drifts, and fine details (like the bail) get reinvented. Doing it by hand means generating, eyeballing the result, and re-rolling until it looks right — slow and unmeasurable.

This tool makes the process deterministic and verifiable. It computes real-world scale, prepares precise inputs, calls the image model with the user's own API key, then **measures the output against spec and automatically retries** if it fails. It can also produce a real-pixel composite with no AI at all.

Scope is **pendants/necklaces only**. The QA logic (pendant height in mm, hang position, near-square aspect) is pendant-specific. Generalizing to rings/earrings is explicitly future work.

---

## 2. Goals and non-goals

**Goals**
- Deterministic prep: background-removed pendant cutout, pixels-per-mm scale math, placement guide, ready-to-paste prompt.
- Optional no-AI composite of the real pendant pixels onto the model photo (so fine details aren't reinvented).
- AI generation via a pluggable backend, defaulting to Google Gemini (Nano Banana), using the **user's** API key supplied per request.
- A QA gate that measures the result and a **generate → measure → reject → regenerate** loop with a retry cap.
- Cleanup/export utilities (watermark removal, crop, Lanczos upscale).
- Three usable surfaces: a Python **engine** (importable library), a **FastAPI** backend exposing the engine, a **CLI**, and a **minimal web UI**.

**Non-goals (do NOT build)**
- User accounts, auth, databases, or any persistence of keys or images.
- Billing or key management of our own — keys are always the user's, passed per request.
- Chain recoloring (out of scope; unreliable on thin chains over skin).
- Auto-detecting the chain endpoint / pendant position (use manual placement args with sensible defaults; mention auto-detect as future work).
- A polished production UI (a separate, nicer UI will be built elsewhere; ship a minimal functional one only).

---

## 3. Architecture

Three layers over one core engine:

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

- The **engine** is pure functions with no global state and no I/O assumptions beyond file paths / PIL images. Everything else calls it.
- The **CLI** and **FastAPI** are both thin adapters over the engine. No business logic duplicated.
- The **generate** step is behind an interface so backends (Gemini default; fal/OpenAI later) are swappable.

---

## 4. Tech stack

- Python 3.10+
- Image: Pillow, NumPy, OpenCV (`opencv-python-headless`)
- Generation SDK: `google-genai` (official Google Gen AI SDK). **Verify the current image model name and call signature from Google's official docs at build time** — do not assume; the model id (e.g. a `gemini-*-image*` string) and SDK API change. Pin versions in `requirements.txt`.
- CLI: Typer (preferred) or argparse
- Backend: FastAPI + Uvicorn
- UI: plain HTML + vanilla JS (single page). No build step.
- Config: environment variables via `python-dotenv`; `.env.example` provided.

---

## 5. Repository structure

```
pendant-swap/
  README.md
  DESIGN.md                 (this file)
  requirements.txt
  .env.example
  .gitignore
  pendant_swap/
    __init__.py
    cutout.py               background removal
    scale.py                pixels-per-mm + target size math
    guide.py                placement guide image
    composite.py            real-pixel composite (no AI)
    generate.py             pluggable generation backend(s)
    qa.py                   measurement + pass/fail report
    finish.py               watermark removal, crop, upscale
    loop.py                 generate -> qa -> retry orchestration
    types.py                dataclasses for params, QA report, etc.
  cli.py                    Typer CLI
  api.py                    FastAPI app
  web/
    index.html              minimal UI
    app.js
    style.css
  samples/                  (gitignored except a README note) place test images here
  tests/
    test_scale.py
    test_qa.py
```

---

## 6. Core engine — module specs

All image functions accept and return PIL Images or numpy arrays where natural; file-path wrappers are fine for CLI. Keep functions small and testable.

### 6.1 `cutout.py` — background removal
- `remove_background(img, tolerance=28) -> RGBA Image`
- Approach: flood-fill from the four image borders treating near-uniform background as transparent. Compute per-pixel distance to a sampled background color (average of corner pixels); seed a BFS/flood from border pixels where distance < tolerance; mark only border-connected background transparent (so light regions *inside* the object are preserved). Output transparent PNG.
- `tolerance` is a flag/param. Also expose `trim_to_alpha(img)` to crop to the alpha bounding box.

### 6.2 `scale.py` — scale math
- `pixels_per_mm(ref_pixel_height, ref_mm) -> float`
- `target_pixels(target_mm, ppm) -> float`
- Pure arithmetic, fully unit-tested. Example from real data: ref pendant ≈ 130 px tall, assumed ≈ 13 mm → 10 px/mm; target 21 mm → 210 px. (These are the default test values.)

### 6.3 `guide.py` — placement guide
- `make_guide(model_img, cutout, target_px_height, hang_xy, opacity=0.78, rotate_deg=0) -> Image`
- Scale the cutout so its height (or wing width — see composite) equals the target, rotate by `rotate_deg`, paste semi-transparently centered on `hang_xy`. Returns a guide image for feeding to the model and for QA comparison.

### 6.4 `composite.py` — real-pixel swap (no AI)
- `composite_pendant(model_img, cutout, *, scale_width_px, hang_xy, rotate_deg=0, top_crop_px=0) -> Image`
- Steps: optionally zero the cutout's alpha above `top_crop_px` (to drop a product-photo chain above the bail); rotate by `rotate_deg` (expand=True) to deskew toward upright; measure wing width as the **max alpha row coverage**; scale so that width == `scale_width_px`; alpha-composite onto the model image centered horizontally at `hang_xy.x` with its top near `hang_xy.y`.
- Sensible defaults: `rotate_deg` a few degrees, `scale_width_px` derived from `target_pixels`, hang point lower-center of the image.
- This must be able to reproduce a clean hand-built composite: real bail + butterfly, leveled, placed under the chain.

### 6.5 `generate.py` — pluggable AI backend (BYO key)
- Define an interface:
  ```python
  class ImageEditor(Protocol):
      def edit(self, *, base_images: list[Image], prompt: str, api_key: str) -> Image: ...
  ```
- `GeminiEditor` implements it using `google-genai`. The **api_key is passed in per call** — never read from a global, never hardcoded. If no key is passed, fall back to `GEMINI_API_KEY` env var (for CLI convenience only).
- The call sends the base image(s) (model photo, optional cutout, optional guide) plus the prompt and returns the generated image. Handle multi-part responses (extract the image part). Surface clear errors on auth failure / quota / safety blocks.
- Keep the model id and request shape in one place; **confirm against current Google docs at build time**. Make it trivial to add `FalEditor` / `OpenAIEditor` later.

### 6.6 `qa.py` — measurement + report
- `measure_pendant(result_img, search_bbox) -> bbox` : within a caller-supplied approximate region, segment the pendant (dark/brown pendant vs skin; e.g. brownish + low-luma mask) and return its bounding box. Caller passes the search bbox to avoid false positives elsewhere.
- `qa_report(result_img, *, target_mm, ppm, search_bbox, aspect_tol=0.20, size_tol=0.10, chain_region=None) -> QAReport`
  - Pendant height & width in px → mm via `ppm`. **Flag** if height outside `target_mm ± size_tol`.
  - Aspect ratio (width/height); flag if not near-square within `aspect_tol`.
  - If `chain_region` provided: classify gold vs silver by mean hue/saturation (gold = warm hue, higher saturation; silver = low saturation). Report which, flag if not gold.
  - `annotate=True` option saves a copy with the measured pendant boxed and labeled with the mm reading.
- `QAReport` (dataclass): per-check `value`, `target`, `pass: bool`, plus overall `passed: bool` and a human-readable summary string.

### 6.7 `finish.py` — cleanup/export
- `remove_watermark(img, bbox, method="inpaint") -> Image`
  - `"inpaint"` (default): within bbox, detect bright + grayish pixels (logo/sparkle), dilate the mask a few px, `cv2.inpaint` (TELEA, small radius). Keep the mask tight to the logo so it doesn't smear across edges.
  - `"corner_patch"`: copy a clean region (from a `source_img` arg or an offset within the same image) over the bbox with a feathered mask. Useful when inpaint leaves a smudge.
- `crop_margins(img, top, right, bottom, left) -> Image`
- `upscale(img, factor) -> Image` : high-quality Lanczos. README must note this is a stopgap and a dedicated upscaler is better.

### 6.8 `loop.py` — orchestration (the headline feature)
- `run_swap(params) -> SwapResult` :
  1. `prep`: cutout + scale + guide + prompt.
  2. Loop up to `max_retries` (default 4):
     - `generate` (model photo + cutout + guide + prompt) via the editor + user key.
     - `qa_report` on the result.
     - If `passed`, stop and return.
     - If failed, append a **corrective hint** to the prompt based on which check failed (e.g. height too large → "make the pendant noticeably smaller"; chain not gold → "ensure the entire chain is warm gold"; aspect wrong → "make the butterfly more compact / near-square"), then retry.
  3. Return the best attempt (prefer a passing one; otherwise the closest by score) plus all QA reports and the attempt count.
- `SwapResult`: final image, chosen attempt index, list of `QAReport`s, the prompt(s) used.

---

## 7. FastAPI backend (`api.py`)

- `POST /swap` — multipart form:
  - files: `model` (required), `pendant` (required)
  - fields: `target_mm`, `ref_px_height`, `ref_mm`, `hang_x`, `hang_y`, `rotate_deg`, `max_retries`, and a `mode` of `composite` (no AI) or `generate` (AI loop, default).
  - **API key**: read from request header `X-API-Key` (preferred) or a form field. Never store it, never log it, never write it to disk.
  - Response (JSON): the result image as base64 (or a temp URL), the `QAReport`(s), attempt count, and the prompt used. Optionally include cutout/guide as base64 for transparency.
- `POST /prep`, `POST /qa`, `POST /composite`, `POST /finish` — thin endpoints over the matching engine functions (handy for the UI and for demos). Keep request/response shapes documented in the OpenAPI schema (FastAPI gives this for free at `/docs`).
- CORS: allow the local UI origin for dev. Do not enable wildcard in any "production" note.
- Validation: reject missing key for `generate` mode with a clear 400; reject oversized uploads with a sane limit.

---

## 8. CLI (`cli.py`)

Subcommands mirroring the engine. Each prints results and writes outputs to a chosen dir.

```
pendant-swap prep      --model M --pendant P --target-mm 21 --ref-px 130 --ref-mm 13 [--x --y --tolerance] --out DIR
pendant-swap composite --model M --pendant P --target-mm 21 --ref-px 130 --ref-mm 13 [--x --y --rotate --top-crop] --out DIR
pendant-swap generate  --model M --pendant P --target-mm 21 --ref-px 130 --ref-mm 13 [--max-retries 4] --out DIR   # needs key
pendant-swap qa        --result R --target-mm 21 --ref-px 130 --ref-mm 13 --search X0 Y0 X1 Y1 [--annotate] 
pendant-swap finish    --image I [--remove-watermark X0 Y0 X1 Y1] [--method inpaint|corner_patch] [--crop T R B L] [--upscale 2] --out DIR
```

- API key for `generate`: from `--api-key` or `GEMINI_API_KEY` env. Never hardcode.
- `--help` on every command, with the defaults visible.

---

## 9. Minimal web UI (`web/`)

Single static page, served by FastAPI (mount `web/` as static) or opened directly against the API.

- Inputs: file picker for model photo, file picker for pendant photo, a **password-type field for the API key**, number fields for `target_mm`, `ref_px_height`, `ref_mm`, optional `hang_x/hang_y/rotate`, a `mode` toggle (Composite / Generate), and a `max_retries` field.
- A "Run" button posts to `/swap` (or `/composite`).
- Output area: shows the result image, the QA report (each check with a ✓/✗ and the number), and the attempt count.
- The key field: include visible copy — "Your API key is sent only to your own backend for this request and is never stored or logged." Do not persist it (no localStorage).
- Keep styling minimal; this will be replaced. Functional over pretty.

---

## 10. BYO-key handling & security (must follow)

- **Never hardcode any API key.** No keys in code, tests, samples, README, or git history.
- The user's key is supplied **per request** (UI field → backend → SDK call) or via env var for the CLI. It is held only in memory for the duration of the call.
- **Never log, print, or persist** the key. Scrub it from any error output.
- `.env.example` contains placeholders only (`GEMINI_API_KEY=your-key-here`). `.gitignore` includes `.env`, `samples/` (except a placeholder note), `__pycache__/`, and output dirs.
- The key must never be embedded in front-end source or returned in any response.

---

## 11. Default config / first-run values

Bake these as defaults so the first real run matches values already verified by hand:
- `ref_px_height = 130`, `ref_mm = 13` → `ppm = 10` px/mm
- `target_mm = 21` → `target_px ≈ 210` (full-res reference frame)
- composite: small `rotate_deg` (a few degrees) to deskew toward upright; hang point lower-center.
- QA: `size_tol = 0.10`, `aspect_tol = 0.20`, `max_retries = 4`.

These are reference-frame numbers; scale proportionally if the working image is a different resolution (derive `ppm` from the actual image, don't assume a fixed px value).

---

## 12. Error handling

- Auth/quota/safety errors from the generation API → clear, user-facing messages (and a hint to check the key / quota), never a raw stack trace with the key in it.
- Background removal on a non-uniform background → warn that the cutout may be poor and suggest a cleaner product image.
- QA segmentation finding nothing in the search bbox → report "pendant not found in search region" rather than crashing.

---

## 13. README requirements

The README must include:
- The problem statement and the one-line pitch: **"It turns 'generate and hope' into 'generate, measure, and auto-retry until it meets spec.'"**
- Install steps, `.env` setup, and the BYO-key note (your key, your usage, never stored).
- The end-to-end workflow for both modes (composite-only, and the AI generate loop), with copy-paste CLI examples and a UI screenshot/gif placeholder.
- A worked example on the sample images with the QA report output shown.
- Architecture diagram (reuse section 3).
- **Roadmap / future work** section: auto-detect hang point, batch/catalog processing via a product feed, additional jewelry types (rings/earrings) with their own QA metrics, additional generation backends (fal/OpenAI), a real upscaler, and a polished UI.
- A short "Design notes" paragraph honestly stating: the generation step is the unreliable middle, so it is wrapped in deterministic prep and a measuring QA gate; the composite path needs no AI at all.

---

## 14. Build order

1. Repo skeleton + README + `.env.example` + `.gitignore` + `requirements.txt`.
2. `scale.py` + `tests/test_scale.py` (pure, fast).
3. `cutout.py`, then `guide.py`, then `composite.py`. Verify `composite` reproduces a clean real-pixel swap on the sample images.
4. `qa.py` + `tests/test_qa.py`. Verify it reports a sensible mm reading on a known image.
5. `finish.py` (watermark removal both methods, crop, upscale).
6. `generate.py` (Gemini backend; confirm current model/SDK from docs) and `loop.py` (retry orchestration).
7. `cli.py` wiring all of the above.
8. `api.py` FastAPI endpoints + static UI mount.
9. `web/` minimal UI against `/swap` and `/composite`.
10. End-to-end test run; fill the README example with real output.

**After scaffolding step 1–3, ask me for the sample images** (model photo, pendant product photo, and a known-good result) so you can test `composite` and `qa` on real inputs before wiring the API call.

---

## 15. Definition of done

- `pip install -r requirements.txt` works; CLI `--help` works for every command.
- `composite` produces a real-pixel swap on the samples with no API key.
- `qa` prints a numeric pass/fail report and can annotate.
- With a user key, `generate` runs the full generate→measure→retry loop and returns the best attempt + reports.
- FastAPI `/docs` lists the endpoints; the minimal UI can run a swap end-to-end.
- No key is ever stored, logged, or committed. Tests pass.
