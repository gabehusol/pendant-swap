"""FastAPI backend, thin adapter over the core engine.

Key handling rules (DESIGN.md §10):
  - API key is read from the X-API-Key request header (preferred) or a form field.
  - It is NEVER stored, logged, or written to disk.
  - It is NEVER returned in any response.
  - Auth/quota errors from the SDK are returned as 400 JSON — never raw tracebacks.

Run locally:
  uvicorn api:app --reload
  Docs: http://localhost:8000/docs
"""

from __future__ import annotations

import base64
import io
import os
from typing import Annotated, Optional

from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image as PILImage

from pendant_swap.composite import composite_pendant
from pendant_swap.cutout import remove_background, trim_to_alpha
from pendant_swap.finish import crop_margins, remove_watermark, upscale
from pendant_swap.guide import make_guide
from pendant_swap.qa import qa_report
from pendant_swap.scale import pixels_per_mm, target_pixels
from pendant_swap.types import Point, SwapParams

app = FastAPI(
    title="pendant-swap API",
    description=(
        "Bring-your-own-key pendant swap toolkit. "
        "Your API key is used only for this request and is never stored or logged."
    ),
    version="0.1.0",
)

@app.exception_handler(Exception)
async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
    import traceback
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "type": type(exc).__name__,
                 "trace": traceback.format_exc()},
    )


# CORS: allow the local UI and localhost during development.
# Do NOT enable wildcard in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000",
                   "http://localhost:5500", "http://127.0.0.1:5500"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Serve the minimal web UI at /
app.mount("/web", StaticFiles(directory="web", html=True), name="web")

# Upload size limit: 20 MB per file
_MAX_BYTES = 20 * 1024 * 1024

# Default calibration values (DESIGN.md §11)
_REF_PX  = 130
_REF_MM  = 13.0
_TARGET  = 21.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_image(upload: UploadFile) -> PILImage.Image:
    data = upload.file.read()
    if len(data) > _MAX_BYTES:
        raise HTTPException(413, "Upload too large (max 20 MB).")
    return PILImage.open(io.BytesIO(data)).convert("RGB")


def _img_to_b64(img: PILImage.Image, fmt: str = "JPEG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=92)
    return base64.b64encode(buf.getvalue()).decode()


def _get_key(x_api_key: Optional[str], form_key: Optional[str]) -> Optional[str]:
    """Return the API key from header (preferred) or form field. Never log it."""
    return x_api_key or form_key or None


def _prep_cutout_and_scale(
    pendant_file: UploadFile,
    ref_px: int,
    ref_mm: float,
    target_mm: float,
    tolerance: int = 28,
) -> tuple[PILImage.Image, float, int]:
    raw = _load_image(pendant_file)
    cutout = trim_to_alpha(remove_background(raw, tolerance=tolerance))
    ppm = pixels_per_mm(ref_px, ref_mm)
    target_px = int(round(target_pixels(target_mm, ppm)))
    return cutout, ppm, target_px


def _hang_point(hang_x: Optional[int], hang_y: Optional[int],
                model_img: PILImage.Image) -> Point:
    w, h = model_img.size
    return Point(
        x=hang_x if hang_x is not None else w // 2,
        y=hang_y if hang_y is not None else int(h * 0.60),
    )


# ---------------------------------------------------------------------------
# POST /swap: full pipeline (composite or generate)
# ---------------------------------------------------------------------------

@app.post("/swap")
async def swap(
    model:       UploadFile = File(...),
    pendant:     UploadFile = File(...),
    target_mm:   float      = Form(_TARGET),
    ref_px:      int        = Form(_REF_PX),
    ref_mm:      float      = Form(_REF_MM),
    hang_x:      Optional[int]   = Form(None),
    hang_y:      Optional[int]   = Form(None),
    rotate_deg:  float      = Form(0.0),
    top_crop_px: int        = Form(0),
    max_retries: int        = Form(4),
    mode:        str        = Form("composite"),
    model_id:    str        = Form("gemini-3.1-flash-image"),
    extra_prompt: str       = Form(""),
    composite_finish: str   = Form("false"),
    replace_chain: str      = Form("false"),
    api_key_form: Optional[str] = Form(None, alias="api_key"),
    x_api_key:   Optional[str] = Header(None),
) -> JSONResponse:
    """Full swap pipeline.

    mode='composite': no AI key required.
    mode='generate':  requires API key in X-API-Key header or api_key form field.
    """
    key = _get_key(x_api_key, api_key_form)

    if mode == "generate" and not key:
        raise HTTPException(400, "API key required for generate mode. "
                            "Pass it in the X-API-Key header or api_key field.")

    model_img = _load_image(model)
    cutout, ppm, target_px = _prep_cutout_and_scale(pendant, ref_px, ref_mm, target_mm)
    hang = _hang_point(hang_x, hang_y, model_img)

    if mode == "composite":
        result = composite_pendant(
            model_img, cutout,
            scale_width_px=target_px,
            hang_xy=hang,
            rotate_deg=rotate_deg,
            top_crop_px=top_crop_px,
        )
        return JSONResponse({
            "mode": "composite",
            "result_image": _img_to_b64(result),
            "ppm": round(ppm, 3),
            "target_px": target_px,
            "hang_xy": {"x": hang.x, "y": hang.y},
        })

    # generate mode
    from pendant_swap.loop import run_swap
    w, h = model_img.size
    params = SwapParams(
        model_path="",   # loop.py will use the image objects directly via prep bypass
        pendant_path="",
        target_mm=target_mm,
        ref_px_height=ref_px,
        ref_mm=ref_mm,
        hang_x=hang.x,
        hang_y=hang.y,
        rotate_deg=rotate_deg,
        top_crop_px=top_crop_px,
        max_retries=max_retries,
        mode="generate",
        api_key=key,
        model_id=model_id,
        extra_prompt=extra_prompt,
        composite_finish=composite_finish.lower() not in ("false", "0", "no", "off"),
        replace_chain=replace_chain.lower() in ("true", "1", "yes", "on"),
    )
    # loop.py uses file paths for prep; for the API we write to temp files
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as tmp:
        mp = pathlib.Path(tmp) / "model.png"
        pp = pathlib.Path(tmp) / "pendant.png"
        model_img.save(mp)
        # Re-save the raw pendant (pre-cutout) so loop.py can do its own prep
        pendant.file.seek(0)
        raw_pendant = PILImage.open(io.BytesIO(pendant.file.read())).convert("RGB")
        raw_pendant.save(pp)
        params.model_path = str(mp)
        params.pendant_path = str(pp)
        try:
            swap_result = run_swap(params)
        except (RuntimeError, Exception) as exc:
            raise HTTPException(400, str(exc))

    # Build a per-attempt list: clean image, annotated (QA box) image, summary, score, pass.
    attempts = []
    for i, rep in enumerate(swap_result.qa_reports):
        clean = swap_result.attempt_images[i] if i < len(swap_result.attempt_images) else None
        attempts.append({
            "index": i,
            "image": _img_to_b64(clean) if clean is not None else None,
            "annotated": _img_to_b64(rep.annotated_image) if rep.annotated_image else None,
            "summary": rep.summary,
            "passed": rep.passed,
            "score": swap_result.attempt_scores[i] if i < len(swap_result.attempt_scores) else None,
            "height_mm": round(rep.pendant_height_mm.value, 2),
            "aspect": round(rep.aspect_ratio.value, 3),
        })

    payload: dict = {
        "mode": "generate",
        "result_image": _img_to_b64(swap_result.final_image),
        "chosen_attempt": swap_result.chosen_attempt,
        "qa_reports": [r.summary for r in swap_result.qa_reports],
        "attempts": attempts,
        "prompts_used": swap_result.prompts_used,
        "ppm": round(ppm, 3),
    }
    if swap_result.gen_image_size:
        payload["gen_image_size"] = list(swap_result.gen_image_size)
    if swap_result.final_qa:
        payload["final_qa"] = swap_result.final_qa.summary
    return JSONResponse(payload)


# ---------------------------------------------------------------------------
# POST /prep
# ---------------------------------------------------------------------------

@app.post("/prep")
async def prep(
    model:      UploadFile = File(...),
    pendant:    UploadFile = File(...),
    target_mm:  float = Form(_TARGET),
    ref_px:     int   = Form(_REF_PX),
    ref_mm:     float = Form(_REF_MM),
    hang_x:     Optional[int] = Form(None),
    hang_y:     Optional[int] = Form(None),
    tolerance:  int   = Form(28),
) -> JSONResponse:
    """Return the cutout and guide overlay as base64 images."""
    model_img = _load_image(model)
    cutout, ppm, target_px = _prep_cutout_and_scale(pendant, ref_px, ref_mm, target_mm, tolerance)
    hang = _hang_point(hang_x, hang_y, model_img)
    guide = make_guide(model_img, cutout, target_px_height=target_px, hang_xy=hang)
    return JSONResponse({
        "cutout": _img_to_b64(cutout.convert("RGB")),
        "guide":  _img_to_b64(guide),
        "ppm": round(ppm, 3),
        "target_px": target_px,
        "hang_xy": {"x": hang.x, "y": hang.y},
    })


# ---------------------------------------------------------------------------
# POST /composite
# ---------------------------------------------------------------------------

@app.post("/composite")
async def composite(
    model:       UploadFile = File(...),
    pendant:     UploadFile = File(...),
    target_mm:   float = Form(_TARGET),
    ref_px:      int   = Form(_REF_PX),
    ref_mm:      float = Form(_REF_MM),
    hang_x:      Optional[int] = Form(None),
    hang_y:      Optional[int] = Form(None),
    rotate_deg:  float = Form(0.0),
    top_crop_px: int   = Form(0),
    tolerance:   int   = Form(28),
) -> JSONResponse:
    """Real-pixel composite. No API key required."""
    model_img = _load_image(model)
    cutout, ppm, target_px = _prep_cutout_and_scale(pendant, ref_px, ref_mm, target_mm, tolerance)
    hang = _hang_point(hang_x, hang_y, model_img)
    result = composite_pendant(
        model_img, cutout,
        scale_width_px=target_px,
        hang_xy=hang,
        rotate_deg=rotate_deg,
        top_crop_px=top_crop_px,
    )
    return JSONResponse({
        "result_image": _img_to_b64(result),
        "ppm": round(ppm, 3),
        "target_px": target_px,
        "hang_xy": {"x": hang.x, "y": hang.y},
    })


# ---------------------------------------------------------------------------
# POST /qa
# ---------------------------------------------------------------------------

@app.post("/qa")
async def qa_endpoint(
    result:    UploadFile = File(...),
    target_mm: float = Form(_TARGET),
    ref_px:    int   = Form(_REF_PX),
    ref_mm:    float = Form(_REF_MM),
    search_x0: int   = Form(...),
    search_y0: int   = Form(...),
    search_x1: int   = Form(...),
    search_y1: int   = Form(...),
    annotate:  bool  = Form(False),
) -> JSONResponse:
    """Measure a result image and return a QA report."""
    img = _load_image(result)
    ppm = pixels_per_mm(ref_px, ref_mm)
    report = qa_report(
        img,
        target_mm=target_mm,
        ppm=ppm,
        search_bbox=(search_x0, search_y0, search_x1, search_y1),
        annotate=annotate,
    )
    payload: dict = {
        "passed": report.passed,
        "summary": report.summary,
        "checks": {
            "pendant_height_mm": {
                "value": report.pendant_height_mm.value,
                "target": report.pendant_height_mm.target,
                "passed": report.pendant_height_mm.passed,
            },
            "aspect_ratio": {
                "value": report.aspect_ratio.value,
                "target": report.aspect_ratio.target,
                "passed": report.aspect_ratio.passed,
            },
        },
    }
    if report.chain_color:
        payload["checks"]["chain_color"] = {
            "value": report.chain_color.value,
            "target": report.chain_color.target,
            "passed": report.chain_color.passed,
        }
    if annotate and report.annotated_image:
        payload["annotated_image"] = _img_to_b64(report.annotated_image)
    return JSONResponse(payload)


# ---------------------------------------------------------------------------
# POST /finish
# ---------------------------------------------------------------------------

@app.post("/finish")
async def finish(
    image:             UploadFile    = File(...),
    watermark_bbox:    Optional[str] = Form(None, description="'X0 Y0 X1 Y1'"),
    watermark_method:  str           = Form("inpaint"),
    crop_trbl:         Optional[str] = Form(None, description="'TOP RIGHT BOTTOM LEFT'"),
    upscale_factor:    Optional[float] = Form(None),
) -> JSONResponse:
    """Watermark removal, crop, and upscale."""
    img = _load_image(image)

    if watermark_bbox:
        x0, y0, x1, y1 = [int(v) for v in watermark_bbox.split()]
        img = remove_watermark(img, (x0, y0, x1, y1), method=watermark_method)

    if crop_trbl:
        t, r, b, l = [int(v) for v in crop_trbl.split()]
        img = crop_margins(img, top=t, right=r, bottom=b, left=l)

    if upscale_factor:
        img = upscale(img, upscale_factor)

    return JSONResponse({"result_image": _img_to_b64(img)})


# ---------------------------------------------------------------------------
# GET /
# ---------------------------------------------------------------------------

@app.get("/")
def root() -> dict:
    return {
        "service": "pendant-swap API",
        "docs": "/docs",
        "ui": "/web/index.html",
    }
