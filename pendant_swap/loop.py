"""Generate → QA → retry orchestration.

run_swap() is the headline feature: it wraps an unreliable AI generation step
in deterministic prep (cutout, scale, guide, prompt) and a measuring QA gate
that rejects bad results and feeds corrective hints back into the next attempt.

Key invariant: the api_key in SwapParams is never logged, stored, or included
in any exception message.  It flows directly to GeminiEditor.edit() and is
held only in memory for the duration of each generate call.
"""

from __future__ import annotations

import io
from typing import Optional

from PIL import Image

from .composite import composite_pendant
from .cutout import remove_background, trim_to_alpha
from .generate import GeminiEditor, ImageEditor
from .guide import make_guide
from .qa import qa_report
from .scale import pixels_per_mm, target_pixels
from .types import Point, QAReport, SwapParams, SwapResult


# ---------------------------------------------------------------------------
# Default prompt template
# ---------------------------------------------------------------------------

_BASE_PROMPT = """\
You are editing a product photography image.

Task: Replace the existing pendant on the necklace COMPLETELY with the exact \
butterfly pendant shown in the second and third reference images. \
The butterfly pendant should appear approximately {target_mm:.0f} mm tall in real life.

Rules:
- Remove the original pendant entirely — it must not appear in the output.
- Reproduce the butterfly pendant EXACTLY as shown in the reference cutout:
    * Two rounded butterfly wings made of tiger-eye stone (warm brown, chatoyant).
    * A thin gold/rose-gold outline border around each wing.
    * A vertical centre bar set with small round diamonds or CZs.
    * The overall shape is wide and compact — roughly as wide as it is tall.
- Do NOT simplify, stylize, or redesign the pendant. Match the reference precisely.
- Keep the chain, model, clothing, background, and lighting exactly as they are.
- Hang the butterfly naturally from the chain at the same position as the \
original pendant.
- The semi-transparent guide overlay in the third image shows the target size \
and position — match it exactly.
- Output the full photo at the same crop and framing as the input.
""".strip()

# Corrective hints appended on retry when a specific check fails.
_HINTS = {
    "too_large":   "The pendant is too large. Make it noticeably smaller — about {target_mm:.0f} mm tall.",
    "too_small":   "The pendant is too small. Make it noticeably larger — about {target_mm:.0f} mm tall.",
    "not_square":  "The butterfly shape looks stretched. Make it more compact and near-square.",
    "not_gold":    "Ensure the entire chain is warm gold coloured, not silver or grey.",
}


def _build_prompt(params: SwapParams, corrections: list[str]) -> str:
    prompt = _BASE_PROMPT.format(target_mm=params.target_mm)
    if params.extra_prompt and params.extra_prompt.strip():
        prompt += "\n\nAdditional instructions:\n" + params.extra_prompt.strip()
    if corrections:
        prompt += "\n\nCorrections from previous attempt:\n" + "\n".join(
            "- " + c for c in corrections
        )
    return prompt


# ---------------------------------------------------------------------------
# Prep helpers
# ---------------------------------------------------------------------------

def _prep(params: SwapParams) -> tuple[Image.Image, Image.Image, float, int, Point]:
    """Load images, remove background, compute scale, return working objects.

    Returns:
        (model_img, cutout, ppm, target_px, hang_xy)
    """
    model_img = Image.open(params.model_path).convert("RGB")
    raw_pendant = Image.open(params.pendant_path).convert("RGB")

    cutout = remove_background(raw_pendant, tolerance=params.tolerance)
    cutout = trim_to_alpha(cutout)

    ppm = pixels_per_mm(params.ref_px_height, params.ref_mm)
    target_px = int(round(target_pixels(params.target_mm, ppm)))

    w, h = model_img.size
    hang_x = params.hang_x if params.hang_x is not None else w // 2
    hang_y = params.hang_y if params.hang_y is not None else int(h * 0.60)
    hang_xy = Point(x=hang_x, y=hang_y)

    return model_img, cutout, ppm, target_px, hang_xy


def _search_bbox(hang_xy: Point, target_px: int, model_size: tuple[int, int]) -> tuple:
    """Derive a QA search bbox from the hang point and expected pendant size."""
    margin_x = int(target_px * 0.8)
    margin_y_up = int(target_px * 0.3)
    margin_y_down = int(target_px * 1.5)
    w, h = model_size
    x0 = max(0, hang_xy.x - margin_x)
    y0 = max(0, hang_xy.y - margin_y_up)
    x1 = min(w, hang_xy.x + margin_x)
    y1 = min(h, hang_xy.y + margin_y_down)
    return (x0, y0, x1, y1)


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def run_swap(
    params: SwapParams,
    editor: Optional[ImageEditor] = None,
) -> SwapResult:
    """Run the full generate → measure → retry loop.

    Args:
        params: All swap parameters including the api_key (passed through to
            the editor per-call; never stored).
        editor: Optional ImageEditor override for testing.  Defaults to
            GeminiEditor.

    Returns:
        SwapResult with the best attempt image, all QA reports, and prompts used.
    """
    if editor is None:
        editor = GeminiEditor()

    model_img, cutout, ppm, target_px, hang_xy = _prep(params)
    search_bbox = _search_bbox(hang_xy, target_px, model_img.size)

    guide = make_guide(
        model_img, cutout,
        target_px_height=target_px,
        hang_xy=hang_xy,
        opacity=0.78,
        rotate_deg=params.rotate_deg,
    )

    corrections: list[str] = []
    qa_reports: list[QAReport] = []
    prompts_used: list[str] = []
    attempts: list[Image.Image] = []

    for attempt in range(params.max_retries + 1):
        prompt = _build_prompt(params, corrections)
        prompts_used.append(prompt)

        # api_key flows directly to the editor — never logged here
        result_img = editor.edit(
            base_images=[model_img, cutout, guide],
            prompt=prompt,
            api_key=params.api_key or "",
            model_id=params.model_id,
        )
        attempts.append(result_img)

        # Gemini may output at a different resolution than the input.
        # Scale search_bbox and ppm proportionally so QA measures correctly.
        gen_w, gen_h = result_img.size
        mod_w, mod_h = model_img.size
        if (gen_w, gen_h) != (mod_w, mod_h):
            sx, sy = gen_w / mod_w, gen_h / mod_h
            scaled_bbox = (
                int(search_bbox[0] * sx), int(search_bbox[1] * sy),
                int(search_bbox[2] * sx), int(search_bbox[3] * sy),
            )
            scaled_ppm = ppm * ((sx + sy) / 2)
        else:
            scaled_bbox, scaled_ppm = search_bbox, ppm

        report = qa_report(
            result_img,
            target_mm=params.target_mm,
            ppm=scaled_ppm,
            search_bbox=scaled_bbox,
        )
        qa_reports.append(report)

        if report.passed:
            final = result_img
            fqa = None
            if params.composite_finish:
                final = _composite_finish(result_img, model_img, cutout,
                                          target_px, hang_xy, params)
                fqa = _final_qa(final, params, ppm, model_img.size, hang_xy, target_px)
            return SwapResult(
                final_image=final,
                chosen_attempt=attempt,
                qa_reports=qa_reports,
                prompts_used=prompts_used,
                final_qa=fqa,
                gen_image_size=(gen_w, gen_h),
            )

        # Build corrective hints for the next attempt
        corrections = _corrections_from_report(report, params)

    # No attempt passed — return the closest one by score
    best_idx = _best_attempt(qa_reports)
    best_img = attempts[best_idx]
    gen_w, gen_h = attempts[best_idx].size
    fqa = None

    if params.composite_finish:
        best_img = _composite_finish(best_img, model_img, cutout, target_px,
                                     hang_xy, params)
        fqa = _final_qa(best_img, params, ppm, model_img.size, hang_xy, target_px)

    return SwapResult(
        final_image=best_img,
        chosen_attempt=best_idx,
        qa_reports=qa_reports,
        prompts_used=prompts_used,
        final_qa=fqa,
        gen_image_size=(gen_w, gen_h),
    )


def _composite_finish(
    gen_img: Image.Image,
    model_img: Image.Image,
    cutout: Image.Image,
    target_px: int,
    hang_xy: Point,
    params: SwapParams,
) -> Image.Image:
    """Erase the AI-generated pendant, then paste the exact real cutout.

    Two-step:
      1. Inpaint the pendant zone in the AI image so the AI's pendant disappears
         and surrounding skin/background fills in naturally.
      2. Composite the real cutout at the precise calculated size/position.

    This avoids the 'double pendant' artifact caused by overlaying the real
    cutout on top of an AI-drawn pendant that sits at a slightly different position.
    """
    import cv2
    import numpy as np
    from PIL import Image as _PIL

    gen_w, gen_h = gen_img.size
    mod_w, mod_h = model_img.size
    if (gen_w, gen_h) != (mod_w, mod_h):
        sx, sy = gen_w / mod_w, gen_h / mod_h
        scaled_hang = Point(x=int(hang_xy.x * sx), y=int(hang_xy.y * sy))
        scaled_px = int(target_px * (sx + sy) / 2)
    else:
        scaled_hang = hang_xy
        scaled_px = target_px

    # Build an erase mask covering the expected pendant area.
    # Centre of pendant body sits ~half-height below the hang point.
    cw_r = cutout.width / cutout.height if cutout.height > 0 else 1.0
    half_h = max(scaled_px // 2, 4)
    half_w = max(int(half_h * cw_r), 4)
    cx = scaled_hang.x
    cy = scaled_hang.y + half_h          # centre of pendant body

    mask = np.zeros((gen_h, gen_w), dtype=np.uint8)
    # Ellipse padded by 30 % to cover any bleed from the AI-drawn pendant
    cv2.ellipse(mask, (cx, cy),
                (int(half_w * 1.3), int(half_h * 1.3)),
                0, 0, 360, 255, -1)

    # Inpaint: fill the masked area from surrounding pixels
    gen_bgr = cv2.cvtColor(np.array(gen_img.convert("RGB")), cv2.COLOR_RGB2BGR)
    clean_bgr = cv2.inpaint(gen_bgr, mask, inpaintRadius=8, flags=cv2.INPAINT_TELEA)
    base = _PIL.fromarray(cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2RGB))

    # Detect where the chain actually terminates in the AI image so the real
    # pendant connects at the right spot rather than our precomputed hang_xy.
    actual_hang = _find_chain_end(gen_img, scaled_hang, half_w, half_h) or scaled_hang

    # Paste the real cutout onto the clean base
    return composite_pendant(
        base, cutout,
        scale_width_px=scaled_px,
        hang_xy=actual_hang,
        rotate_deg=params.rotate_deg,
        top_crop_px=params.top_crop_px,
    )


def _find_chain_end(
    gen_img: Image.Image,
    approx_hang: Point,
    search_half_w: int,
    search_half_h: int,
) -> "Optional[Point]":
    """Find the lowest silver chain pixel above the pendant zone.

    Scans upward from approx_hang in a narrow column to find the last
    bright/neutral pixel that looks like a chain link. Returns None if
    detection is uncertain (caller falls back to approx_hang).
    """
    import numpy as np

    arr = np.array(gen_img.convert("RGB")).astype(float)
    h, w = arr.shape[:2]

    # Search column: narrow band centred on hang x, from above the pendant up
    x0 = max(0, approx_hang.x - search_half_w // 3)
    x1 = min(w, approx_hang.x + search_half_w // 3)
    y_top = max(0, approx_hang.y - search_half_h * 2)
    y_bot = approx_hang.y   # don't go into pendant zone

    if y_bot <= y_top or x1 <= x0:
        return None

    col = arr[y_top:y_bot, x0:x1]
    r, g, b = col[:, :, 0], col[:, :, 1], col[:, :, 2]
    luma = (r + g + b) / 3.0
    # Silver chain: bright (luma > 160) and low saturation (r≈g≈b)
    warmth = np.abs(r - b)
    is_chain = (luma > 160) & (warmth < 25)

    # Find the last (lowest) row that has chain pixels
    row_has_chain = is_chain.any(axis=1)
    chain_rows = np.where(row_has_chain)[0]
    if len(chain_rows) == 0:
        return None

    last_row = int(chain_rows.max())
    # x centre of chain pixels in that row
    chain_xs = np.where(is_chain[last_row])[0]
    cx = int(chain_xs.mean()) + x0
    cy = last_row + y_top

    return Point(x=cx, y=cy)


def _final_qa(
    final_img: Image.Image,
    params: SwapParams,
    ppm: float,
    model_size: tuple,
    hang_xy: Point,
    target_px: int,
) -> QAReport:
    """Run QA on the final composited image, scaling ppm and search_bbox to match
    the final image's resolution (which may differ from the original model image)."""
    fw, fh = final_img.size
    mw, mh = model_size
    if (fw, fh) != (mw, mh):
        sx, sy = fw / mw, fh / mh
        scaled_ppm = ppm * (sx + sy) / 2
        scaled_hang = Point(x=int(hang_xy.x * sx), y=int(hang_xy.y * sy))
        scaled_px = int(target_px * (sx + sy) / 2)
    else:
        scaled_ppm, scaled_hang, scaled_px = ppm, hang_xy, target_px
    search_bbox = _search_bbox(scaled_hang, scaled_px, (fw, fh))
    return qa_report(final_img, target_mm=params.target_mm, ppm=scaled_ppm,
                     search_bbox=search_bbox)


def _corrections_from_report(report: QAReport, params: SwapParams) -> list[str]:
    """Derive corrective hint strings from a failed QA report."""
    hints = []
    h = report.pendant_height_mm
    if not h.passed:
        if h.value > h.target:
            hints.append(_HINTS["too_large"].format(target_mm=params.target_mm))
        else:
            hints.append(_HINTS["too_small"].format(target_mm=params.target_mm))
    if not report.aspect_ratio.passed:
        hints.append(_HINTS["not_square"])
    if report.chain_color and not report.chain_color.passed:
        hints.append(_HINTS["not_gold"])
    return hints


def _best_attempt(reports: list[QAReport]) -> int:
    """Score each attempt and return the index of the closest-to-passing one."""
    def _score(r: QAReport) -> float:
        # Count passing checks; break ties by how close height is to target
        passing = sum([
            r.pendant_height_mm.passed,
            r.aspect_ratio.passed,
            (r.chain_color.passed if r.chain_color else True),
        ])
        height_err = abs(r.pendant_height_mm.value - r.pendant_height_mm.target)
        return passing - height_err * 0.01   # lower height error is slightly better

    return max(range(len(reports)), key=lambda i: _score(reports[i]))
