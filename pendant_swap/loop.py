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

from .cutout import isolate_pendant, remove_background, trim_to_alpha
from .generate import GeminiEditor, ImageEditor
from .guide import make_guide
from .qa import qa_report, report_from_bbox
from .scale import pixels_per_mm, target_pixels
from .types import Point, QAReport, SwapParams, SwapResult


# ---------------------------------------------------------------------------
# Default prompt template
# ---------------------------------------------------------------------------

_BASE_PROMPT = """\
You are editing a product photography image.

The FIRST image is the exact pendant to use — study it carefully.
The SECOND image is the model photo to edit.
The THIRD image is a placement guide showing target size and position.

Task: Replace the existing pendant on the necklace with the pendant from the \
first image. Target real-world size: approximately {target_mm:.0f} mm tall.

Rules:
- Remove the original pendant completely. It must not appear anywhere in the output.
- Reproduce the pendant from image 1 EXACTLY — the same shape, colours, materials, \
textures, gemstones, and details. Do NOT invent, simplify, restyle, or change it.
- SIZE IS CRITICAL: the pendant must match the size of the faint guide overlay in \
image 3 — it must NOT be larger. Err on the side of slightly too small rather than \
too large. A common mistake is making the pendant too big; do not.
- Hang the pendant from the chain at the position shown in the guide overlay.
- CONNECTION POINT: the chain attaches to the pendant's bail (the small connector \
loop) at the TOP of the pendant. Connect at the top only — do NOT attach the chain \
to the centre or body of the pendant, and do NOT let it overlap or cross the pendant.
- Do NOT add drop shadows, glows, or blur around the pendant.
{chain_rule}
- Keep the model, pose, skin, clothing, background, and lighting exactly as they are.
- Output the full photo at the same crop and framing as the second image.
""".strip()

# Refinement prompt: used on retries. Feeds the best attempt so far back to the
# model and asks ONLY for targeted adjustments, so attempts converge instead of
# being independent random samples.
_REFINE_PROMPT = """\
The FIRST image already shows the butterfly pendant on the necklace, but it needs \
small adjustments. The SECOND image is the exact butterfly pendant for reference.

Keep EVERYTHING in the first image identical — the model, the chain, the \
lighting, the background, and the butterfly's tiger-eye appearance and its position \
on the chain. Apply ONLY these specific changes:

{corrections}

Do not change anything else. Do not add shadows, glows, or blur. \
Output the same photo with only those adjustments applied.
""".strip()


# Variant used when replace_chain is set: the reference image shows the whole
# product (gold beaded chain + butterfly), so we replace the entire necklace.
_REPLACE_NECKLACE = """\
- ALSO replace the existing chain to match the chain shown in the reference product \
image (image 1) — the same style, colour, and links.
- The bail is a SMALL loop at the TOP of the pendant. The chain threads through this \
loop and the pendant hangs cleanly BELOW it. Keep the bail small, neat, and clearly \
separate — it must NOT be elongated, tube-like, connect to the centre/body of the \
pendant, or merge into the pendant.
- The whole necklace (chain + bail + pendant) must read as one matched set."""

_KEEP_CHAIN = """\
- Keep the existing chain exactly as it is — do not change it."""


def _build_prompt(params: SwapParams, corrections: list[str]) -> str:
    chain_rule = _REPLACE_NECKLACE if params.replace_chain else _KEEP_CHAIN
    prompt = _BASE_PROMPT.format(target_mm=params.target_mm, chain_rule=chain_rule)
    if params.extra_prompt and params.extra_prompt.strip():
        prompt += "\n\nAdditional instructions:\n" + params.extra_prompt.strip()
    if corrections:
        prompt += "\n\nCorrections from previous attempt:\n" + "\n".join(
            "- " + c for c in corrections
        )
    return prompt


def _build_refine_prompt(params: SwapParams, corrections: list[str]) -> str:
    body = "\n".join("- " + c for c in corrections) if corrections else \
        "- Fine-tune the pendant so it matches the reference exactly."
    prompt = _REFINE_PROMPT.format(corrections=body)
    if params.extra_prompt and params.extra_prompt.strip():
        prompt += "\n\nAlso: " + params.extra_prompt.strip()
    return prompt


# ---------------------------------------------------------------------------
# Prep helpers
# ---------------------------------------------------------------------------

def _prep(params: SwapParams) -> tuple[Image.Image, Image.Image, Image.Image, float, int, Point]:
    """Load images, remove background, compute scale, return working objects.

    Returns:
        (model_img, cutout, full_cutout, ppm, target_px, hang_xy)
        - cutout: pendant only (chain removed) — used for the guide and as the
          pendant-appearance reference.
        - full_cutout: pendant WITH its chain (background removed) — used as the
          reference when the user wants the chain replaced too.
    """
    model_img = Image.open(params.model_path).convert("RGB")
    raw_pendant = Image.open(params.pendant_path).convert("RGB")

    full_cutout = trim_to_alpha(remove_background(raw_pendant, tolerance=params.tolerance))
    cutout = isolate_pendant(full_cutout)   # drop any attached chain from the product photo

    ppm = pixels_per_mm(params.ref_px_height, params.ref_mm)
    target_px = int(round(target_pixels(params.target_mm, ppm)))

    w, h = model_img.size
    hang_x = params.hang_x if params.hang_x is not None else w // 2
    hang_y = params.hang_y if params.hang_y is not None else int(h * 0.60)
    hang_xy = Point(x=hang_x, y=hang_y)

    return model_img, cutout, full_cutout, ppm, target_px, hang_xy


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

    model_img, cutout, full_cutout, ppm, target_px, hang_xy = _prep(params)
    search_bbox = _search_bbox(hang_xy, target_px, model_img.size)

    # Pendant-appearance reference (image 1). When replacing the chain too, use the
    # full product cutout (pendant + chain) so the AI can copy the chain style;
    # otherwise use the chain-stripped pendant so no stray chain bleeds in.
    pendant_ref = full_cutout if params.replace_chain else cutout

    # The model renders the pendant ~40% larger than the guide shows, so draw the
    # guide smaller (guide_size_bias) to nudge the result toward the true target.
    # The guide always uses the clean pendant-only cutout (avoids a boxy frame).
    guide_px = max(8, int(target_px * params.guide_size_bias))
    guide = make_guide(
        model_img, cutout,
        target_px_height=guide_px,
        hang_xy=hang_xy,
        opacity=0.45,   # subtle hint — too strong and the AI redraws its frame
        rotate_deg=params.rotate_deg,
    )

    qa_reports: list[QAReport] = []
    prompts_used: list[str] = []
    attempts: list[Image.Image] = []

    # Each attempt is INDEPENDENT (fresh generation from cutout+model+guide), not a
    # refinement of the previous one. The guide bias controls size at generation
    # time, so we no longer need a feedback loop — independent attempts give the
    # user real variety to choose from and avoid compounding artifacts (e.g. a
    # stray guide frame getting re-drawn each pass). We generate the full set so
    # there's always a spread to pick the best from.
    for attempt in range(params.max_retries + 1):
        prompt = _build_prompt(params, [])
        base = [pendant_ref, model_img, guide]
        prompts_used.append(prompt)

        # api_key flows directly to the editor — never logged here
        result_img = editor.edit(
            base_images=base,
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
            annotate=True,
        )
        qa_reports.append(report)

    # Choose the best attempt (the passing one, or closest by score)
    scores = [_score(r) for r in qa_reports]
    best_idx = _best_attempt(qa_reports)
    best_img = attempts[best_idx]
    gen_w, gen_h = best_img.size
    fqa = None

    final_img = best_img
    if params.composite_finish:
        final_img, placed_bbox = _size_lock(best_img, model_img, target_px,
                                            hang_xy, ppm, params)
        if placed_bbox is not None:
            # We placed the pendant; measure the exact bbox we produced (no
            # re-segmentation noise from the inpainted background).
            fw, fh = final_img.size
            sx, sy = fw / model_img.size[0], fh / model_img.size[1]
            scaled_ppm = ppm * (sx + sy) / 2
            fqa = report_from_bbox(placed_bbox, target_mm=params.target_mm,
                                   ppm=scaled_ppm)

    return SwapResult(
        final_image=final_img,
        chosen_attempt=best_idx,
        qa_reports=qa_reports,
        prompts_used=prompts_used,
        final_qa=fqa,
        gen_image_size=(gen_w, gen_h),
        attempt_images=attempts,
        attempt_scores=scores,
    )


def _size_lock(
    gen_img: Image.Image,
    model_img: Image.Image,
    target_px: int,
    hang_xy: Point,
    ppm: float,
    params: SwapParams,
) -> "tuple[Image.Image, Optional[tuple]]":
    """Rescale the AI's OWN rendered pendant to the exact target size, in place.

    The AI renders a beautifully-integrated pendant (correct lighting, shadow,
    chain connection) but won't hit a precise physical size. So we:
      1. Segment the AI's pendant (reusing the QA mask).
      2. Cut it out as RGBA, scale it by target/measured.
      3. Inpaint the original (too-large) pendant out of the base.
      4. Paste the resized pendant back, anchoring the bail (top-centre) so the
         chain connection is preserved.

    This keeps the AI's rendering quality while guaranteeing the size.
    Returns the original image unchanged if the pendant can't be measured or is
    already within tolerance.
    """
    import cv2
    import numpy as np
    from PIL import Image as _PIL
    from .qa import measure_pendant

    fw, fh = gen_img.size
    mw, mh = model_img.size
    sx, sy = fw / mw, fh / mh
    scaled_ppm = ppm * (sx + sy) / 2
    scaled_hang = Point(x=int(hang_xy.x * sx), y=int(hang_xy.y * sy))
    scaled_px = int(target_px * (sx + sy) / 2)
    search_bbox = _search_bbox(scaled_hang, scaled_px, (fw, fh))

    bbox, mask = measure_pendant(gen_img, search_bbox, return_mask=True)
    if bbox is None:
        return gen_img, None   # couldn't find the pendant; leave AI output as-is

    bx0, by0, bx1, by1 = bbox
    measured_h = by1 - by0
    target_h_px = params.target_mm * scaled_ppm
    scale = target_h_px / measured_h if measured_h > 0 else 1.0

    # Skip if implausible or already close enough (avoid needless quality loss)
    if not (0.25 < scale < 1.25) or (0.93 <= scale <= 1.07):
        return gen_img, bbox

    arr = np.array(gen_img.convert("RGB"))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # 1. Extract the pendant as RGBA. Dilate the dark-wing mask outward so the
    #    adjacent bright gold border is included, then close holes so the whole
    #    pendant body is opaque. Feather the edge for a clean composite.
    pad = max(int(measured_h * 0.12), 3)
    px0, py0 = max(0, bx0 - pad), max(0, by0 - pad)
    px1, py1 = min(fw, bx1 + pad), min(fh, by1 + pad)
    patch = arr[py0:py1, px0:px1].copy()
    pm = mask[py0:py1, px0:px1]
    pm = cv2.dilate(pm, kernel, iterations=3)               # grab the gold border
    pm = cv2.morphologyEx(pm, cv2.MORPH_CLOSE, kernel, iterations=3)  # fill holes
    patch_alpha = cv2.GaussianBlur(pm, (0, 0), 1.5)         # feathered edge
    pend = _PIL.fromarray(np.dstack([patch, patch_alpha]).astype(np.uint8), "RGBA")

    # 2. Inpaint the original pendant out of the base.
    #    Use a filled ellipse over the WHOLE pendant region (not the warm-dark QA
    #    mask) — otherwise the bright gold border (luma>165, absent from the mask)
    #    is left behind as a halo ring. The new pendant covers the centre; the
    #    inpainted skin around it is smooth so a generous fill looks natural.
    #    Margin is generous (40% of the pendant size) so no edge of the original
    #    AI pendant survives as a faint 'second pendant' ghost.
    base_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    inpaint_mask = np.zeros(arr.shape[:2], dtype=np.uint8)
    ecx, ecy = (bx0 + bx1) // 2, (by0 + by1) // 2
    eax = int((bx1 - bx0) / 2 * 1.4 + pad)
    eay = int((by1 - by0) / 2 * 1.4 + pad)
    cv2.ellipse(inpaint_mask, (ecx, ecy), (eax, eay), 0, 0, 360, 255, -1)
    clean_bgr = cv2.inpaint(base_bgr, inpaint_mask, 8, cv2.INPAINT_TELEA)
    base = _PIL.fromarray(cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2RGB))

    # 3. Scale the pendant
    new_w = max(1, int(round(pend.width * scale)))
    new_h = max(1, int(round(pend.height * scale)))
    pend_small = pend.resize((new_w, new_h), _PIL.LANCZOS)

    # 4. Paste anchored at the bail (top-centre) so the chain connection is kept.
    #    The bail point in patch coords is (bx0+bx1)/2 - px0 horizontally, by0 - py0
    #    vertically; scale those offsets so the bail stays at the same image point.
    bail_x = (bx0 + bx1) / 2.0
    off_x = (bail_x - px0) * scale
    off_y = (by0 - py0) * scale
    paste_x = int(round(bail_x - off_x))
    paste_y = int(round(by0 - off_y))
    base.paste(pend_small, (paste_x, paste_y), pend_small)

    # The placed pendant's tight bbox = original tight bbox mapped through the
    # same scale/anchor. We know it exactly, so QA can use it without re-segmenting.
    placed = (
        paste_x + int((bx0 - px0) * scale),
        paste_y + int((by0 - py0) * scale),
        paste_x + int((bx1 - px0) * scale),
        paste_y + int((by1 - py0) * scale),
    )
    return base, placed


def _corrections_from_report(report: QAReport, params: SwapParams) -> list[str]:
    """Derive corrective hint strings from a failed QA report.

    Size hints are quantitative: we know the measured height and the target, so we
    tell the model the exact scale factor (e.g. 'make it 31% smaller') instead of
    a vague 'make it smaller'. This makes the refinement loop actually converge.
    """
    hints = []
    h = report.pendant_height_mm
    if not h.passed and h.value > 0:
        ratio = h.target / h.value           # e.g. 21 / 30.67 = 0.685
        pct = abs(round((1 - ratio) * 100))
        scale_to = round(ratio * 100)
        if h.value > h.target:
            hints.append(
                "Make the butterfly pendant about %d%% SMALLER "
                "(scale it down to roughly %d%% of its current size). "
                "Keep it centred at the same spot on the chain." % (pct, scale_to)
            )
        else:
            hints.append(
                "Make the butterfly pendant about %d%% LARGER "
                "(scale it up to roughly %d%% of its current size). "
                "Keep it centred at the same spot on the chain." % (pct, scale_to)
            )
    if not report.aspect_ratio.passed:
        a = report.aspect_ratio
        if a.value < a.target:
            hints.append("The butterfly looks too narrow/tall — widen the wings so it is "
                         "slightly wider than it is tall.")
        else:
            hints.append("The butterfly looks too wide/flat — make it a little taller so the "
                         "wings are more balanced.")
    if report.chain_color and not report.chain_color.passed:
        hints.append("Ensure the chain stays its original silver colour.")
    return hints


def _score(r: QAReport) -> float:
    """Score a QA report: count passing checks, break ties by height accuracy."""
    passing = sum([
        r.pendant_height_mm.passed,
        r.aspect_ratio.passed,
        (r.chain_color.passed if r.chain_color else True),
    ])
    height_err = abs(r.pendant_height_mm.value - r.pendant_height_mm.target)
    return round(passing - height_err * 0.01, 3)  # lower height error is slightly better


def _best_attempt(reports: list[QAReport]) -> int:
    """Return the index of the closest-to-passing attempt."""
    return max(range(len(reports)), key=lambda i: _score(reports[i]))
