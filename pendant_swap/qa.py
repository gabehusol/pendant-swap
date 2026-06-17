"""QA measurement and pass/fail report.

Segmentation strategy: within the caller-supplied search_bbox, look for pixels
that are warm (R significantly > B) and darker than plain skin/background — this
isolates the pendant (tiger-eye brown + gold border) while rejecting the
silver chain and skin tone.  A simple density-based row/column scan then derives
the tightest bounding box around the cluster.

Callers must supply a search_bbox that excludes other warm regions (e.g. keep it
centered around where the pendant is expected).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .types import CheckResult, QAReport

BBox = Tuple[int, int, int, int]  # x0, y0, x1, y1


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

def measure_pendant(
    result_img: Image.Image,
    search_bbox: BBox,
) -> Optional[BBox]:
    """Segment the pendant within search_bbox and return its tight bounding box.

    Uses a warm-and-dark colour mask to isolate the pendant against skin/chain.
    Returns None if no pendant-like region is found.

    Args:
        result_img: RGB result image.
        search_bbox: (x0, y0, x1, y1) in image coords — region to search.

    Returns:
        (x0, y0, x1, y1) tight bbox around the detected pendant, or None.
    """
    x0, y0, x1, y1 = search_bbox
    img = result_img.convert("RGB")
    arr = np.array(img)

    region = arr[y0:y1, x0:x1].astype(float)
    if region.size == 0:
        return None

    r, g, b = region[:, :, 0], region[:, :, 1], region[:, :, 2]
    luma = (r + g + b) / 3.0

    # Tiger-eye/gold pendant: warm (R > B) and noticeably darker than skin/chain.
    # Skin and silver chain both have luma > 200 in typical model photos; the
    # pendant tiger-eye core is luma < 160 with strong warmth.
    warm = (r - b) > 20
    dark_enough = luma < 165
    bright_enough = luma > 15

    mask = (warm & dark_enough & bright_enough).astype(np.uint8) * 255

    # Morphological cleanup: close gaps inside the pendant body.
    # The bright diamond center bar separates the two wings into distinct blobs;
    # a larger kernel is needed to bridge that gap in scaled-down generated images.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)

    # Find the largest connected component — that's the pendant, not stray pixels.
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    if n_labels < 2:
        return None

    # stats columns: LEFT, TOP, WIDTH, HEIGHT, AREA (label 0 is background)
    areas = stats[1:, cv2.CC_STAT_AREA]
    best_label = int(areas.argmax()) + 1
    if areas[best_label - 1] < 50:   # ignore tiny specks
        return None

    best_mask = (labels == best_label).astype(np.uint8)
    ys, xs = np.where(best_mask > 0)
    if len(ys) == 0:
        return None

    # Convert back to full-image coords
    bx0 = int(xs.min()) + x0
    bx1 = int(xs.max()) + x0
    by0 = int(ys.min()) + y0
    by1 = int(ys.max()) + y0

    if bx1 <= bx0 or by1 <= by0:
        return None

    return (bx0, by0, bx1, by1)


# ---------------------------------------------------------------------------
# QA report
# ---------------------------------------------------------------------------

def qa_report(
    result_img: Image.Image,
    *,
    target_mm: float,
    ppm: float,
    search_bbox: BBox,
    aspect_target: float = 1.3,
    aspect_tol: float = 0.35,
    size_tol: float = 0.10,
    chain_region: Optional[BBox] = None,
    annotate: bool = False,
) -> QAReport:
    """Measure the result image and return a pass/fail QAReport.

    Args:
        result_img: RGB image to evaluate.
        target_mm: Expected pendant height in real-world mm.
        ppm: Scale factor in pixels per mm (from scale.pixels_per_mm).
        search_bbox: (x0, y0, x1, y1) region to search for the pendant.
        aspect_target: Target width/height ratio (default 1.3 — butterfly is wider than tall).
        aspect_tol: Max allowed deviation from aspect_target.
        size_tol: Fractional tolerance on pendant height (e.g. 0.10 = ±10%).
        chain_region: Optional (x0, y0, x1, y1) region to classify chain colour.
        annotate: If True, draw the measured bbox on a copy saved as
            QAReport.annotated_image.

    Returns:
        QAReport with per-check results and an overall passed flag.
    """
    bbox = measure_pendant(result_img, search_bbox)

    if bbox is None:
        # Can't measure — all checks fail
        not_found = CheckResult(value=0.0, target=target_mm, passed=False,
                                label="Pendant height (mm) — NOT FOUND in search region")
        ar_fail = CheckResult(value=0.0, target=1.0, passed=False,
                              label="Aspect ratio — pendant not found")
        chain_check = _classify_chain(result_img, chain_region) if chain_region else None
        return QAReport(
            pendant_height_mm=not_found,
            aspect_ratio=ar_fail,
            chain_color=chain_check,
            passed=False,
            summary="FAIL — pendant not found in search region",
        )

    bx0, by0, bx1, by1 = bbox
    width_px = bx1 - bx0
    height_px = by1 - by0

    # Height check
    height_mm = height_px / ppm
    height_pass = abs(height_mm - target_mm) / target_mm <= size_tol
    height_check = CheckResult(
        value=round(height_mm, 3),
        target=target_mm,
        passed=height_pass,
        label="Pendant height (mm)",
    )

    # Aspect ratio (width/height).
    # Butterfly pendant is wider than tall — default target ~1.3.
    aspect = width_px / height_px if height_px > 0 else 0.0
    aspect_pass = abs(aspect - aspect_target) <= aspect_tol
    aspect_check = CheckResult(
        value=round(aspect, 3),
        target=aspect_target,
        passed=aspect_pass,
        label="Aspect ratio (w/h, target wider-than-tall)",
    )

    # Chain colour (optional)
    chain_check = _classify_chain(result_img, chain_region) if chain_region else None

    checks = [height_check, aspect_check]
    if chain_check:
        checks.append(chain_check)

    passed_count = sum(c.passed for c in checks)
    overall = all(c.passed for c in checks)

    lines = [str(c) for c in checks]
    lines.append("Overall: %s (%d/%d checks)" % ("PASSED" if overall else "FAILED",
                                                   passed_count, len(checks)))
    summary = "\n".join(lines)

    annotated = None
    if annotate:
        annotated = _annotate(result_img, bbox, height_mm, target_mm, overall)

    return QAReport(
        pendant_height_mm=height_check,
        aspect_ratio=aspect_check,
        chain_color=chain_check,
        passed=overall,
        summary=summary,
        annotated_image=annotated,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _classify_chain(img: Image.Image, region: BBox) -> CheckResult:
    """Classify chain colour as gold (warm, saturated) or silver (cool/neutral)."""
    x0, y0, x1, y1 = region
    arr = np.array(img.convert("RGB"))[y0:y1, x0:x1].astype(float)
    if arr.size == 0:
        return CheckResult(value=0.0, target=1.0, passed=False, label="Chain color")

    r, g, b = arr[:, :, 0].mean(), arr[:, :, 1].mean(), arr[:, :, 2].mean()
    warmth = r - b          # positive → warm/gold; near-zero → silver
    luma = (r + g + b) / 3
    sat = (max(r, g, b) - min(r, g, b)) / (luma + 1e-6)

    is_gold = warmth > 15 and sat > 0.15
    label = "Chain color (gold=pass)" if is_gold else "Chain color (silver — expected gold)"
    return CheckResult(
        value=round(warmth, 1),
        target=15.0,
        passed=is_gold,
        label=label,
    )


def _annotate(
    img: Image.Image,
    bbox: BBox,
    measured_mm: float,
    target_mm: float,
    passed: bool,
) -> Image.Image:
    """Draw the measurement bbox and label on a copy of the image."""
    out = img.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    bx0, by0, bx1, by1 = bbox
    colour = (0, 200, 0) if passed else (220, 40, 40)
    draw.rectangle([bx0, by0, bx1, by1], outline=colour, width=4)
    label = "%.1fmm (target %.1fmm) %s" % (measured_mm, target_mm, "PASS" if passed else "FAIL")
    # Try to place text above the box; fall back below if near top
    ty = by0 - 40 if by0 > 50 else by1 + 10
    draw.rectangle([bx0, ty, bx0 + len(label) * 14, ty + 32], fill=(0, 0, 0))
    draw.text((bx0 + 4, ty + 4), label, fill=colour)
    return out
