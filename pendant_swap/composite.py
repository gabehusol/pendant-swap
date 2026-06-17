"""Real-pixel pendant composite — no AI required.

Steps (per spec):
1. Optionally zero the cutout's alpha above `top_crop_px` (drops product-photo
   chain above the bail so it doesn't ghost over the model's chain).
2. Rotate by `rotate_deg` (clockwise, expand=True) to deskew toward upright.
3. Measure the pendant's **wing width** as the maximum alpha-covered span across
   all rows — this is the widest point of the pendant face.
4. Scale so that wing width == `scale_width_px`.
5. Alpha-composite onto the model image, centred horizontally at hang_xy.x with
   the top of the (scaled) cutout placed at hang_xy.y.

Design note: we scale by *width* (the wing span) rather than height because for
a butterfly/medallion pendant the width is the most visible and stable dimension.
The height naturally follows from the aspect ratio of the real cutout.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from .types import Point


def composite_pendant(
    model_img: Image.Image,
    cutout: Image.Image,
    *,
    scale_width_px: int,
    hang_xy: Point,
    rotate_deg: float = 0.0,
    top_crop_px: int = 0,
) -> Image.Image:
    """Composite the pendant cutout onto the model photo at real-pixel quality.

    Args:
        model_img: The model/lifestyle photo.
        cutout: RGBA pendant cutout produced by :func:`cutout.remove_background`.
        scale_width_px: Target wing-width in pixels.  Derive from
            ``scale.target_pixels(target_mm, ppm)`` before calling.
        hang_xy: (x, y) in the model image.  The pendant is centred at x and
            its top is placed at y.
        rotate_deg: Clockwise rotation to deskew the pendant.  A few degrees
            (e.g. 3°) corrects typical product-photo tilt.
        top_crop_px: Number of rows from the top of the *cutout* whose alpha is
            forced to zero before placement.  Use to strip the product-photo
            chain/hook above the bail.

    Returns:
        RGB composite image (same size as model_img).
    """
    pendant = cutout.convert("RGBA")

    # 1. Zero alpha above bail (strip product-photo chain).
    if top_crop_px > 0:
        arr = np.array(pendant)
        arr[:top_crop_px, :, 3] = 0
        pendant = Image.fromarray(arr, "RGBA")

    # 2. Rotate to deskew (clockwise).
    if rotate_deg != 0:
        pendant = pendant.rotate(-rotate_deg, expand=True, resample=Image.BICUBIC)

    # 3. Measure wing width = widest alpha-covered row span.
    arr = np.array(pendant)
    alpha = arr[:, :, 3]
    row_coverage = (alpha > 0).sum(axis=1)  # non-zero pixels per row
    max_row_width = int(row_coverage.max())

    if max_row_width == 0:
        # Cutout is fully transparent — return model unchanged.
        return model_img.convert("RGB")

    # 4. Scale so wing width == scale_width_px.
    orig_w, orig_h = pendant.size
    scale = scale_width_px / max_row_width
    new_w = max(1, round(orig_w * scale))
    new_h = max(1, round(orig_h * scale))
    scaled = pendant.resize((new_w, new_h), Image.LANCZOS)

    # 5. Composite: top at hang_xy.y, centred horizontally at hang_xy.x.
    paste_x = hang_xy.x - scaled.width // 2
    paste_y = hang_xy.y

    result = model_img.copy().convert("RGBA")
    result.paste(scaled, (paste_x, paste_y), scaled)
    return result.convert("RGB")
