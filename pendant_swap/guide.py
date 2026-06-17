"""Placement guide overlay.

Scales the cutout to the target height, optionally rotates it, applies an
opacity mask, and composites it semi-transparently onto the model image.
The guide is used both as a visual reference and as an input to the generation
backend (so the model knows exactly where and how big the pendant should be).
"""

from __future__ import annotations

from PIL import Image

from .types import Point


def make_guide(
    model_img: Image.Image,
    cutout: Image.Image,
    target_px_height: int,
    hang_xy: Point,
    opacity: float = 0.78,
    rotate_deg: float = 0,
) -> Image.Image:
    """Overlay the cutout semi-transparently on the model image.

    The cutout is scaled so its height equals `target_px_height`, then placed
    with its **top edge at hang_xy.y** and **horizontally centred at hang_xy.x**.

    Args:
        model_img: Background model photo.
        cutout: RGBA pendant cutout (any mode accepted; converted internally).
        target_px_height: Desired height of the pendant in the guide (pixels).
        hang_xy: (x, y) pixel coordinate.  The pendant is centred on x and its
            top is aligned to y.
        opacity: Alpha multiplier applied to the cutout (0.0–1.0).
        rotate_deg: Clockwise rotation in degrees applied after scaling.

    Returns:
        RGB guide image (same size as model_img).
    """
    pendant = cutout.convert("RGBA")
    orig_w, orig_h = pendant.size
    if orig_h == 0 or target_px_height <= 0:
        return model_img.convert("RGB")

    # Scale to target height, preserve aspect ratio.
    scale = target_px_height / orig_h
    new_w = max(1, round(orig_w * scale))
    new_h = max(1, round(orig_h * scale))
    scaled = pendant.resize((new_w, new_h), Image.LANCZOS)

    # Rotate (clockwise) with transparent fill; expand keeps it fully visible.
    if rotate_deg != 0:
        scaled = scaled.rotate(-rotate_deg, expand=True, resample=Image.BICUBIC)

    # Apply opacity to the alpha channel.
    r, g, b, a = scaled.split()
    a = a.point(lambda v: int(v * opacity))
    scaled = Image.merge("RGBA", (r, g, b, a))

    # Paste: centred on hang_xy.x, top at hang_xy.y.
    paste_x = hang_xy.x - scaled.width // 2
    paste_y = hang_xy.y
    guide = model_img.copy().convert("RGBA")
    guide.paste(scaled, (paste_x, paste_y), scaled)

    return guide.convert("RGB")
