"""Cleanup and export utilities.

finish.py is always run on an already-generated or already-composited image.
None of these functions call any external API.

Notes:
- `upscale` uses Lanczos resampling, which is a stopgap.  A dedicated upscaler
  model (e.g. Real-ESRGAN) will produce noticeably sharper results and is the
  recommended upgrade for production use.
- `remove_watermark` with method="inpaint" uses OpenCV TELEA inpainting.  It
  works well for small, localized marks (logos, sparkle overlays) with a radius
  of a few pixels.  Large marks or coloured gradients may need corner_patch.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image

BBox = Tuple[int, int, int, int]   # x0, y0, x1, y1


# ---------------------------------------------------------------------------
# Watermark removal
# ---------------------------------------------------------------------------

def remove_watermark(
    img: Image.Image,
    bbox: BBox,
    method: str = "inpaint",
    source_img: Optional[Image.Image] = None,
) -> Image.Image:
    """Remove a watermark / logo within bbox.

    Args:
        img: RGB image containing the watermark.
        bbox: (x0, y0, x1, y1) bounding box of the watermark region.
        method: ``"inpaint"`` (default) — detect bright/grayish pixels,
            dilate the mask slightly, and inpaint with TELEA.
            ``"corner_patch"`` — copy a clean patch from another region of
            the image (or from ``source_img``) over the watermark bbox with a
            feathered blend.
        source_img: Optional clean source image for corner_patch mode.  If
            None, the patch is taken from a horizontally offset region of
            ``img`` itself.

    Returns:
        Cleaned RGB image.
    """
    if method == "inpaint":
        return _inpaint_watermark(img, bbox)
    elif method == "corner_patch":
        return _corner_patch_watermark(img, bbox, source_img)
    else:
        raise ValueError("method must be 'inpaint' or 'corner_patch', got %r" % method)


def _inpaint_watermark(img: Image.Image, bbox: BBox) -> Image.Image:
    """Detect bright+grayish (logo/sparkle) pixels and TELEA-inpaint them."""
    x0, y0, x1, y1 = bbox
    arr = np.array(img.convert("RGB"))
    h, w = arr.shape[:2]

    # Clamp bbox to image bounds
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)

    region = arr[y0:y1, x0:x1].astype(float)
    r, g, b = region[:, :, 0], region[:, :, 1], region[:, :, 2]
    luma = (r + g + b) / 3.0
    # Low colour saturation: max channel - min channel is small relative to luma
    sat = (np.maximum(r, np.maximum(g, b)) - np.minimum(r, np.minimum(g, b))) / (luma + 1e-6)

    # Watermark pixels: bright (luma > 180) AND near-neutral (low saturation)
    logo_mask = ((luma > 180) & (sat < 0.25)).astype(np.uint8) * 255

    # Dilate slightly to feather edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.dilate(logo_mask, kernel, iterations=1)

    # Paste mask back into a full-image mask
    full_mask = np.zeros((h, w), dtype=np.uint8)
    full_mask[y0:y1, x0:x1] = dilated

    # OpenCV TELEA inpaint (works on BGR)
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    inpainted_bgr = cv2.inpaint(bgr, full_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
    inpainted_rgb = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(inpainted_rgb, "RGB")


def _corner_patch_watermark(
    img: Image.Image,
    bbox: BBox,
    source_img: Optional[Image.Image],
) -> Image.Image:
    """Copy a clean patch over the watermark bbox with a feathered alpha blend."""
    x0, y0, x1, y1 = bbox
    bw, bh = x1 - x0, y1 - y0

    arr = np.array(img.convert("RGB")).astype(float)
    h, w = arr.shape[:2]

    if source_img is not None:
        src_arr = np.array(source_img.convert("RGB")).astype(float)
        patch = src_arr[y0:y0+bh, x0:x0+bw]
    else:
        # Offset patch horizontally (mirror or shift by bbox width, wrap if needed)
        offset_x = x0 + bw + 10
        if offset_x + bw > w:
            offset_x = max(0, x0 - bw - 10)
        patch = arr[y0:y0+bh, offset_x:offset_x+bw]

    if patch.shape[0] != bh or patch.shape[1] != bw:
        # Fallback: just use a copy of the region itself (no-op, but safe)
        patch = arr[y0:y0+bh, x0:x0+bw].copy()

    # Feather mask: gaussian-weighted blend, full in centre, fade at edges
    feather = _feather_mask(bh, bw, sigma_fraction=0.35)
    blended = feather * patch + (1 - feather) * arr[y0:y0+bh, x0:x0+bw]
    result = arr.copy()
    result[y0:y0+bh, x0:x0+bw] = blended
    return Image.fromarray(result.clip(0, 255).astype(np.uint8), "RGB")


def _feather_mask(h: int, w: int, sigma_fraction: float = 0.35) -> np.ndarray:
    """Return an (h, w, 1) Gaussian feather mask, peak 1.0 at centre."""
    cy, cx = h / 2.0, w / 2.0
    sy, sx = h * sigma_fraction, w * sigma_fraction
    y = np.arange(h) - cy
    x = np.arange(w) - cx
    Y, X = np.meshgrid(y, x, indexing="ij")
    mask = np.exp(-0.5 * ((Y / sy) ** 2 + (X / sx) ** 2))
    return mask[:, :, np.newaxis]


# ---------------------------------------------------------------------------
# Crop
# ---------------------------------------------------------------------------

def crop_margins(
    img: Image.Image,
    top: int,
    right: int,
    bottom: int,
    left: int,
) -> Image.Image:
    """Crop fixed margins from each edge.

    Args:
        img: Input image.
        top, right, bottom, left: Pixels to remove from each edge.

    Returns:
        Cropped image.
    """
    w, h = img.size
    x0 = max(0, left)
    y0 = max(0, top)
    x1 = max(x0, w - right)
    y1 = max(y0, h - bottom)
    return img.crop((x0, y0, x1, y1))


# ---------------------------------------------------------------------------
# Upscale
# ---------------------------------------------------------------------------

def upscale(img: Image.Image, factor: float) -> Image.Image:
    """Lanczos upscale by a given factor.

    This is a stopgap; a dedicated upscaler model (e.g. Real-ESRGAN) will
    produce substantially sharper results.

    Args:
        img: Input image.
        factor: Scale multiplier (e.g. 2.0 → 2× upscale).

    Returns:
        Upscaled image.
    """
    if factor <= 0:
        raise ValueError("factor must be positive, got %s" % factor)
    w, h = img.size
    new_w = max(1, round(w * factor))
    new_h = max(1, round(h * factor))
    return img.resize((new_w, new_h), Image.LANCZOS)
