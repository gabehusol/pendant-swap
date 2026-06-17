"""Scale math: pixels-per-mm calibration and target-size derivation.

Pure arithmetic — no I/O, no images. Fully unit-tested.

Reference calibration (verified by hand):
  ref_px_height = 130 px  →  ref_mm = 13 mm  →  ppm = 10 px/mm
  target_mm = 21  →  target_px = 210
"""

from __future__ import annotations


def pixels_per_mm(ref_pixel_height: float, ref_mm: float) -> float:
    """Return the px/mm scale factor given a reference pendant's pixel height and real height.

    Args:
        ref_pixel_height: Height of the reference pendant in pixels.
        ref_mm: Real-world height of that reference pendant in millimetres.

    Returns:
        Scale factor in pixels per millimetre.

    Raises:
        ValueError: If either argument is non-positive.
    """
    if ref_pixel_height <= 0:
        raise ValueError(f"ref_pixel_height must be positive, got {ref_pixel_height}")
    if ref_mm <= 0:
        raise ValueError(f"ref_mm must be positive, got {ref_mm}")
    return ref_pixel_height / ref_mm


def target_pixels(target_mm: float, ppm: float) -> float:
    """Return the pixel size that corresponds to target_mm at the given scale.

    Args:
        target_mm: Desired real-world size in millimetres.
        ppm: Scale factor in pixels per millimetre (from :func:`pixels_per_mm`).

    Returns:
        Target size in pixels (float; round or int-cast as needed by the caller).

    Raises:
        ValueError: If either argument is non-positive.
    """
    if target_mm <= 0:
        raise ValueError(f"target_mm must be positive, got {target_mm}")
    if ppm <= 0:
        raise ValueError(f"ppm must be positive, got {ppm}")
    return float(target_mm * ppm)


def scale_for_image(
    ref_pixel_height: float,
    ref_mm: float,
    target_mm: float,
) -> tuple[float, float]:
    """Convenience wrapper: compute ppm and target_px in one call.

    Returns:
        (ppm, target_px) tuple.
    """
    ppm = pixels_per_mm(ref_pixel_height, ref_mm)
    target_px = target_pixels(target_mm, ppm)
    return ppm, target_px
