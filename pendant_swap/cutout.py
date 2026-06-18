"""Background removal via border-connected BFS flood fill.

Approach: average the four corner pixels to get an estimate of the background
colour. Compute per-pixel Euclidean distance to that colour in RGB space.
Seed a BFS queue from every border pixel whose distance is below `tolerance`,
then flood only connected pixels that are also below the threshold. This keeps
light regions *inside* the object opaque because they are not border-connected.
"""

from __future__ import annotations

from collections import deque

import numpy as np
from PIL import Image


def remove_background(img: Image.Image, tolerance: int = 28) -> Image.Image:
    """Return `img` with the border-connected background made transparent.

    Args:
        img: Input image (any mode; converted to RGBA internally).
        tolerance: Maximum Euclidean RGB distance from the estimated background
            colour for a pixel to be treated as background.  Lower → more
            conservative removal; higher → removes more fringe.

    Returns:
        RGBA image with background pixels set to alpha=0.
    """
    rgba = img.convert("RGBA")
    data = np.array(rgba, dtype=np.uint8)
    h, w = data.shape[:2]

    # Estimate background colour from the four corners.
    corners = np.array(
        [data[0, 0, :3], data[0, w - 1, :3], data[h - 1, 0, :3], data[h - 1, w - 1, :3]],
        dtype=float,
    )
    bg_color = corners.mean(axis=0)  # shape (3,)

    # Per-pixel Euclidean distance to bg_color (RGB channels only).
    rgb = data[:, :, :3].astype(float)
    diff = rgb - bg_color
    dist = np.sqrt((diff ** 2).sum(axis=2))  # shape (h, w)

    # BFS: seed from every border pixel that looks like background.
    is_bg = np.zeros((h, w), dtype=bool)
    queue: deque[tuple[int, int]] = deque()

    def _seed(y: int, x: int) -> None:
        if not is_bg[y, x] and dist[y, x] < tolerance:
            is_bg[y, x] = True
            queue.append((y, x))

    for y in range(h):
        _seed(y, 0)
        _seed(y, w - 1)
    for x in range(w):
        _seed(0, x)
        _seed(h - 1, x)

    # Flood-fill through connected background pixels.
    neighbours = ((-1, 0), (1, 0), (0, -1), (0, 1))
    while queue:
        y, x = queue.popleft()
        for dy, dx in neighbours:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and not is_bg[ny, nx] and dist[ny, nx] < tolerance:
                is_bg[ny, nx] = True
                queue.append((ny, nx))

    # Zero the alpha channel for every background pixel.
    result = data.copy()
    result[is_bg, 3] = 0
    return Image.fromarray(result, "RGBA")


def isolate_pendant(img: Image.Image) -> Image.Image:
    """Crop an RGBA cutout down to just the pendant, discarding an attached chain.

    Product photos often show the pendant hanging from a long chain. After
    background removal the chain survives (it's not background), so the cutout is
    pendant + a tall thin chain. The pendant is a WIDE compact blob; the chain is
    NARROW. We keep the contiguous band of rows whose opaque width is a healthy
    fraction of the widest row - that band is the pendant - and crop to it.

    Safe no-op: if the shape is already compact (no thin chain), the band spans
    the whole thing and nothing is cropped.
    """
    rgba = img.convert("RGBA")
    a = np.array(rgba)[:, :, 3]
    h, w = a.shape
    opaque = a > 128

    # Use opaque pixel COUNT per row, not width: the chain is thin (few pixels
    # even where its two strands splay wide), the pendant (wings + bail) is solid.
    row_count = opaque.sum(axis=1).astype(float)
    if row_count.max() <= 0:
        return rgba

    # Smooth so the bail's open loop (a low-count dip) doesn't fool the scan.
    k = max(1, int(h * 0.02))
    smooth = np.convolve(row_count, np.ones(k) / k, mode="same")
    max_c = smooth.max()

    # The chain sits far below the pendant in density. Separate them at 18% of max:
    # chain rows are ~8-14%, the bail is ~21%+, the wings higher. Scan top-down for
    # the first row that crosses into pendant density - that's the top of the bail.
    pendant = smooth >= 0.18 * max_c
    ys = np.where(pendant)[0]
    if len(ys) == 0:
        return rgba
    y0 = int(ys.min())                       # top of bail (chain excluded)
    # Lower wing tips taper thin; keep them with a more lenient bottom threshold.
    lenient = smooth >= 0.12 * max_c
    y1 = int(np.where(lenient)[0].max())

    band = rgba.crop((0, y0, w, y1 + 1))
    return trim_to_alpha(band)               # tighten to the pendant's x-extent


def trim_to_alpha(img: Image.Image) -> Image.Image:
    """Crop the image to the bounding box of non-transparent pixels.

    Args:
        img: RGBA image (or any mode; converted internally).

    Returns:
        Cropped image.  If the image is entirely transparent, the original is
        returned unchanged.
    """
    rgba = img.convert("RGBA")
    bbox = rgba.getbbox()
    if bbox is None:
        return img
    return rgba.crop(bbox)
