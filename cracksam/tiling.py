"""
tiling.py
---------
Shared helpers for splitting images into tile_size x tile_size patches
at NATIVE resolution (no resizing) -- used by dataset.py (training and
validation) and by infer.py (inference on large/4K images).

Keeping this logic in exactly one place matters: it guarantees training
and inference tile images in the same way, so the model always sees the
same crack-pixel-width statistics regardless of the source image's
overall resolution. This does NOT solve a difference in ground sample
distance (cm/pixel) between two genuinely different cameras/altitudes --
it only removes the extra, avoidable resolution loss that a single
whole-image resize would otherwise introduce.
"""

from typing import List

import numpy as np
from PIL import Image


def compute_tile_origins(size: int, tile: int, overlap: int = 0) -> List[int]:
    """
    Top-left coordinates (along one axis) covering [0, size) with the
    given tile size and overlap. Guarantees full coverage: the last
    tile is pulled back so it never runs past the image border.
    """
    assert overlap < tile, "overlap must be smaller than tile size"
    stride = tile - overlap
    origins = list(range(0, max(size - tile, 0) + 1, stride))
    if not origins:
        origins = [0]
    if origins[-1] + tile < size:
        origins.append(size - tile)
    return sorted(set(origins))


def pad_to_tile_rgb(image: Image.Image, tile: int) -> Image.Image:
    """Reflect-pads an RGB image so both dimensions are >= tile."""
    W, H = image.size
    pad_w = max(0, tile - W)
    pad_h = max(0, tile - H)
    if pad_w or pad_h:
        arr = np.pad(np.array(image), ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
        image = Image.fromarray(arr)
    return image


def pad_to_tile_gray(mask: Image.Image, tile: int) -> Image.Image:
    """Reflect-pads a single-channel (mask) image so both dims are >= tile."""
    W, H = mask.size
    pad_w = max(0, tile - W)
    pad_h = max(0, tile - H)
    if pad_w or pad_h:
        arr = np.pad(np.array(mask), ((0, pad_h), (0, pad_w)), mode="reflect")
        mask = Image.fromarray(arr)
    return mask


def make_blend_weight(tile: int, overlap: int) -> np.ndarray:
    """
    2D weight map for one tile: high in the center, tapering (but never
    to zero) over the overlap region at the edges. Overlapping tiles are
    combined with this weighting so stitched predictions don't show hard
    seams at tile borders. Only used at inference time (grid/validation
    tiling with overlap=0 doesn't need it, since tiles don't overlap).
    """
    if overlap <= 0:
        return np.ones((tile, tile), dtype=np.float32)
    ramp_len = max(1, overlap // 2)
    w1d = np.ones(tile, dtype=np.float32)
    ramp = np.linspace(0.1, 1.0, ramp_len, dtype=np.float32)
    w1d[:ramp_len] = ramp
    w1d[-ramp_len:] = ramp[::-1]
    return np.outer(w1d, w1d)
