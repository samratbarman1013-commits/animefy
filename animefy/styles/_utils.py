"""Shared image-processing helpers for the classical style engines."""

from __future__ import annotations

import cv2
import numpy as np


def learn_palette(sample_frames: list[np.ndarray], k: int, max_pixels: int = 40000) -> np.ndarray:
    """Learn a shared colour palette (k x 3, BGR uint8) from sample frames.

    A palette learned once for the whole clip keeps colours stable between
    frames and stops the output from shimmering, which is the main artefact
    of naive per-frame quantisation.
    """
    if not sample_frames:
        return np.array(
            [[40, 40, 40], [80, 80, 90], [120, 120, 130], [170, 170, 180],
             [210, 210, 215], [245, 245, 245], [60, 70, 140], [90, 120, 190],
             [120, 170, 210], [150, 190, 120], [200, 180, 90], [190, 120, 110],
             [230, 160, 170], [150, 90, 160], [110, 150, 160], [70, 110, 90]],
            dtype=np.uint8,
        )

    pixels = []
    step = max(1, len(sample_frames) // 8)
    for frame in sample_frames[::step]:
        small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
        pixels.append(small.reshape(-1, 3))
    data = np.concatenate(pixels, axis=0).astype(np.float32)
    if len(data) > max_pixels:
        idx = np.random.default_rng(0).choice(len(data), max_pixels, replace=False)
        data = data[idx]

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    k = int(max(2, min(k, len(data))))
    _, _, centers = cv2.kmeans(data, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    return centers.astype(np.uint8)


_LUT_BITS = 6  # 6 bits/channel -> a 64x64x64 lookup table
_LUT_CACHE: dict[bytes, np.ndarray] = {}


def _build_palette_lut(palette: np.ndarray) -> np.ndarray:
    """Precompute a Lab->palette-colour lookup table for fast quantisation."""
    key = palette.astype(np.uint8).tobytes()
    cached = _LUT_CACHE.get(key)
    if cached is not None:
        return cached

    pal_lab = cv2.cvtColor(palette.reshape(1, -1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    levels = 1 << _LUT_BITS
    # Representative Lab value at the centre of each bin.
    step = 256 // levels
    reps = (np.arange(levels) * step + step / 2).astype(np.float32)
    grid = np.stack(np.meshgrid(reps, reps, reps, indexing="ij"), axis=-1).reshape(-1, 3)
    d = np.linalg.norm(grid[:, None, :] - pal_lab[None, :, :], axis=2)
    nearest = np.argmin(d, axis=1)
    lut = palette.reshape(-1, 3)[nearest].reshape(levels, levels, levels, 3)
    _LUT_CACHE[key] = lut
    return lut


def apply_palette(img_bgr: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Snap every pixel to its nearest palette colour (perceptually, in Lab).

    Uses a precomputed 3D lookup table, so it is effectively O(pixels) and
    stays fast even at HD resolutions.
    """
    lut = _build_palette_lut(palette)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    idx = (lab >> (8 - _LUT_BITS)).astype(np.int32)
    return lut[idx[..., 0], idx[..., 1], idx[..., 2]]


def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    """Drop connected blobs smaller than `min_area` from a binary mask."""
    if min_area <= 1:
        return mask
    num, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    keep = np.zeros_like(mask)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep[labels == i] = 255
    return keep


def clean_lines(
    gray: np.ndarray,
    low: int = 50,
    high: int = 150,
    min_area: int = 60,
    thickness: int = 1,
) -> np.ndarray:
    """Extract thin ink lines (black-on-white) with Canny, speckle removed."""
    g = cv2.medianBlur(gray, 5)
    edges = cv2.Canny(g, low, high)
    edges = remove_small_components(edges, min_area)
    if thickness > 0:
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=thickness)
    return 255 - edges


def soft_outline(gray: np.ndarray, sigma_small: float = 1.0, sigma_large: float = 2.5) -> np.ndarray:
    """Difference-of-Gaussians line art - softer, more 'anime' than threshold."""
    g1 = cv2.GaussianBlur(gray, (0, 0), sigma_small)
    g2 = cv2.GaussianBlur(gray, (0, 0), sigma_large)
    dog = cv2.subtract(g1, g2)
    dog = cv2.normalize(dog, None, 0, 255, cv2.NORM_MINMAX)
    _, mask = cv2.threshold(dog, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return mask
