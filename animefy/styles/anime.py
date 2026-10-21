"""Classical 'anime' engine - a cel-shaded, pastel look, still CPU-only.

This is not a neural network; it approximates the flat-shaded anime
aesthetic with signal processing: edge-preserving smoothing, a pastel
palette, contrast-gated cel shading and soft line art. For a true learned
anime style use the optional `neural` engine.

The cel shading is gated by local contrast so it only flattens regions
that actually have shading to flatten - smooth gradients like skies are
left alone, which avoids the false contour "bands" a naive posteriser
produces on them.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .base import Style
from ._utils import apply_palette, clean_lines, learn_palette


@dataclass
class AnimeStyle(Style):
    name = "anime"
    description = "Cel-shaded pastel look with soft line art (fast, CPU)"

    strength: float = 0.92
    num_colors: int = 20
    #: edge-preserving smoothing strength (larger = flatter)
    smooth_scale: float = 95.0
    #: how much of the palette-quantised image to use vs. the smoothed one
    #: (1 = fully flat, lower keeps gradients smooth)
    flatten: float = 0.82
    #: luminance bands for cel shading (3-5 reads as anime)
    cel_bands: int = 4
    #: contrast threshold below which cel shading is suppressed (higher =
    #: only strong shading is banded, smoother output)
    cel_gate: float = 18.0
    #: extra damping of the cel effect (0 = full, 1 = off)
    cel_softness: float = 0.0
    #: how much to push saturation of the flat colours
    saturation: float = 1.4
    #: pastel mix (0 = keep colours, 1 = fully washed toward white)
    pastel: float = 0.05
    #: line-art presence (0..1)
    line_amount: float = 0.65
    edge_low: int = 40
    edge_high: int = 120
    edge_min_area: int = 80

    _palette: np.ndarray | None = None

    def prepare(self, sample_frames):
        self._palette = learn_palette(sample_frames, self.num_colors)
        if self._palette is not None:
            self._palette = self._pastelize(self._palette)

    def _pastelize(self, colors: np.ndarray) -> np.ndarray:
        c = colors.astype(np.float32)
        c = c * (1 - self.pastel) + 255.0 * self.pastel
        hsv = cv2.cvtColor(c.reshape(1, -1, 3).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 1] = np.clip(hsv[..., 1] * self.saturation, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).reshape(-1, 3)

    def apply(self, frame_bgr: np.ndarray) -> np.ndarray:
        palette = self._palette
        if palette is None:
            palette = self._pastelize(learn_palette([frame_bgr], self.num_colors))

        # 1. Edge-preserving smoothing to flatten texture but keep boundaries.
        smoothed = cv2.edgePreservingFilter(
            frame_bgr, flags=cv2.RECURS_FILTER, sigma_s=self.smooth_scale, sigma_r=0.4
        )

        # 2. Snap to the pastel palette, then blend back toward the smoothed
        #    image so smooth gradients do not step.
        quant = apply_palette(smoothed, palette)
        flatten = float(np.clip(self.flatten, 0.0, 1.0))
        flat = cv2.addWeighted(smoothed, 1 - flatten, quant, flatten, 0)

        # 3. Contrast-gated cel shading.
        lab = cv2.cvtColor(flat, cv2.COLOR_BGR2LAB)
        l = lab[..., 0].astype(np.float32)
        step = 255.0 / max(2, self.cel_bands)
        banded = (np.floor(l / step) + 0.5) * step
        gx = cv2.Sobel(l, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(l, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.GaussianBlur(cv2.magnitude(gx, gy), (0, 0), 4)
        gate = np.clip(mag / max(1e-3, self.cel_gate), 0, 1) * (1 - float(np.clip(self.cel_softness, 0, 1)))
        lab[..., 0] = np.clip(l + (banded - l) * gate, 0, 255).astype(np.uint8)
        cel = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # 4. Soft line art over the flat colours.
        gray = cv2.cvtColor(smoothed, cv2.COLOR_BGR2GRAY)
        lines = clean_lines(gray, low=self.edge_low, high=self.edge_high,
                            min_area=self.edge_min_area, thickness=1)
        line_rgb = cv2.cvtColor(lines, cv2.COLOR_GRAY2BGR)
        anime = cv2.bitwise_and(cel, line_rgb)

        if self.line_amount < 0.999:
            anime = cv2.addWeighted(anime, self.line_amount, cel, 1 - self.line_amount, 0)

        return self._blend(frame_bgr, anime, self.strength)
