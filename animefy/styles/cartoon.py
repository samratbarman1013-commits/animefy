"""Classical 'cartoon' engine - the fast, always-available default.

Pipeline per frame:
  1. OpenCV's edge-preserving stylisation gives a painterly, flat base,
  2. colours are snapped onto a shared palette learned from the clip,
  3. bold ink lines are extracted with Canny (speckle removed),
  4. lines are multiplied over the flat colours and blended to taste.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .base import Style
from ._utils import apply_palette, clean_lines, learn_palette


@dataclass
class CartoonStyle(Style):
    name = "cartoon"
    description = "Bold outlines and flat colours - classic cartoon look (fast, CPU)"

    #: how strongly the stylised result replaces the original (0..1)
    strength: float = 0.9
    #: number of palette colours (fewer = flatter / more cartoonish)
    num_colors: int = 16
    #: spatial sigma of the painterly base (higher = larger flat regions)
    stylize_scale: float = 60.0
    #: edge detection thresholds - higher = fewer, bolder lines
    edge_low: int = 50
    edge_high: int = 150
    #: minimum line-blob size in pixels; kills foliage speckle
    edge_min_area: int = 60
    #: outline thickness in dilation steps
    edge_thickness: int = 1
    #: outline darkness / presence (0..1)
    edge_amount: float = 1.0

    _palette: np.ndarray | None = None

    def prepare(self, sample_frames):
        self._palette = learn_palette(sample_frames, self.num_colors)

    def apply(self, frame_bgr: np.ndarray) -> np.ndarray:
        palette = self._palette
        if palette is None:
            palette = learn_palette([frame_bgr], self.num_colors)

        base = cv2.stylization(frame_bgr, sigma_s=self.stylize_scale, sigma_r=0.45)
        flat = apply_palette(base, palette)

        gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
        lines = clean_lines(
            gray, low=self.edge_low, high=self.edge_high,
            min_area=self.edge_min_area, thickness=self.edge_thickness,
        )
        line_rgb = cv2.cvtColor(lines, cv2.COLOR_GRAY2BGR)
        cartoon = cv2.bitwise_and(flat, line_rgb)

        if self.edge_amount < 0.999:
            # Soften the ink toward a dark grey rather than pure black.
            softened = cv2.addWeighted(flat, 0.75, np.zeros_like(flat), 0.0, 0)
            cartoon = cv2.addWeighted(
                cartoon, self.edge_amount, softened, 1 - self.edge_amount, 0
            )

        return self._blend(frame_bgr, cartoon, self.strength)
