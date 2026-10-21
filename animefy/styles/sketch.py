"""Pencil-sketch engine - graphite line art on paper.

Uses OpenCV's pencil-sketch (it returns a greyscale drawing and a colour
version of the same); `colorize` mixes a little of the colour version back
in for a tinted-pencil look.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2

from .base import Style


@dataclass
class SketchStyle(Style):
    name = "sketch"
    description = "Pencil-sketch line art on paper (fast, CPU)"

    strength: float = 0.9
    #: detail scale - larger keeps broader strokes
    sigma_s: float = 60.0
    #: edge sensitivity - smaller picks up more lines
    sigma_r: float = 0.07
    #: how dark the shading is (0..1)
    shade_factor: float = 0.05
    #: 0 = pure graphite, 1 = the coloured pencil sketch
    colorize: float = 0.0

    def apply(self, frame_bgr):
        gray, color = cv2.pencilSketch(
            frame_bgr, sigma_s=self.sigma_s, sigma_r=self.sigma_r,
            shade_factor=self.shade_factor,
        )
        gray3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        if self.colorize > 0.001:
            out = cv2.addWeighted(gray3, 1 - self.colorize, color, self.colorize, 0)
        else:
            out = gray3
        return self._blend(frame_bgr, out, self.strength)
