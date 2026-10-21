"""Base class shared by every stylisation engine."""

from __future__ import annotations

import numpy as np


class Style:
    """A frame-to-frame styliser.

    Engines subclass this and implement :meth:`apply`. If an engine needs
    to see the whole clip before it can style a single frame (e.g. to learn
    a shared colour palette) it overrides :meth:`prepare`, which the video
    pipeline calls once with a list of sample frames.
    """

    #: short, machine-friendly identifier used on the CLI / UI
    name: str = "base"
    #: human description
    description: str = ""

    def prepare(self, sample_frames: list[np.ndarray]) -> None:
        """Optional one-time setup from a few representative BGR frames."""
        return None

    def apply(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Return a stylised copy of a single BGR uint8 frame."""
        raise NotImplementedError

    # -- small helpers shared by engines ---------------------------------
    @staticmethod
    def _blend(original: np.ndarray, stylised: np.ndarray, amount: float) -> np.ndarray:
        """Linear mix between two uint8 images (amount 0..1)."""
        amount = float(np.clip(amount, 0.0, 1.0))
        if amount >= 0.999:
            return stylised
        if amount <= 0.001:
            return original
        return np.clip(
            original.astype(np.float32) * (1.0 - amount)
            + stylised.astype(np.float32) * amount,
            0,
            255,
        ).astype(np.uint8)
