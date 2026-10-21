"""Optional neural 'neural' engine backed by AnimeGANv2 (PyTorch).

Only usable when `torch` is installed. It is imported lazily so the rest
of animefy works without PyTorch.

Weights are downloaded on first use to a local cache (~/.cache/animefy)
from the bryandlee/animegan2-pytorch release, or you can point at your
own .pt file with `--model` / the ANIMEFY_MODEL env var.

Presets: paprika (general photos), celeba_distill, face_paint_512_v1,
face_paint_512_v2 (portraits).
"""

from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass

import cv2
import numpy as np

from .base import Style

_WEIGHTS_BASE = "https://github.com/bryandlee/animegan2-pytorch/raw/main/weights"
_PRESETS = {"paprika", "celeba_distill", "face_paint_512_v1", "face_paint_512_v2"}
_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "animefy")


def _download_weights(name: str) -> str:
    os.makedirs(_CACHE_DIR, exist_ok=True)
    dest = os.path.join(_CACHE_DIR, f"{name}.pt")
    if os.path.exists(dest) and os.path.getsize(dest) > 1_000_000:
        return dest
    url = f"{_WEIGHTS_BASE}/{name}.pt"
    tmp = dest + ".part"
    with urllib.request.urlopen(url, timeout=120) as resp, open(tmp, "wb") as fh:
        fh.write(resp.read())
    os.replace(tmp, dest)
    return dest


@dataclass
class NeuralAnimeStyle(Style):
    name = "neural"
    description = "AnimeGANv2 (PyTorch) - truest anime style, needs torch + weights"

    #: preset name or path to a .pt state dict
    model_path: str | None = None
    #: preset to use when no explicit path is given
    preset: str = "paprika"
    #: inference size (AnimeGANv2 is trained at 256; higher = sharper, slower)
    size: int = 256
    strength: float = 1.0
    #: temporal EMA smoothing across frames (0 = off, 0.3 = mild)
    temporal: float = 0.0

    _net = None
    _device = "cpu"
    _prev: np.ndarray | None = None

    def _resolve_weights(self) -> str:
        path = self.model_path or os.environ.get("ANIMEFY_MODEL")
        if path and os.path.exists(path):
            return path
        if path and not os.path.exists(path):
            raise RuntimeError(f"weights not found: {path}")
        if self.preset not in _PRESETS:
            raise RuntimeError(
                f"unknown preset {self.preset!r}; choose from {sorted(_PRESETS)} "
                "or pass --model /path/to/weights.pt"
            )
        return _download_weights(self.preset)

    def _load(self):
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - depends on user env
            raise RuntimeError(
                "The 'neural' engine needs PyTorch. Install it with:\n"
                "  pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
                "or run with --style cartoon / --style anime instead."
            ) from exc

        from ._animegan2 import Generator

        path = self._resolve_weights()
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        net = Generator().to(self._device)
        state = torch.load(path, map_location=self._device)
        state = {k.replace("module.", ""): v for k, v in state.items()}
        net.load_state_dict(state)
        net.eval()
        self._net = net

    def prepare(self, sample_frames):
        if self._net is None:
            self._load()

    def apply(self, frame_bgr: np.ndarray) -> np.ndarray:
        import torch

        if self._net is None:
            self._load()

        h, w = frame_bgr.shape[:2]
        resized = cv2.resize(frame_bgr, (self.size, self.size), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(self._device)

        with torch.no_grad():
            out = self._net(tensor)[0]

        out = out.clamp(-1, 1).permute(1, 2, 0).cpu().numpy()
        out = ((out + 1.0) * 127.5).astype(np.uint8)
        out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        out = cv2.resize(out, (w, h), interpolation=cv2.INTER_CUBIC)

        if self.temporal > 0 and self._prev is not None:
            out = cv2.addWeighted(out, 1 - self.temporal, self._prev, self.temporal, 0)
        self._prev = out

        return self._blend(frame_bgr, out, self.strength)
