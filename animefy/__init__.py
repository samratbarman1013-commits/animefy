"""animefy - turn real video into anime / cartoon video.

A small, dependency-light toolkit with pluggable stylisation engines:

    cartoon  - classical OpenCV pipeline (edge-preserving smoothing,
               colour flattening, ink outlines). Fast, CPU-only, no
               model downloads.
    anime    - classical "cel-shaded" pipeline tuned for a flatter,
               softer, anime-like look with pastel colour and line art.
    neural   - optional AnimeGANv2 (PyTorch). Truest anime result but
               needs `torch` + model weights and is much slower on CPU.

Typical use::

    from animefy import stylize_video
    stylize_video("clip.mp4", "clip_cartoon.mp4", style="cartoon")
"""

from .pipeline import stylize_video, stylize_frames
from .styles import available_styles, get_style, STYLES

__version__ = "1.1.0"
__all__ = [
    "stylize_video",
    "stylize_frames",
    "available_styles",
    "get_style",
    "STYLES",
]
