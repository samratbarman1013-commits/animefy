"""Comic-book engine: heavy ink, few colours, high contrast.

A thin variation on the cartoon pipeline with bolder outlines and a
smaller palette.
"""

from __future__ import annotations

from dataclasses import dataclass

from .cartoon import CartoonStyle


@dataclass
class ComicStyle(CartoonStyle):
    name = "comic"
    description = "Heavy ink and few colours - comic book look (fast, CPU)"

    strength: float = 0.95
    num_colors: int = 6
    stylize_scale: float = 70.0
    edge_low: int = 40
    edge_high: int = 120
    edge_min_area: int = 40
    edge_thickness: int = 2
    edge_amount: float = 1.0
