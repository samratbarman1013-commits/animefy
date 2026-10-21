"""Style registry and factory."""

from __future__ import annotations

from dataclasses import fields

from .base import Style
from .cartoon import CartoonStyle
from .anime import AnimeStyle
from .neural import NeuralAnimeStyle

#: name -> class. Add your own engine here to make it available everywhere.
STYLES: dict[str, type[Style]] = {
    CartoonStyle.name: CartoonStyle,
    AnimeStyle.name: AnimeStyle,
    NeuralAnimeStyle.name: NeuralAnimeStyle,
}


def available_styles() -> list[dict[str, str]]:
    """List the registered styles with their descriptions."""
    return [
        {"name": cls.name, "description": cls.description}
        for cls in STYLES.values()
    ]


def style_params(name: str) -> dict[str, object]:
    """Return the tunable parameters (and defaults) for a style."""
    cls = STYLES.get(name)
    if cls is None:
        raise KeyError(f"unknown style {name!r}; choose from {list(STYLES)}")
    out: dict[str, object] = {}
    for f in fields(cls):
        if f.name.startswith("_"):
            continue
        default = f.default if f.default is not f.default_factory else None  # type: ignore[comparison-overlap]
        out[f.name] = default
    return out


def get_style(name: str = "cartoon", **params) -> Style:
    """Instantiate a style by name, ignoring params it does not declare."""
    cls = STYLES.get(name)
    if cls is None:
        raise KeyError(
            f"unknown style {name!r}; choose from {sorted(STYLES)}"
        )
    valid = {f.name for f in fields(cls)}
    clean = {k: v for k, v in params.items() if k in valid and v is not None}
    return cls(**clean)


__all__ = ["Style", "STYLES", "available_styles", "get_style", "style_params"]
