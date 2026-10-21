"""Smoke tests. Run with:  pytest -q"""
import numpy as np
import pytest

from animefy import stylize_frames, available_styles, get_style


def _frame(seed=0):
    rng = np.random.default_rng(seed)
    img = np.zeros((90, 160, 3), np.uint8)
    img[..., 0] = np.linspace(0, 255, 160)          # gradient background
    img[20:70, 40:120] = (30, 180, 240)             # a coloured block
    img = (img + rng.integers(-8, 8, img.shape)).clip(0, 255).astype(np.uint8)
    return img


def test_styles_registered():
    names = {s["name"] for s in available_styles()}
    assert {"cartoon", "anime", "comic", "sketch", "neural"} <= names


@pytest.mark.parametrize("style", ["cartoon", "anime", "comic", "sketch"])
def test_engine_shapes_and_dtype(style):
    frames = [_frame(i) for i in range(5)]
    out = stylize_frames(frames, style=style)
    assert len(out) == len(frames)
    for a, b in zip(frames, out):
        assert b.shape == a.shape
        assert b.dtype == np.uint8


@pytest.mark.parametrize("style", ["cartoon", "anime", "comic", "sketch"])
def test_engine_changes_pixels(style):
    frames = [_frame(i) for i in range(5)]
    out = stylize_frames(frames, style=style, strength=1.0)
    diff = np.abs(frames[0].astype(int) - out[0].astype(int)).mean()
    assert diff > 1.0  # the styliser actually did something


def test_unknown_style_raises():
    with pytest.raises(KeyError):
        get_style("nope")
