# animefy

Turn real, live video into **anime / cartoon** video.

**Live demo (no install):** https://samratbarman1013-commits.github.io/animefy/
— a browser-only version (cartoon + anime) that runs entirely on your device.

`animefy` is a small, dependency-light toolkit with three pluggable style
engines. Two of them run on plain CPU with no model downloads; the third is
an optional PyTorch AnimeGANv2 model for the truest anime look.

| engine    | what it does                                            | speed (CPU)        | needs                     |
|-----------|---------------------------------------------------------|--------------------|---------------------------|
| `cartoon` | Bold outlines + flat colours (classic cartoon)          | fast (~0.8s/frame @960×540) | nothing extra      |
| `anime`   | Cel-shaded pastel look with soft line art               | fast (~0.4s/frame) | nothing extra             |
| `comic`   | Heavy ink, few colours, high contrast (comic book)      | fast (~0.8s/frame) | nothing extra             |
| `sketch`  | Pencil-sketch line art on paper                         | very fast (~0.2s/frame) | nothing extra        |
| `neural`  | AnimeGANv2 — truest anime style                         | slow (~3s/frame)   | `torch` (+ auto weights)  |

## Install

```bash
git clone <this repo> && cd animefy
pip install -e .                 # core: cartoon + anime engines
pip install -e ".[web]"          # + Gradio web UI
pip install torch --index-url https://download.pytorch.org/whl/cpu   # + neural engine
```

Requires Python 3.9+. `opencv-python-headless` and `imageio-ffmpeg` are pulled
in automatically, and `imageio-ffmpeg` supplies a bundled `ffmpeg` binary, so
you do **not** need to install ffmpeg yourself (audio is copied from the source
clip by that binary).

## Quick start (command line)

```bash
# classic cartoon
animefy input.mp4 output_cartoon.mp4 --style cartoon

# cel-shaded anime look
animefy input.mp4 output_anime.mp4 --style anime

# true anime via AnimeGANv2 (downloads weights on first run)
animefy input.mp4 output_anime.mp4 --style neural

# quick 2-second preview while you tune
animefy input.mp4 preview.mp4 --style cartoon --max-frames 48

# see every option
animefy --list-styles
animefy --show-params anime
```

Five styles ship in the box: `cartoon`, `anime`, `comic`, `sketch` and
`neural`.

`python -m animefy ...` works identically if you prefer not to install the
console script.

## Quick start (web UI)

```bash
pip install -e ".[web]"
python -m animefy.app          # then open http://localhost:7860
```

Drag in a video, pick a style, drag the sliders, hit **Stylise**.

## Quick start (Python)

```python
from animefy import stylize_video

stylize_video("clip.mp4", "clip_anime.mp4", style="anime",
              strength=0.92, num_colors=20, cel_bands=4)

# or work frame by frame
from animefy import stylize_frames, get_style
import cv2
engine = get_style("cartoon", edge_thickness=2)
frames = [engine.apply(f) for f in my_frames]
```

## How it works

Every engine is a `Style` with two methods: `prepare()` (called once, with a
few sample frames) and `apply()` (called per frame). `prepare()` is where a
shared colour palette is learned for the whole clip — learning it once instead
of per frame is what stops the output from shimmering frame to frame.

**`cartoon`** — OpenCV's edge-preserving stylisation gives a painterly flat
base; colours are snapped onto the clip's palette through a precomputed 3-D
Lab lookup table (fast at any resolution); Canny lines are cleaned of speckle
and multiplied over the flat colours.

**`anime`** — edge-preserving smoothing flattens texture, colours go onto a
pastel palette, cel shading bands the luminance, and soft line art is laid on
top. The cel shading is *gated by local contrast*, so it only flattens regions
that actually have shading to flatten — smooth gradients like skies are left
alone, which avoids the false "contour band" a naive posteriser produces on
them.

**`neural`** — the AnimeGANv2 generator (PyTorch). Weights download to
`~/.cache/animefy` on first use. Presets: `paprika` (general photos),
`celeba_distill`, `face_paint_512_v1`, `face_paint_512_v2` (portraits).
Point at your own file with `--model /path/to/weights.pt` or `ANIMEFY_MODEL`.

```bash
animefy portrait.mp4 out.mp4 --style neural --preset face_paint_512_v2 --size 512
```

## Tuning

```
--strength       how strongly the effect replaces the original (0..1)
--colors         palette size; fewer = flatter
--smoothing      edge-preserving smoothing strength
--cel-bands      luminance bands (anime)
--saturation     colour saturation boost (anime)
--pastel         pastel wash, 0..1 (anime)
--edge-thickness outline thickness (cartoon)
--edge-amount    outline presence (cartoon)
```

Run `animefy --show-params <style>` for the full list and defaults.

## Adding your own engine

Subclass `animefy.styles.base.Style`, implement `apply()` (and optionally
`prepare()`), and register it in `animefy/styles/__init__.py`:

```python
from animefy.styles.base import Style

class MyStyle(Style):
    name = "mine"
    description = "my look"

    def apply(self, frame_bgr):
        return self._blend(frame_bgr, do_something(frame_bgr), 0.8)
```

## Performance notes

These figures are from a CPU-only run (no GPU) on 960×540 frames:

- `cartoon` ≈ 0.8 s/frame, `anime` ≈ 0.4 s/frame, `neural` ≈ 3 s/frame.
- A 5-second clip at 24 fps is 120 frames → ~1.5 min for `cartoon`,
  ~1 min for `anime`, ~6 min for `neural` on CPU. A GPU makes `neural`
  roughly an order of magnitude faster.
- For long clips, consider `--max-frames` to preview, then run the full clip.

## Limitations

- The classical engines are signal processing, not learned stylisation: on
  very busy scenes (dense foliage, crowds) they simplify less than a human
  artist would. `neural` handles those better.
- `neural` runs at a fixed internal resolution (`--size`, default 256) and is
  upscaled to the frame size, so fine detail is soft. This matches how
  AnimeGANv2 is normally used.
- Output video is re-encoded with OpenCV (`mp4v`) then remuxed; for archival
  quality re-encode with ffmpeg/CRF afterwards.

## License

MIT. The vendored AnimeGANv2 architecture (`animefy/styles/_animegan2.py`)
and its downloadable weights are from
[bryandlee/animegan2-pytorch](https://github.com/bryandlee/animegan2-pytorch)
(MIT). See `LICENSE`.
