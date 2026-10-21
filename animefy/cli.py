"""Command-line interface for animefy."""

from __future__ import annotations

import argparse
import sys
import time

from . import __version__
from .pipeline import stylize_video
from .styles import available_styles, style_params, STYLES


def _add_style_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("style tuning")
    g.add_argument("--strength", type=float, default=None,
                   help="how strongly the effect replaces the original (0..1)")
    g.add_argument("--colors", type=int, default=None, dest="num_colors",
                   help="palette size; fewer = flatter")
    g.add_argument("--smoothing", type=int, default=None,
                   help="edge-preserving smoothing passes")
    g.add_argument("--edge-thickness", type=int, default=None,
                   help="outline thickness (cartoon)")
    g.add_argument("--edge-amount", type=float, default=None,
                   help="outline presence (cartoon)")
    g.add_argument("--cel-bands", type=int, default=None,
                   help="shading bands (anime)")
    g.add_argument("--saturation", type=float, default=None,
                   help="colour saturation boost (anime)")
    g.add_argument("--pastel", type=float, default=None,
                   help="pastel wash 0..1 (anime)")
    g.add_argument("--line-amount", type=float, default=None,
                   help="line-art presence (anime)")
    g.add_argument("--model", type=str, default=None, dest="model_path",
                   help="path to AnimeGANv2 .pth weights (neural)")
    g.add_argument("--size", type=int, default=None,
                   help="inference resolution (neural)")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="animefy",
        description="Turn real video into anime / cartoon video.",
    )
    p.add_argument("input", nargs="?", help="input video file")
    p.add_argument("output", nargs="?", help="output video file")
    p.add_argument("--style", default="cartoon",
                   choices=sorted(STYLES), help="stylisation engine")
    p.add_argument("--max-frames", type=int, default=None,
                   help="stop after N frames (handy for a quick preview)")
    p.add_argument("--no-audio", action="store_true", help="do not copy audio")
    p.add_argument("--list-styles", action="store_true",
                   help="show styles and exit")
    p.add_argument("--show-params", metavar="STYLE",
                   help="show tunable parameters for a style and exit")
    p.add_argument("--version", action="version", version=f"animefy {__version__}")
    _add_style_args(p)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_styles:
        print("Available styles:\n")
        for s in available_styles():
            print(f"  {s['name']:<9} {s['description']}")
        return 0

    if args.show_params:
        params = style_params(args.show_params)
        print(f"Parameters for '{args.show_params}':\n")
        for k, v in params.items():
            print(f"  --{k.replace('_', '-'):<16} default: {v}")
        return 0

    if not args.input or not args.output:
        build_parser().print_help()
        return 2

    params = {
        "strength": args.strength,
        "num_colors": args.num_colors,
        "smoothing": args.smoothing,
        "edge_thickness": args.edge_thickness,
        "edge_amount": args.edge_amount,
        "cel_bands": args.cel_bands,
        "saturation": args.saturation,
        "pastel": args.pastel,
        "line_amount": args.line_amount,
        "model_path": args.model_path,
        "size": args.size,
    }

    start = time.time()

    def progress(done: int, total: int) -> None:
        if total <= 0:
            return
        pct = done / total * 100
        bar = "#" * int(pct // 3)
        sys.stderr.write(f"\r[{bar:<33}] {pct:5.1f}%  frame {done}/{total}")
        sys.stderr.flush()

    print(f"animefy: {args.input} -> {args.output}  (style={args.style})")
    try:
        stylize_video(
            args.input, args.output, style=args.style,
            progress=progress, keep_audio=not args.no_audio,
            max_frames=args.max_frames, **params,
        )
    except (FileNotFoundError, RuntimeError, KeyError) as exc:
        sys.stderr.write(f"\nerror: {exc}\n")
        return 1

    sys.stderr.write("\n")
    print(f"done in {time.time() - start:.1f}s -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
