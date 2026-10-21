"""Build a side-by-side comparison video: original | cartoon | anime.

Usage:
    python examples/make_comparison.py input.mp4 out_dir
"""
import os
import sys

import cv2
import numpy as np

from animefy.styles import get_style


def main(src: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    if not frames:
        raise SystemExit(f"no frames read from {src}")

    h, w = frames[0].shape[:2]
    engines = {}
    for name in ("cartoon", "anime"):
        eng = get_style(name)
        eng.prepare(frames[:12])
        engines[name] = eng

    pw, ph = 640, 360
    out = cv2.VideoWriter(
        os.path.join(out_dir, "comparison.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"), fps, (pw * 3, ph),
    )
    for f in frames:
        panels = [cv2.resize(f, (pw, ph))]
        for eng in engines.values():
            panels.append(cv2.resize(eng.apply(f), (pw, ph)))
        out.write(np.hstack(panels))
    out.release()
    print(f"wrote {os.path.join(out_dir, 'comparison.mp4')}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2])
