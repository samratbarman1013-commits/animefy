"""Video I/O and the frame-processing loop.

Reading and writing go through OpenCV's bundled FFmpeg, and audio is
carried over from the source with the ffmpeg binary that imageio-ffmpeg
ships, so no system ffmpeg install is required.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from typing import Callable, Iterable

import cv2
import numpy as np

from .styles import Style, get_style


def _ffmpeg_exe() -> str | None:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def has_audio(path: str) -> bool:
    exe = _ffmpeg_exe()
    if not exe:
        return False
    try:
        out = subprocess.run(
            [exe, "-i", path, "-hide_banner"],
            capture_output=True, text=True,
        ).stderr
        return "Audio:" in out
    except Exception:
        return False


def _mux_audio(video_no_audio: str, source: str, dest: str) -> bool:
    """Copy the audio track from `source` onto `video_no_audio` -> dest."""
    exe = _ffmpeg_exe()
    if not exe:
        return False
    cmd = [
        exe, "-y", "-hide_banner", "-loglevel", "error",
        "-i", video_no_audio, "-i", source,
        "-map", "0:v:0", "-map", "1:a:0?",
        "-c:v", "copy", "-c:a", "aac", "-shortest",
        dest,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except Exception:
        return False


def _sample_frames(cap: cv2.VideoCapture, count: int = 12) -> list[np.ndarray]:
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    picks = np.linspace(0, max(0, total - 1), num=min(count, total), dtype=int)
    frames = []
    for idx in picks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return frames


def stylize_frames(
    frames: Iterable[np.ndarray],
    style: str | Style = "cartoon",
    progress: Callable[[int, int], None] | None = None,
    **params,
) -> list[np.ndarray]:
    """Stylise an in-memory sequence of BGR frames and return the results."""
    engine = style if isinstance(style, Style) else get_style(style, **params)
    frames = list(frames)
    engine.prepare(frames[: min(len(frames), 12)])
    out = []
    total = len(frames)
    for i, frame in enumerate(frames):
        out.append(engine.apply(frame))
        if progress:
            progress(i + 1, total)
    return out


def stylize_video(
    src: str,
    dst: str,
    style: str | Style = "cartoon",
    progress: Callable[[int, int], None] | None = None,
    keep_audio: bool = True,
    max_frames: int | None = None,
    **params,
) -> str:
    """Stylise a video file end to end and write `dst`.

    Returns the output path. Audio (if present) is muxed back in.
    """
    if not os.path.exists(src):
        raise FileNotFoundError(src)

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {src}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    engine = style if isinstance(style, Style) else get_style(style, **params)
    engine.prepare(_sample_frames(cap, 12))

    os.makedirs(os.path.dirname(os.path.abspath(dst)) or ".", exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
    os.close(tmp_fd)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_path, fourcc, fps, (w, h))

    count = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(engine.apply(frame))
            count += 1
            if progress:
                progress(count, total if total > 0 else count)
            if max_frames and count >= max_frames:
                break
    finally:
        cap.release()
        writer.release()

    if keep_audio and has_audio(src):
        if not _mux_audio(tmp_path, src, dst):
            shutil.move(tmp_path, dst)
    else:
        shutil.move(tmp_path, dst)
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    return dst
