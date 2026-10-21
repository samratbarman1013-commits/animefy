"""Gradio web app: drag in a video, pick a style, get anime back.

Run with:  python -m animefy.app
Requires:  pip install gradio
"""

from __future__ import annotations

import os
import tempfile

from .pipeline import stylize_video

#: styles offered by the web UI (neural is added only if torch is importable)
_BASE_STYLES = ["cartoon", "anime", "comic", "sketch"]


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def _params_for(style, strength, colors, detail, cel_bands, saturation,
                pastel, edge_thickness, edge_amount, line_amount):
    """Map the shared UI sliders onto each engine's own parameter names."""
    params = {"strength": strength, "num_colors": int(colors)}
    if style in ("cartoon", "comic"):
        params["stylize_scale"] = detail
        params["edge_thickness"] = int(edge_thickness)
        params["edge_amount"] = edge_amount
    elif style == "sketch":
        params["sigma_s"] = detail
        params["colorize"] = line_amount
    elif style == "anime":
        params["smooth_scale"] = detail
        params["cel_bands"] = int(cel_bands)
        params["saturation"] = saturation
        params["pastel"] = pastel
        params["line_amount"] = line_amount
    return params


def build_demo():
    import gradio as gr

    styles = list(_BASE_STYLES)
    if _torch_available():
        styles.append("neural")

    def run(video_path, style, strength, colors, detail, cel_bands, saturation,
            pastel, edge_thickness, edge_amount, line_amount, max_seconds,
            progress=gr.Progress()):
        if not video_path:
            raise gr.Error("Please upload a video first.")
        out_dir = tempfile.mkdtemp(prefix="animefy_")
        stem = os.path.splitext(os.path.basename(video_path))[0]
        out_path = os.path.join(out_dir, f"{stem}_{style}.mp4")

        import cv2
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        cap.release()
        max_frames = max(1, int(max_seconds * fps))

        def cb(done, total):
            if total > 0:
                progress(min(done / total, 1.0), desc=f"Stylising frame {done}/{total}")

        stylize_video(
            video_path, out_path, style=style, progress=cb, max_frames=max_frames,
            **_params_for(style, strength, colors, detail, cel_bands, saturation,
                          pastel, edge_thickness, edge_amount, line_amount),
        )
        return out_path

    with gr.Blocks(title="animefy") as demo:
        gr.Markdown(
            "# animefy\nTurn real video into **anime / cartoon** video.\n"
            "Upload a clip, choose a style, hit *Stylise*. Everything runs on "
            "CPU — a few seconds of footage takes under a minute."
        )
        with gr.Row():
            with gr.Column():
                video_in = gr.Video(label="Input video")
                style = gr.Dropdown(styles, value="cartoon", label="Style")
                strength = gr.Slider(0.0, 1.0, 0.9, label="Effect strength")
                colors = gr.Slider(4, 32, 16, step=1, label="Palette colours")
                detail = gr.Slider(30, 150, 70, step=5, label="Smoothing / flatness")
                with gr.Accordion("Anime options", open=False):
                    cel_bands = gr.Slider(2, 6, 4, step=1, label="Cel shading bands")
                    saturation = gr.Slider(0.5, 2.0, 1.4, label="Saturation")
                    pastel = gr.Slider(0.0, 0.6, 0.05, label="Pastel wash")
                    line_amount = gr.Slider(0.0, 1.0, 0.65, label="Line-art presence")
                with gr.Accordion("Cartoon options", open=False):
                    edge_thickness = gr.Slider(0, 3, 1, step=1, label="Outline thickness")
                    edge_amount = gr.Slider(0.0, 1.0, 1.0, label="Outline presence")
                max_seconds = gr.Slider(1, 10, 5, step=1, label="Max seconds to process")
                go = gr.Button("Stylise", variant="primary")
            with gr.Column():
                video_out = gr.Video(label="Result")

        go.click(
            run,
            inputs=[video_in, style, strength, colors, detail, cel_bands,
                    saturation, pastel, edge_thickness, edge_amount,
                    line_amount, max_seconds],
            outputs=video_out,
        )
    return demo


def main():
    build_demo().launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
