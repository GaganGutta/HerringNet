"""Gradio web demo for HerringNet.

Provides an interactive interface for fish detection and species
classification on camera trap images and videos. Designed for
quick demos and visual inspection of model results.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

from herringnet.config import HerringNetConfig, load_config
from herringnet.inference.result_types import FrameResult, VideoResult
from herringnet.models.pipeline import HerringNetPipeline
from herringnet.visualization.draw_detections import draw_detections

logger = logging.getLogger(__name__)


def create_demo(config: HerringNetConfig) -> gr.Blocks:
    """Create the Gradio demo interface.

    Args:
        config: HerringNet configuration.

    Returns:
        Gradio Blocks application ready to launch.
    """
    pipeline = HerringNetPipeline(config)

    def process_image(image_path: str | None) -> tuple:
        """Process a single image and return annotated result."""
        if image_path is None:
            return None, "No image uploaded."

        result = pipeline.process_image(image_path)
        image = cv2.imread(image_path)
        if image is None:
            return None, "Failed to read image."

        annotated = draw_detections(image, result.detections)
        # Convert BGR to RGB for Gradio display
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        summary = _format_image_summary(result)
        return annotated_rgb, summary

    def process_video(video_path: str | None, max_frames: int) -> tuple:
        """Process a video and return results."""
        if video_path is None:
            return [], "No video uploaded."

        result = pipeline.process_video(
            video_path,
            max_frames=max_frames if max_frames > 0 else None,
        )

        # Generate annotated frame gallery
        gallery_images = _generate_gallery(video_path, result)
        summary = _format_video_summary(result)

        return gallery_images, summary

    # Build the Gradio interface
    with gr.Blocks(
        title="HerringNet - Fish Detection",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            "# HerringNet - Fish Detection for Camera Traps\n"
            "Upload camera trap images or video to detect and classify fish.\n"
            "Built for Jordaan Labs, UMass Amherst."
        )

        with gr.Tab("Image Detection"):
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(
                        type="filepath",
                        label="Upload Image",
                    )
                    detect_btn = gr.Button(
                        "Detect Fish", variant="primary"
                    )
                with gr.Column():
                    output_image = gr.Image(label="Detection Results")

            results_text = gr.Textbox(
                label="Detection Summary",
                lines=8,
                interactive=False,
            )

            detect_btn.click(
                fn=process_image,
                inputs=input_image,
                outputs=[output_image, results_text],
            )

        with gr.Tab("Video Processing"):
            with gr.Row():
                with gr.Column():
                    input_video = gr.Video(label="Upload Video")
                    max_frames_input = gr.Slider(
                        minimum=0,
                        maximum=500,
                        value=50,
                        step=10,
                        label="Max Frames (0 = all)",
                    )
                    process_btn = gr.Button(
                        "Process Video", variant="primary"
                    )
                with gr.Column():
                    output_gallery = gr.Gallery(
                        label="Frames with Detections",
                        columns=3,
                        height=400,
                    )

            video_summary = gr.Textbox(
                label="Video Summary",
                lines=10,
                interactive=False,
            )

            process_btn.click(
                fn=process_video,
                inputs=[input_video, max_frames_input],
                outputs=[output_gallery, video_summary],
            )

        with gr.Tab("About"):
            _mode = "Detection + Classification" if pipeline.classifier else "Detection Only"
            gr.Markdown(
                f"## Model Information\n\n"
                f"- **Mode:** {_mode}\n"
                f"- **Detector:** {config.detector.model_name}\n"
                f"- **Detection Threshold:** {config.detector.confidence_threshold}\n"
                f"- **Classifier:** {config.classifier.model_path or 'Not loaded'}\n"
                f"- **FRR (video):** {config.frame_extraction.mean_frr}\n"
                f"- **Frame Interval:** {config.frame_extraction.frame_interval}\n\n"
                f"## About HerringNet\n\n"
                f"HerringNet is a fish detection and species classification tool "
                f"built for monitoring juvenile river herring emigration from "
                f"underwater camera traps. Developed at Jordaan Labs, UMass Amherst.\n\n"
                f"The two-stage pipeline uses pretrained fish detection (CFD) "
                f"followed by species-level classification."
            )

    return demo


def _format_image_summary(result: FrameResult) -> str:
    """Format detection results as a readable text summary."""
    lines = [f"Total fish detected: {result.fish_count}"]

    if not result.detections:
        lines.append("No fish found in this image.")
        return "\n".join(lines)

    lines.append("")
    for i, det in enumerate(result.detections, 1):
        bbox = det.detection.bbox
        if det.classification is not None:
            species = det.classification.species.replace("_", " ").title()
            conf = det.classification.confidence
            lines.append(
                f"  {i}. {species} (confidence: {conf:.1%})"
            )
        else:
            conf = det.detection.confidence
            lines.append(
                f"  {i}. Fish (detection confidence: {conf:.1%})"
            )

        lines.append(
            f"     Box: ({bbox[0]:.0f}, {bbox[1]:.0f}) to "
            f"({bbox[2]:.0f}, {bbox[3]:.0f})"
        )
        if det.is_uncertain:
            lines.append(f"     [UNCERTAIN] {det.uncertainty_reason}")

    if result.flagged_for_review:
        lines.append("")
        lines.append("* Some detections flagged for human review.")

    return "\n".join(lines)


def _format_video_summary(result: VideoResult) -> str:
    """Format video processing results as a readable text summary."""
    counts = result.counts
    lines = [
        "Video Processing Results",
        "=" * 30,
        f"Frames extracted:    {result.total_frames_extracted}",
        f"Frames with fish:    {counts.frames_with_fish}",
        f"Raw fish count:      {counts.raw_total}",
        f"Corrected count:     {counts.corrected_total:.1f}",
        f"Processing time:     {result.processing_time_seconds:.1f}s",
        f"Video FPS:           {result.video_fps:.1f}",
        f"Video duration:      {result.video_duration:.1f}s",
    ]

    if counts.species_counts:
        lines.append("")
        lines.append("Species Breakdown:")
        for species, pair in sorted(counts.species_counts.items()):
            display = species.replace("_", " ").title()
            lines.append(
                f"  {display}: {pair.raw} raw, {pair.corrected:.1f} corrected"
            )

    return "\n".join(lines)


def _generate_gallery(
    video_path: str,
    result: VideoResult,
    max_gallery: int = 20,
) -> list:
    """Generate annotated frame images for the gallery display."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    gallery = []
    shown = 0

    try:
        for frame_result in result.per_frame_results:
            if shown >= max_gallery:
                break
            if frame_result.fish_count == 0:
                continue
            if frame_result.frame_number is None:
                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_result.frame_number)
            ret, frame = cap.read()
            if not ret:
                continue

            annotated = draw_detections(frame, frame_result.detections)
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

            caption = (
                f"Frame {frame_result.frame_number}: "
                f"{frame_result.fish_count} fish"
            )
            gallery.append((annotated_rgb, caption))
            shown += 1
    finally:
        cap.release()

    return gallery
