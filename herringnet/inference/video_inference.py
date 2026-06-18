"""Video inference module for processing video files through the pipeline.

Provides convenience functions for running the HerringNet pipeline on
video files, with options for saving annotated frames, generating
CSV summaries, and exporting annotated videos.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import cv2

from herringnet.config import HerringNetConfig
from herringnet.inference.result_types import VideoResult
from herringnet.models.pipeline import HerringNetPipeline
from herringnet.visualization.draw_detections import (
    draw_count_overlay,
    draw_detections,
)

logger = logging.getLogger(__name__)


def process_video_file(
    video_path: str | Path,
    config: HerringNetConfig,
    output_dir: str | Path | None = None,
    save_annotated_frames: bool = False,
    save_csv: bool = False,
    max_frames: int | None = None,
) -> VideoResult:
    """Process a video file and optionally save annotated outputs.

    Args:
        video_path: Path to the video file.
        config: HerringNet configuration.
        output_dir: Directory for saving outputs.
        save_annotated_frames: Whether to save frames with drawn detections.
        save_csv: Whether to save a CSV summary of counts.
        max_frames: Maximum number of frames to process.

    Returns:
        VideoResult with per-frame detections and corrected counts.
    """
    pipeline = HerringNetPipeline(config)
    video_path = Path(video_path)

    if output_dir is None:
        output_dir = Path(config.output.output_dir) / video_path.stem
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = pipeline.process_video(
        video_path,
        output_dir=output_dir,
        max_frames=max_frames,
        save_frames=save_annotated_frames,
    )

    # Save annotated frames if requested
    if save_annotated_frames:
        _save_annotated_frames(result, video_path, output_dir)

    # Save CSV summary if requested
    if save_csv:
        csv_path = output_dir / f"{video_path.stem}_counts.csv"
        save_counts_csv(result, csv_path)

    return result


def save_counts_csv(result: VideoResult, output_path: str | Path) -> str:
    """Save detection counts as a CSV file.

    Columns: frame_number, timestamp, fish_count, and per-species counts.

    Args:
        result: VideoResult from pipeline processing.
        output_path: Path to save the CSV file.

    Returns:
        Path to the saved CSV file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect all species seen across all frames
    all_species: set[str] = set()
    for frame in result.per_frame_results:
        all_species.update(frame.species_counts.keys())
    species_list = sorted(all_species)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)

        # Header
        header = ["frame_number", "timestamp_sec", "total_fish"]
        header.extend(species_list)
        header.append("flagged_for_review")
        writer.writerow(header)

        # Data rows
        for frame in result.per_frame_results:
            row = [
                frame.frame_number or 0,
                f"{frame.timestamp:.3f}" if frame.timestamp is not None else "",
                frame.fish_count,
            ]
            for species in species_list:
                row.append(frame.species_counts.get(species, 0))
            row.append(frame.flagged_for_review)
            writer.writerow(row)

        # Summary row
        counts = result.counts
        summary = [
            "TOTAL (raw)",
            "",
            counts.raw_total,
        ]
        for species in species_list:
            pair = counts.species_counts.get(species)
            summary.append(pair.raw if pair else 0)
        summary.append("")
        writer.writerow(summary)

        corrected = [
            "TOTAL (corrected)",
            "",
            f"{counts.corrected_total:.1f}",
        ]
        for species in species_list:
            pair = counts.species_counts.get(species)
            corrected.append(f"{pair.corrected:.1f}" if pair else "0.0")
        corrected.append("")
        writer.writerow(corrected)

    logger.info("Saved counts CSV: %s", output_path)
    return str(output_path)


def _save_annotated_frames(
    result: VideoResult,
    video_path: Path,
    output_dir: Path,
) -> None:
    """Save annotated frame images for frames that contain detections.

    Only saves frames with at least one detection to avoid filling
    disk with empty frames.
    """
    annotated_dir = output_dir / "annotated"
    annotated_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning("Cannot open video for annotated frame export: %s", video_path)
        return

    try:
        for frame_result in result.per_frame_results:
            if frame_result.fish_count == 0:
                continue
            if frame_result.frame_number is None:
                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_result.frame_number)
            ret, frame = cap.read()
            if not ret:
                continue

            annotated = draw_detections(frame, frame_result.detections)
            draw_count_overlay(
                annotated,
                frame_result.fish_count,
                frame_number=frame_result.frame_number,
            )

            filename = f"annotated_frame{frame_result.frame_number:06d}.jpg"
            cv2.imwrite(str(annotated_dir / filename), annotated)
    finally:
        cap.release()

    logger.info("Saved annotated frames to %s", annotated_dir)
