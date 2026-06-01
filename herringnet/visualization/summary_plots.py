"""Summary visualization plots for detection results.

Generates time-series count plots, confidence score distributions,
and species breakdown charts for video processing results.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from herringnet.inference.result_types import FrameResult, VideoResult


def plot_detection_timeline(
    result: VideoResult,
    output_path: str | Path,
    bin_minutes: int = 5,
) -> str:
    """Plot fish detections over time from a video.

    Creates a bar chart showing the number of detections per time bin.

    Args:
        result: VideoResult from pipeline processing.
        output_path: Path to save the plot image.
        bin_minutes: Duration of each time bin in minutes.

    Returns:
        Path to the saved plot image.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    timestamps = []
    counts = []
    for frame in result.per_frame_results:
        if frame.timestamp is not None:
            timestamps.append(frame.timestamp / 60.0)  # Convert to minutes
            counts.append(frame.fish_count)

    if not timestamps:
        # Create an empty plot with a message
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "No timestamped frames available",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel("Time (minutes)")
        ax.set_ylabel("Fish Count")
        ax.set_title("Detection Timeline")
        plt.tight_layout()
        plt.savefig(str(output_path), dpi=150)
        plt.close()
        return str(output_path)

    fig, ax = plt.subplots(figsize=(10, 4))

    # Bin the data
    max_time = max(timestamps)
    bins = np.arange(0, max_time + bin_minutes, bin_minutes)
    binned_counts = np.zeros(len(bins) - 1)

    for t, c in zip(timestamps, counts):
        bin_idx = int(t / bin_minutes)
        if 0 <= bin_idx < len(binned_counts):
            binned_counts[bin_idx] += c

    bin_centers = (bins[:-1] + bins[1:]) / 2
    ax.bar(bin_centers, binned_counts, width=bin_minutes * 0.8,
           color="#2196F3", alpha=0.8, edgecolor="#1565C0")

    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("Fish Detections (raw)")
    ax.set_title(f"Fish Detection Timeline - {Path(result.video_path).name}")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150)
    plt.close()

    return str(output_path)


def plot_confidence_distribution(
    frame_results: list[FrameResult],
    output_path: str | Path,
) -> str:
    """Plot the distribution of detection confidence scores.

    Args:
        frame_results: List of frame results with detections.
        output_path: Path to save the plot image.

    Returns:
        Path to the saved plot image.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    confidences = []
    for frame in frame_results:
        for det in frame.detections:
            confidences.append(det.detection.confidence)

    fig, ax = plt.subplots(figsize=(8, 4))

    if confidences:
        ax.hist(confidences, bins=20, range=(0, 1),
                color="#4CAF50", alpha=0.8, edgecolor="#2E7D32")
        ax.axvline(x=np.mean(confidences), color="red", linestyle="--",
                   label=f"Mean: {np.mean(confidences):.2f}")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No detections to plot",
                ha="center", va="center", transform=ax.transAxes)

    ax.set_xlabel("Detection Confidence")
    ax.set_ylabel("Count")
    ax.set_title("Detection Confidence Distribution")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150)
    plt.close()

    return str(output_path)


def plot_species_breakdown(
    frame_results: list[FrameResult],
    output_path: str | Path,
) -> str:
    """Plot a species breakdown bar chart.

    Args:
        frame_results: List of frame results with classifications.
        output_path: Path to save the plot image.

    Returns:
        Path to the saved plot image.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    species_counts: dict[str, int] = {}
    for frame in frame_results:
        for species, count in frame.species_counts.items():
            species_counts[species] = species_counts.get(species, 0) + count

    fig, ax = plt.subplots(figsize=(8, 4))

    if species_counts:
        species = list(species_counts.keys())
        counts = list(species_counts.values())

        # Clean up species names for display
        display_names = [s.replace("_", " ").title() for s in species]

        colors = plt.cm.Set2(np.linspace(0, 1, len(species)))
        bars = ax.bar(display_names, counts, color=colors, edgecolor="gray")

        # Add count labels on bars
        for bar, count in zip(bars, counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                str(count),
                ha="center",
                va="bottom",
                fontsize=10,
            )
    else:
        ax.text(0.5, 0.5, "No species data to plot",
                ha="center", va="center", transform=ax.transAxes)

    ax.set_xlabel("Species")
    ax.set_ylabel("Detection Count")
    ax.set_title("Species Breakdown")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150)
    plt.close()

    return str(output_path)
