"""Visualization utilities for drawing detection results on images.

Draws bounding boxes, species labels, and confidence scores on camera
trap images. Uses species-specific colors for easy visual identification.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from herringnet.inference.result_types import PipelineDetection

# Default species color map (BGR format for OpenCV).
DEFAULT_COLORS: dict[str, tuple[int, int, int]] = {
    "fish": (0, 255, 0),           # Green
    "river_herring": (0, 255, 0),  # Green
    "bass": (255, 0, 0),           # Blue
    "perch": (0, 165, 255),        # Orange
    "trout": (255, 255, 0),        # Cyan
    "unknown_fish": (128, 128, 128),  # Gray
}

# Color for uncertain detections
UNCERTAIN_COLOR = (0, 0, 255)  # Red


def get_color(species: str) -> tuple[int, int, int]:
    """Get the display color for a species.

    Args:
        species: Species name.

    Returns:
        BGR color tuple.
    """
    return DEFAULT_COLORS.get(species, (0, 255, 0))


def draw_detections(
    image: np.ndarray,
    detections: list[PipelineDetection],
    show_confidence: bool = True,
    show_species: bool = True,
    line_thickness: int = 2,
    font_scale: float = 0.6,
) -> np.ndarray:
    """Draw bounding boxes and labels on an image.

    Args:
        image: Source image as a numpy array (BGR format). Will be copied.
        detections: List of pipeline detection results to draw.
        show_confidence: Whether to show confidence scores in labels.
        show_species: Whether to show species names in labels.
        line_thickness: Thickness of bounding box lines in pixels.
        font_scale: Font size scale for labels.

    Returns:
        Annotated copy of the image.
    """
    annotated = image.copy()

    for det in detections:
        bbox = det.detection.bbox
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        # Choose color based on species or uncertainty
        if det.is_uncertain:
            color = UNCERTAIN_COLOR
        elif det.classification is not None:
            color = get_color(det.classification.species)
        else:
            color = get_color("fish")

        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, line_thickness)

        # Build label text
        label_parts = []
        if show_species and det.classification is not None:
            species = det.classification.species.replace("_", " ").title()
            label_parts.append(species)
        elif show_species:
            label_parts.append("Fish")

        if show_confidence:
            if det.classification is not None:
                conf = det.classification.confidence
            else:
                conf = det.detection.confidence
            label_parts.append(f"{conf:.0%}")

        if det.is_uncertain:
            label_parts.append("?")

        label = " ".join(label_parts)

        if label:
            _draw_label(annotated, label, (x1, y1), color, font_scale)

    return annotated


def draw_count_overlay(
    image: np.ndarray,
    fish_count: int,
    frame_number: int | None = None,
    font_scale: float = 0.8,
) -> np.ndarray:
    """Draw a count overlay in the top-left corner of the image.

    Args:
        image: Source image (will be modified in place).
        fish_count: Number of fish detected.
        frame_number: Optional frame number to display.
        font_scale: Font size scale.

    Returns:
        The image with the overlay drawn.
    """
    text = f"Fish: {fish_count}"
    if frame_number is not None:
        text = f"Frame {frame_number} | {text}"

    # Draw background rectangle
    (text_w, text_h), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2
    )
    cv2.rectangle(
        image,
        (5, 5),
        (15 + text_w, 15 + text_h + baseline),
        (0, 0, 0),
        cv2.FILLED,
    )

    # Draw text
    cv2.putText(
        image,
        text,
        (10, 10 + text_h),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        2,
    )

    return image


def save_annotated_image(
    image: np.ndarray,
    detections: list[PipelineDetection],
    output_path: str | Path,
    show_confidence: bool = True,
    show_species: bool = True,
) -> str:
    """Draw detections on an image and save it to disk.

    Args:
        image: Source image (BGR format).
        detections: Detection results to draw.
        output_path: Path to save the annotated image.
        show_confidence: Whether to show confidence scores.
        show_species: Whether to show species names.

    Returns:
        Path to the saved image file.
    """
    annotated = draw_detections(
        image, detections,
        show_confidence=show_confidence,
        show_species=show_species,
    )
    draw_count_overlay(annotated, len(detections))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), annotated)

    return str(output_path)


def _draw_label(
    image: np.ndarray,
    label: str,
    position: tuple[int, int],
    color: tuple[int, int, int],
    font_scale: float,
) -> None:
    """Draw a label with a background rectangle above a bounding box.

    Args:
        image: Image to draw on (modified in place).
        label: Text label to draw.
        position: Top-left corner of the bounding box (x, y).
        color: Color for the label background.
        font_scale: Font size scale.
    """
    x, y = position
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1

    (text_w, text_h), baseline = cv2.getTextSize(
        label, font, font_scale, thickness
    )

    # Draw filled rectangle behind text
    label_y = max(y - text_h - baseline - 4, 0)
    cv2.rectangle(
        image,
        (x, label_y),
        (x + text_w + 4, label_y + text_h + baseline + 4),
        color,
        cv2.FILLED,
    )

    # Draw text in white or black depending on background brightness
    brightness = sum(color) / 3
    text_color = (0, 0, 0) if brightness > 127 else (255, 255, 255)

    cv2.putText(
        image,
        label,
        (x + 2, label_y + text_h + 2),
        font,
        font_scale,
        text_color,
        thickness,
    )
