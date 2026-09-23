"""Turn results.json into the two flat views: one row per box, one row per frame.

This module derives; it does not decide. Every detection the model produced
above the recording floor becomes a row in detections.csv, unchanged. No box is
dropped, demoted, or annotated with a verdict, because no such rule here has
ever been measured against labelled frames.

One number separates "frame holds fish" from "frame does not": the reporting
threshold. It is applied in exactly one place, `n_boxes_above_threshold`, and
because detections.csv keeps every box and its confidence, any other threshold
can be evaluated later without re-running the detector.

Frame statistics (blur, brightness) are carried through as recorded. They are
there to slice evaluation by condition, not to filter anything.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DETECTIONS_HEADER = ["frame", "x1", "y1", "x2", "y2", "confidence", "area_frac"]
FRAMES_HEADER = [
    "frame",
    "max_conf",
    "n_boxes_above_threshold",
    "blur",
    "brightness",
    "error",
]


@dataclass(slots=True)
class ReportSummary:
    """What the run found, for the console summary."""

    out_dir: Path
    threshold: float
    frames: int
    frames_with_detections: int
    frames_above_threshold: int
    detections: int
    detections_above_threshold: int
    unreadable: int


def write_reports(out_dir: Path, *, threshold: float) -> ReportSummary:
    """Read out_dir/results.json; write detections.csv and frames.csv.

    Both files are sorted by frame path. Every frame gets a row in frames.csv,
    including frames with no detections and frames that could not be read, so
    the frame list is a complete census of the input folder.
    """
    payload = json.loads((out_dir / "results.json").read_text(encoding="utf-8"))
    images = sorted(payload["images"], key=lambda image: str(image["file"]))

    summary = ReportSummary(
        out_dir=out_dir,
        threshold=threshold,
        frames=len(images),
        frames_with_detections=0,
        frames_above_threshold=0,
        detections=0,
        detections_above_threshold=0,
        unreadable=0,
    )

    detections_path = out_dir / "detections.csv"
    frames_path = out_dir / "frames.csv"
    with (
        detections_path.open("w", encoding="utf-8", newline="") as detections_file,
        frames_path.open("w", encoding="utf-8", newline="") as frames_file,
    ):
        detections_writer = csv.writer(detections_file)
        detections_writer.writerow(DETECTIONS_HEADER)
        frames_writer = csv.writer(frames_file)
        frames_writer.writerow(FRAMES_HEADER)

        for image in images:
            if image.get("error") is not None:
                summary.unreadable += 1
                frames_writer.writerow([image["file"], "", "", "", "", image["error"]])
                continue
            detections = image.get("detections", [])
            area = float(image["width"]) * float(image["height"])
            confidences = [float(det["confidence"]) for det in detections]
            above = sum(conf >= threshold for conf in confidences)

            for det, conf in zip(detections, confidences, strict=True):
                detections_writer.writerow(_detection_row(image["file"], det, conf, area))
            frames_writer.writerow(_frame_row(image, confidences, above))

            summary.detections += len(detections)
            summary.detections_above_threshold += above
            summary.frames_with_detections += len(detections) > 0
            summary.frames_above_threshold += above > 0

    return summary


def _detection_row(frame: str, det: dict[str, Any], conf: float, area: float) -> list[str]:
    x1, y1, x2, y2 = det["box"]
    box_area = max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))
    area_frac = box_area / area if area > 0 else 0.0
    return [frame, str(x1), str(y1), str(x2), str(y2), f"{conf:.3f}", f"{area_frac:.6f}"]


def _frame_row(image: dict[str, Any], confidences: list[float], above: int) -> list[str]:
    return [
        image["file"],
        f"{max(confidences, default=0.0):.3f}",
        str(above),
        _number(image.get("blur")),
        _number(image.get("brightness")),
        "",
    ]


def _number(value: float | None) -> str:
    """Format a recorded statistic, leaving it blank when it was not recorded."""
    return "" if value is None else f"{value:.1f}"
