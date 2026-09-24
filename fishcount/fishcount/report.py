"""Turn the journal into the two flat views: one row per box, one row per frame.

This module derives; it does not decide. Every detection the model produced
above the recording floor becomes a row in detections.csv, unchanged. No box is
dropped, demoted, or annotated with a verdict, because no such rule here has
ever been measured against labelled frames.

One number separates "frame holds fish" from "frame does not": the reporting
threshold. It is applied here and nowhere else -- to `n_boxes_above_threshold`,
and to which frames get an annotated image. Detection never sees it, which is
what lets it change: re-reporting an existing journal at a new threshold costs
a pass over a text file and a few image writes, not hours of inference.

Memory stays flat. The journal is read back one frame at a time in sorted
order, so reporting on fifty thousand frames costs no more than fifty.
"""

from __future__ import annotations

import csv
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2

from fishcount.batch import load_image
from fishcount.detector import Detection
from fishcount.draw import annotate
from fishcount.journal import JOURNAL_FILENAME, stream_sorted

DETECTIONS_HEADER = ["frame", "x1", "y1", "x2", "y2", "confidence", "area_frac"]
FRAMES_HEADER = [
    "frame",
    "max_conf",
    "n_boxes_above_threshold",
    "blur",
    "brightness",
    "error",
]

# The whole answer in two columns, for when the question is only "which frames
# do I need to look at". The path is relative to the run's input folder and
# never a bare filename: GoPro reuses basenames between cards, so across this
# project's frames there are thousands of names that belong to two different
# photographs.
FISH_OR_NOT_HEADER = ["file", "has_fish"]


@dataclass(slots=True)
class ReportSummary:
    """What the run found, for the console summary."""

    out_dir: Path
    dest: Path
    threshold: float
    frames: int
    frames_with_detections: int
    frames_above_threshold: int
    detections: int
    detections_above_threshold: int
    unreadable: int
    annotated: int = 0


def write_reports(
    out_dir: Path,
    *,
    threshold: float,
    input_dir: Path | None = None,
    write_images: bool = False,
    only: Sequence[str] | None = None,
    dest: Path | None = None,
) -> ReportSummary:
    """Read the journal; write detections.csv, frames.csv and fish_or_not.csv.

    All three are sorted by frame path. Every frame gets a row in frames.csv
    and in fish_or_not.csv, including frames with no detections and frames
    that could not be read, so the list is a complete census of what was
    scanned. An unreadable frame reads as `no`, since nothing was found in it;
    frames.csv is where the distinction between "nothing there" and "could not
    look" is kept.

    `only` keeps just the frames whose path starts with one of the given
    prefixes, which is how one folder is reported on without re-running the
    detector over it. `dest` sends the CSVs somewhere other than the run
    folder, so a filtered report cannot overwrite the whole run's.

    With `write_images` and `input_dir`, frames at or above the threshold are
    re-read and written to dest/annotated/ with every recorded box drawn on
    them. The folder is rebuilt from scratch so it always matches the threshold
    that produced it, never a leftover from an earlier one.
    """
    journal = out_dir / JOURNAL_FILENAME
    dest = dest if dest is not None else out_dir
    dest.mkdir(parents=True, exist_ok=True)
    summary = ReportSummary(
        out_dir=out_dir,
        dest=dest,
        threshold=threshold,
        frames=0,
        frames_with_detections=0,
        frames_above_threshold=0,
        detections=0,
        detections_above_threshold=0,
        unreadable=0,
    )

    annotated_dir = dest / "annotated"
    annotating = write_images and input_dir is not None
    if annotating:
        shutil.rmtree(annotated_dir, ignore_errors=True)

    with (
        (dest / "detections.csv").open("w", encoding="utf-8", newline="") as detections_file,
        (dest / "frames.csv").open("w", encoding="utf-8", newline="") as frames_file,
        (dest / "fish_or_not.csv").open("w", encoding="utf-8", newline="") as verdict_file,
    ):
        detections_writer = csv.writer(detections_file)
        detections_writer.writerow(DETECTIONS_HEADER)
        frames_writer = csv.writer(frames_file)
        frames_writer.writerow(FRAMES_HEADER)
        verdict_writer = csv.writer(verdict_file)
        verdict_writer.writerow(FISH_OR_NOT_HEADER)

        for image in stream_sorted(journal):
            if only is not None and not _matches(str(image["file"]), only):
                continue
            summary.frames += 1
            if image.get("error") is not None:
                summary.unreadable += 1
                frames_writer.writerow([image["file"], "", "", "", "", image["error"]])
                verdict_writer.writerow([image["file"], "no"])
                continue
            detections = image.get("detections", [])
            area = float(image["width"]) * float(image["height"])
            confidences = [float(det["confidence"]) for det in detections]
            above = sum(conf >= threshold for conf in confidences)

            for det, conf in zip(detections, confidences, strict=True):
                detections_writer.writerow(_detection_row(image["file"], det, conf, area))
            frames_writer.writerow(_frame_row(image, confidences, above))
            verdict_writer.writerow([image["file"], "yes" if above else "no"])

            summary.detections += len(detections)
            summary.detections_above_threshold += above
            summary.frames_with_detections += len(detections) > 0
            summary.frames_above_threshold += above > 0

            if annotating and above > 0 and input_dir is not None:
                summary.annotated += _annotate_frame(
                    input_dir / image["file"], annotated_dir / image["file"], detections
                )

    return summary


def _matches(file: str, prefixes: Sequence[str]) -> bool:
    """Whether a frame's path sits under one of the wanted folders."""
    return any(file == prefix or file.startswith(prefix.rstrip("/") + "/") for prefix in prefixes)


def _annotate_frame(source: Path, dest: Path, detections: list[dict[str, Any]]) -> int:
    """Draw every recorded box on one frame. Returns 1 if written, 0 if not.

    A frame that has vanished or gone unreadable since detection is skipped
    rather than failing the report: the CSVs are the product, the images are
    a convenience for looking at them.
    """
    array = load_image(source)
    if array is None:
        return 0
    boxes = [_to_detection(det) for det in detections]
    _write_annotated(annotate(array, boxes), dest)
    return 1


def _to_detection(det: dict[str, Any]) -> Detection:
    x1, y1, x2, y2 = (float(value) for value in det["box"])
    return Detection(x1, y1, x2, y2, float(det["confidence"]))


def _write_annotated(image: Any, dest: Path) -> None:
    """Encode with the original extension (PNG as a fallback); unicode-path safe."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(dest.suffix.lower(), image)
    if not ok:
        dest = dest.with_suffix(".png")
        ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise OSError(f"could not encode annotated image for {dest}")
    encoded.tofile(str(dest))


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
