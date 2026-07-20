"""Counting and aggregation: per-image detections in, CSV/JSON/totals out.

Phase 1 counts each image independently: total fish = sum of per-image counts.

Phase 2 design (not built yet): image sequences and video frames need
double-count correction, because one fish stays in view across many consecutive
frames. The seam is `total_fish` below. It consumes ordered per-image results
that already carry pixel boxes, which is exactly what a sequence-aware
aggregator needs, so Phase 2 slots in here without touching detection:

- frame-residence-rate correction: unique fish ~= raw detections scaled by how
  many frames the average fish stays in view, or
- lightweight tracking: match boxes across consecutive frames (IoU/centroid)
  and count track births instead of detections.

Either lands as e.g. `total_fish_sequence(results, strategy=...)` plus a CLI
flag. fishcount.detector and fishcount.batch stay unchanged.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from fishcount.detector import Detection

TOTAL_ROW_LABEL = "TOTAL"


@dataclass(slots=True)
class ImageResult:
    """Detections for one image; `file` is the path relative to the input folder."""

    file: str
    width: int = 0
    height: int = 0
    detections: list[Detection] = field(default_factory=list)
    error: str | None = None

    @property
    def count(self) -> int:
        return len(self.detections)

    @property
    def ok(self) -> bool:
        return self.error is None


def total_fish(results: Sequence[ImageResult]) -> int:
    """Phase 1 aggregate: every detection in every image counts once."""
    return sum(result.count for result in results if result.ok)


def write_counts_csv(results: Sequence[ImageResult], path: Path) -> None:
    """One row per successfully processed image, plus a grand-total row."""
    rows: list[dict[str, object]] = [
        {"filename": result.file, "fish_count": result.count} for result in results if result.ok
    ]
    rows.append({"filename": TOTAL_ROW_LABEL, "fish_count": total_fish(results)})
    pd.DataFrame(rows, columns=["filename", "fish_count"]).to_csv(path, index=False)


def write_results_json(
    results: Sequence[ImageResult],
    path: Path,
    *,
    input_dir: Path,
    model_path: Path,
    conf: float,
    imgsz: int,
) -> None:
    """Full per-detection dump: boxes as [x1, y1, x2, y2] pixels plus confidence."""
    payload = {
        "input": str(input_dir),
        "model": str(model_path),
        "conf": conf,
        "imgsz": imgsz,
        "total_fish": total_fish(results),
        "images": [_image_entry(result) for result in results],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _image_entry(result: ImageResult) -> dict[str, object]:
    if not result.ok:
        return {"file": result.file, "error": result.error}
    return {
        "file": result.file,
        "width": result.width,
        "height": result.height,
        "fish_count": result.count,
        "detections": [
            {"box": list(detection.int_box()), "confidence": round(detection.confidence, 3)}
            for detection in result.detections
        ],
    }
