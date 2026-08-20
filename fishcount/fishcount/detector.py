"""Model loading and inference: the only module that talks to the detector.

Detection is the product. The detector runs one full-frame pass per image at a
recall-first operating point (imgsz 1536, confidence floor 0.10, IoU 0.7,
max_det 3000); everything above the floor is recorded and later tiered by
fishcount.classify, never silently dropped.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from fishcount.config import AppConfig

ImageArray = NDArray[np.uint8]

MODEL_FILENAME = "cfd-yolov12x.pt"


class ModelNotFoundError(FileNotFoundError):
    """The fish detector weights are not where they should be."""


@dataclass(frozen=True, slots=True)
class Detection:
    """One detection, as pixel coordinates on the original image."""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float

    def int_box(self) -> tuple[int, int, int, int]:
        """The box rounded to integer pixels: (x1, y1, x2, y2)."""
        return (round(self.x1), round(self.y1), round(self.x2), round(self.y2))


class Detector(Protocol):
    """Anything that turns a batch of BGR images into per-image detections."""

    def detect_batch(self, images: Sequence[ImageArray]) -> list[list[Detection]]:
        """Return one list of detections per input image, in input order."""
        ...


def resolve_model_path(configured: Path) -> Path:
    """Locate the model weights, or raise ModelNotFoundError with instructions.

    Relative paths are tried against the current working directory first, then
    against the project root (the folder containing the fishcount package), so
    the CLI works from any directory in an editable install.
    """
    if configured.is_absolute():
        if configured.is_file():
            return configured
        raise ModelNotFoundError(_missing_message([configured]))
    tried: list[Path] = []
    for base in (Path.cwd(), Path(__file__).resolve().parents[1]):
        candidate = (base / configured).resolve()
        if candidate.is_file():
            return candidate
        if candidate not in tried:
            tried.append(candidate)
    raise ModelNotFoundError(_missing_message(tried))


def _missing_message(tried: Sequence[Path]) -> str:
    locations = "\n".join(f"  - {path}" for path in tried)
    return (
        "Fish detector weights not found. Looked for:\n"
        f"{locations}\n"
        f"Place {MODEL_FILENAME} (the Community Fish Detector) in the models/ folder.\n"
        "This tool never downloads, retrains, or substitutes another model."
    )


class YoloDetector:
    """Batched Ultralytics YOLO inference. Loads the model once and reuses it."""

    def __init__(self, weights: Path, config: AppConfig) -> None:
        if not weights.is_file():
            raise ModelNotFoundError(_missing_message([weights]))
        # Lazy import: keeps --help, error paths, and the test suite fast.
        from ultralytics import YOLO

        self._model = YOLO(str(weights))
        self._conf = config.conf
        self._iou = config.iou
        self._imgsz = config.imgsz
        self._max_det = config.max_det

    def detect_batch(self, images: Sequence[ImageArray]) -> list[list[Detection]]:
        """One forward pass over the whole batch."""
        if not images:
            return []
        outputs = self._model.predict(
            source=list(images),
            conf=self._conf,
            iou=self._iou,
            imgsz=self._imgsz,
            max_det=self._max_det,
            verbose=False,
        )
        return [_to_detections(output) for output in outputs]


def create_detector(weights: Path, config: AppConfig) -> Detector:
    """Build the detector for a run."""
    return YoloDetector(weights, config)


def _to_detections(output: Any) -> list[Detection]:
    boxes = output.boxes
    if boxes is None or len(boxes) == 0:
        return []
    coords = boxes.xyxy.cpu().numpy()
    scores = boxes.conf.cpu().numpy()
    return [
        Detection(float(x1), float(y1), float(x2), float(y2), float(score))
        for (x1, y1, x2, y2), score in zip(coords, scores, strict=True)
    ]
