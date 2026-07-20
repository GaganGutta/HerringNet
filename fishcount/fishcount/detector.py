"""Model loading and inference: the only module that talks to detector backends.

Counting lives in fishcount.count. Keeping detection behind the small Detector
protocol is what lets Phase 2 (sequence-aware counting for video frames) land
without touching inference, and lets tests substitute a scripted fake detector
that needs no model weights.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np
from numpy.typing import NDArray

from fishcount.config import AppConfig

ImageArray = NDArray[np.uint8]

MODEL_FILENAME = "cfd-yolov12x.pt"


class ModelNotFoundError(FileNotFoundError):
    """The fish detector weights are not where they should be."""


@dataclass(frozen=True, slots=True)
class Detection:
    """One detected fish, as pixel coordinates on the original image."""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float

    def int_box(self) -> tuple[int, int, int, int]:
        """The box rounded to integer pixels: (x1, y1, x2, y2)."""
        return (round(self.x1), round(self.y1), round(self.x2), round(self.y2))


class Detector(Protocol):
    """Anything that turns a batch of BGR images into per-image fish detections."""

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

    def detect_batch(self, images: Sequence[ImageArray]) -> list[list[Detection]]:
        """One forward pass over the whole batch. Every detection is a fish."""
        if not images:
            return []
        outputs = self._model.predict(
            source=list(images),
            conf=self._conf,
            iou=self._iou,
            imgsz=self._imgsz,
            verbose=False,
        )
        return [_to_detections(output) for output in outputs]


class SahiDetector:
    """Sliced (tiled) inference via SAHI, to catch very small fish. Much slower.

    SAHI processes one image at a time; the Detector interface stays batched so
    the pipeline does not care which backend it got.
    """

    def __init__(self, weights: Path, config: AppConfig) -> None:
        if not weights.is_file():
            raise ModelNotFoundError(_missing_message([weights]))
        try:
            from sahi import AutoDetectionModel
            from sahi.predict import get_sliced_prediction
        except ImportError as exc:
            raise RuntimeError(
                "--thorough needs the optional sahi package. "
                'Install it with: pip install -e ".[thorough]"'
            ) from exc
        self._sliced_prediction = get_sliced_prediction
        self._model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=str(weights),
            confidence_threshold=config.conf,
            device="cpu",
            image_size=config.imgsz,
        )
        self._slice = min(config.imgsz, 1024)

    def detect_batch(self, images: Sequence[ImageArray]) -> list[list[Detection]]:
        results: list[list[Detection]] = []
        for image in images:
            prediction = self._sliced_prediction(
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                self._model,
                slice_height=self._slice,
                slice_width=self._slice,
                overlap_height_ratio=0.2,
                overlap_width_ratio=0.2,
                verbose=0,
            )
            detections: list[Detection] = []
            for obj in prediction.object_prediction_list:
                x1, y1, x2, y2 = obj.bbox.to_xyxy()
                detections.append(
                    Detection(float(x1), float(y1), float(x2), float(y2), float(obj.score.value))
                )
            results.append(detections)
        return results


def create_detector(weights: Path, config: AppConfig, *, thorough: bool = False) -> Detector:
    """Build the configured detector; thorough=True swaps in SAHI tiling."""
    if thorough:
        return SahiDetector(weights, config)
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
