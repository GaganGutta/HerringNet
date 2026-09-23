"""Model loading and inference: the only module that talks to the detector.

Detection is the product. The detector runs one full-frame pass per image at a
recall-first operating point (imgsz 1536, recording floor 0.10, IoU 0.7,
max_det 3000); everything above the floor is recorded, never silently dropped.

The GPU is used when there is one and the CPU otherwise, decided once at
startup. Batch size is a throughput knob, not a correctness one, so running out
of VRAM is treated as a hint rather than a failure: an out-of-memory batch is
split in half and retried until it fits. A run that would have died hours in
instead slows down for one batch and carries on.
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


def select_device(requested: str) -> str:
    """Resolve config.device to a device string torch will accept.

    "auto" means the GPU when one is actually usable and the CPU otherwise.
    Torch is imported lazily and a missing or broken CUDA stack falls back to
    the CPU rather than raising: a slow run beats no run.
    """
    if requested != "auto":
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def default_batch_size(device: str) -> int:
    """How many images to push through the model at once when nothing is set.

    Measured, not guessed. On an 8 GB RTX 4070 at imgsz 1536 this model needs
    ~2.4 GB for one image, ~6.6 GB for three, and more than the card for four;
    once it overflows, Windows lets CUDA spill into system RAM over PCIe and
    throughput collapses (0.31 s/frame at batch 1, 1.05 s/frame at batch 4).
    Batching buys nothing on a GPU this model already saturates, so the default
    is one image: the fastest measured setting and the one leaving the most
    headroom on a smaller card. The CPU keeps the batch of 8 it has always had.
    """
    return 1 if device.startswith("cuda") else 8


def describe_device(device: str) -> str:
    """A human-readable name for the device in use, for the run log."""
    if not device.startswith("cuda"):
        return "CPU"
    try:
        import torch

        index = int(device.split(":")[1]) if ":" in device else 0
        total = torch.cuda.get_device_properties(index).total_memory / 1024**3
        return f"{torch.cuda.get_device_name(index)} ({total:.0f} GB)"
    except Exception:
        return device


def _is_out_of_memory(exc: BaseException) -> bool:
    """Whether an exception is the GPU (or host) running out of memory.

    Matched on the message as well as the type because the exact class has
    moved between torch releases.
    """
    if isinstance(exc, MemoryError):
        return True
    text = str(exc).lower()
    return "out of memory" in text or "cuda error: out of memory" in text


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

        self.device = select_device(config.device)
        self._model = YOLO(str(weights))
        self._model.to(self.device)
        self._conf = config.conf
        self._iou = config.iou
        self._imgsz = config.imgsz
        self._max_det = config.max_det
        self.oom_splits = 0  # how often a batch had to be halved, for the run log

    def detect_batch(self, images: Sequence[ImageArray]) -> list[list[Detection]]:
        """One forward pass over the batch, halving it if VRAM runs out."""
        if not images:
            return []
        return self._detect(list(images))

    def _detect(self, images: list[ImageArray]) -> list[list[Detection]]:
        try:
            outputs = self._model.predict(
                source=images,
                conf=self._conf,
                iou=self._iou,
                imgsz=self._imgsz,
                max_det=self._max_det,
                device=self.device,
                verbose=False,
            )
        except (RuntimeError, MemoryError) as exc:
            if len(images) == 1 or not _is_out_of_memory(exc):
                raise
            self.oom_splits += 1
            self._empty_cache()
            half = len(images) // 2
            return self._detect(images[:half]) + self._detect(images[half:])
        return [_to_detections(output) for output in outputs]

    def _empty_cache(self) -> None:
        if not self.device.startswith("cuda"):
            return
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass  # freeing cache is best-effort; the retry is what matters


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
