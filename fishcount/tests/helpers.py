"""Shared test helpers: a scripted detector and tiny on-disk test images."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from fishcount.detector import Detection, ImageArray


class FakeDetector:
    """Returns queued detections in image order; needs no model weights.

    `batch_sizes` records how many images each detect_batch call received, so
    tests can assert that inference is actually batched.
    """

    def __init__(self, script: list[list[Detection]] | None = None) -> None:
        self._script = list(script or [])
        self.batch_sizes: list[int] = []

    def detect_batch(self, images: Sequence[ImageArray]) -> list[list[Detection]]:
        self.batch_sizes.append(len(images))
        return [self._script.pop(0) if self._script else [] for _ in images]


def fish(
    x1: float = 4, y1: float = 6, x2: float = 30, y2: float = 24, conf: float = 0.9
) -> Detection:
    return Detection(x1, y1, x2, y2, conf)


def write_image(path: Path, *, width: int = 64, height: int = 48) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.full((height, width, 3), (40, 80, 120), dtype=np.uint8)
    assert cv2.imwrite(str(path), array)
