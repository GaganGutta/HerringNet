"""Drawing: detection boxes with confidences. Never mutates its input."""

from __future__ import annotations

from collections.abc import Sequence

import cv2

from fishcount.detector import Detection, ImageArray

_BOX_COLOR = (60, 220, 60)  # BGR green
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def annotate(image: ImageArray, detections: Sequence[Detection]) -> ImageArray:
    """Return a copy of `image` with a box and confidence label per detection."""
    out: ImageArray = image.copy()
    height, width = out.shape[:2]
    thickness = max(2, round(min(height, width) / 500))
    for detection in detections:
        _draw_box(out, detection, thickness)
    return out


def _draw_box(out: ImageArray, detection: Detection, thickness: int) -> None:
    x1, y1, x2, y2 = detection.int_box()
    cv2.rectangle(out, (x1, y1), (x2, y2), _BOX_COLOR, thickness)
    label = f"{detection.confidence:.2f}"
    scale = max(0.4, thickness * 0.3)
    offset = 4 + thickness
    position = (x1, y1 - offset) if y1 > 10 * offset else (x1, y2 + 6 * offset)
    cv2.putText(out, label, position, _FONT, scale, _BOX_COLOR, max(1, thickness - 1), cv2.LINE_AA)
