"""Drawing: fish boxes and the per-image count banner. Never mutates its input."""

from __future__ import annotations

from collections.abc import Sequence

import cv2

from fishcount.detector import Detection, ImageArray

_BOX_COLOR = (60, 220, 60)  # BGR green
_TEXT_COLOR = (255, 255, 255)
_BANNER_COLOR = (30, 30, 30)
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def annotate(image: ImageArray, detections: Sequence[Detection]) -> ImageArray:
    """Return a copy of `image` with boxes around fish and a 'Fish: N' banner."""
    out: ImageArray = image.copy()
    height, width = out.shape[:2]
    thickness = max(2, round(min(height, width) / 500))
    for detection in detections:
        _draw_box(out, detection, thickness)
    _draw_banner(out, f"Fish: {len(detections)}", thickness)
    return out


def _draw_box(out: ImageArray, detection: Detection, thickness: int) -> None:
    x1, y1, x2, y2 = detection.int_box()
    cv2.rectangle(out, (x1, y1), (x2, y2), _BOX_COLOR, thickness)
    label = f"{detection.confidence:.2f}"
    scale = max(0.4, thickness * 0.3)
    offset = 4 + thickness
    position = (x1, y1 - offset) if y1 > 10 * offset else (x1, y2 + 6 * offset)
    cv2.putText(out, label, position, _FONT, scale, _BOX_COLOR, max(1, thickness - 1), cv2.LINE_AA)


def _draw_banner(out: ImageArray, text: str, thickness: int) -> None:
    scale = max(0.7, thickness * 0.45)
    text_thickness = max(2, thickness)
    (text_width, text_height), baseline = cv2.getTextSize(text, _FONT, scale, text_thickness)
    pad = max(6, text_height // 2)
    banner_width = text_width + 2 * pad
    banner_height = text_height + baseline + 2 * pad
    original = out[:banner_height, :banner_width].copy()
    cv2.rectangle(out, (0, 0), (banner_width, banner_height), _BANNER_COLOR, -1)
    region = out[:banner_height, :banner_width]
    region[:] = cv2.addWeighted(original, 0.35, region, 0.65, 0)
    cv2.putText(
        out, text, (pad, pad + text_height), _FONT, scale, _TEXT_COLOR, text_thickness, cv2.LINE_AA
    )
