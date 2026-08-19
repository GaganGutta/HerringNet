"""Cross-frame logic for a fixed camera shooting an image sequence.

The camera does not move and fish do. A detection whose box recurs at
(nearly) the same pixels across many separate frames is a stationary object
in the scene: a rock, a shell, debris on the seabed. It gets re-detected in
frame after frame at 0.5-0.7 confidence and can flag hundreds of frames.

`static_detections` finds those. A box must overlap (IoU >= iou) a box in at
least `min_frames` DISTINCT frames before it is called static, so a fish that
holds still for a few frames is untouched. The IoU default is loose (0.3)
because a small rock's box jitters by tens of pixels between frames; on the
first 999-frame set, 0.3 caught the jittering pebbles that 0.5 missed while
every additionally demoted box inspected by hand was still a rock, and the
moving fish stayed flagged.
The pipeline demotes static-only frames to their own tier rather than
deleting anything, so the decision stays auditable.

This module is also the natural home for the Phase 2 sequence-aware counting
(tracking / frame-residence correction) when that lands.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

# Detections at (frame_index, detection_index) that are stationary background.
StaticMask = set[tuple[int, int]]


def static_detections(results_json: Path, *, min_frames: int = 8, iou: float = 0.3) -> StaticMask:
    """Return the (frame_index, det_index) pairs that are stationary objects.

    A detection is static if boxes overlapping it at >= `iou` occur in at
    least `min_frames` distinct frames (counting its own frame). Frame order
    and gaps do not matter: a rock is a rock whether it is re-detected in
    consecutive frames or sporadically.
    """
    payload = json.loads(results_json.read_text(encoding="utf-8"))
    boxes: list[tuple[int, int, np.ndarray]] = []
    for frame_index, image in enumerate(payload["images"]):
        for det_index, det in enumerate(image.get("detections", [])):
            boxes.append((frame_index, det_index, np.asarray(det["box"], dtype=np.float64)))
    if len(boxes) < min_frames:
        return set()

    coords = np.stack([box for _, _, box in boxes])  # (N, 4) x1 y1 x2 y2
    frames = np.asarray([frame_index for frame_index, _, _ in boxes])
    overlap = _pairwise_iou(coords) >= iou  # (N, N) bool, diagonal True

    static: StaticMask = set()
    for i, (frame_index, det_index, _) in enumerate(boxes):
        distinct_frames = len(set(frames[overlap[i]].tolist()))
        if distinct_frames >= min_frames:
            static.add((frame_index, det_index))
    return static


def static_frame_counts(mask: StaticMask, n_frames: int) -> list[int]:
    """How many static detections each frame has, indexed by frame order."""
    counts = [0] * n_frames
    per_frame: dict[int, int] = defaultdict(int)
    for frame_index, _ in mask:
        per_frame[frame_index] += 1
    for frame_index, count in per_frame.items():
        counts[frame_index] = count
    return counts


def _pairwise_iou(boxes: np.ndarray) -> np.ndarray:
    """IoU matrix for boxes in x1, y1, x2, y2 pixel coordinates."""
    x1 = np.maximum(boxes[:, None, 0], boxes[None, :, 0])
    y1 = np.maximum(boxes[:, None, 1], boxes[None, :, 1])
    x2 = np.minimum(boxes[:, None, 2], boxes[None, :, 2])
    y2 = np.minimum(boxes[:, None, 3], boxes[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area[:, None] + area[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        result: np.ndarray = np.where(union > 0, inter / union, 0.0)
    return result
