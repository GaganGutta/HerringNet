import json
from pathlib import Path

from fishcount.sequence import static_detections


def _results(path: Path, images: list[dict]) -> Path:
    payload = {"input": "x", "model": "m", "conf": 0.1, "imgsz": 1536, "images": images}
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_recurring_box_is_static_and_moving_box_is_not(tmp_path: Path) -> None:
    rock = {"box": [50, 50, 120, 100], "confidence": 0.6}
    images = []
    for i in range(9):
        dets = [dict(rock)]
        if i in (3, 7):  # a fish appears at two different places
            dets.append({"box": [300 + 200 * i, 400, 380 + 200 * i, 450], "confidence": 0.9})
        images.append({"file": f"f{i}.jpg", "width": 1000, "height": 800, "detections": dets})
    path = _results(tmp_path / "results.json", images)

    static = static_detections(path, min_frames=8, iou=0.3)

    assert len(static) == 9  # the rock in every frame
    assert all(det_index == 0 for _, det_index in static)  # never the moving fish


def test_below_min_frames_is_not_static(tmp_path: Path) -> None:
    rock = {"box": [50, 50, 120, 100], "confidence": 0.6}
    images = [
        {"file": f"f{i}.jpg", "width": 1000, "height": 800, "detections": [dict(rock)]}
        for i in range(5)
    ]
    path = _results(tmp_path / "results.json", images)
    assert static_detections(path, min_frames=8, iou=0.3) == set()
