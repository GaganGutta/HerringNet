"""detections.csv and frames.csv are derived views: complete, and decision-free."""

import csv
import json
from pathlib import Path

from fishcount.report import write_reports


def _results(out_dir: Path, images: list[dict]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "input": "x",
        "model": "m",
        "conf": 0.1,
        "threshold": 0.25,
        "imgsz": 1536,
        "images": images,
    }
    (out_dir / "results.json").write_text(json.dumps(payload), encoding="utf-8")
    return out_dir


def _frame(file: str, detections: list[dict], **kwargs: object) -> dict:
    entry = {
        "file": file,
        "width": 1000,
        "height": 800,
        "blur": 42.0,
        "brightness": 100.0,
        "detections": detections,
    }
    entry.update(kwargs)
    return entry


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_every_recorded_box_becomes_a_row_with_its_area_fraction(tmp_path: Path) -> None:
    out = _results(
        tmp_path / "out",
        [
            _frame(
                "a.jpg",
                [
                    {"box": [0, 0, 100, 80], "confidence": 0.9},  # 8000 / 800000 = 0.01
                    {"box": [10, 10, 20, 20], "confidence": 0.12},  # below threshold, still a row
                ],
            )
        ],
    )

    summary = write_reports(out, threshold=0.25)

    rows = _read(out / "detections.csv")
    assert [row["confidence"] for row in rows] == ["0.900", "0.120"]
    assert rows[0]["area_frac"] == "0.010000"
    assert summary.detections == 2
    assert summary.detections_above_threshold == 1


def test_frames_csv_is_a_complete_census_including_empty_frames(tmp_path: Path) -> None:
    out = _results(
        tmp_path / "out",
        [
            _frame("b.jpg", [{"box": [0, 0, 10, 10], "confidence": 0.8}]),
            _frame("a.jpg", []),  # no detections: still gets a row
        ],
    )

    summary = write_reports(out, threshold=0.25)

    rows = _read(out / "frames.csv")
    assert [row["frame"] for row in rows] == ["a.jpg", "b.jpg"]  # sorted by frame
    assert rows[0]["max_conf"] == "0.000"
    assert rows[0]["n_boxes_above_threshold"] == "0"
    assert rows[0]["blur"] == "42.0"
    assert rows[0]["brightness"] == "100.0"
    assert rows[1]["n_boxes_above_threshold"] == "1"
    assert summary.frames == 2
    assert summary.frames_with_detections == 1
    assert summary.frames_above_threshold == 1
    # a frame with no boxes contributes no detection rows
    assert [row["frame"] for row in _read(out / "detections.csv")] == ["b.jpg"]


def test_threshold_only_changes_the_counts_never_the_rows(tmp_path: Path) -> None:
    detections = [
        {"box": [0, 0, 10, 10], "confidence": 0.20},
        {"box": [0, 0, 10, 10], "confidence": 0.40},
    ]
    out = _results(tmp_path / "out", [_frame("a.jpg", detections)])

    write_reports(out, threshold=0.25)
    at_25 = _read(out / "detections.csv")
    frames_25 = _read(out / "frames.csv")
    write_reports(out, threshold=0.50)
    at_50 = _read(out / "detections.csv")
    frames_50 = _read(out / "frames.csv")

    assert at_25 == at_50  # the box rows are the raw record, threshold-independent
    assert frames_25[0]["n_boxes_above_threshold"] == "1"
    assert frames_50[0]["n_boxes_above_threshold"] == "0"
    assert frames_25[0]["max_conf"] == frames_50[0]["max_conf"] == "0.400"


def test_unreadable_frame_is_marked_not_silently_a_zero_detection_frame(tmp_path: Path) -> None:
    out = _results(
        tmp_path / "out",
        [{"file": "bad.jpg", "error": "unreadable image"}, _frame("good.jpg", [])],
    )

    summary = write_reports(out, threshold=0.25)

    rows = {row["frame"]: row for row in _read(out / "frames.csv")}
    assert rows["bad.jpg"]["error"] == "unreadable image"
    assert rows["bad.jpg"]["max_conf"] == ""  # blank, not 0.000
    assert rows["bad.jpg"]["blur"] == ""
    assert rows["good.jpg"]["error"] == ""
    assert rows["good.jpg"]["max_conf"] == "0.000"
    assert summary.unreadable == 1
    assert summary.frames == 2
