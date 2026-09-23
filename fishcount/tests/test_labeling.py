"""Labeling has to survive being quit halfway through, and must not lose a verdict."""

import csv
from pathlib import Path

from fishcount.labeling import LabelStore, load_tasks


def _sample(path: Path, rows: list[tuple[str, str, str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "frame_id",
                "source",
                "frame",
                "camera",
                "folder",
                "blur",
                "brightness",
                "blur_bin",
                "brightness_bin",
                "max_conf",
                "conf_band",
                "n_boxes",
                "max_area_frac",
                "large_box",
            ]
        )
        for source, frame, band in rows:
            writer.writerow(
                [
                    f"{source}/{frame}",
                    source,
                    frame,
                    "Primary",
                    f"{source}/dir",
                    "30.0",
                    "90.0",
                    "low",
                    "mid",
                    "0.400",
                    band,
                    "1",
                    "0.010000",
                    "no",
                ]
            )
    return path


def _detections(out_dir: Path, rows: list[tuple[str, float]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "detections.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "x1", "y1", "x2", "y2", "confidence", "area_frac"])
        for frame, conf in rows:
            writer.writerow([frame, 10, 20, 30, 40, f"{conf:.3f}", "0.010000"])


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_tasks_carry_the_detector_boxes_for_the_right_frame(tmp_path: Path) -> None:
    rows = [("RUN", "sub/a.jpg", "mid"), ("RUN", "b.jpg", "none")]
    sample = _sample(tmp_path / "sample.csv", rows)
    _detections(tmp_path / "run", [("sub/a.jpg", 0.4), ("sub/a.jpg", 0.9)])

    tasks = load_tasks(sample, {"RUN": tmp_path / "images"}, {"RUN": tmp_path / "run"})

    assert [task.frame_id for task in tasks] == ["RUN/sub/a.jpg", "RUN/b.jpg"]
    assert len(tasks[0].boxes) == 2
    assert tasks[0].boxes[0]["confidence"] == 0.4
    assert tasks[1].boxes == []  # a zero-detection frame is still a task
    assert tasks[0].path == tmp_path / "images" / "sub/a.jpg"


def test_saving_writes_both_files_after_every_frame(tmp_path: Path) -> None:
    store = LabelStore(tmp_path)

    store.save_frame(
        "RUN/a.jpg",
        "RUN",
        "a.jpg",
        [
            {
                "box_index": 0,
                "origin": "model",
                "label": "fish",
                "x1": 1,
                "y1": 2,
                "x2": 3,
                "y2": 4,
                "confidence": 0.8,
            },
            {
                "box_index": 1,
                "origin": "model",
                "label": "not_fish",
                "x1": 5,
                "y1": 6,
                "x2": 7,
                "y2": 8,
                "confidence": 0.3,
            },
            {
                "box_index": 2,
                "origin": "missed",
                "label": "fish",
                "x1": 9,
                "y1": 9,
                "x2": 20,
                "y2": 20,
                "confidence": "",
            },
        ],
    )

    labels = _read(tmp_path / "labels.csv")
    assert [row["label"] for row in labels] == ["fish", "not_fish", "fish"]
    assert [row["origin"] for row in labels] == ["model", "model", "missed"]
    assert labels[2]["confidence"] == ""  # the model never saw a hand-drawn box
    assert all(row["frame_id"] == "RUN/a.jpg" for row in labels)

    frames = _read(tmp_path / "labeled_frames.csv")
    assert frames[0]["n_model_boxes"] == "2"
    assert frames[0]["n_fish"] == "1"
    assert frames[0]["n_not_fish"] == "1"
    assert frames[0]["n_missed"] == "1"
    assert frames[0]["has_fish"] == "yes"


def test_a_reviewed_empty_frame_is_recorded_as_a_true_negative(tmp_path: Path) -> None:
    """Otherwise "no fish here" is indistinguishable from "not looked at yet"."""
    store = LabelStore(tmp_path)

    store.save_frame("RUN/empty.jpg", "RUN", "empty.jpg", [])

    assert _read(tmp_path / "labels.csv") == []  # no boxes, so no box rows
    frames = _read(tmp_path / "labeled_frames.csv")
    assert frames[0]["frame_id"] == "RUN/empty.jpg"
    assert frames[0]["has_fish"] == "no"
    assert frames[0]["n_model_boxes"] == "0"


def test_a_frame_the_model_missed_entirely_still_counts_as_holding_fish(
    tmp_path: Path,
) -> None:
    store = LabelStore(tmp_path)
    missed = {
        "box_index": 0,
        "origin": "missed",
        "label": "fish",
        "x1": 1,
        "y1": 1,
        "x2": 9,
        "y2": 9,
        "confidence": "",
    }

    store.save_frame("RUN/miss.jpg", "RUN", "miss.jpg", [missed])

    frames = _read(tmp_path / "labeled_frames.csv")
    assert frames[0]["n_model_boxes"] == "0"
    assert frames[0]["n_missed"] == "1"
    assert frames[0]["has_fish"] == "yes"  # a false negative, and recorded as one


def test_quitting_and_reopening_resumes_with_the_verdicts_intact(tmp_path: Path) -> None:
    box = {
        "box_index": 0,
        "origin": "model",
        "label": "fish",
        "x1": 1,
        "y1": 2,
        "x2": 3,
        "y2": 4,
        "confidence": 0.8,
    }
    first = LabelStore(tmp_path)
    first.save_frame("RUN/a.jpg", "RUN", "a.jpg", [box])
    first.save_frame("RUN/b.jpg", "RUN", "b.jpg", [])

    reopened = LabelStore(tmp_path)

    assert set(reopened.frames) == {"RUN/a.jpg", "RUN/b.jpg"}
    assert reopened.frames["RUN/a.jpg"]["boxes"][0]["label"] == "fish"
    assert reopened.frames["RUN/a.jpg"]["boxes"][0]["x2"] == 3
    assert reopened.frames["RUN/b.jpg"]["boxes"] == []


def test_relabeling_a_frame_replaces_its_rows_rather_than_adding_more(
    tmp_path: Path,
) -> None:
    """Going back and changing your mind must leave one verdict, not two."""
    store = LabelStore(tmp_path)
    box = {
        "box_index": 0,
        "origin": "model",
        "label": "fish",
        "x1": 1,
        "y1": 2,
        "x2": 3,
        "y2": 4,
        "confidence": 0.8,
    }
    store.save_frame("RUN/a.jpg", "RUN", "a.jpg", [box])

    store.save_frame("RUN/a.jpg", "RUN", "a.jpg", [{**box, "label": "not_fish"}])

    rows = _read(tmp_path / "labels.csv")
    assert len(rows) == 1
    assert rows[0]["label"] == "not_fish"
    assert len(_read(tmp_path / "labeled_frames.csv")) == 1


def test_rows_are_sorted_by_frame_id_whatever_order_they_were_labeled(
    tmp_path: Path,
) -> None:
    store = LabelStore(tmp_path)
    for name in ("c.jpg", "a.jpg", "b.jpg"):
        store.save_frame(f"RUN/{name}", "RUN", name, [])

    frames = _read(tmp_path / "labeled_frames.csv")
    assert [row["frame"] for row in frames] == ["a.jpg", "b.jpg", "c.jpg"]
