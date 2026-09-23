import json
from pathlib import Path

import pytest

from fishcount.batch import NoImagesFoundError, run_batch
from fishcount.config import AppConfig
from helpers import FakeDetector, fish, write_image


def _input_folder(tmp_path: Path) -> Path:
    folder = tmp_path / "dive1"
    write_image(folder / "a.jpg")
    write_image(folder / "c.jpg")
    write_image(folder / "sub" / "b.png")
    return folder  # discovery order: a.jpg, c.jpg, sub/b.png


def test_end_to_end_records_detections_and_frame_stats(tmp_path: Path) -> None:
    folder = _input_folder(tmp_path)
    originals = {path: path.read_bytes() for path in folder.rglob("*") if path.is_file()}
    detector = FakeDetector([[fish(), fish(conf=0.5)], [], [fish()]])
    out = tmp_path / "out"

    summary = run_batch(folder, out, AppConfig(batch_size=2), detector, show_progress=False)

    assert summary.processed == 3
    assert summary.skipped == 0
    assert summary.conf == 0.10
    assert detector.batch_sizes == [2, 1]  # batched inference, not one call per image

    payload = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert payload["threshold"] == 0.25
    entries = {entry["file"]: entry for entry in payload["images"]}
    assert entries["a.jpg"]["detections"][0]["box"] == [4, 6, 30, 24]
    assert entries["a.jpg"]["detections"][1]["confidence"] == 0.5
    assert entries["c.jpg"]["detections"] == []
    for entry in entries.values():
        # blur and brightness recorded for every readable frame, acted on by nothing
        assert isinstance(entry["blur"], float)
        assert isinstance(entry["brightness"], float)

    # input files are never modified
    assert {path: path.read_bytes() for path in folder.rglob("*") if path.is_file()} == originals


def test_every_box_above_the_floor_is_recorded(tmp_path: Path) -> None:
    """Sub-threshold boxes stay in the record: no rule drops a detection."""
    folder = tmp_path / "dive"
    write_image(folder / "a.jpg")
    weak = [fish(conf=0.11), fish(0, 0, 60, 46, conf=0.12)]  # weak, and nearly frame-filling
    out = tmp_path / "out"

    run_batch(folder, out, AppConfig(threshold=0.9), FakeDetector([weak]), show_progress=False)

    payload = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert len(payload["images"][0]["detections"]) == 2


def test_annotated_images_only_for_frames_at_or_above_threshold(tmp_path: Path) -> None:
    folder = _input_folder(tmp_path)
    # a.jpg: 0.30 -> annotated; c.jpg: none; sub/b.png: 0.20 -> below threshold, not annotated
    detector = FakeDetector([[fish(conf=0.30)], [], [fish(conf=0.20)]])
    out = tmp_path / "out"

    run_batch(folder, out, AppConfig(threshold=0.25), detector, show_progress=False)

    assert (out / "annotated" / "a.jpg").is_file()
    assert not (out / "annotated" / "c.jpg").exists()
    assert not (out / "annotated" / "sub" / "b.png").exists()


def test_corrupt_image_is_skipped_with_run_continuing(tmp_path: Path) -> None:
    folder = tmp_path / "dive2"
    write_image(folder / "good.jpg")
    (folder / "bad.jpg").write_bytes(b"this is not an image")
    out = tmp_path / "out"

    summary = run_batch(folder, out, AppConfig(), FakeDetector([[fish()]]), show_progress=False)

    assert summary.processed == 1
    assert summary.skipped == 1
    payload = json.loads((out / "results.json").read_text(encoding="utf-8"))
    entries = {entry["file"]: entry for entry in payload["images"]}
    assert entries["bad.jpg"]["error"] == "unreadable image"
    assert len(entries["good.jpg"]["detections"]) == 1


def test_no_images_flag_skips_annotated_output(tmp_path: Path) -> None:
    folder = _input_folder(tmp_path)
    out = tmp_path / "out"

    run_batch(
        folder, out, AppConfig(), FakeDetector([[fish()]]), write_images=False, show_progress=False
    )

    assert not (out / "annotated").exists()
    assert (out / "results.json").is_file()


def test_empty_folder_raises_and_writes_nothing(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()

    with pytest.raises(NoImagesFoundError):
        run_batch(empty, tmp_path / "out", AppConfig(), FakeDetector(), show_progress=False)
    assert not (tmp_path / "out").exists()
