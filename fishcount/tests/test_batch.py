import json
from pathlib import Path

import pandas as pd
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


def test_end_to_end_outputs(tmp_path: Path) -> None:
    folder = _input_folder(tmp_path)
    originals = {path: path.read_bytes() for path in folder.rglob("*") if path.is_file()}
    detector = FakeDetector([[fish(), fish(conf=0.5)], [], [fish()]])
    out = tmp_path / "out"

    summary = run_batch(folder, out, AppConfig(batch_size=2), detector, show_progress=False)

    assert summary.processed == 3
    assert summary.skipped == 0
    assert summary.total_fish == 3
    assert detector.batch_sizes == [2, 1]  # batched inference, not one call per image

    for relative in ("a.jpg", "c.jpg", "sub/b.png"):
        assert (out / "annotated" / relative).is_file()

    table = pd.read_csv(out / "counts.csv")
    assert list(table.columns) == ["filename", "fish_count"]
    assert dict(zip(table["filename"], table["fish_count"], strict=True)) == {
        "a.jpg": 2,
        "c.jpg": 0,
        "sub/b.png": 1,
        "TOTAL": 3,
    }
    assert table.iloc[-1]["filename"] == "TOTAL"

    payload = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert payload["total_fish"] == 3
    first = payload["images"][0]
    assert first["file"] == "a.jpg"
    assert first["fish_count"] == 2
    assert first["detections"][0]["box"] == [4, 6, 30, 24]
    assert first["detections"][1]["confidence"] == 0.5

    # input files are never modified
    assert {path: path.read_bytes() for path in folder.rglob("*") if path.is_file()} == originals


def test_corrupt_image_is_skipped_with_run_continuing(tmp_path: Path) -> None:
    folder = tmp_path / "dive2"
    write_image(folder / "good.jpg")
    (folder / "bad.jpg").write_bytes(b"this is not an image")
    out = tmp_path / "out"

    summary = run_batch(folder, out, AppConfig(), FakeDetector([[fish()]]), show_progress=False)

    assert summary.processed == 1
    assert summary.skipped == 1
    assert summary.total_fish == 1

    table = pd.read_csv(out / "counts.csv")
    assert list(table["filename"]) == ["good.jpg", "TOTAL"]  # bad.jpg not counted

    payload = json.loads((out / "results.json").read_text(encoding="utf-8"))
    entries = {entry["file"]: entry for entry in payload["images"]}
    assert entries["bad.jpg"]["error"] == "unreadable image"
    assert entries["good.jpg"]["fish_count"] == 1


def test_max_box_frac_drops_oversized_boxes(tmp_path: Path) -> None:
    folder = tmp_path / "gate"
    write_image(folder / "a.jpg", width=100, height=100)  # frame area 10000
    # one compact fish (100 px^2 = 1% of frame) and one huge box (3600 = 36%).
    small = fish(0, 0, 10, 10)
    huge = fish(10, 10, 70, 70)
    detector = FakeDetector([[small, huge]])
    out = tmp_path / "out"

    summary = run_batch(folder, out, AppConfig(max_box_frac=0.25), detector, show_progress=False)

    assert summary.total_fish == 1  # the 36% box is dropped, the 1% box stays
    payload = json.loads((out / "results.json").read_text(encoding="utf-8"))
    boxes = payload["images"][0]["detections"]
    assert boxes == [{"box": [0, 0, 10, 10], "confidence": 0.9}]


def test_no_images_flag_skips_annotated_output(tmp_path: Path) -> None:
    folder = _input_folder(tmp_path)
    out = tmp_path / "out"

    run_batch(folder, out, AppConfig(), FakeDetector(), write_images=False, show_progress=False)

    assert not (out / "annotated").exists()
    assert (out / "counts.csv").is_file()
    assert (out / "results.json").is_file()


def test_empty_folder_raises_and_writes_nothing(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()

    with pytest.raises(NoImagesFoundError):
        run_batch(empty, tmp_path / "out", AppConfig(), FakeDetector(), show_progress=False)
    assert not (tmp_path / "out").exists()
