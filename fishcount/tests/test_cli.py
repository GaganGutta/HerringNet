import csv
from pathlib import Path

import pytest

from fishcount.cli import main
from helpers import FakeDetector, fish, write_image


def test_detect_end_to_end_writes_both_csvs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    folder = tmp_path / "photos"
    write_image(folder / "a.jpg")
    write_image(folder / "b.jpg")
    write_image(folder / "c.jpg")
    captured: dict[str, object] = {}
    # a: above threshold; b: recorded but below it; c: nothing
    fake = FakeDetector([[fish(4, 6, 12, 12, conf=0.9)], [fish(4, 6, 12, 12, conf=0.15)], []])
    monkeypatch.setattr(
        "fishcount.detector.resolve_model_path", lambda configured: tmp_path / "fake.pt"
    )

    def spy(weights: object, config: object) -> FakeDetector:
        captured["config"] = config
        return fake

    monkeypatch.setattr("fishcount.detector.create_detector", spy)
    out = tmp_path / "out"

    code = main(["detect", str(folder), "--out", str(out)])

    assert code == 0
    config = captured["config"]
    assert config.conf == 0.10  # recall-first recording floor
    assert config.threshold == 0.25
    assert config.imgsz == 1536
    assert config.max_det == 3000

    with (out / "detections.csv").open(encoding="utf-8", newline="") as handle:
        detections = list(csv.DictReader(handle))
    assert [row["frame"] for row in detections] == ["a.jpg", "b.jpg"]  # both kept

    with (out / "frames.csv").open(encoding="utf-8", newline="") as handle:
        frames = list(csv.DictReader(handle))
    assert [row["frame"] for row in frames] == ["a.jpg", "b.jpg", "c.jpg"]  # every frame
    assert [row["n_boxes_above_threshold"] for row in frames] == ["1", "0", "0"]

    # annotated images only for frames at or above the threshold
    assert (out / "annotated" / "a.jpg").is_file()
    assert not (out / "annotated" / "b.jpg").exists()


def test_threshold_flag_overrides_the_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    folder = tmp_path / "photos"
    write_image(folder / "a.jpg")
    monkeypatch.setattr(
        "fishcount.detector.resolve_model_path", lambda configured: tmp_path / "fake.pt"
    )
    fake = FakeDetector([[fish(4, 6, 12, 12, conf=0.30)]])
    monkeypatch.setattr("fishcount.detector.create_detector", lambda weights, config: fake)
    out = tmp_path / "out"

    assert main(["detect", str(folder), "--out", str(out), "--threshold", "0.5"]) == 0

    with (out / "frames.csv").open(encoding="utf-8", newline="") as handle:
        frames = list(csv.DictReader(handle))
    assert frames[0]["n_boxes_above_threshold"] == "0"  # 0.30 is below 0.5
    assert frames[0]["max_conf"] == "0.300"  # but the box is still on the record
    assert not (out / "annotated").exists()


def test_missing_model_is_a_clear_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = tmp_path / "photos"
    write_image(folder / "a.jpg")
    missing = tmp_path / "nowhere" / "cfd-yolov12x.pt"
    (tmp_path / "config.yaml").write_text(f"model_path: {missing.as_posix()}\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    code = main(["detect", str(folder)])

    assert code == 1
    err = capsys.readouterr().err
    assert "cfd-yolov12x.pt" in err
    assert "never downloads" in err


def test_nonexistent_folder_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["detect", str(tmp_path / "missing")])
    assert code == 1
    assert "Not a folder" in capsys.readouterr().err
