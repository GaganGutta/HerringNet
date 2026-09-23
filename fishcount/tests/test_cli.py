import csv
from pathlib import Path

import pytest

from fishcount.cli import main
from helpers import FakeDetector, fish, write_image


def _fake_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, detector: FakeDetector) -> None:
    monkeypatch.setattr(
        "fishcount.detector.resolve_model_path", lambda configured: tmp_path / "fake.pt"
    )
    monkeypatch.setattr("fishcount.detector.create_detector", lambda weights, config: detector)


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_detect_end_to_end_writes_both_csvs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    folder = tmp_path / "photos"
    for name in ("a.jpg", "b.jpg", "c.jpg"):
        write_image(folder / name)
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
    assert config.device == "auto"

    detections = _read(out / "detections.csv")
    assert [row["frame"] for row in detections] == ["a.jpg", "b.jpg"]  # both kept

    frames = _read(out / "frames.csv")
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
    _fake_model(tmp_path, monkeypatch, FakeDetector([[fish(4, 6, 12, 12, conf=0.30)]]))
    out = tmp_path / "out"

    assert main(["detect", str(folder), "--out", str(out), "--threshold", "0.5"]) == 0

    frames = _read(out / "frames.csv")
    assert frames[0]["n_boxes_above_threshold"] == "0"  # 0.30 is below 0.5
    assert frames[0]["max_conf"] == "0.300"  # but the box is still on the record
    assert not (out / "annotated").exists()


def test_report_reruns_a_threshold_without_touching_the_detector(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The point of splitting report out: trying a threshold costs no inference."""
    monkeypatch.chdir(tmp_path)
    folder = tmp_path / "photos"
    write_image(folder / "a.jpg")
    detector = FakeDetector([[fish(4, 6, 12, 12, conf=0.30)]])
    _fake_model(tmp_path, monkeypatch, detector)
    out = tmp_path / "out"
    assert main(["detect", str(folder), "--out", str(out)]) == 0
    assert _read(out / "frames.csv")[0]["n_boxes_above_threshold"] == "1"
    calls = len(detector.batch_sizes)

    code = main(["report", str(out), "--threshold", "0.6", "--folder", str(folder)])

    assert code == 0
    assert len(detector.batch_sizes) == calls  # the model was never invoked again
    assert _read(out / "frames.csv")[0]["n_boxes_above_threshold"] == "0"
    assert not (out / "annotated" / "a.jpg").exists()  # stale image cleared


def test_report_on_a_folder_with_no_run_is_a_clear_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    empty = tmp_path / "nothing"
    empty.mkdir()

    code = main(["report", str(empty)])

    assert code == 1
    assert "results.jsonl" in capsys.readouterr().err


def test_rerunning_detect_resumes_instead_of_redoing_the_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    folder = tmp_path / "photos"
    for name in ("a.jpg", "b.jpg"):
        write_image(folder / name)
    _fake_model(tmp_path, monkeypatch, FakeDetector([[fish()], []]))
    out = tmp_path / "out"
    assert main(["detect", str(folder), "--out", str(out)]) == 0

    second = FakeDetector([[fish()], []])
    _fake_model(tmp_path, monkeypatch, second)
    assert main(["detect", str(folder), "--out", str(out)]) == 0

    assert second.batch_sizes == []  # nothing left to do, so nothing was run
    assert len(_read(out / "frames.csv")) == 2  # and no duplicate rows


def test_rerunning_with_different_detection_settings_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    folder = tmp_path / "photos"
    write_image(folder / "a.jpg")
    _fake_model(tmp_path, monkeypatch, FakeDetector([[fish()]]))
    out = tmp_path / "out"
    assert main(["detect", str(folder), "--out", str(out)]) == 0

    _fake_model(tmp_path, monkeypatch, FakeDetector([[fish()]]))
    code = main(["detect", str(folder), "--out", str(out), "--conf", "0.4"])

    assert code == 1
    err = capsys.readouterr().err
    assert "conf" in err and "--restart" in err


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
