import csv
from pathlib import Path

import pytest

from fishcount.cli import main
from helpers import FakeDetector, fish, write_image


def test_detect_end_to_end_tiers_frames(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    folder = tmp_path / "photos"
    write_image(folder / "a.jpg")
    write_image(folder / "b.jpg")
    write_image(folder / "c.jpg")
    captured: dict[str, object] = {}
    # a: strong detection -> confident; b: one moderate -> under_review; c: none
    fake = FakeDetector([[fish(4, 6, 12, 12, conf=0.9)], [fish(4, 6, 12, 12, conf=0.3)], []])
    monkeypatch.setattr(
        "fishcount.detector.resolve_model_path", lambda configured: tmp_path / "fake.pt"
    )

    def spy(weights: object, config: object) -> FakeDetector:
        captured["config"] = config
        return fake

    monkeypatch.setattr("fishcount.detector.create_detector", spy)
    out = tmp_path / "out"

    code = main(["detect", str(folder), "--out", str(out), "--blur-percentile", "0"])

    assert code == 0
    config = captured["config"]
    assert config.conf == 0.10  # recall-first floor is the default
    assert config.imgsz == 1536
    assert config.max_det == 3000
    for tier in ("confident", "under_review", "not_confident"):
        assert (out / f"{tier}.csv").is_file()
    with (out / "confident.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))[1:]
    assert [r[0] for r in rows] == ["a.jpg"]
    assert (out / "confident" / "a.jpg").is_file()
    assert (out / "under_review" / "b.jpg").is_file()
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
