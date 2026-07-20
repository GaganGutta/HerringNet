import json
from pathlib import Path

import pytest

from fishcount.cli import main
from helpers import FakeDetector, fish, write_image


def test_detect_end_to_end_with_flag_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    folder = tmp_path / "photos"
    write_image(folder / "a.jpg")
    write_image(folder / "b.jpg")
    fake = FakeDetector([[fish()], [fish(), fish()]])
    monkeypatch.setattr(
        "fishcount.detector.resolve_model_path", lambda configured: tmp_path / "fake.pt"
    )
    monkeypatch.setattr(
        "fishcount.detector.create_detector",
        lambda weights, config, *, thorough=False: fake,
    )
    out = tmp_path / "out"

    code = main(["detect", str(folder), "--out", str(out), "--conf", "0.4", "--batch-size", "1"])

    assert code == 0
    assert (out / "counts.csv").is_file()
    payload = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert payload["total_fish"] == 3
    assert payload["conf"] == 0.4  # --conf reached the config
    assert fake.batch_sizes == [1, 1]  # --batch-size reached the batching


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
    assert "models/" in err
    assert "never downloads" in err


def test_nonexistent_folder_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["detect", str(tmp_path / "missing")])
    assert code == 1
    assert "Not a folder" in capsys.readouterr().err
