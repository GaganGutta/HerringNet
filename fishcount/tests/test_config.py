from pathlib import Path

import pytest

from fishcount.config import AppConfig, ConfigError, load_config, merge_overrides


def test_defaults_when_no_config_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    config = load_config()
    assert config.conf == 0.10
    assert config.threshold == 0.25
    assert config.iou == 0.7
    assert config.imgsz == 1536
    assert config.batch_size is None  # unset: chosen from the device at run time
    assert config.max_det == 3000
    assert config.device == "auto"
    assert config.model_path == Path("models") / "cfd-yolov12x.pt"


def test_yaml_overrides_defaults(tmp_path: Path) -> None:
    file = tmp_path / "config.yaml"
    file.write_text("conf: 0.5\nimgsz: 640\n", encoding="utf-8")
    config = load_config(file)
    assert config.conf == 0.5
    assert config.imgsz == 640
    assert config.batch_size is None  # untouched settings keep their defaults


def test_invalid_value_rejected(tmp_path: Path) -> None:
    file = tmp_path / "config.yaml"
    file.write_text("conf: 1.5\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(file)


def test_unknown_key_rejected(tmp_path: Path) -> None:
    file = tmp_path / "config.yaml"
    file.write_text("connf: 0.5\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(file)


def test_merge_overrides_validates_and_skips_none() -> None:
    config = AppConfig()
    assert merge_overrides(config, conf=None) is config
    assert merge_overrides(config, conf=0.7).conf == 0.7
    assert merge_overrides(config, threshold=0.4).threshold == 0.4
    assert merge_overrides(config, device="cpu").device == "cpu"
    with pytest.raises(ConfigError):
        merge_overrides(config, conf=3.0)
    with pytest.raises(ConfigError):
        merge_overrides(config, device="gpu")  # not a device torch understands
