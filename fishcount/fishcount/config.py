"""Typed runtime configuration: built-in defaults, optional config.yaml, CLI overrides."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

DEFAULT_MODEL_PATH = Path("models") / "cfd-yolov12x.pt"
CONFIG_FILENAME = "config.yaml"


class ConfigError(RuntimeError):
    """config.yaml is malformed or a setting has an invalid value."""


class AppConfig(BaseModel):
    """Detection thresholds and runtime settings.

    Precedence: CLI flags > config.yaml > these defaults.
    """

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model_path: Path = DEFAULT_MODEL_PATH
    conf: float = Field(default=0.25, ge=0.0, le=1.0)
    iou: float = Field(default=0.7, ge=0.0, le=1.0)
    imgsz: int = Field(default=1024, ge=32, le=8192)
    batch_size: int = Field(default=8, ge=1, le=256)
    # Per-image detection cap. Ultralytics silently defaults this to 300, which
    # truncates dense schools; we set it high and expose it.
    max_det: int = Field(default=1000, ge=1, le=100000)
    # SAHI tiled-inference settings (only used with --thorough). Smaller tiles
    # magnify tiny fish; more overlap recovers fish cut by tile seams.
    slice_size: int = Field(default=640, ge=64, le=2048)
    overlap_ratio: float = Field(default=0.2, ge=0.0, lt=0.9)
    # Drop any detection whose box covers more than this fraction of the frame.
    # Real fish are compact (<5% of frame here); murky/empty water gets misread
    # as one huge fish-shaped box, so a size cap removes those false positives.
    # 1.0 disables the filter (the default; counting passes leave it off).
    max_box_frac: float = Field(default=1.0, ge=0.01, le=1.0)


def load_config(path: Path | None = None) -> AppConfig:
    """Load settings from config.yaml if present (default: ./config.yaml), else defaults."""
    candidate = path if path is not None else Path(CONFIG_FILENAME)
    if not candidate.is_file():
        return AppConfig()
    try:
        raw: Any = yaml.safe_load(candidate.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Could not parse {candidate}: {exc}") from exc
    if raw is None:
        return AppConfig()
    if not isinstance(raw, dict):
        raise ConfigError(f"{candidate} must contain a mapping of settings")
    try:
        return AppConfig(**raw)
    except ValidationError as exc:
        raise ConfigError(f"Invalid settings in {candidate}:\n{exc}") from exc


def merge_overrides(config: AppConfig, **overrides: object) -> AppConfig:
    """Return a validated copy of `config` with the non-None overrides applied."""
    updates = {key: value for key, value in overrides.items() if value is not None}
    if not updates:
        return config
    try:
        return AppConfig(**{**config.model_dump(), **updates})
    except ValidationError as exc:
        raise ConfigError(f"Invalid settings: {exc}") from exc
