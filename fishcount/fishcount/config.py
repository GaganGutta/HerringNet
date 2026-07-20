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
    iou: float = Field(default=0.5, ge=0.0, le=1.0)
    imgsz: int = Field(default=1024, ge=32, le=8192)
    batch_size: int = Field(default=8, ge=1, le=256)


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
