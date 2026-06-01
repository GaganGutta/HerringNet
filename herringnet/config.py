"""Configuration loading and validation for HerringNet.

Supports layered YAML configuration with site-specific overrides.
Loading order: default.yaml -> sites/<site>.yaml -> CLI overrides.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

# Default path to the configs directory, relative to the project root.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIGS_DIR = _PROJECT_ROOT / "configs"


class DetectorConfig(BaseModel):
    """Configuration for the fish detection model."""

    model_name: str = "cfd"
    confidence_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    iou_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    image_size: int = Field(default=1024, gt=0)
    device: str = "cpu"
    # SAHI sliced inference for small fish detection.
    # Set use_sahi to true to tile images into overlapping slices.
    use_sahi: bool = False
    sahi_slice_size: int = Field(default=512, gt=0)
    sahi_overlap_ratio: float = Field(default=0.3, ge=0.0, le=0.5)


class ClassifierConfig(BaseModel):
    """Configuration for the species classification model."""

    model_path: str | None = None
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    image_size: int = Field(default=224, gt=0)


class FrameExtractionConfig(BaseModel):
    """Configuration for video frame extraction.

    Based on Marjadi et al. (2024) frame-residence-rate methodology.
    """

    frame_interval: int = Field(default=5, gt=0)
    mean_frr: float = Field(default=4.55, gt=0.0)


class ActiveLearningConfig(BaseModel):
    """Configuration for the active learning review system."""

    enabled: bool = True
    uncertainty_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    margin_threshold: float = Field(default=0.15, ge=0.0, le=1.0)
    max_review_queue: int = Field(default=500, gt=0)
    review_output_dir: str = "data/review_queue"


class OutputConfig(BaseModel):
    """Configuration for output behavior."""

    output_dir: str = "outputs"
    save_json: bool = True
    save_images: bool = False
    save_crops: bool = False
    save_csv: bool = False


class HerringNetConfig(BaseModel):
    """Top-level configuration for HerringNet."""

    detector: DetectorConfig = DetectorConfig()
    classifier: ClassifierConfig = ClassifierConfig()
    frame_extraction: FrameExtractionConfig = FrameExtractionConfig()
    active_learning: ActiveLearningConfig = ActiveLearningConfig()
    output: OutputConfig = OutputConfig()
    site_name: str = "default"


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge override dict into base dict. Override values take precedence.

    Args:
        base: Base configuration dictionary.
        override: Override values to merge on top.

    Returns:
        Merged dictionary. The base dict is not modified.
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(
    config_path: str | Path | None = None,
    site: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> HerringNetConfig:
    """Load HerringNet configuration from YAML files.

    Args:
        config_path: Path to the base config YAML. Defaults to configs/default.yaml.
        site: Site name for loading site-specific overrides from configs/sites/.
        overrides: Additional overrides to apply on top (from CLI args).

    Returns:
        Validated HerringNetConfig instance.
    """
    if config_path is None:
        config_path = _CONFIGS_DIR / "default.yaml"
    config_path = Path(config_path)

    # Load base config
    with open(config_path) as f:
        config_data = yaml.safe_load(f) or {}

    # Apply site-specific overrides
    if site is not None:
        site_path = _CONFIGS_DIR / "sites" / f"{site}.yaml"
        if site_path.exists():
            with open(site_path) as f:
                site_data = yaml.safe_load(f) or {}
            config_data = deep_merge(config_data, site_data)

    # Apply CLI overrides
    if overrides:
        config_data = deep_merge(config_data, overrides)

    return HerringNetConfig(**config_data)


def load_species_config(
    species_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load species class definitions from YAML.

    Args:
        species_path: Path to species.yaml. Defaults to configs/species.yaml.

    Returns:
        Dictionary with species class definitions.
    """
    if species_path is None:
        species_path = _CONFIGS_DIR / "species.yaml"
    species_path = Path(species_path)

    with open(species_path) as f:
        return yaml.safe_load(f) or {}
