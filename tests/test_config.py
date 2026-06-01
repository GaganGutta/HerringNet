"""Tests for the configuration loading and validation system."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

from herringnet.config import (
    DetectorConfig,
    HerringNetConfig,
    deep_merge,
    load_config,
    load_species_config,
)


class TestDeepMerge:
    """Tests for the deep_merge utility."""

    def test_simple_override(self):
        base = {"a": 1, "b": 2}
        override = {"b": 3}
        result = deep_merge(base, override)
        assert result == {"a": 1, "b": 3}

    def test_nested_override(self):
        base = {"detector": {"confidence": 0.25, "iou": 0.45}}
        override = {"detector": {"confidence": 0.5}}
        result = deep_merge(base, override)
        assert result["detector"]["confidence"] == 0.5
        assert result["detector"]["iou"] == 0.45

    def test_add_new_key(self):
        base = {"a": 1}
        override = {"b": 2}
        result = deep_merge(base, override)
        assert result == {"a": 1, "b": 2}

    def test_does_not_modify_base(self):
        base = {"a": {"b": 1}}
        override = {"a": {"b": 2}}
        deep_merge(base, override)
        assert base["a"]["b"] == 1


class TestDetectorConfig:
    """Tests for DetectorConfig validation."""

    def test_defaults(self):
        config = DetectorConfig()
        assert config.model_name == "cfd"
        assert config.confidence_threshold == 0.25
        assert config.device == "cpu"

    def test_invalid_confidence(self):
        with pytest.raises(Exception):
            DetectorConfig(confidence_threshold=1.5)

    def test_invalid_image_size(self):
        with pytest.raises(Exception):
            DetectorConfig(image_size=-1)


class TestHerringNetConfig:
    """Tests for the full configuration."""

    def test_defaults(self):
        config = HerringNetConfig()
        assert config.site_name == "default"
        assert config.detector.model_name == "cfd"
        assert config.frame_extraction.mean_frr == 4.55

    def test_from_dict(self):
        data = {
            "site_name": "test_site",
            "detector": {"confidence_threshold": 0.3},
        }
        config = HerringNetConfig(**data)
        assert config.site_name == "test_site"
        assert config.detector.confidence_threshold == 0.3


class TestLoadConfig:
    """Tests for loading config from YAML files."""

    def test_load_default_config(self):
        config = load_config()
        assert isinstance(config, HerringNetConfig)
        assert config.detector.model_name == "cfd"

    def test_load_with_site_override(self):
        config = load_config(site="monument_river")
        assert config.site_name == "monument_river"
        assert config.detector.confidence_threshold == 0.20

    def test_load_with_cli_overrides(self):
        config = load_config(
            overrides={"detector": {"confidence_threshold": 0.9}}
        )
        assert config.detector.confidence_threshold == 0.9

    def test_load_custom_yaml(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump({
                "site_name": "custom",
                "detector": {"confidence_threshold": 0.1},
            }, f)
            f.flush()

            config = load_config(config_path=f.name)
            assert config.site_name == "custom"
            assert config.detector.confidence_threshold == 0.1


class TestLoadSpeciesConfig:
    """Tests for loading species definitions."""

    def test_load_species(self):
        species = load_species_config()
        assert "classes" in species
        assert 0 in species["classes"]
        assert species["classes"][0]["name"] == "river_herring"
