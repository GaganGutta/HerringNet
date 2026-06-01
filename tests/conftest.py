"""Shared test fixtures for HerringNet tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from herringnet.config import (
    ActiveLearningConfig,
    ClassifierConfig,
    DetectorConfig,
    FrameExtractionConfig,
    HerringNetConfig,
    OutputConfig,
)
from herringnet.inference.result_types import (
    Classification,
    Detection,
    FrameResult,
    PipelineDetection,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_config() -> HerringNetConfig:
    """Create a test configuration."""
    return HerringNetConfig(
        detector=DetectorConfig(
            model_name="yolov8n",
            confidence_threshold=0.25,
            image_size=640,
            device="cpu",
        ),
        classifier=ClassifierConfig(
            model_path=None,
            confidence_threshold=0.5,
        ),
        frame_extraction=FrameExtractionConfig(
            frame_interval=5,
            mean_frr=4.55,
        ),
        active_learning=ActiveLearningConfig(
            enabled=True,
            uncertainty_threshold=0.6,
            margin_threshold=0.15,
        ),
        output=OutputConfig(
            output_dir="test_outputs",
        ),
        site_name="test",
    )


@pytest.fixture
def sample_detection() -> Detection:
    """Create a sample detection for testing."""
    return Detection(
        bbox=(100.0, 200.0, 300.0, 400.0),
        confidence=0.85,
        class_id=0,
        class_name="fish",
    )


@pytest.fixture
def sample_classification() -> Classification:
    """Create a sample classification for testing."""
    return Classification(
        species="river_herring",
        confidence=0.92,
        all_probabilities={
            "river_herring": 0.92,
            "bass": 0.05,
            "perch": 0.02,
            "unknown_fish": 0.01,
        },
    )


@pytest.fixture
def sample_pipeline_detection(
    sample_detection, sample_classification
) -> PipelineDetection:
    """Create a sample pipeline detection for testing."""
    return PipelineDetection(
        detection=sample_detection,
        classification=sample_classification,
        is_uncertain=False,
    )


@pytest.fixture
def sample_frame_result(sample_pipeline_detection) -> FrameResult:
    """Create a sample frame result for testing."""
    return FrameResult(
        source_path="test_frame.jpg",
        frame_number=0,
        timestamp=0.0,
        detections=[sample_pipeline_detection],
        flagged_for_review=False,
    )


@pytest.fixture
def sample_image() -> np.ndarray:
    """Create a synthetic test image (640x480, 3 channels)."""
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    # Add some colored rectangles to simulate fish-like shapes
    image[200:300, 100:250] = (0, 128, 0)   # Green rectangle
    image[150:250, 350:500] = (0, 0, 128)   # Red rectangle
    return image


@pytest.fixture
def fixtures_dir() -> Path:
    """Return the path to the test fixtures directory."""
    return FIXTURES_DIR
