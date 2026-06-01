"""Tests for result types and pipeline data flow.

These tests verify the data structures and serialization without
requiring model weights. Integration tests that load actual models
are marked with @pytest.mark.integration.
"""

from __future__ import annotations

import pytest

from herringnet.inference.result_types import (
    Classification,
    CountPair,
    CountResult,
    Detection,
    FrameResult,
    PipelineDetection,
    VideoResult,
)


class TestDetection:
    """Tests for the Detection dataclass."""

    def test_properties(self, sample_detection):
        assert sample_detection.width == 200.0
        assert sample_detection.height == 200.0
        assert sample_detection.area == 40000.0
        assert sample_detection.center == (200.0, 300.0)

    def test_to_dict(self, sample_detection):
        d = sample_detection.to_dict()
        assert d["bbox"] == [100.0, 200.0, 300.0, 400.0]
        assert d["confidence"] == 0.85
        assert d["class_name"] == "fish"


class TestClassification:
    """Tests for the Classification dataclass."""

    def test_to_dict(self, sample_classification):
        d = sample_classification.to_dict()
        assert d["species"] == "river_herring"
        assert d["confidence"] == 0.92
        assert "all_probabilities" in d

    def test_without_life_stage(self):
        cls = Classification(species="bass", confidence=0.8)
        d = cls.to_dict()
        assert "life_stage" not in d

    def test_with_life_stage(self):
        cls = Classification(
            species="river_herring",
            life_stage="juvenile",
            confidence=0.9,
        )
        d = cls.to_dict()
        assert d["life_stage"] == "juvenile"


class TestPipelineDetection:
    """Tests for the PipelineDetection dataclass."""

    def test_label_with_classification(self, sample_pipeline_detection):
        label = sample_pipeline_detection.label
        assert "River Herring" in label
        assert "92%" in label

    def test_label_without_classification(self, sample_detection):
        det = PipelineDetection(detection=sample_detection)
        label = det.label
        assert "Fish" in label
        assert "85%" in label

    def test_to_dict(self, sample_pipeline_detection):
        d = sample_pipeline_detection.to_dict()
        assert "detection" in d
        assert "classification" in d
        assert d["is_uncertain"] is False


class TestFrameResult:
    """Tests for the FrameResult dataclass."""

    def test_fish_count(self, sample_frame_result):
        assert sample_frame_result.fish_count == 1

    def test_species_counts(self, sample_frame_result):
        counts = sample_frame_result.species_counts
        assert counts["river_herring"] == 1

    def test_empty_frame(self):
        frame = FrameResult(source_path="empty.jpg")
        assert frame.fish_count == 0
        assert frame.species_counts == {}

    def test_to_dict(self, sample_frame_result):
        d = sample_frame_result.to_dict()
        assert d["fish_count"] == 1
        assert "detections" in d
        assert "species_counts" in d


class TestCountResult:
    """Tests for the CountResult dataclass."""

    def test_to_dict(self):
        result = CountResult(
            raw_total=10,
            corrected_total=2.2,
            species_counts={
                "river_herring": CountPair(raw=10, corrected=2.2),
            },
            frames_processed=5,
            frames_with_fish=3,
        )
        d = result.to_dict()
        assert d["raw_total"] == 10
        assert d["corrected_total"] == 2.2
        assert "river_herring" in d["species_counts"]


class TestVideoResult:
    """Tests for the VideoResult dataclass."""

    def test_to_dict(self):
        result = VideoResult(
            video_path="test.mp4",
            total_frames_extracted=10,
            video_fps=30.0,
            video_duration=60.0,
            processing_time_seconds=5.5,
        )
        d = result.to_dict()
        assert d["video_path"] == "test.mp4"
        assert d["total_frames_extracted"] == 10
        assert d["video_fps"] == 30.0
