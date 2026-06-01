"""Tests for the FRR-corrected fish counter."""

from __future__ import annotations

import pytest

from herringnet.inference.result_types import (
    Classification,
    Detection,
    FrameResult,
    PipelineDetection,
)
from herringnet.postprocessing.counter import FishCounter


def _make_frame(
    fish_count: int,
    species: str = "fish",
    timestamp: float | None = None,
    frame_number: int | None = None,
) -> FrameResult:
    """Helper to create a FrameResult with the given number of detections."""
    detections = []
    for i in range(fish_count):
        det = Detection(
            bbox=(i * 100.0, 0.0, i * 100.0 + 50.0, 50.0),
            confidence=0.9,
        )
        cls = Classification(species=species, confidence=0.9)
        detections.append(
            PipelineDetection(detection=det, classification=cls)
        )
    return FrameResult(
        source_path="test.jpg",
        frame_number=frame_number,
        timestamp=timestamp,
        detections=detections,
    )


class TestFishCounter:
    """Tests for the FishCounter class."""

    def test_basic_counting(self):
        counter = FishCounter(mean_frr=4.55)
        frames = [_make_frame(3), _make_frame(5), _make_frame(0)]
        result = counter.count_from_frames(frames)

        assert result.raw_total == 8
        assert result.corrected_total == pytest.approx(8 / 4.55, abs=0.01)
        assert result.frames_processed == 3
        assert result.frames_with_fish == 2

    def test_empty_frames(self):
        counter = FishCounter(mean_frr=4.55)
        result = counter.count_from_frames([])

        assert result.raw_total == 0
        assert result.corrected_total == 0.0
        assert result.frames_processed == 0
        assert result.frames_with_fish == 0

    def test_all_empty_frames(self):
        counter = FishCounter(mean_frr=4.55)
        frames = [_make_frame(0), _make_frame(0)]
        result = counter.count_from_frames(frames)

        assert result.raw_total == 0
        assert result.frames_processed == 2
        assert result.frames_with_fish == 0

    def test_species_breakdown(self):
        counter = FishCounter(mean_frr=5.0)
        frames = [
            _make_frame(3, species="river_herring"),
            _make_frame(2, species="bass"),
            _make_frame(1, species="river_herring"),
        ]
        result = counter.count_from_frames(frames)

        assert result.raw_total == 6
        assert "river_herring" in result.species_counts
        assert "bass" in result.species_counts
        assert result.species_counts["river_herring"].raw == 4
        assert result.species_counts["bass"].raw == 2
        assert result.species_counts["river_herring"].corrected == pytest.approx(0.8)
        assert result.species_counts["bass"].corrected == pytest.approx(0.4)

    def test_custom_frr(self):
        counter = FishCounter(mean_frr=10.0)
        frames = [_make_frame(10)]
        result = counter.count_from_frames(frames)

        assert result.raw_total == 10
        assert result.corrected_total == pytest.approx(1.0)

    def test_invalid_frr(self):
        with pytest.raises(ValueError):
            FishCounter(mean_frr=0)
        with pytest.raises(ValueError):
            FishCounter(mean_frr=-1.0)

    def test_to_dict(self):
        counter = FishCounter(mean_frr=4.55)
        frames = [_make_frame(3, species="river_herring")]
        result = counter.count_from_frames(frames)

        d = result.to_dict()
        assert "raw_total" in d
        assert "corrected_total" in d
        assert "species_counts" in d

    def test_time_series(self):
        counter = FishCounter(mean_frr=4.55)
        frames = [
            _make_frame(2, timestamp=0.0),
            _make_frame(3, timestamp=30.0),
            _make_frame(1, timestamp=120.0),
        ]
        df = counter.generate_time_series(frames, bin_duration_minutes=1)

        assert not df.empty
        assert "raw_count" in df.columns
        assert "corrected_count" in df.columns

    def test_time_series_empty(self):
        counter = FishCounter(mean_frr=4.55)
        df = counter.generate_time_series([], bin_duration_minutes=1)
        assert df.empty
