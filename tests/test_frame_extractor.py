"""Tests for the video frame extractor."""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from herringnet.config import FrameExtractionConfig
from herringnet.data.frame_extractor import FrameExtractor


def _create_test_video(
    output_path: str,
    num_frames: int = 30,
    fps: float = 10.0,
    width: int = 320,
    height: int = 240,
) -> str:
    """Create a short synthetic test video."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for i in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Draw frame number for visual debugging
        cv2.putText(
            frame, str(i), (width // 2 - 20, height // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
        )
        writer.write(frame)

    writer.release()
    return output_path


class TestFrameExtractor:
    """Tests for the FrameExtractor class."""

    def test_extract_frames(self, tmp_path):
        video_path = str(tmp_path / "test.mp4")
        _create_test_video(video_path, num_frames=30, fps=10.0)

        config = FrameExtractionConfig(frame_interval=5, mean_frr=4.55)
        extractor = FrameExtractor(config)

        output_dir = tmp_path / "frames"
        frames = extractor.extract_frames(video_path, str(output_dir))

        # With 30 frames and interval 5, should get frames at 0, 5, 10, 15, 20, 25
        assert len(frames) == 6
        assert frames[0].frame_number == 0
        assert frames[1].frame_number == 5
        assert frames[-1].frame_number == 25

        # Check files exist
        for f in frames:
            assert Path(f.file_path).exists()

    def test_extract_max_frames(self, tmp_path):
        video_path = str(tmp_path / "test.mp4")
        _create_test_video(video_path, num_frames=50, fps=10.0)

        config = FrameExtractionConfig(frame_interval=5, mean_frr=4.55)
        extractor = FrameExtractor(config)

        output_dir = tmp_path / "frames"
        frames = extractor.extract_frames(
            video_path, str(output_dir), max_frames=3
        )

        assert len(frames) == 3

    def test_video_metadata(self, tmp_path):
        video_path = str(tmp_path / "test.mp4")
        _create_test_video(video_path, num_frames=30, fps=10.0)

        config = FrameExtractionConfig(frame_interval=5, mean_frr=4.55)
        extractor = FrameExtractor(config)
        metadata = extractor.get_video_metadata(video_path)

        assert metadata.fps == pytest.approx(10.0, abs=0.5)
        assert metadata.total_frames == 30
        assert metadata.width == 320
        assert metadata.height == 240
        assert metadata.duration_seconds == pytest.approx(3.0, abs=0.5)

    def test_extract_in_memory(self, tmp_path):
        video_path = str(tmp_path / "test.mp4")
        _create_test_video(video_path, num_frames=20, fps=10.0)

        config = FrameExtractionConfig(frame_interval=5, mean_frr=4.55)
        extractor = FrameExtractor(config)

        frames = extractor.extract_frames_in_memory(video_path)

        assert len(frames) == 4  # Frames at 0, 5, 10, 15
        for info, img in frames:
            assert isinstance(img, np.ndarray)
            assert img.shape == (240, 320, 3)

    def test_missing_video(self):
        config = FrameExtractionConfig(frame_interval=5, mean_frr=4.55)
        extractor = FrameExtractor(config)

        with pytest.raises(FileNotFoundError):
            extractor.extract_frames("nonexistent.mp4", "output")

    def test_timestamps(self, tmp_path):
        video_path = str(tmp_path / "test.mp4")
        _create_test_video(video_path, num_frames=30, fps=10.0)

        config = FrameExtractionConfig(frame_interval=5, mean_frr=4.55)
        extractor = FrameExtractor(config)

        output_dir = tmp_path / "frames"
        frames = extractor.extract_frames(video_path, str(output_dir))

        # Frame 0 at 0.0s, frame 5 at 0.5s, etc.
        assert frames[0].timestamp_seconds == pytest.approx(0.0)
        assert frames[1].timestamp_seconds == pytest.approx(0.5, abs=0.1)
