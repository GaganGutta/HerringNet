"""Video frame extraction with frame-residence-rate (FRR) methodology.

Implements the frame extraction approach from Marjadi et al. (2024).
Extracts frames at intervals calibrated to the mean frame residence
rate to avoid overcounting the same fish across consecutive frames.

Reference:
    Marjadi, M.N., et al. (2024). Automated video monitoring estimates
    high abundances of juvenile anadromous fish emigrating from a coastal
    river. Limnology and Oceanography: Methods, 22, 295-310.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from herringnet.config import FrameExtractionConfig
from herringnet.inference.result_types import FrameInfo, VideoMetadata

logger = logging.getLogger(__name__)


class FrameExtractor:
    """Extract frames from video using FRR-corrected intervals.

    The frame residence rate (FRR) is the average number of consecutive
    frames a fish appears in as it transits the camera's field of view.
    Extracting every Nth frame (where N >= FRR) ensures that consecutive
    extracted frames do not contain the same fish, preventing overcounting.

    Args:
        config: Frame extraction configuration with interval and FRR values.
    """

    def __init__(self, config: FrameExtractionConfig):
        self.frame_interval = config.frame_interval
        self.mean_frr = config.mean_frr

    def extract_frames(
        self,
        video_path: str | Path,
        output_dir: str | Path,
        max_frames: int | None = None,
        image_format: str = "jpg",
    ) -> list[FrameInfo]:
        """Extract frames from a video at FRR-corrected intervals.

        Saves frames as individual image files in the output directory.
        Filenames include the frame number and timestamp for traceability.

        Args:
            video_path: Path to the source video file.
            output_dir: Directory to save extracted frame images.
            max_frames: Maximum number of frames to extract. None for all.
            image_format: Output image format ("jpg" or "png").

        Returns:
            List of FrameInfo objects with frame metadata and file paths.

        Raises:
            FileNotFoundError: If the video file does not exist.
            RuntimeError: If the video cannot be opened.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        metadata = self._read_metadata(cap)
        logger.info(
            "Video: %s, %.1f fps, %d frames, %.1f seconds",
            video_path.name,
            metadata.fps,
            metadata.total_frames,
            metadata.duration_seconds,
        )

        frames: list[FrameInfo] = []
        frame_number = 0
        extracted_count = 0
        stem = video_path.stem

        try:
            while True:
                if max_frames is not None and extracted_count >= max_frames:
                    break

                # Seek to the target frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = cap.read()

                if not ret:
                    break

                timestamp = frame_number / metadata.fps if metadata.fps > 0 else 0.0

                filename = f"{stem}_frame{frame_number:06d}.{image_format}"
                file_path = output_dir / filename
                cv2.imwrite(str(file_path), frame)

                frames.append(
                    FrameInfo(
                        frame_number=frame_number,
                        timestamp_seconds=timestamp,
                        file_path=str(file_path),
                    )
                )

                extracted_count += 1
                frame_number += self.frame_interval
        finally:
            cap.release()

        logger.info(
            "Extracted %d frames from %s (interval=%d)",
            len(frames),
            video_path.name,
            self.frame_interval,
        )

        return frames

    def extract_frames_in_memory(
        self,
        video_path: str | Path,
        max_frames: int | None = None,
    ) -> list[tuple[FrameInfo, np.ndarray]]:
        """Extract frames from video and return them in memory.

        Similar to extract_frames but does not save to disk. Returns
        (FrameInfo, image_array) tuples for direct pipeline processing.

        Args:
            video_path: Path to the source video file.
            max_frames: Maximum number of frames to extract.

        Returns:
            List of (FrameInfo, numpy_array) tuples.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        metadata = self._read_metadata(cap)
        frames: list[tuple[FrameInfo, np.ndarray]] = []
        frame_number = 0
        extracted_count = 0

        try:
            while True:
                if max_frames is not None and extracted_count >= max_frames:
                    break

                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = cap.read()

                if not ret:
                    break

                timestamp = frame_number / metadata.fps if metadata.fps > 0 else 0.0

                info = FrameInfo(
                    frame_number=frame_number,
                    timestamp_seconds=timestamp,
                    file_path="",  # No file path for in-memory frames
                )

                frames.append((info, frame))
                extracted_count += 1
                frame_number += self.frame_interval
        finally:
            cap.release()

        return frames

    def get_video_metadata(self, video_path: str | Path) -> VideoMetadata:
        """Extract metadata from a video file without reading frames.

        Args:
            video_path: Path to the video file.

        Returns:
            VideoMetadata with fps, dimensions, duration, and frame count.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        try:
            return self._read_metadata(cap)
        finally:
            cap.release()

    @staticmethod
    def _read_metadata(cap: cv2.VideoCapture) -> VideoMetadata:
        """Read video metadata from an open VideoCapture.

        Args:
            cap: An open cv2.VideoCapture instance.

        Returns:
            VideoMetadata with properties read from the capture.
        """
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0.0

        return VideoMetadata(
            fps=fps,
            total_frames=total_frames,
            duration_seconds=duration,
            width=width,
            height=height,
        )
