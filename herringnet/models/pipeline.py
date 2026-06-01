"""Two-stage fish detection and classification pipeline.

This is the central orchestrator for HerringNet. It combines:
  Stage 1: Fish detection (locating all fish in a frame)
  Stage 2: Species classification (identifying each detected fish)

It also handles active learning uncertainty flagging and integrates
with the FRR-corrected counting system for video processing.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import cv2
from tqdm import tqdm

from herringnet.config import HerringNetConfig
from herringnet.data.frame_extractor import FrameExtractor
from herringnet.inference.result_types import (
    Classification,
    FrameResult,
    PipelineDetection,
    VideoResult,
)
from herringnet.models.classifier import SpeciesClassifier
from herringnet.models.detector import FishDetector
from herringnet.postprocessing.counter import FishCounter

logger = logging.getLogger(__name__)

# File extensions recognized as images
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# File extensions recognized as videos
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"}


class HerringNetPipeline:
    """Two-stage fish detection and classification pipeline.

    Stage 1 uses the FishDetector (e.g., CFD) to locate all fish in
    a frame and produce bounding boxes. Stage 2 crops each detection
    and classifies species/life-stage using the SpeciesClassifier.

    If the classifier model is not configured, the pipeline runs in
    detection-only mode (Stage 1 only).

    Args:
        config: Full HerringNet configuration.
    """

    def __init__(self, config: HerringNetConfig):
        self.config = config
        self.detector = FishDetector(config.detector)

        # Classifier is optional. If model_path is None, run detection-only.
        self.classifier: SpeciesClassifier | None = None
        if config.classifier.model_path is not None:
            try:
                self.classifier = SpeciesClassifier(config.classifier)
            except (FileNotFoundError, ValueError) as e:
                logger.warning(
                    "Classifier not loaded, running detection-only: %s", e
                )

        self.frame_extractor = FrameExtractor(config.frame_extraction)
        self.counter = FishCounter(config.frame_extraction.mean_frr)

        # Active learning settings
        self._al_enabled = config.active_learning.enabled
        self._al_threshold = config.active_learning.uncertainty_threshold
        self._al_margin = config.active_learning.margin_threshold

    def process_image(
        self,
        image_path: str | Path,
    ) -> FrameResult:
        """Process a single image through the full pipeline.

        1. Run detection to find fish bounding boxes.
        2. Crop each detection.
        3. Classify each crop for species/life-stage (if classifier loaded).
        4. Flag uncertain detections for review.

        Args:
            image_path: Path to the image file.

        Returns:
            FrameResult with all detections and classifications.
        """
        image_path = str(image_path)
        image = self.detector.load_image(image_path)

        # Stage 1: Detection (use SAHI sliced inference if configured)
        det_config = self.config.detector
        if det_config.use_sahi:
            detections = self.detector.detect_sliced(
                image_path,
                slice_size=det_config.sahi_slice_size,
                overlap_ratio=det_config.sahi_overlap_ratio,
            )
        else:
            detections = self.detector.detect(image)

        # Stage 2: Classification (if available)
        pipeline_detections: list[PipelineDetection] = []
        if self.classifier is not None and detections:
            crops = self.detector.crop_detections(image, detections)
            classifications = self.classifier.classify_batch(crops)

            for det, cls in zip(detections, classifications):
                is_uncertain, reason = self._check_uncertainty(det, cls)
                pipeline_detections.append(
                    PipelineDetection(
                        detection=det,
                        classification=cls,
                        is_uncertain=is_uncertain,
                        uncertainty_reason=reason,
                    )
                )
        else:
            for det in detections:
                is_uncertain = det.confidence < self._al_threshold
                reason = (
                    f"Low detection confidence: {det.confidence:.2f}"
                    if is_uncertain
                    else None
                )
                pipeline_detections.append(
                    PipelineDetection(
                        detection=det,
                        is_uncertain=is_uncertain,
                        uncertainty_reason=reason,
                    )
                )

        flagged = any(d.is_uncertain for d in pipeline_detections)

        return FrameResult(
            source_path=image_path,
            detections=pipeline_detections,
            flagged_for_review=flagged,
        )

    def process_video(
        self,
        video_path: str | Path,
        output_dir: str | Path | None = None,
        max_frames: int | None = None,
        save_frames: bool = False,
    ) -> VideoResult:
        """Process a video file with FRR-corrected frame extraction.

        1. Extract frames at FRR-corrected intervals.
        2. Run process_image on each extracted frame.
        3. Aggregate counts with FRR correction.

        Args:
            video_path: Path to the video file.
            output_dir: Directory for saving frames and results.
            max_frames: Maximum number of frames to process.
            save_frames: Whether to save extracted frames to disk.

        Returns:
            VideoResult with per-frame detections and corrected counts.
        """
        video_path = Path(video_path)
        start_time = time.time()

        metadata = self.frame_extractor.get_video_metadata(video_path)

        if save_frames and output_dir is not None:
            frames_dir = Path(output_dir) / "frames"
            frame_infos = self.frame_extractor.extract_frames(
                video_path, frames_dir, max_frames=max_frames
            )
            # Process saved frame files
            per_frame: list[FrameResult] = []
            for info in tqdm(frame_infos, desc="Processing frames"):
                result = self.process_image(info.file_path)
                result.frame_number = info.frame_number
                result.timestamp = info.timestamp_seconds
                per_frame.append(result)
        else:
            # Process frames in memory (no disk I/O)
            in_memory = self.frame_extractor.extract_frames_in_memory(
                video_path, max_frames=max_frames
            )
            per_frame = []
            for info, frame_img in tqdm(in_memory, desc="Processing frames"):
                if det_config.use_sahi:
                    detections = self.detector.detect_sliced(
                        frame_img,
                        slice_size=det_config.sahi_slice_size,
                        overlap_ratio=det_config.sahi_overlap_ratio,
                    )
                else:
                    detections = self.detector.detect(frame_img)

                pipeline_dets: list[PipelineDetection] = []
                if self.classifier is not None and detections:
                    crops = self.detector.crop_detections(frame_img, detections)
                    classifications = self.classifier.classify_batch(crops)
                    for det, cls in zip(detections, classifications):
                        is_unc, reason = self._check_uncertainty(det, cls)
                        pipeline_dets.append(
                            PipelineDetection(
                                detection=det,
                                classification=cls,
                                is_uncertain=is_unc,
                                uncertainty_reason=reason,
                            )
                        )
                else:
                    for det in detections:
                        is_unc = det.confidence < self._al_threshold
                        reason = (
                            f"Low detection confidence: {det.confidence:.2f}"
                            if is_unc
                            else None
                        )
                        pipeline_dets.append(
                            PipelineDetection(
                                detection=det,
                                is_uncertain=is_unc,
                                uncertainty_reason=reason,
                            )
                        )

                flagged = any(d.is_uncertain for d in pipeline_dets)
                per_frame.append(
                    FrameResult(
                        source_path=str(video_path),
                        frame_number=info.frame_number,
                        timestamp=info.timestamp_seconds,
                        detections=pipeline_dets,
                        flagged_for_review=flagged,
                    )
                )

        # Compute FRR-corrected counts
        counts = self.counter.count_from_frames(per_frame)

        elapsed = time.time() - start_time

        return VideoResult(
            video_path=str(video_path),
            total_frames_extracted=len(per_frame),
            per_frame_results=per_frame,
            counts=counts,
            video_fps=metadata.fps,
            video_duration=metadata.duration_seconds,
            processing_time_seconds=elapsed,
        )

    def process_directory(
        self,
        input_dir: str | Path,
        recursive: bool = False,
    ) -> list[FrameResult]:
        """Process all images in a directory.

        Args:
            input_dir: Directory containing image files.
            recursive: Whether to search subdirectories.

        Returns:
            List of FrameResult objects, one per image.
        """
        input_dir = Path(input_dir)
        pattern = "**/*" if recursive else "*"

        image_paths = sorted(
            p
            for p in input_dir.glob(pattern)
            if p.suffix.lower() in IMAGE_EXTENSIONS
        )

        if not image_paths:
            logger.warning("No images found in %s", input_dir)
            return []

        results = []
        for path in tqdm(image_paths, desc="Processing images"):
            result = self.process_image(path)
            results.append(result)

        return results

    def _check_uncertainty(
        self,
        det: "Detection",
        cls: Classification,
    ) -> tuple[bool, str | None]:
        """Check if a detection should be flagged for human review.

        Flags when:
        - Detection confidence is below the uncertainty threshold.
        - Classification confidence is below the uncertainty threshold.
        - Top-2 classification probabilities are within the margin threshold.

        Args:
            det: The detection from Stage 1.
            cls: The classification from Stage 2.

        Returns:
            Tuple of (is_uncertain, reason_string_or_None).
        """
        if not self._al_enabled:
            return False, None

        # Low detection confidence
        if det.confidence < self._al_threshold:
            return True, f"Low detection confidence: {det.confidence:.2f}"

        # Low classification confidence
        if cls.confidence < self._al_threshold:
            return True, f"Low classification confidence: {cls.confidence:.2f}"

        # Close margin between top-2 predictions
        if cls.all_probabilities:
            sorted_probs = sorted(
                cls.all_probabilities.values(), reverse=True
            )
            if len(sorted_probs) >= 2:
                margin = sorted_probs[0] - sorted_probs[1]
                if margin < self._al_margin:
                    return True, (
                        f"Close margin between top predictions: {margin:.2f}"
                    )

        return False, None


def save_results_json(
    results: list[FrameResult] | VideoResult,
    output_path: str | Path,
) -> str:
    """Save pipeline results as a JSON file.

    Args:
        results: Either a list of FrameResults or a single VideoResult.
        output_path: Path to save the JSON file.

    Returns:
        Path to the saved JSON file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(results, VideoResult):
        data = results.to_dict()
    else:
        data = {"frames": [r.to_dict() for r in results]}

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    return str(output_path)
