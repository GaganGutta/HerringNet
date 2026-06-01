"""Data classes for detection, classification, and pipeline results.

These types define the structured output of each stage of the HerringNet
pipeline, from raw detections through species classification to final
video-level counting results.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Detection:
    """A single fish detection from the detector model.

    Attributes:
        bbox: Bounding box coordinates (x1, y1, x2, y2) in pixels.
        confidence: Detection confidence score, 0.0 to 1.0.
        class_id: Class index from the detection model.
        class_name: Class label from the detection model (e.g., "fish").
    """

    bbox: tuple[float, float, float, float]
    confidence: float
    class_id: int = 0
    class_name: str = "fish"

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return (
            (self.bbox[0] + self.bbox[2]) / 2,
            (self.bbox[1] + self.bbox[3]) / 2,
        )

    def to_dict(self) -> dict:
        return {
            "bbox": list(self.bbox),
            "confidence": round(self.confidence, 4),
            "class_id": self.class_id,
            "class_name": self.class_name,
        }


@dataclass
class Classification:
    """Species and life-stage classification result for a cropped fish region.

    Attributes:
        species: Predicted species name (e.g., "river_herring").
        life_stage: Predicted life stage if available (e.g., "juvenile").
        confidence: Classification confidence score, 0.0 to 1.0.
        all_probabilities: Full probability distribution over all classes.
    """

    species: str
    life_stage: str | None = None
    confidence: float = 0.0
    all_probabilities: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        result: dict = {
            "species": self.species,
            "confidence": round(self.confidence, 4),
        }
        if self.life_stage is not None:
            result["life_stage"] = self.life_stage
        if self.all_probabilities:
            result["all_probabilities"] = {
                k: round(v, 4) for k, v in self.all_probabilities.items()
            }
        return result


@dataclass
class PipelineDetection:
    """Combined detection + classification result from the two-stage pipeline.

    Attributes:
        detection: The bounding box detection from Stage 1.
        classification: The species classification from Stage 2, or None
            if the classifier is not loaded.
        is_uncertain: Whether this detection was flagged for human review.
        uncertainty_reason: Why the detection was flagged, if applicable.
    """

    detection: Detection
    classification: Classification | None = None
    is_uncertain: bool = False
    uncertainty_reason: str | None = None

    @property
    def label(self) -> str:
        """Human-readable label for display on annotated images."""
        if self.classification is not None:
            species = self.classification.species.replace("_", " ").title()
            conf = self.classification.confidence
            return f"{species} ({conf:.0%})"
        conf = self.detection.confidence
        return f"Fish ({conf:.0%})"

    def to_dict(self) -> dict:
        result = {
            "detection": self.detection.to_dict(),
            "is_uncertain": self.is_uncertain,
        }
        if self.classification is not None:
            result["classification"] = self.classification.to_dict()
        if self.uncertainty_reason is not None:
            result["uncertainty_reason"] = self.uncertainty_reason
        return result


@dataclass
class FrameResult:
    """Results from processing a single image or video frame.

    Attributes:
        source_path: Path to the source image file.
        frame_number: Frame index if extracted from video, or None for images.
        timestamp: Timestamp in seconds if from video, or None.
        detections: List of pipeline detection results.
        flagged_for_review: Whether any detections were flagged for review.
    """

    source_path: str
    frame_number: int | None = None
    timestamp: float | None = None
    detections: list[PipelineDetection] = field(default_factory=list)
    flagged_for_review: bool = False

    @property
    def fish_count(self) -> int:
        return len(self.detections)

    @property
    def species_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for det in self.detections:
            if det.classification is not None:
                species = det.classification.species
            else:
                species = "fish"
            counts[species] = counts.get(species, 0) + 1
        return counts

    def to_dict(self) -> dict:
        result: dict = {
            "source_path": self.source_path,
            "fish_count": self.fish_count,
            "detections": [d.to_dict() for d in self.detections],
            "flagged_for_review": self.flagged_for_review,
        }
        if self.frame_number is not None:
            result["frame_number"] = self.frame_number
        if self.timestamp is not None:
            result["timestamp"] = round(self.timestamp, 3)
        result["species_counts"] = self.species_counts
        return result


@dataclass
class CountPair:
    """Raw and FRR-corrected count for a species.

    Attributes:
        raw: Raw detection count across all frames.
        corrected: FRR-adjusted count estimating true passage.
    """

    raw: int = 0
    corrected: float = 0.0

    def to_dict(self) -> dict:
        return {"raw": self.raw, "corrected": round(self.corrected, 2)}


@dataclass
class CountResult:
    """FRR-corrected fish counting results for a video.

    Attributes:
        raw_total: Total raw detections across all extracted frames.
        corrected_total: FRR-adjusted total passage estimate.
        species_counts: Per-species raw and corrected counts.
        frames_processed: Number of frames that were processed.
        frames_with_fish: Number of frames with at least one detection.
    """

    raw_total: int = 0
    corrected_total: float = 0.0
    species_counts: dict[str, CountPair] = field(default_factory=dict)
    frames_processed: int = 0
    frames_with_fish: int = 0

    def to_dict(self) -> dict:
        return {
            "raw_total": self.raw_total,
            "corrected_total": round(self.corrected_total, 2),
            "species_counts": {
                k: v.to_dict() for k, v in self.species_counts.items()
            },
            "frames_processed": self.frames_processed,
            "frames_with_fish": self.frames_with_fish,
        }


@dataclass
class VideoResult:
    """Full results from processing a video file.

    Attributes:
        video_path: Path to the source video file.
        total_frames_extracted: Number of frames extracted from the video.
        per_frame_results: Detection results for each extracted frame.
        counts: FRR-corrected counting results.
        video_fps: Original video frame rate.
        video_duration: Video duration in seconds.
        processing_time_seconds: Wall-clock time for processing.
    """

    video_path: str
    total_frames_extracted: int = 0
    per_frame_results: list[FrameResult] = field(default_factory=list)
    counts: CountResult = field(default_factory=CountResult)
    video_fps: float = 0.0
    video_duration: float = 0.0
    processing_time_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "video_path": self.video_path,
            "total_frames_extracted": self.total_frames_extracted,
            "video_fps": round(self.video_fps, 2),
            "video_duration": round(self.video_duration, 2),
            "processing_time_seconds": round(self.processing_time_seconds, 2),
            "counts": self.counts.to_dict(),
            "per_frame_results": [r.to_dict() for r in self.per_frame_results],
        }


@dataclass
class FrameInfo:
    """Metadata for an extracted video frame.

    Attributes:
        frame_number: The frame index in the original video.
        timestamp_seconds: Timestamp of this frame in seconds.
        file_path: Path to the saved frame image file.
    """

    frame_number: int
    timestamp_seconds: float
    file_path: str


@dataclass
class VideoMetadata:
    """Metadata extracted from a video file.

    Attributes:
        fps: Frames per second.
        total_frames: Total number of frames in the video.
        duration_seconds: Duration of the video in seconds.
        width: Frame width in pixels.
        height: Frame height in pixels.
    """

    fps: float
    total_frames: int
    duration_seconds: float
    width: int
    height: int
