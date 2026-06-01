"""Active learning module for uncertainty sampling and human review.

Identifies frames where the model is uncertain and queues them for
human annotation. Corrected labels are fed back into the training
dataset to improve model performance iteratively.
"""

from __future__ import annotations

import csv
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from herringnet.config import ActiveLearningConfig
from herringnet.inference.result_types import Classification, Detection

logger = logging.getLogger(__name__)


class ActiveLearningManager:
    """Manage uncertainty sampling and the human review workflow.

    Uncertain detections are saved to a review queue directory as
    cropped images with JSON metadata sidecar files. The review queue
    can be exported as CSV for use with annotation tools, and corrected
    labels can be imported back into the training dataset.

    Args:
        config: Active learning configuration.
    """

    def __init__(self, config: ActiveLearningConfig):
        self.config = config
        self.review_dir = Path(config.review_output_dir)
        self.review_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_uncertainty(
        self,
        detection: Detection,
        classification: Classification | None,
    ) -> tuple[bool, str | None]:
        """Determine if a detection should be flagged for human review.

        Checks multiple uncertainty criteria:
        - Detection confidence below threshold
        - Classification confidence below threshold
        - Top-2 class probabilities within the margin threshold

        Args:
            detection: The bounding box detection.
            classification: The species classification, if available.

        Returns:
            Tuple of (is_uncertain, reason_string_or_None).
        """
        if not self.config.enabled:
            return False, None

        threshold = self.config.uncertainty_threshold
        margin = self.config.margin_threshold

        if detection.confidence < threshold:
            return True, f"Low detection confidence: {detection.confidence:.2f}"

        if classification is not None:
            if classification.confidence < threshold:
                return (
                    True,
                    f"Low classification confidence: {classification.confidence:.2f}",
                )

            if classification.all_probabilities:
                sorted_probs = sorted(
                    classification.all_probabilities.values(), reverse=True
                )
                if len(sorted_probs) >= 2:
                    prob_margin = sorted_probs[0] - sorted_probs[1]
                    if prob_margin < margin:
                        return (
                            True,
                            f"Close margin between top predictions: {prob_margin:.2f}",
                        )

        return False, None

    def add_to_review_queue(
        self,
        frame_path: str,
        detection: Detection,
        classification: Classification | None,
        crop: np.ndarray,
        reason: str,
    ) -> str:
        """Save an uncertain detection to the review queue.

        Saves the cropped fish image and a JSON sidecar file with
        detection metadata, model predictions, and the flagging reason.

        Args:
            frame_path: Path to the source frame image.
            detection: The detection bounding box.
            classification: The model's classification prediction.
            crop: Cropped fish image (BGR numpy array).
            reason: Why this detection was flagged.

        Returns:
            Path to the saved crop image.
        """
        # Check queue size limit
        existing = list(self.review_dir.glob("*.json"))
        if len(existing) >= self.config.max_review_queue:
            logger.warning(
                "Review queue at capacity (%d). Skipping.",
                self.config.max_review_queue,
            )
            return ""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        crop_filename = f"review_{timestamp}.jpg"
        json_filename = f"review_{timestamp}.json"

        crop_path = self.review_dir / crop_filename
        json_path = self.review_dir / json_filename

        # Save the crop image
        cv2.imwrite(str(crop_path), crop)

        # Save metadata
        metadata = {
            "source_frame": frame_path,
            "crop_path": str(crop_path),
            "timestamp": timestamp,
            "uncertainty_reason": reason,
            "detection": detection.to_dict(),
            "model_prediction": (
                classification.to_dict() if classification else None
            ),
            "corrected_label": None,  # To be filled by human reviewer
        }

        with open(json_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.debug("Added to review queue: %s", crop_filename)
        return str(crop_path)

    def export_review_queue(self, output_path: str) -> str:
        """Export the review queue as a CSV file for annotation.

        The CSV includes crop paths, model predictions, and empty columns
        for the human reviewer to fill in corrected labels.

        Args:
            output_path: Path to save the CSV file.

        Returns:
            Path to the saved CSV file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        json_files = sorted(self.review_dir.glob("*.json"))
        if not json_files:
            logger.warning("Review queue is empty, nothing to export.")
            return str(output_path)

        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "crop_path",
                "source_frame",
                "model_species",
                "model_confidence",
                "uncertainty_reason",
                "corrected_species",
                "corrected_life_stage",
                "notes",
            ])

            for jf in json_files:
                with open(jf) as jfile:
                    data = json.load(jfile)

                model_pred = data.get("model_prediction", {}) or {}
                writer.writerow([
                    data.get("crop_path", ""),
                    data.get("source_frame", ""),
                    model_pred.get("species", ""),
                    model_pred.get("confidence", ""),
                    data.get("uncertainty_reason", ""),
                    data.get("corrected_label", ""),
                    "",  # corrected_life_stage
                    "",  # notes
                ])

        logger.info("Exported %d items to %s", len(json_files), output_path)
        return str(output_path)

    def import_corrections(
        self,
        corrections_path: str,
        dataset_dir: str,
    ) -> int:
        """Import human-corrected labels back into the training dataset.

        Reads a CSV with corrected_species labels, copies the crop images
        into the appropriate class folders in the training dataset.

        Args:
            corrections_path: Path to the corrected CSV file.
            dataset_dir: Path to the training dataset directory with
                class subfolders (ImageNet-style structure).

        Returns:
            Number of corrections successfully imported.
        """
        corrections_path = Path(corrections_path)
        dataset_dir = Path(dataset_dir)

        if not corrections_path.exists():
            raise FileNotFoundError(f"Corrections file not found: {corrections_path}")

        imported = 0
        with open(corrections_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                corrected = row.get("corrected_species", "").strip()
                crop_path = row.get("crop_path", "").strip()

                if not corrected or not crop_path:
                    continue

                crop = Path(crop_path)
                if not crop.exists():
                    logger.warning("Crop not found, skipping: %s", crop_path)
                    continue

                # Copy to the appropriate class folder
                class_dir = dataset_dir / "train" / corrected
                class_dir.mkdir(parents=True, exist_ok=True)

                dest = class_dir / crop.name
                shutil.copy2(str(crop), str(dest))
                imported += 1

        logger.info("Imported %d corrections into %s", imported, dataset_dir)
        return imported
