"""Species and life-stage classification module.

Uses a YOLOv8 classification model to identify the species and
optionally the life stage of cropped fish regions produced by
the FishDetector in Stage 1 of the pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from herringnet.config import ClassifierConfig
from herringnet.inference.result_types import Classification

logger = logging.getLogger(__name__)


class SpeciesClassifier:
    """YOLOv8 classification model for species and life-stage identification.

    Takes cropped fish regions from the detector and classifies them
    into species categories. The classifier is trained separately from
    the detector and can be updated independently.

    Args:
        config: Classifier configuration specifying model path and thresholds.
    """

    def __init__(self, config: ClassifierConfig):
        self.config = config
        if config.model_path is None:
            raise ValueError(
                "Classifier model_path is not set. Train a classifier first "
                "using 'herringnet train classify' or set classifier.model_path "
                "in the config."
            )

        model_path = Path(config.model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Classifier model not found: {model_path}. "
                "Train a classifier first using 'herringnet train classify'."
            )

        logger.info("Loading classifier model: %s", model_path)
        self.model = YOLO(str(model_path))
        self._class_names: dict[int, str] = self.model.names or {}

    @property
    def class_names(self) -> dict[int, str]:
        """Map of class indices to species names from the model."""
        return self._class_names

    def classify(self, crop: np.ndarray) -> Classification:
        """Classify a single cropped fish image.

        Args:
            crop: Cropped fish image as a numpy array (BGR format).

        Returns:
            Classification with species, confidence, and full probabilities.
        """
        results = self.model.predict(
            source=crop,
            imgsz=self.config.image_size,
            verbose=False,
        )

        if not results or results[0].probs is None:
            return Classification(
                species="unknown_fish",
                confidence=0.0,
            )

        probs = results[0].probs
        top_class = int(probs.top1)
        top_conf = float(probs.top1conf.cpu().numpy())
        species_name = self._class_names.get(top_class, "unknown_fish")

        # Build full probability distribution
        all_probs = {}
        prob_data = probs.data.cpu().numpy()
        for idx, prob in enumerate(prob_data):
            name = self._class_names.get(idx, f"class_{idx}")
            all_probs[name] = float(prob)

        return Classification(
            species=species_name,
            confidence=top_conf,
            all_probabilities=all_probs,
        )

    def classify_batch(self, crops: list[np.ndarray]) -> list[Classification]:
        """Classify a batch of cropped fish images.

        Args:
            crops: List of cropped fish images (BGR format).

        Returns:
            List of Classification results, one per input crop.
        """
        if not crops:
            return []

        results = self.model.predict(
            source=crops,
            imgsz=self.config.image_size,
            verbose=False,
        )

        classifications = []
        for result in results:
            if result.probs is None:
                classifications.append(
                    Classification(species="unknown_fish", confidence=0.0)
                )
                continue

            probs = result.probs
            top_class = int(probs.top1)
            top_conf = float(probs.top1conf.cpu().numpy())
            species_name = self._class_names.get(top_class, "unknown_fish")

            all_probs = {}
            prob_data = probs.data.cpu().numpy()
            for idx, prob in enumerate(prob_data):
                name = self._class_names.get(idx, f"class_{idx}")
                all_probs[name] = float(prob)

            classifications.append(
                Classification(
                    species=species_name,
                    confidence=top_conf,
                    all_probabilities=all_probs,
                )
            )

        return classifications
