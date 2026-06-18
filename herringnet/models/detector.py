"""Fish detection module wrapping YOLO-based detection models.

Supports loading the Community Fish Detector (CFD) weights or any
Ultralytics-compatible YOLO detection model. Provides a clean interface
for running detection, extracting crops, and batch processing.

Includes SAHI (Sliced Aided Hyper Inference) support for detecting
small fish that would otherwise be missed at full-frame resolution.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from herringnet.config import DetectorConfig
from herringnet.inference.result_types import Detection
from herringnet.models.model_registry import ModelRegistry

logger = logging.getLogger(__name__)


class FishDetector:
    """Wrapper around a YOLO detection model for finding fish in images.

    The detector locates fish in camera trap images and returns bounding
    boxes with confidence scores. It does not classify species. That is
    handled by the SpeciesClassifier in the second pipeline stage.

    Supports two inference modes:
    - Standard: runs the model on the full image at the configured resolution.
    - Sliced (SAHI): tiles the image into overlapping slices so small fish
      are seen at a larger relative scale. Much better for juvenile herring
      and other small-bodied fish in camera trap footage.

    Args:
        config: Detector configuration specifying model, thresholds, etc.
        model_registry: Optional ModelRegistry instance for resolving
            model paths. Created automatically if not provided.
    """

    def __init__(
        self,
        config: DetectorConfig,
        model_registry: ModelRegistry | None = None,
    ):
        self.config = config
        registry = model_registry or ModelRegistry()
        model_path = registry.get_model_path(config.model_name)
        logger.info("Loading detection model: %s", model_path)
        self.model = YOLO(model_path)

        # SAHI detection model (lazy-loaded on first sliced inference call)
        self._sahi_model = None

    def detect(self, source: str | Path | np.ndarray) -> list[Detection]:
        """Run fish detection on a single image.

        Args:
            source: Image file path or numpy array in BGR format.

        Returns:
            List of Detection objects with bounding boxes and confidence
            scores, filtered by the configured confidence threshold.
        """
        results = self.model.predict(
            source=source,
            conf=self.config.confidence_threshold,
            iou=self.config.iou_threshold,
            imgsz=self.config.image_size,
            device=self.config.device,
            verbose=False,
        )

        detections = []
        for result in results:
            if result.boxes is None:
                continue
            boxes = result.boxes
            for i in range(len(boxes)):
                bbox = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())
                cls_name = result.names.get(cls_id, "fish")
                detections.append(
                    Detection(
                        bbox=(float(bbox[0]), float(bbox[1]),
                              float(bbox[2]), float(bbox[3])),
                        confidence=conf,
                        class_id=cls_id,
                        class_name=cls_name,
                    )
                )

        logger.debug("Detected %d fish in %s", len(detections), source)
        return detections

    def detect_sliced(
        self,
        source: str | Path | np.ndarray,
        slice_size: int = 512,
        overlap_ratio: float = 0.3,
    ) -> list[Detection]:
        """Run sliced inference (SAHI) for detecting small fish.

        Tiles the image into overlapping slices of the given size, runs
        detection on each slice, then merges results back into full-image
        coordinates with NMS to remove duplicates at slice boundaries.

        This is critical for juvenile river herring and other small-bodied
        fish that occupy only a tiny fraction of the full camera frame.

        Args:
            source: Image file path or numpy array in BGR format.
            slice_size: Width and height of each tile in pixels.
            overlap_ratio: Fractional overlap between adjacent tiles (0.0 to 0.5).

        Returns:
            List of Detection objects in full-image coordinates.
        """
        from sahi import AutoDetectionModel
        from sahi.predict import get_sliced_prediction

        # Load image if given a path; otherwise write a temp file for SAHI.
        temp_path: Path | None = None
        if isinstance(source, (str, Path)):
            image_path = str(source)
        else:
            import tempfile
            tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            tmp.close()  # release the handle so cv2 can write (and Windows can unlink)
            cv2.imwrite(tmp.name, source)
            image_path = tmp.name
            temp_path = Path(tmp.name)

        try:
            # Lazy-load the SAHI detection model wrapper
            if self._sahi_model is None:
                registry = ModelRegistry()
                model_path = registry.get_model_path(self.config.model_name)
                self._sahi_model = AutoDetectionModel.from_pretrained(
                    model_type="ultralytics",
                    model_path=model_path,
                    confidence_threshold=self.config.confidence_threshold,
                    device=self.config.device,
                )

            result = get_sliced_prediction(
                image=image_path,
                detection_model=self._sahi_model,
                slice_height=slice_size,
                slice_width=slice_size,
                overlap_height_ratio=overlap_ratio,
                overlap_width_ratio=overlap_ratio,
                verbose=0,
            )
        finally:
            # Always clean up the temp frame, even on error, to avoid filling
            # the disk when processing video frame-by-frame with SAHI.
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

        detections = []
        for pred in result.object_prediction_list:
            bbox = pred.bbox
            detections.append(
                Detection(
                    bbox=(
                        float(bbox.minx),
                        float(bbox.miny),
                        float(bbox.maxx),
                        float(bbox.maxy),
                    ),
                    confidence=float(pred.score.value),
                    class_id=pred.category.id if pred.category else 0,
                    class_name=pred.category.name if pred.category else "fish",
                )
            )

        logger.debug(
            "Sliced inference: %d detections (slice=%d, overlap=%.1f)",
            len(detections), slice_size, overlap_ratio,
        )
        return detections

    def detect_batch(
        self,
        sources: list[str | Path],
        batch_size: int = 8,
    ) -> list[list[Detection]]:
        """Run detection on a batch of images.

        Args:
            sources: List of image file paths.
            batch_size: Number of images to process at once.

        Returns:
            List of detection lists, one per input image.
        """
        all_detections: list[list[Detection]] = []

        for i in range(0, len(sources), batch_size):
            batch = sources[i : i + batch_size]
            results = self.model.predict(
                source=batch,
                conf=self.config.confidence_threshold,
                iou=self.config.iou_threshold,
                imgsz=self.config.image_size,
                device=self.config.device,
                verbose=False,
            )

            for result in results:
                frame_detections = []
                if result.boxes is not None:
                    boxes = result.boxes
                    for j in range(len(boxes)):
                        bbox = boxes.xyxy[j].cpu().numpy()
                        conf = float(boxes.conf[j].cpu().numpy())
                        cls_id = int(boxes.cls[j].cpu().numpy())
                        cls_name = result.names.get(cls_id, "fish")
                        frame_detections.append(
                            Detection(
                                bbox=(float(bbox[0]), float(bbox[1]),
                                      float(bbox[2]), float(bbox[3])),
                                confidence=conf,
                                class_id=cls_id,
                                class_name=cls_name,
                            )
                        )
                all_detections.append(frame_detections)

        return all_detections

    @staticmethod
    def crop_detections(
        image: np.ndarray,
        detections: list[Detection],
        padding: float = 0.1,
    ) -> list[np.ndarray]:
        """Crop detected fish regions from an image.

        Extracts the bounding box region for each detection, with optional
        padding to include surrounding context. These crops are passed to
        the species classifier in Stage 2.

        Args:
            image: Source image as a numpy array (BGR format).
            detections: List of detections to crop.
            padding: Fractional padding to add around each box (0.1 = 10%).

        Returns:
            List of cropped image arrays, one per detection.
        """
        h, w = image.shape[:2]
        crops = []

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            box_w = x2 - x1
            box_h = y2 - y1

            # Add padding
            pad_x = box_w * padding
            pad_y = box_h * padding

            # Clamp to image bounds
            cx1 = max(0, int(x1 - pad_x))
            cy1 = max(0, int(y1 - pad_y))
            cx2 = min(w, int(x2 + pad_x))
            cy2 = min(h, int(y2 + pad_y))

            crop = image[cy1:cy2, cx1:cx2]
            if crop.size > 0:
                crops.append(crop)

        return crops

    def load_image(self, image_path: str | Path) -> np.ndarray:
        """Load an image from disk using OpenCV.

        Args:
            image_path: Path to the image file.

        Returns:
            Image as a numpy array in BGR format.

        Raises:
            FileNotFoundError: If the image file does not exist.
            ValueError: If the image cannot be read.
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Failed to read image: {path}")

        return image
