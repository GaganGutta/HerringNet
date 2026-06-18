"""Training orchestration for fine-tuning detection and classification models.

Wraps the Ultralytics training API with HerringNet-specific defaults
and logging. Supports fine-tuning from pretrained YOLO models with
optional layer freezing for transfer learning.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ultralytics import YOLO

logger = logging.getLogger(__name__)


class HerringNetTrainer:
    """Orchestrate fine-tuning of detection and classification models.

    Uses the Ultralytics training API with sensible defaults for
    CPU-based training on small to medium fish datasets.
    """

    def train_classifier(
        self,
        data_path: str,
        base_model: str = "yolov8n-cls.pt",
        epochs: int = 50,
        image_size: int = 224,
        batch_size: int = 16,
        freeze_layers: int | None = None,
        project: str = "runs/classify",
        name: str = "herringnet-cls",
    ) -> str:
        """Fine-tune a YOLOv8 classification model for species ID.

        The classifier is trained on cropped fish images organized in
        ImageNet-style directory structure (one folder per class).

        Args:
            data_path: Path to the dataset directory (with train/val subdirs).
            base_model: Pretrained model to start from.
            epochs: Number of training epochs.
            image_size: Input image size for training.
            batch_size: Training batch size.
            freeze_layers: Number of backbone layers to freeze. None to
                train all layers.
            project: Directory to save training outputs.
            name: Name for this training run.

        Returns:
            Path to the best model weights file.
        """
        logger.info("Starting classifier training")
        logger.info("  Base model: %s", base_model)
        logger.info("  Data: %s", data_path)
        logger.info(
            "  Epochs: %d, Batch: %d, ImgSz: %d",
            epochs, batch_size, image_size,
        )

        model = YOLO(base_model)

        train_kwargs: dict = {
            "data": data_path,
            "epochs": epochs,
            "imgsz": image_size,
            "batch": batch_size,
            "project": project,
            "name": name,
            "device": "cpu",
            "workers": 0,  # Avoid multiprocessing issues on Windows/CPU
            "exist_ok": True,
        }

        if freeze_layers is not None:
            train_kwargs["freeze"] = freeze_layers

        model.train(**train_kwargs)

        best_path = Path(project) / name / "weights" / "best.pt"
        if best_path.exists():
            logger.info("Best weights saved: %s", best_path)
            return str(best_path)

        # Fallback: return last.pt if best.pt is not found
        last_path = Path(project) / name / "weights" / "last.pt"
        if last_path.exists():
            logger.info("Last weights saved: %s", last_path)
            return str(last_path)

        raise FileNotFoundError(
            f"Training completed but weights not found in {project}/{name}/weights/"
        )

    def train_detector(
        self,
        data_path: str,
        base_model: str = "yolov8s.pt",
        epochs: int = 100,
        image_size: int = 640,
        batch_size: int = 8,
        freeze_layers: int | None = 10,
        project: str = "runs/detect",
        name: str = "herringnet-det",
    ) -> str:
        """Fine-tune a YOLOv8 detection model for fish detection.

        This is an alternative to using the CFD pretrained detector.
        Fine-tunes a YOLOv8 model on fish detection data in YOLO format.

        Args:
            data_path: Path to data.yaml with train/val image paths and classes.
            base_model: Pretrained model to start from.
            epochs: Number of training epochs.
            image_size: Input image size for training.
            batch_size: Training batch size.
            freeze_layers: Number of backbone layers to freeze.
            project: Directory to save training outputs.
            name: Name for this training run.

        Returns:
            Path to the best model weights file.
        """
        logger.info("Starting detector training")
        logger.info("  Base model: %s", base_model)
        logger.info("  Data: %s", data_path)
        logger.info(
            "  Epochs: %d, Batch: %d, ImgSz: %d",
            epochs, batch_size, image_size,
        )

        model = YOLO(base_model)

        train_kwargs: dict = {
            "data": data_path,
            "epochs": epochs,
            "imgsz": image_size,
            "batch": batch_size,
            "project": project,
            "name": name,
            "device": "cpu",
            "workers": 0,
            "exist_ok": True,
        }

        if freeze_layers is not None:
            train_kwargs["freeze"] = freeze_layers

        model.train(**train_kwargs)

        best_path = Path(project) / name / "weights" / "best.pt"
        if best_path.exists():
            logger.info("Best weights saved: %s", best_path)
            return str(best_path)

        last_path = Path(project) / name / "weights" / "last.pt"
        if last_path.exists():
            logger.info("Last weights saved: %s", last_path)
            return str(last_path)

        raise FileNotFoundError(
            f"Training completed but weights not found in {project}/{name}/weights/"
        )

    @staticmethod
    def evaluate_model(
        model_path: str,
        data_path: str,
        task: str = "detect",
    ) -> dict:
        """Run validation on a trained model and return metrics.

        Args:
            model_path: Path to model weights.
            data_path: Path to data.yaml or dataset directory.
            task: "detect" or "classify".

        Returns:
            Dictionary of evaluation metrics (mAP, precision, recall, etc.).
        """
        model = YOLO(model_path)
        metrics = model.val(data=data_path, device="cpu", verbose=False)

        if task == "detect":
            return {
                "mAP50": float(metrics.box.map50),
                "mAP50-95": float(metrics.box.map),
                "precision": float(metrics.box.mp),
                "recall": float(metrics.box.mr),
            }
        else:
            return {
                "top1_accuracy": float(metrics.top1),
                "top5_accuracy": float(metrics.top5),
            }
