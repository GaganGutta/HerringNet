"""Dataset management for downloading and organizing training data.

Supports downloading datasets from Roboflow Universe and organizing
them into the directory structures expected by Ultralytics for
training detection and classification models.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class DatasetManager:
    """Download and organize datasets from Roboflow and other sources.

    Args:
        data_dir: Base directory for storing datasets.
        roboflow_api_key: Roboflow API key. Can also be set via the
            ROBOFLOW_API_KEY environment variable.
    """

    def __init__(
        self,
        data_dir: str = "data/raw",
        roboflow_api_key: str | None = None,
    ):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._api_key = roboflow_api_key or os.environ.get("ROBOFLOW_API_KEY")

    def download_roboflow_dataset(
        self,
        workspace: str,
        project: str,
        version: int = 1,
        export_format: str = "yolov8",
    ) -> str:
        """Download a dataset from Roboflow Universe.

        Args:
            workspace: Roboflow workspace name.
            project: Roboflow project name.
            version: Dataset version number.
            export_format: Export format (e.g., "yolov8", "coco").

        Returns:
            Path to the downloaded dataset directory.

        Raises:
            ValueError: If no Roboflow API key is configured.
            RuntimeError: If the download fails.
        """
        if not self._api_key:
            raise ValueError(
                "Roboflow API key is required. Set ROBOFLOW_API_KEY env var "
                "or pass roboflow_api_key to DatasetManager."
            )

        try:
            from roboflow import Roboflow
        except ImportError:
            raise ImportError(
                "roboflow package is required. Install with: pip install roboflow"
            )

        logger.info(
            "Downloading %s/%s v%d in %s format",
            workspace, project, version, export_format,
        )

        rf = Roboflow(api_key=self._api_key)
        rf_project = rf.workspace(workspace).project(project)
        dataset = rf_project.version(version).download(
            export_format,
            location=str(self.data_dir / f"{project}-v{version}"),
        )

        dataset_path = str(self.data_dir / f"{project}-v{version}")
        logger.info("Dataset downloaded to: %s", dataset_path)
        return dataset_path

    def prepare_classification_dataset(
        self,
        source_dirs: list[str],
        output_dir: str,
        train_ratio: float = 0.8,
        val_ratio: float = 0.15,
        test_ratio: float = 0.05,
    ) -> str:
        """Organize images into ImageNet-style classification structure.

        Creates train/val/test splits with class subfolders. This is the
        structure expected by YOLOv8 for classification training.

        Expected output structure:
            output_dir/
                train/
                    river_herring/
                    bass/
                    ...
                val/
                    river_herring/
                    bass/
                    ...
                test/
                    river_herring/
                    bass/
                    ...

        Args:
            source_dirs: List of source dataset directories to merge.
            output_dir: Directory to create the organized dataset.
            train_ratio: Fraction of data for training.
            val_ratio: Fraction of data for validation.
            test_ratio: Fraction of data for testing.

        Returns:
            Path to the output dataset directory.
        """
        import random
        import shutil

        output_path = Path(output_dir)
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.01:
            raise ValueError(
                "train_ratio + val_ratio + test_ratio must equal 1.0"
            )

        # Collect all images by class
        class_images: dict[str, list[Path]] = {}
        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

        for source in source_dirs:
            source_path = Path(source)
            if not source_path.exists():
                logger.warning("Source directory not found: %s", source)
                continue

            # Look for class subdirectories
            for class_dir in source_path.iterdir():
                if not class_dir.is_dir():
                    continue
                class_name = class_dir.name
                images = [
                    f for f in class_dir.iterdir()
                    if f.suffix.lower() in image_exts
                ]
                if images:
                    if class_name not in class_images:
                        class_images[class_name] = []
                    class_images[class_name].extend(images)

        if not class_images:
            raise ValueError("No class directories with images found in source dirs")

        # Create splits
        for class_name, images in class_images.items():
            random.shuffle(images)
            n = len(images)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)

            splits = {
                "train": images[:n_train],
                "val": images[n_train : n_train + n_val],
                "test": images[n_train + n_val :],
            }

            for split_name, split_images in splits.items():
                dest_dir = output_path / split_name / class_name
                dest_dir.mkdir(parents=True, exist_ok=True)
                for img in split_images:
                    shutil.copy2(str(img), str(dest_dir / img.name))

            logger.info(
                "Class '%s': %d train, %d val, %d test",
                class_name,
                len(splits["train"]),
                len(splits["val"]),
                len(splits["test"]),
            )

        logger.info("Classification dataset prepared at: %s", output_path)
        return str(output_path)
