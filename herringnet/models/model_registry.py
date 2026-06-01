"""Model weight management: downloading, caching, and versioning.

Handles automatic download of pretrained model weights (e.g., Community
Fish Detector) and local caching so they are only downloaded once.
"""

from __future__ import annotations

import hashlib
import logging
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_MODELS_DIR = _PROJECT_ROOT / "models"

# Known model registry. URLs and checksums are populated as models
# are verified. For Ultralytics built-in models (yolov8n, etc.),
# the YOLO() constructor handles downloading automatically.
KNOWN_MODELS: dict[str, dict] = {
    "cfd": {
        "description": "Community Fish Detector (YOLOv5-based, 1.9M training images)",
        "url": None,  # Set after verifying the CFD release URL
        "filename": "cfd.pt",
        "license": "AGPL-3.0",
        "notes": (
            "Download manually from the CFD GitHub releases page: "
            "https://github.com/WildHackers/community-fish-detector/releases "
            "and place in the models/ directory."
        ),
    },
    "yolov8n": {
        "description": "YOLOv8 Nano detection model (COCO pretrained)",
        "url": "ultralytics-builtin",
        "filename": "yolov8n.pt",
        "license": "AGPL-3.0",
    },
    "yolov8s": {
        "description": "YOLOv8 Small detection model (COCO pretrained)",
        "url": "ultralytics-builtin",
        "filename": "yolov8s.pt",
        "license": "AGPL-3.0",
    },
    "yolov8n-cls": {
        "description": "YOLOv8 Nano classification model (ImageNet pretrained)",
        "url": "ultralytics-builtin",
        "filename": "yolov8n-cls.pt",
        "license": "AGPL-3.0",
    },
    "yolov8s-cls": {
        "description": "YOLOv8 Small classification model (ImageNet pretrained)",
        "url": "ultralytics-builtin",
        "filename": "yolov8s-cls.pt",
        "license": "AGPL-3.0",
    },
}


class ModelRegistry:
    """Manage downloading, caching, and retrieval of model weights.

    Model weights are stored in the models/ directory at the project root.
    Ultralytics built-in models are downloaded automatically by the YOLO
    constructor. Other models (like CFD) require manual download or have
    a URL configured for automatic download.
    """

    def __init__(self, models_dir: str | Path | None = None):
        self.models_dir = Path(models_dir) if models_dir else _DEFAULT_MODELS_DIR
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def get_model_path(self, model_name: str) -> str:
        """Get the local path to model weights, downloading if needed.

        For Ultralytics built-in models, returns the model name directly
        (the YOLO constructor handles the download). For other models,
        checks for local weights and downloads if a URL is available.

        Args:
            model_name: Model name from the registry or a direct file path.

        Returns:
            Path string suitable for passing to YOLO().

        Raises:
            FileNotFoundError: If the model is not found and cannot be
                downloaded automatically.
        """
        # If it is a direct path to an existing file, use it as-is
        if Path(model_name).exists():
            return model_name

        # Check the registry
        if model_name in KNOWN_MODELS:
            model_info = KNOWN_MODELS[model_name]

            # Ultralytics built-in models are handled by the YOLO constructor
            if model_info.get("url") == "ultralytics-builtin":
                return model_info["filename"]

            # Check if already downloaded locally
            local_path = self.models_dir / model_info["filename"]
            if local_path.exists():
                logger.info("Found cached model: %s", local_path)
                return str(local_path)

            # Try to download if URL is available
            url = model_info.get("url")
            if url is not None:
                return self._download_model(url, local_path)

            # No URL available, provide guidance
            notes = model_info.get("notes", "")
            raise FileNotFoundError(
                f"Model '{model_name}' not found at {local_path}. {notes}"
            )

        # Check if it is a filename in the models directory
        local_path = self.models_dir / model_name
        if local_path.exists():
            return str(local_path)

        # Try as a filename with .pt extension
        if not model_name.endswith(".pt"):
            local_path_pt = self.models_dir / f"{model_name}.pt"
            if local_path_pt.exists():
                return str(local_path_pt)

        raise FileNotFoundError(
            f"Model '{model_name}' not found. Check the models/ directory "
            f"or use a registered model name: {list(KNOWN_MODELS.keys())}"
        )

    def _download_model(self, url: str, destination: Path) -> str:
        """Download model weights from a URL.

        Args:
            url: URL to download from.
            destination: Local path to save the file.

        Returns:
            Path string to the downloaded file.
        """
        logger.info("Downloading model to %s ...", destination)
        destination.parent.mkdir(parents=True, exist_ok=True)

        try:
            urllib.request.urlretrieve(url, str(destination))
        except Exception as e:
            raise RuntimeError(
                f"Failed to download model from {url}: {e}"
            ) from e

        logger.info("Download complete: %s", destination)
        return str(destination)

    def list_models(self) -> dict[str, dict]:
        """List all known models and their availability.

        Returns:
            Dictionary mapping model names to their info and local status.
        """
        result = {}
        for name, info in KNOWN_MODELS.items():
            local_path = self.models_dir / info["filename"]
            result[name] = {
                **info,
                "available_locally": local_path.exists(),
                "local_path": str(local_path),
            }
        return result

    @staticmethod
    def verify_checksum(file_path: str | Path, expected_sha256: str) -> bool:
        """Verify a file's SHA256 checksum.

        Args:
            file_path: Path to the file to verify.
            expected_sha256: Expected SHA256 hex digest.

        Returns:
            True if the checksum matches.
        """
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest() == expected_sha256
