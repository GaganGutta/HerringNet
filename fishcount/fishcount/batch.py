"""Folder pipeline: discover images, run batched detection, record every detection.

Each image is read from disk exactly once; the same decoded array feeds
inference, the per-frame blur score, and annotation. Input files are never
written to. Output is detections only: results.json (every frame, every
detection above the floor, plus the blur score) and annotated copies of the
frames that have detections. Tiering happens afterwards in fishcount.classify.
"""

from __future__ import annotations

import io
import json
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from fishcount.config import AppConfig
from fishcount.detector import Detection, Detector, ImageArray
from fishcount.draw import annotate

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}

# EXIF orientation tag values -> transform that upright-orients the decoded pixels
# (cv2.imdecode ignores EXIF, unlike cv2.imread, so this is applied by hand).
_EXIF_TRANSFORMS: dict[int, Callable[[ImageArray], ImageArray]] = {
    2: lambda a: cv2.flip(a, 1),
    3: lambda a: cv2.rotate(a, cv2.ROTATE_180),
    4: lambda a: cv2.flip(a, 0),
    5: lambda a: cv2.transpose(a),
    6: lambda a: cv2.rotate(a, cv2.ROTATE_90_CLOCKWISE),
    7: lambda a: cv2.flip(cv2.transpose(a), -1),
    8: lambda a: cv2.rotate(a, cv2.ROTATE_90_COUNTERCLOCKWISE),
}


class NoImagesFoundError(RuntimeError):
    """The input folder contains no files with a supported image extension."""


@dataclass(slots=True)
class ImageResult:
    """Everything recorded for one frame."""

    file: str
    width: int = 0
    height: int = 0
    blur: float | None = None  # variance of Laplacian at quarter resolution
    detections: list[Detection] = field(default_factory=list)
    error: str | None = None


@dataclass(slots=True)
class BatchSummary:
    """What happened during one run, for the console summary."""

    input_dir: Path
    out_dir: Path
    processed: int
    skipped: int
    seconds: float

    @property
    def seconds_per_image(self) -> float:
        return self.seconds / self.processed if self.processed else 0.0


def discover_images(input_dir: Path, exclude_dir: Path | None = None) -> list[Path]:
    """All images under input_dir (recursive), in sorted order.

    Anything under exclude_dir is skipped so a previous run's output inside the
    input folder is never re-detected.
    """
    exclude: Path | None = None
    if exclude_dir is not None:
        resolved = exclude_dir.resolve()
        if _is_within(resolved, input_dir.resolve()):
            exclude = resolved
    images: list[Path] = []
    for path in sorted(input_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if exclude is not None and _is_within(path.resolve(), exclude):
            continue
        images.append(path)
    return images


def load_image(path: Path) -> ImageArray | None:
    """Read one image with a single disk read, honoring EXIF orientation.

    Returns None for unreadable or corrupt files. Bytes go through cv2.imdecode
    instead of cv2.imread so non-ASCII Windows paths work.
    """
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if not data:
        return None
    array = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if array is None:
        return None
    transform = _EXIF_TRANSFORMS.get(_exif_orientation(data))
    result: ImageArray = transform(array) if transform is not None else array
    return result


def blur_score(array: ImageArray) -> float:
    """Variance of the Laplacian at quarter resolution.

    The absolute value tracks turbidity and lighting as much as focus, so it is
    only meaningful relative to the same run's distribution (fishcount.classify
    normalizes by percentile per folder).
    """
    height, width = array.shape[:2]
    small = cv2.resize(array, (max(1, width // 4), max(1, height // 4)))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def run_batch(
    input_dir: Path,
    out_dir: Path,
    config: AppConfig,
    detector: Detector,
    *,
    write_images: bool = True,
    show_progress: bool = True,
    model_path: Path | None = None,
) -> BatchSummary:
    """Detect in every image under input_dir; write results.json and annotated frames.

    Inference runs in batches of config.batch_size. Unreadable images are
    skipped with a warning and recorded in results.json. Annotated copies are
    written (to out_dir/annotated/) only for frames that have detections.
    """
    images = discover_images(input_dir, exclude_dir=out_dir)
    if not images:
        extensions = ", ".join(sorted(IMAGE_EXTENSIONS))
        raise NoImagesFoundError(f"No images found under {input_dir} (extensions: {extensions})")

    out_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir = out_dir / "annotated"
    results: list[ImageResult] = []
    start = time.perf_counter()
    progress = tqdm(total=len(images), unit="img", desc="Detecting", disable=not show_progress)
    try:
        for chunk in _chunks(images, config.batch_size):
            per_path: dict[Path, ImageResult] = {}
            loaded: list[tuple[Path, ImageArray]] = []
            for path in chunk:
                array = load_image(path)
                if array is None:
                    tqdm.write(f"WARNING: skipping unreadable image: {path}")
                    per_path[path] = ImageResult(
                        file=path.relative_to(input_dir).as_posix(), error="unreadable image"
                    )
                    progress.update(1)
                else:
                    loaded.append((path, array))
            if loaded:
                batch_detections = detector.detect_batch([array for _, array in loaded])
                for (path, array), detections in zip(loaded, batch_detections, strict=True):
                    relative = path.relative_to(input_dir)
                    height, width = array.shape[:2]
                    if write_images and detections:
                        _write_annotated(annotate(array, detections), annotated_dir / relative)
                    per_path[path] = ImageResult(
                        relative.as_posix(), width, height, blur_score(array), detections
                    )
                    progress.update(1)
            results.extend(per_path[path] for path in chunk)
    finally:
        progress.close()
    seconds = time.perf_counter() - start

    write_results_json(
        results,
        out_dir / "results.json",
        input_dir=input_dir,
        model_path=model_path if model_path is not None else config.model_path,
        conf=config.conf,
        imgsz=config.imgsz,
    )
    processed = sum(1 for result in results if result.error is None)
    return BatchSummary(
        input_dir=input_dir,
        out_dir=out_dir,
        processed=processed,
        skipped=len(results) - processed,
        seconds=seconds,
    )


def write_results_json(
    results: Sequence[ImageResult],
    path: Path,
    *,
    input_dir: Path,
    model_path: Path,
    conf: float,
    imgsz: int,
) -> None:
    """Full per-detection dump: boxes as [x1, y1, x2, y2] pixels plus confidence."""
    payload = {
        "input": str(input_dir),
        "model": str(model_path),
        "conf": conf,
        "imgsz": imgsz,
        "images": [_image_entry(result) for result in results],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _image_entry(result: ImageResult) -> dict[str, object]:
    if result.error is not None:
        return {"file": result.file, "error": result.error}
    return {
        "file": result.file,
        "width": result.width,
        "height": result.height,
        "blur": round(result.blur, 1) if result.blur is not None else None,
        "detections": [
            {"box": list(detection.int_box()), "confidence": round(detection.confidence, 3)}
            for detection in result.detections
        ],
    }


def _write_annotated(image: ImageArray, dest: Path) -> None:
    """Encode with the original extension (PNG as a fallback); unicode-path safe."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(dest.suffix.lower(), image)
    if not ok:
        dest = dest.with_suffix(".png")
        ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise OSError(f"could not encode annotated image for {dest}")
    encoded.tofile(str(dest))


def _exif_orientation(data: bytes) -> int:
    try:
        with Image.open(io.BytesIO(data)) as image:
            return int(image.getexif().get(0x0112, 1))
    except Exception:
        return 1  # any EXIF weirdness just means "no rotation"


def _is_within(path: Path, ancestor: Path) -> bool:
    try:
        path.relative_to(ancestor)
    except ValueError:
        return False
    return True


def _chunks(items: Sequence[Path], size: int) -> Iterator[list[Path]]:
    for start in range(0, len(items), size):
        yield list(items[start : start + size])
