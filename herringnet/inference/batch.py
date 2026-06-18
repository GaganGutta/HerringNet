"""Fast folder-in, results-out batch detection.

Built for the common case: point at a folder of camera-trap images and
get annotated images plus a counts CSV back, quickly. It deliberately
does NOT use SAHI sliced inference (that runs ~dozens of model passes per
image and is far too slow for bulk work); it runs one batched pass over
the images instead, reading each image from disk exactly once.

For maximum small-fish recall on a handful of images, use
``herringnet detect ... --sahi`` instead. For getting through a whole
folder, use this.
"""

from __future__ import annotations

import csv
import json
import logging
import time
from collections.abc import Callable
from pathlib import Path

import cv2

from herringnet.config import HerringNetConfig
from herringnet.inference.result_types import (
    Detection,
    FrameResult,
    PipelineDetection,
)
from herringnet.models.detector import FishDetector
from herringnet.visualization.draw_detections import (
    draw_count_overlay,
    draw_detections,
)

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def run_batch(
    config: HerringNetConfig,
    input_dir: str | Path,
    output_dir: str | Path | None = None,
    batch_size: int = 8,
    save_annotated: bool = True,
    save_empty: bool = False,
    progress: Callable[[int, int], None] | None = None,
) -> dict:
    """Detect fish across every image in a folder and write results.

    Args:
        config: HerringNet configuration (SAHI is ignored here by design).
        input_dir: Folder of images (searched recursively).
        output_dir: Where to write results. Defaults to
            ``<config.output.output_dir>/<input folder name>``.
        batch_size: Images per model pass.
        save_annotated: Write annotated copies of images with detections.
        save_empty: Also write annotated copies of images with no fish.
        progress: Optional callback(done, total) for progress reporting.

    Returns:
        Summary dict: images, images_with_fish, total_fish, output_dir,
        seconds, seconds_per_image.
    """
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")

    images = sorted(
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise ValueError(f"No images found in {input_dir}")

    out = Path(output_dir) if output_dir else (
        Path(config.output.output_dir) / input_dir.name
    )
    out.mkdir(parents=True, exist_ok=True)
    annotated_dir = out / "annotated"
    if save_annotated:
        annotated_dir.mkdir(parents=True, exist_ok=True)

    detector = FishDetector(config.detector)
    dc = config.detector

    results: list[FrameResult] = []
    total_fish = 0
    images_with_fish = 0
    start = time.time()

    for i in range(0, len(images), batch_size):
        chunk = images[i : i + batch_size]
        arrays = []
        valid_paths = []
        for p in chunk:
            img = cv2.imread(str(p))
            if img is None:
                logger.warning("Could not read image, skipping: %s", p)
                continue
            arrays.append(img)
            valid_paths.append(p)
        if not arrays:
            continue

        # One batched, full-image inference pass (no SAHI).
        preds = detector.model.predict(
            source=arrays,
            conf=dc.confidence_threshold,
            iou=dc.iou_threshold,
            imgsz=dc.image_size,
            device=dc.device,
            verbose=False,
        )

        for path, img, result in zip(valid_paths, arrays, preds):
            detections = _parse_result(result)
            pdets = [PipelineDetection(detection=d) for d in detections]
            results.append(
                FrameResult(source_path=str(path), detections=pdets)
            )
            total_fish += len(detections)
            if detections:
                images_with_fish += 1

            if save_annotated and (detections or save_empty):
                annotated = draw_detections(img, pdets)
                draw_count_overlay(annotated, len(detections))
                cv2.imwrite(
                    str(annotated_dir / f"{path.stem}_annotated{path.suffix}"),
                    annotated,
                )

        if progress is not None:
            progress(min(i + batch_size, len(images)), len(images))

    _write_json(results, out / "results.json")
    _write_csv(results, out / "counts.csv")

    elapsed = time.time() - start
    return {
        "images": len(results),
        "images_with_fish": images_with_fish,
        "total_fish": total_fish,
        "output_dir": str(out),
        "seconds": round(elapsed, 1),
        "seconds_per_image": round(elapsed / max(len(results), 1), 2),
    }


def _parse_result(result) -> list[Detection]:
    """Convert one Ultralytics result into Detection objects."""
    detections: list[Detection] = []
    if result.boxes is None:
        return detections
    boxes = result.boxes
    for j in range(len(boxes)):
        xy = boxes.xyxy[j].cpu().numpy()
        cls_id = int(boxes.cls[j].cpu().numpy())
        detections.append(
            Detection(
                bbox=(float(xy[0]), float(xy[1]), float(xy[2]), float(xy[3])),
                confidence=float(boxes.conf[j].cpu().numpy()),
                class_id=cls_id,
                class_name=result.names.get(cls_id, "fish"),
            )
        )
    return detections


def _write_json(results: list[FrameResult], path: Path) -> None:
    with open(path, "w") as f:
        json.dump({"frames": [r.to_dict() for r in results]}, f, indent=2)


def _write_csv(results: list[FrameResult], path: Path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "fish_count"])
        for r in results:
            writer.writerow([Path(r.source_path).name, r.fish_count])
        writer.writerow([])
        writer.writerow(["TOTAL", sum(r.fish_count for r in results)])
