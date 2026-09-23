"""Folder pipeline: discover images, run batched detection, journal every frame.

Each image is read from disk exactly once; the same decoded array feeds
inference and the per-frame statistics (blur and brightness). Input files are
never written to.

Nothing accumulates. Frames are streamed a batch at a time and appended to the
journal as each batch finishes, so memory is flat whether the folder holds
fifty frames or fifty thousand, and a run killed halfway through keeps
everything it had already recorded. Re-running picks up where it left off.

Detection is all that happens here. The CSVs and the annotated images are
written afterwards by fishcount.report, which reads the journal.
"""

from __future__ import annotations

import io
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from fishcount.config import AppConfig
from fishcount.detector import Detection, Detector, ImageArray, default_batch_size
from fishcount.journal import (
    JOURNAL_FILENAME,
    JournalWriter,
    ResumeMismatchError,
    RunParams,
    completed_frames,
    describe_mismatch,
    read_params,
    write_params,
)

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

_LOG_EVERY = 100  # frames between throughput lines, for when output is a log file


class NoImagesFoundError(RuntimeError):
    """The input folder contains no files with a supported image extension."""


@dataclass(frozen=True, slots=True)
class FrameStats:
    """Per-frame image statistics, both measured at quarter resolution."""

    blur: float  # variance of the Laplacian
    brightness: float  # mean grayscale value, 0-255


@dataclass(slots=True)
class BatchSummary:
    """What happened during one run, for the console summary."""

    input_dir: Path
    out_dir: Path
    total: int  # frames in the input folder
    already_done: int  # frames the journal already held, skipped this run
    processed: int  # frames detected in this run
    skipped: int  # frames this run could not read
    seconds: float
    conf: float
    device: str
    oom_splits: int = 0

    @property
    def seconds_per_image(self) -> float:
        return self.seconds / self.processed if self.processed else 0.0

    @property
    def images_per_second(self) -> float:
        return self.processed / self.seconds if self.seconds > 0 else 0.0


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


def frame_stats(array: ImageArray) -> FrameStats:
    """Blur and brightness for one frame, from a single quarter-resolution pass.

    Both are recorded, never acted on: they exist so that evaluation can break
    detector performance down by condition. Blur is the variance of the
    Laplacian, whose absolute value tracks turbidity and lighting as much as
    focus; brightness is the mean grayscale value.
    """
    height, width = array.shape[:2]
    small = cv2.resize(array, (max(1, width // 4), max(1, height // 4)))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return FrameStats(float(cv2.Laplacian(gray, cv2.CV_64F).var()), float(gray.mean()))


def run_batch(
    input_dir: Path,
    out_dir: Path,
    config: AppConfig,
    detector: Detector,
    *,
    show_progress: bool = True,
    model_path: Path | None = None,
    resume: bool = True,
) -> BatchSummary:
    """Detect in every image under input_dir, appending each frame to the journal.

    With `resume` (the default), frames already in the journal are skipped and
    the run continues from where a previous one stopped; the settings the
    journal was built under must match, or ResumeMismatchError is raised rather
    than mixing results. Without it, the journal is discarded and rebuilt.
    """
    images = discover_images(input_dir, exclude_dir=out_dir)
    if not images:
        extensions = ", ".join(sorted(IMAGE_EXTENSIONS))
        raise NoImagesFoundError(f"No images found under {input_dir} (extensions: {extensions})")

    out_dir.mkdir(parents=True, exist_ok=True)
    journal = out_dir / JOURNAL_FILENAME
    params = RunParams(
        model=str(model_path if model_path is not None else config.model_path),
        conf=config.conf,
        iou=config.iou,
        imgsz=config.imgsz,
        max_det=config.max_det,
    )

    done: set[str] = set()
    if resume:
        stored = read_params(out_dir)
        if stored is not None and stored != params:
            raise ResumeMismatchError(describe_mismatch(stored, params))
        done = completed_frames(journal)
    else:
        journal.unlink(missing_ok=True)
    write_params(out_dir, params)

    pending = [path for path in images if path.relative_to(input_dir).as_posix() not in done]
    device = getattr(detector, "device", "cpu")
    batch_size = config.batch_size if config.batch_size is not None else default_batch_size(device)

    processed = 0
    unreadable = 0
    start = time.perf_counter()
    progress = tqdm(
        total=len(pending), unit="img", desc="Detecting", disable=not show_progress, smoothing=0.1
    )
    try:
        with JournalWriter(journal) as writer:
            for chunk in _chunks(pending, batch_size):
                loaded: list[tuple[Path, ImageArray]] = []
                for path in chunk:
                    array = load_image(path)
                    if array is None:
                        tqdm.write(f"WARNING: skipping unreadable image: {path}")
                        writer.append(
                            {
                                "file": path.relative_to(input_dir).as_posix(),
                                "error": "unreadable image",
                            }
                        )
                        unreadable += 1
                        progress.update(1)
                    else:
                        loaded.append((path, array))
                if loaded:
                    detections = detector.detect_batch([array for _, array in loaded])
                    for (path, array), boxes in zip(loaded, detections, strict=True):
                        writer.append(_entry(path.relative_to(input_dir), array, boxes))
                        processed += 1
                        progress.update(1)
                writer.flush()  # a kill after this point cannot lose this batch
                _log_throughput(processed + unreadable, len(pending), start, show_progress)
    finally:
        progress.close()

    return BatchSummary(
        input_dir=input_dir,
        out_dir=out_dir,
        total=len(images),
        already_done=len(images) - len(pending),
        processed=processed,
        skipped=unreadable,
        seconds=time.perf_counter() - start,
        conf=config.conf,
        device=device,
        oom_splits=getattr(detector, "oom_splits", 0),
    )


def _entry(relative: Path, array: ImageArray, detections: Sequence[Detection]) -> dict[str, object]:
    """One frame's journal record: boxes as [x1, y1, x2, y2] pixels, plus stats."""
    height, width = array.shape[:2]
    stats = frame_stats(array)
    return {
        "file": relative.as_posix(),
        "width": width,
        "height": height,
        "blur": round(stats.blur, 1),
        "brightness": round(stats.brightness, 1),
        "detections": [
            {"box": list(detection.int_box()), "confidence": round(detection.confidence, 3)}
            for detection in detections
        ],
    }


def _log_throughput(done: int, total: int, start: float, show_progress: bool) -> None:
    """Throughput and ETA, at intervals, so a redirected log still shows progress."""
    if not show_progress or done == 0 or total == 0:
        return
    if done % _LOG_EVERY and done != total:
        return
    elapsed = time.perf_counter() - start
    rate = done / elapsed if elapsed > 0 else 0.0
    if rate <= 0:
        tqdm.write(f"  {done}/{total} frames")
        return
    tqdm.write(
        f"  {done}/{total} frames | {rate:.2f} img/s ({1 / rate:.2f} s/frame) "
        f"| elapsed {_duration(elapsed)} | ETA {_duration((total - done) / rate)}"
    )


def _duration(seconds: float) -> str:
    """Compact h/m/s, because a 15,000-frame ETA in seconds means nothing."""
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


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
