"""Detection-first pipeline: flag every frame that holds a fish, then count.

Stage 1 (base gate):  sensitive pass over every image        -> <out>/base
Stage 2 (dense):      --dense settings on the flagged frames  -> <out>/dense
Stage 3 (thorough):   --dense --thorough (SAHI), same frames  -> <out>/thorough

Detection matters more than the count, so the gate never silently drops a
frame: it records every detection down to a low confidence floor and TIERS
frames by how sure it is, instead of thresholding them away.

    confident  2+ detections at >= detect_conf, or any detection >= CONFIDENT_CONF
    review     exactly one detection at detect_conf..CONFIDENT_CONF (a lone
               moderate box; genuinely ambiguous between a distant fish and
               water-surface ripple, so worth a human glance)
    possible   only weak detections (below detect_conf, above the gate floor)
    static     only stationary objects (rocks/debris re-detected at the same
               pixels across many frames; see fishcount.sequence). Not fish,
               but kept visible instead of deleted.
    none       nothing at all

Outputs, in priority order:
    detections.csv        one row per frame: has_fish, tier, max confidence
    detected/<tier>/      annotated copies of every flagged frame, by tier
    summary.csv           base/dense/thorough counts for the counted frames
"""

from __future__ import annotations

import csv
import json
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from fishcount.batch import BatchSummary, run_batch
from fishcount.config import AppConfig
from fishcount.count import TOTAL_ROW_LABEL
from fishcount.detector import Detector
from fishcount.sequence import StaticMask, static_detections

# Builds a detector for a stage: (config, thorough) -> Detector.
DetectorFactory = Callable[[AppConfig, bool], Detector]

# A single detection at or above this confidence is enough to call the frame
# confident on its own.
CONFIDENT_CONF = 0.50

TIERS = ("confident", "review", "possible", "static", "none")


@dataclass(slots=True)
class FrameDetection:
    """What the base gate saw in one frame."""

    file: str
    n_detect: int  # detections at >= detect_conf (after the box-size filter)
    n_weak: int  # detections below detect_conf but above the gate floor
    max_conf: float
    tier: str
    n_static: int = 0  # stationary-object detections ignored by the tiering

    @property
    def has_fish(self) -> bool:
        return self.tier not in ("none", "static")


@dataclass(slots=True)
class PipelineSummary:
    out_dir: Path
    min_count: int
    base: BatchSummary
    dense: BatchSummary | None
    thorough: BatchSummary | None
    frames: list[FrameDetection] = field(default_factory=list)
    counted: int = 0

    def tier_count(self, tier: str) -> int:
        return sum(1 for frame in self.frames if frame.tier == tier)

    @property
    def flagged(self) -> int:
        return sum(1 for frame in self.frames if frame.has_fish)


def run_pipeline(
    input_dir: Path,
    out_dir: Path,
    *,
    base_config: AppConfig,
    dense_config: AppConfig,
    make_detector: DetectorFactory,
    min_count: int = 1,
    detect_conf: float = 0.25,
    count_possible: bool = False,
    static_min_frames: int | None = 8,
    write_images: bool = True,
    show_progress: bool = True,
    model_path: Path | None = None,
) -> PipelineSummary:
    """Run the base gate, tier every frame, then count the flagged frames.

    Frames with at least `min_count` detections at or above `detect_conf` go to
    the dense and thorough passes; `count_possible=True` also counts the
    possible tier. `base_config.conf` is the gate floor and should sit below
    `detect_conf` so weak detections are recorded rather than lost.
    `static_min_frames` demotes stationary objects (a box recurring in that
    many distinct frames) to the static tier; None disables the filter.
    """
    base_dir = out_dir / "base"
    base_summary = run_batch(
        input_dir,
        base_dir,
        base_config,
        make_detector(base_config, False),
        write_images=write_images,
        show_progress=show_progress,
        model_path=model_path,
    )

    static: StaticMask = set()
    if static_min_frames is not None:
        static = static_detections(base_dir / "results.json", min_frames=static_min_frames)
    frames = tier_frames(base_dir / "results.json", detect_conf=detect_conf, static=static)
    write_detections_csv(frames, out_dir / "detections.csv")
    if write_images:
        _copy_flagged(frames, base_dir / "annotated", out_dir / "detected")

    to_count = [
        frame
        for frame in frames
        if frame.n_detect >= min_count or (count_possible and frame.tier == "possible")
    ]
    subset = [(input_dir / frame.file).resolve() for frame in to_count]

    dense_summary: BatchSummary | None = None
    thorough_summary: BatchSummary | None = None
    if subset:
        dense_summary = run_batch(
            input_dir,
            out_dir / "dense",
            dense_config,
            make_detector(dense_config, False),
            write_images=write_images,
            show_progress=show_progress,
            model_path=model_path,
            only=subset,
        )
        thorough_summary = run_batch(
            input_dir,
            out_dir / "thorough",
            dense_config,
            make_detector(dense_config, True),
            write_images=write_images,
            show_progress=show_progress,
            model_path=model_path,
            only=subset,
        )

    _write_summary(out_dir / "summary.csv", out_dir, to_count)
    return PipelineSummary(
        out_dir=out_dir,
        min_count=min_count,
        base=base_summary,
        dense=dense_summary,
        thorough=thorough_summary,
        frames=frames,
        counted=len(to_count),
    )


def tier_frames(
    results_json: Path, *, detect_conf: float, static: StaticMask | None = None
) -> list[FrameDetection]:
    """Tier every frame in a stage's results.json by what the gate saw.

    Detections listed in `static` are stationary objects and are excluded from
    the tiering; a frame with only static detections gets the "static" tier.
    """
    static = static or set()
    payload = json.loads(results_json.read_text(encoding="utf-8"))
    frames: list[FrameDetection] = []
    for frame_index, image in enumerate(payload["images"]):
        if "error" in image:
            frames.append(FrameDetection(image["file"], 0, 0, 0.0, "none"))
            continue
        confs: list[float] = []
        n_static = 0
        for det_index, det in enumerate(image.get("detections", [])):
            if (frame_index, det_index) in static:
                n_static += 1
            else:
                confs.append(float(det["confidence"]))
        n_detect = sum(1 for conf in confs if conf >= detect_conf)
        n_weak = len(confs) - n_detect
        max_conf = max(confs, default=0.0)
        tier = _tier(n_detect, n_weak, max_conf, n_static)
        frames.append(FrameDetection(image["file"], n_detect, n_weak, max_conf, tier, n_static))
    return frames


def _tier(n_detect: int, n_weak: int, max_conf: float, n_static: int = 0) -> str:
    if n_detect >= 2 or max_conf >= CONFIDENT_CONF:
        return "confident"
    if n_detect == 1:
        return "review"
    if n_weak >= 1:
        return "possible"
    if n_static >= 1:
        return "static"
    return "none"


def write_detections_csv(frames: list[FrameDetection], path: Path) -> None:
    """The headline output: one row per frame, flagged frames first."""
    order = {tier: index for index, tier in enumerate(TIERS)}
    ordered = sorted(frames, key=lambda frame: (order[frame.tier], frame.file))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["filename", "has_fish", "tier", "max_confidence", "detections", "weak", "static"]
        )
        for frame in ordered:
            writer.writerow(
                [
                    frame.file,
                    "yes" if frame.has_fish else "no",
                    frame.tier,
                    f"{frame.max_conf:.3f}",
                    frame.n_detect,
                    frame.n_weak,
                    frame.n_static,
                ]
            )


def _copy_flagged(frames: list[FrameDetection], annotated_dir: Path, detected_dir: Path) -> None:
    """Copy each flagged frame's annotated image into detected/<tier>/.

    Static-only frames are copied too (detected/static/) so the stationary
    object filter can be audited; they are not counted as fish.
    """
    for frame in frames:
        if frame.tier == "none":
            continue
        source = annotated_dir / frame.file
        if not source.is_file():
            source = source.with_suffix(".png")  # annotate() falls back to PNG
            if not source.is_file():
                continue
        dest = (detected_dir / frame.tier / frame.file).with_suffix(source.suffix)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)


def _read_counts(counts_csv: Path) -> list[tuple[str, int]]:
    """(filename, count) rows from a counts.csv, excluding the TOTAL row."""
    if not counts_csv.is_file():
        return []
    rows: list[tuple[str, int]] = []
    with counts_csv.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            name = row["filename"]
            if name == TOTAL_ROW_LABEL:
                continue
            rows.append((name, int(row["fish_count"])))
    return rows


def _write_summary(path: Path, out_dir: Path, counted: list[FrameDetection]) -> None:
    """Per-frame tier and base/dense/thorough counts for the counted frames."""
    base = dict(_read_counts(out_dir / "base" / "counts.csv"))
    dense = dict(_read_counts(out_dir / "dense" / "counts.csv"))
    thorough = dict(_read_counts(out_dir / "thorough" / "counts.csv"))
    totals = [0, 0, 0]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filename", "tier", "base_count", "dense_count", "thorough_count"])
        for frame in counted:
            counts = [
                base.get(frame.file, 0),
                dense.get(frame.file, 0),
                thorough.get(frame.file, 0),
            ]
            totals = [t + c for t, c in zip(totals, counts, strict=True)]
            writer.writerow([frame.file, frame.tier, *counts])
        writer.writerow([TOTAL_ROW_LABEL, "", *totals])
