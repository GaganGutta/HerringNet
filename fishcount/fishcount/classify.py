"""Tier every frame's detections into confident / under_review / not_confident.

Detection quality is the product, so the rule is demote-never-delete: every
detection above the run's confidence floor appears in exactly one of the three
output CSVs. Three signals demote; none of them removes a detection:

- static:    the box recurs at the same pixels in many distinct frames, so it
             is a stationary object (rock, shell, debris), not a fish
             (fishcount.sequence).
- oversized: the box covers more than OVERSIZE_FRAC of the frame. Real fish
             here are a few percent of the frame; frame-filling boxes are
             murky water misread as one giant fish. Kept, routed to review.
- blur:      the frame is blurrier than most of its own folder (percentile
             normalized per run, because absolute sharpness tracks turbidity
             and lighting as much as focus). A blurry frame's tier is capped
             at under_review UNLESS it has >= SCHOOL_EXEMPT real detections.
             The exemption exists because dense fish schools are blurry (the
             fish move); it was fit on six school frames from the 104GOPRO and
             120GOPRO deployments and should be revisited on new sites.

Frame tier, from the non-static, non-oversized detections:
    confident      2+ detections at >= DETECT_CONF, or any at >= STRONG_CONF,
                   in a sharp frame (or a blurry frame with the school exemption)
    under_review   one moderate detection; or confident evidence in a blurry
                   frame; or an oversized box at moderate confidence
    not_confident  only weak (floor..DETECT_CONF), static, or weak-oversized
                   detections

Frames with no detections at all appear in no file.
"""

from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from fishcount.sequence import StaticMask, static_detections

DETECT_CONF = 0.25  # a detection at or above this is treated as a real fish
STRONG_CONF = 0.50  # a single detection at or above this makes a frame confident
OVERSIZE_FRAC = 0.10  # boxes covering more of the frame than this are demoted
SCHOOL_EXEMPT = 4  # blurry frames with this many real detections escape the cap

TIERS = ("confident", "under_review", "not_confident")

CSV_HEADER = ["filename", "x1", "y1", "x2", "y2", "confidence", "note", "frame_blurry"]


@dataclass(slots=True)
class FrameClass:
    """One frame's classification and its detection rows."""

    file: str
    tier: str | None  # None: no detections, frame appears in no file
    blurry: bool
    blur_capped: bool
    school_exempt: bool
    rows: list[list[str]] = field(default_factory=list)


@dataclass(slots=True)
class ClassifySummary:
    out_dir: Path
    blur_threshold: float | None
    frames_per_tier: dict[str, int]
    detections_per_tier: dict[str, int]
    no_detection_frames: int
    static_detections: int
    blur_capped: int
    school_exempt: int


def classify(
    out_dir: Path,
    *,
    floor: float,
    blur_percentile: float = 25.0,
    static_min_frames: int = 8,
    move_images: bool = True,
) -> ClassifySummary:
    """Read out_dir/results.json, tier every frame, write the three CSVs.

    `floor` is the run's confidence floor (detections between floor and
    DETECT_CONF are the weak band). `blur_percentile` 0 disables the blur cap;
    `static_min_frames` 0 disables the static demotion. With `move_images`,
    annotated images move from out_dir/annotated/ into the tier subfolders.
    """
    results_json = out_dir / "results.json"
    payload = json.loads(results_json.read_text(encoding="utf-8"))

    static: StaticMask = set()
    if static_min_frames > 0:
        static = static_detections(results_json, min_frames=static_min_frames)

    blur_threshold: float | None = None
    if blur_percentile > 0:
        values = [image["blur"] for image in payload["images"] if image.get("blur") is not None]
        if values:
            blur_threshold = float(np.percentile(values, blur_percentile))

    frames = [
        _classify_frame(frame_index, image, static, blur_threshold)
        for frame_index, image in enumerate(payload["images"])
    ]

    _write_tier_csvs(frames, out_dir)
    if move_images:
        _move_annotated(frames, out_dir)

    summary = ClassifySummary(
        out_dir=out_dir,
        blur_threshold=blur_threshold,
        frames_per_tier={tier: 0 for tier in TIERS},
        detections_per_tier={tier: 0 for tier in TIERS},
        no_detection_frames=0,
        static_detections=len(static),
        blur_capped=0,
        school_exempt=0,
    )
    for frame in frames:
        if frame.tier is None:
            summary.no_detection_frames += 1
            continue
        summary.frames_per_tier[frame.tier] += 1
        summary.detections_per_tier[frame.tier] += len(frame.rows)
        summary.blur_capped += frame.blur_capped
        summary.school_exempt += frame.school_exempt
    return summary


def _classify_frame(
    frame_index: int,
    image: dict,
    static: StaticMask,
    blur_threshold: float | None,
) -> FrameClass:
    if "error" in image:
        return FrameClass(image["file"], None, False, False, False)
    detections = image.get("detections", [])
    if not detections:
        return FrameClass(image["file"], None, False, False, False)

    area = float(image["width"] * image["height"])
    blur = image.get("blur")
    blurry = blur_threshold is not None and blur is not None and blur < blur_threshold

    rows: list[tuple[list[str], bool, bool, float]] = []
    for det_index, det in enumerate(detections):
        x1, y1, x2, y2 = det["box"]
        conf = float(det["confidence"])
        is_static = (frame_index, det_index) in static
        oversized = area > 0 and ((x2 - x1) * (y2 - y1)) / area > OVERSIZE_FRAC
        flags = []
        if is_static:
            flags.append("static")
        if oversized:
            flags.append("oversized")
        if conf < DETECT_CONF:
            flags.append("weak")
        rows.append(
            (
                [
                    image["file"],
                    str(x1),
                    str(y1),
                    str(x2),
                    str(y2),
                    f"{conf:.3f}",
                    ";".join(flags),
                    "yes" if blurry else "no",
                ],
                is_static,
                oversized,
                conf,
            )
        )

    effective = [conf for _, is_static, oversized, conf in rows if not is_static and not oversized]
    n_real = sum(conf >= DETECT_CONF for conf in effective)
    max_conf = max(effective, default=0.0)
    oversized_real = any(
        oversized and not is_static and conf >= DETECT_CONF
        for _, is_static, oversized, conf in rows
    )

    if n_real >= 2 or max_conf >= STRONG_CONF:
        tier = "confident"
    elif n_real == 1 or oversized_real:
        tier = "under_review"
    else:
        tier = "not_confident"

    blur_capped = False
    school_exempt = False
    if tier == "confident" and blurry:
        if n_real >= SCHOOL_EXEMPT:
            school_exempt = True
        else:
            tier = "under_review"
            blur_capped = True

    return FrameClass(
        image["file"], tier, blurry, blur_capped, school_exempt, [row for row, *_ in rows]
    )


def _write_tier_csvs(frames: list[FrameClass], out_dir: Path) -> None:
    """One CSV per tier; every detection row lands in its frame's tier file."""
    for tier in TIERS:
        path = out_dir / f"{tier}.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(CSV_HEADER)
            for frame in sorted(frames, key=lambda f: f.file):
                if frame.tier == tier:
                    writer.writerows(frame.rows)


def _move_annotated(frames: list[FrameClass], out_dir: Path) -> None:
    """Move annotated images from out_dir/annotated/ into the tier subfolders."""
    annotated = out_dir / "annotated"
    for frame in frames:
        if frame.tier is None:
            continue
        source = annotated / frame.file
        if not source.is_file():
            source = source.with_suffix(".png")  # annotate() falls back to PNG
            if not source.is_file():
                continue
        dest = (out_dir / frame.tier / frame.file).with_suffix(source.suffix)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(dest))
    shutil.rmtree(annotated, ignore_errors=True)
