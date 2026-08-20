"""Command-line interface: `fishcount detect <folder>`. Detection only, no counting."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from rich.console import Console
from rich.table import Table

from fishcount import __version__
from fishcount import detector as detector_module
from fishcount.batch import BatchSummary, NoImagesFoundError, run_batch
from fishcount.classify import TIERS, ClassifySummary, classify
from fishcount.config import ConfigError, load_config, merge_overrides
from fishcount.detector import ModelNotFoundError

_TIER_MEANING = {
    "confident": "2+ real detections or one strong, sharp frame (or school exemption)",
    "under_review": "one moderate detection, blur-capped evidence, or an oversized box",
    "not_confident": "only weak, static, or weak-oversized detections",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fishcount",
        description="Detect fish in a folder of images, fully local and offline. "
        "Detection only: frames are tiered by confidence, nothing is counted.",
    )
    parser.add_argument("--version", action="version", version=f"fishcount {__version__}")
    subcommands = parser.add_subparsers(dest="command", required=True)

    detect = subcommands.add_parser(
        "detect", help="One detection pass; tier every frame into three files."
    )
    detect.add_argument("folder", type=Path, help="Folder of images (subfolders included).")
    detect.add_argument(
        "--name",
        default=None,
        help="Output name under output/ (default: the input folder's name).",
    )
    detect.add_argument(
        "--out", type=Path, default=None, metavar="DIR", help="Output folder (overrides --name)."
    )
    detect.add_argument(
        "--conf",
        type=float,
        default=None,
        help="Confidence floor, 0-1 (default 0.10). Everything above it is recorded.",
    )
    detect.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="Inference image size (default 1536; 1024 misses dense schools of small fish).",
    )
    detect.add_argument(
        "--iou",
        type=float,
        default=None,
        help="NMS IoU threshold, 0-1 (default 0.7). Higher keeps tightly packed fish.",
    )
    detect.add_argument(
        "--max-det",
        type=int,
        default=None,
        dest="max_det",
        help="Max detections per image (default 3000).",
    )
    detect.add_argument(
        "--batch-size", type=int, default=None, help="Images per model pass (default 8)."
    )
    detect.add_argument(
        "--blur-percentile",
        type=float,
        default=25.0,
        dest="blur_percentile",
        help="Frames blurrier than this percentile of their own folder cannot reach "
        "confident on thin evidence (default 25). 0 disables the blur cap.",
    )
    detect.add_argument(
        "--static-min-frames",
        type=int,
        default=8,
        dest="static_min_frames",
        help="A box recurring at the same pixels in this many distinct frames is a "
        "stationary object, demoted to not_confident (default 8). 0 disables.",
    )
    detect.add_argument(
        "--no-images",
        action="store_true",
        help="Write the CSVs and results.json only; skip annotated images.",
    )
    detect.add_argument(
        "--open",
        action="store_true",
        dest="open_folder",
        help="Open the output folder when finished.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "detect":
        return _detect_command(args)
    return 2  # unreachable: subparsers are required


def _detect_command(args: argparse.Namespace) -> int:
    console = Console()
    errors = Console(stderr=True, style="red", soft_wrap=True)

    input_dir: Path = args.folder.expanduser()
    if not input_dir.is_dir():
        errors.print(f"Not a folder: {input_dir}")
        return 1
    input_dir = input_dir.resolve()

    try:
        config = merge_overrides(
            load_config(),
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            max_det=args.max_det,
            batch_size=args.batch_size,
        )
    except ConfigError as exc:
        errors.print(str(exc))
        return 1

    name = args.name if args.name is not None else input_dir.name
    out_dir: Path = args.out if args.out is not None else Path("output") / name
    out_dir = out_dir.expanduser().resolve()

    try:
        weights = detector_module.resolve_model_path(config.model_path)
        console.print(f"[dim]Model:[/] {weights}")
        console.print("[dim]Loading model (takes a few seconds on CPU)...[/]")
        fish_detector = detector_module.create_detector(weights, config)
    except (ModelNotFoundError, RuntimeError) as exc:
        errors.print(str(exc))
        return 1

    try:
        batch = run_batch(
            input_dir,
            out_dir,
            config,
            fish_detector,
            write_images=not args.no_images,
            model_path=weights,
        )
    except NoImagesFoundError as exc:
        errors.print(str(exc))
        return 1

    tiers = classify(
        out_dir,
        floor=config.conf,
        blur_percentile=args.blur_percentile,
        static_min_frames=args.static_min_frames,
        move_images=not args.no_images,
    )

    _print_summary(console, batch, tiers)
    if args.open_folder:
        _open_folder(out_dir)
    return 0


def _print_summary(console: Console, batch: BatchSummary, tiers: ClassifySummary) -> None:
    table = Table(title="fishcount detection", show_header=True, title_justify="left")
    table.add_column("Tier", style="dim")
    table.add_column("Frames", justify="right")
    table.add_column("Detections", justify="right")
    table.add_column("Meaning")
    styles = {"confident": "bold green", "under_review": "bold yellow", "not_confident": ""}
    for tier in TIERS:
        table.add_row(
            tier,
            f"[{styles[tier]}]{tiers.frames_per_tier[tier]}[/]"
            if styles[tier]
            else str(tiers.frames_per_tier[tier]),
            str(tiers.detections_per_tier[tier]),
            _TIER_MEANING[tier],
        )
    table.add_row("(no detections)", str(tiers.no_detection_frames), "", "")
    console.print(table)
    notes = [
        f"processed {batch.processed} frames in {batch.seconds:.0f} s "
        f"({batch.seconds_per_image:.1f} s/frame)",
    ]
    if batch.skipped:
        notes.append(f"skipped {batch.skipped} unreadable")
    if tiers.blur_threshold is not None:
        notes.append(
            f"blur threshold {tiers.blur_threshold:.0f} "
            f"(capped {tiers.blur_capped}, school-exempt {tiers.school_exempt})"
        )
    if tiers.static_detections:
        notes.append(f"static detections demoted: {tiers.static_detections}")
    console.print("[dim]" + "; ".join(notes) + "[/]")
    for tier in TIERS:
        console.print(f"[dim]{tier}:[/] {tiers.out_dir / (tier + '.csv')}")


def _open_folder(path: Path) -> None:
    if sys.platform == "win32":
        os.startfile(path)
    elif sys.platform == "darwin":
        subprocess.run(["open", str(path)], check=False)
    else:
        subprocess.run(["xdg-open", str(path)], check=False)
