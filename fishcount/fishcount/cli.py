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
from fishcount.config import ConfigError, load_config, merge_overrides
from fishcount.detector import ModelNotFoundError
from fishcount.report import ReportSummary, write_reports


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fishcount",
        description="Detect fish in a folder of images, fully local and offline. "
        "Detection only: every box is recorded, nothing is counted.",
    )
    parser.add_argument("--version", action="version", version=f"fishcount {__version__}")
    subcommands = parser.add_subparsers(dest="command", required=True)

    detect = subcommands.add_parser(
        "detect", help="One detection pass over a folder; write detections.csv and frames.csv."
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
        help="Recording floor, 0-1 (default 0.10). Every box above it is written out.",
    )
    detect.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Reporting threshold, 0-1 (default 0.25): the one number deciding which "
        "frames count as holding fish and get an annotated image.",
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
            threshold=args.threshold,
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

    report = write_reports(out_dir, threshold=config.threshold)

    _print_summary(console, batch, report)
    if args.open_folder:
        _open_folder(out_dir)
    return 0


def _print_summary(console: Console, batch: BatchSummary, report: ReportSummary) -> None:
    table = Table(title="fishcount detection", show_header=True, title_justify="left")
    table.add_column("", style="dim")
    table.add_column("Frames", justify="right")
    table.add_column("Detections", justify="right")
    table.add_row(
        f"at or above threshold {report.threshold:.2f}",
        f"[bold green]{report.frames_above_threshold}[/]",
        str(report.detections_above_threshold),
    )
    table.add_row(
        f"recorded (floor {batch.conf:.2f})",
        str(report.frames_with_detections),
        str(report.detections),
    )
    table.add_row("scanned", str(report.frames), "")
    console.print(table)

    notes = [
        f"processed {batch.processed} frames in {batch.seconds:.0f} s "
        f"({batch.seconds_per_image:.1f} s/frame)"
    ]
    if report.unreadable:
        notes.append(f"skipped {report.unreadable} unreadable")
    console.print("[dim]" + "; ".join(notes) + "[/]")
    for name in ("detections.csv", "frames.csv"):
        console.print(f"[dim]{name}:[/] {report.out_dir / name}")


def _open_folder(path: Path) -> None:
    if sys.platform == "win32":
        os.startfile(path)
    elif sys.platform == "darwin":
        subprocess.run(["open", str(path)], check=False)
    else:
        subprocess.run(["xdg-open", str(path)], check=False)
