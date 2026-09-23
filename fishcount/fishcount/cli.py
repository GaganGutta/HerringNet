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
from fishcount.detector import ModelNotFoundError, describe_device
from fishcount.journal import JOURNAL_FILENAME, ResumeMismatchError
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
        "detect",
        help="One detection pass over a folder; write detections.csv and frames.csv. "
        "Resumes automatically if the folder was only partly done.",
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
        "--batch-size",
        type=int,
        default=None,
        help="Images per model pass. A batch too big for VRAM is split automatically, "
        "so this is a speed knob rather than a limit.",
    )
    detect.add_argument(
        "--device",
        default=None,
        help="auto (default: the GPU when there is one), cpu, or cuda / cuda:N.",
    )
    detect.add_argument(
        "--restart",
        action="store_true",
        help="Discard any previous results for this output folder and detect it again. "
        "Without this, a rerun resumes and skips frames already done.",
    )
    detect.add_argument(
        "--no-images",
        action="store_true",
        help="Write the CSVs only; skip annotated images.",
    )
    detect.add_argument(
        "--open",
        action="store_true",
        dest="open_folder",
        help="Open the output folder when finished.",
    )

    report = subcommands.add_parser(
        "report",
        help="Rewrite the CSVs from a finished run at a different threshold. No "
        "inference: this is how to try a threshold without paying for it again.",
    )
    report.add_argument("out_dir", type=Path, help="An output folder from a previous detect run.")
    report.add_argument(
        "--threshold", type=float, default=None, help="Reporting threshold, 0-1 (default 0.25)."
    )
    report.add_argument(
        "--folder",
        type=Path,
        default=None,
        help="The original image folder. Needed only to redraw the annotated images.",
    )
    report.add_argument(
        "--no-images",
        action="store_true",
        help="Write the CSVs only; skip annotated images.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "detect":
        return _detect_command(args)
    if args.command == "report":
        return _report_command(args)
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
            device=args.device,
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
        console.print("[dim]Loading model...[/]")
        fish_detector = detector_module.create_detector(weights, config)
    except (ModelNotFoundError, RuntimeError) as exc:
        errors.print(str(exc))
        return 1

    console.print(f"[dim]Device:[/] {describe_device(getattr(fish_detector, 'device', 'cpu'))}")
    if not args.restart and (out_dir / JOURNAL_FILENAME).is_file():
        console.print(f"[dim]Resuming:[/] {out_dir} is partly done; finishing what is left.")

    try:
        batch = run_batch(
            input_dir,
            out_dir,
            config,
            fish_detector,
            model_path=weights,
            resume=not args.restart,
        )
    except (NoImagesFoundError, ResumeMismatchError) as exc:
        errors.print(str(exc))
        return 1

    report = write_reports(
        out_dir,
        threshold=config.threshold,
        input_dir=input_dir,
        write_images=not args.no_images,
    )

    _print_summary(console, batch, report)
    if args.open_folder:
        _open_folder(out_dir)
    return 0


def _report_command(args: argparse.Namespace) -> int:
    console = Console()
    errors = Console(stderr=True, style="red", soft_wrap=True)

    out_dir: Path = args.out_dir.expanduser().resolve()
    if not (out_dir / JOURNAL_FILENAME).is_file():
        errors.print(f"No run to report on: {out_dir / JOURNAL_FILENAME} does not exist.")
        return 1

    try:
        config = merge_overrides(load_config(), threshold=args.threshold)
    except ConfigError as exc:
        errors.print(str(exc))
        return 1

    input_dir: Path | None = args.folder.expanduser().resolve() if args.folder else None
    if input_dir is None and not args.no_images:
        console.print(
            "[dim]Note:[/] redrawing annotated images needs the original folder; "
            "pass --folder to redraw them, or --no-images to say you meant to skip them."
        )

    report = write_reports(
        out_dir,
        threshold=config.threshold,
        input_dir=input_dir,
        write_images=not args.no_images and input_dir is not None,
    )

    table = Table(title="fishcount report", show_header=True, title_justify="left")
    table.add_column("", style="dim")
    table.add_column("Frames", justify="right")
    table.add_column("Detections", justify="right")
    table.add_row(
        f"at or above threshold {report.threshold:.2f}",
        f"[bold green]{report.frames_above_threshold}[/]",
        str(report.detections_above_threshold),
    )
    table.add_row("recorded", str(report.frames_with_detections), str(report.detections))
    table.add_row("scanned", str(report.frames), "")
    console.print(table)
    if report.annotated:
        console.print(f"[dim]{report.annotated} annotated images rewritten[/]")
    for name in ("detections.csv", "frames.csv"):
        console.print(f"[dim]{name}:[/] {out_dir / name}")
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

    notes = []
    if batch.processed:
        notes.append(
            f"detected {batch.processed} frames in {batch.seconds:.0f} s "
            f"({batch.seconds_per_image:.2f} s/frame, {batch.images_per_second:.2f} img/s) "
            f"on {describe_device(batch.device)}"
        )
    if batch.already_done:
        notes.append(f"{batch.already_done} already done, skipped")
    if report.unreadable:
        notes.append(f"{report.unreadable} unreadable")
    if batch.oom_splits:
        notes.append(f"{batch.oom_splits} batches split to fit VRAM (lower --batch-size)")
    if report.annotated:
        notes.append(f"{report.annotated} annotated images written")
    if notes:
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
