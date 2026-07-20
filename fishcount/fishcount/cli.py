"""Command-line interface: `fishcount detect <folder>`."""

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fishcount",
        description="Detect and count fish in a folder of images. Fully local and offline.",
    )
    parser.add_argument("--version", action="version", version=f"fishcount {__version__}")
    subcommands = parser.add_subparsers(dest="command", required=True)

    detect = subcommands.add_parser(
        "detect", help="Detect and count fish in every image under FOLDER."
    )
    detect.add_argument("folder", type=Path, help="Folder of images (subfolders included).")
    detect.add_argument(
        "--out",
        type=Path,
        default=None,
        metavar="DIR",
        help="Output folder (default: output/<folder name>).",
    )
    detect.add_argument(
        "--conf", type=float, default=None, help="Confidence threshold, 0-1 (default 0.25)."
    )
    detect.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="Inference image size (default 1024; lower is faster).",
    )
    detect.add_argument(
        "--batch-size", type=int, default=None, help="Images per model pass (default 8)."
    )
    detect.add_argument(
        "--no-images",
        action="store_true",
        help="Write counts.csv and results.json only; skip annotated images.",
    )
    detect.add_argument(
        "--thorough",
        action="store_true",
        help="SAHI tiled inference to catch very small fish (much slower).",
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
            load_config(), conf=args.conf, imgsz=args.imgsz, batch_size=args.batch_size
        )
    except ConfigError as exc:
        errors.print(str(exc))
        return 1

    out_dir: Path = args.out if args.out is not None else Path("output") / input_dir.name
    out_dir = out_dir.expanduser().resolve()

    try:
        weights = detector_module.resolve_model_path(config.model_path)
        console.print(f"[dim]Model:[/] {weights}")
        console.print("[dim]Loading model (takes a few seconds on CPU)...[/]")
        fish_detector = detector_module.create_detector(weights, config, thorough=args.thorough)
    except (ModelNotFoundError, RuntimeError) as exc:
        errors.print(str(exc))
        return 1

    try:
        summary = run_batch(
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

    _print_summary(console, summary, wrote_images=not args.no_images)
    if args.open_folder:
        _open_folder(out_dir)
    return 0


def _print_summary(console: Console, summary: BatchSummary, *, wrote_images: bool) -> None:
    table = Table(title="fishcount", show_header=False, title_justify="left")
    table.add_column(style="dim")
    table.add_column()
    table.add_row("Images processed", str(summary.processed))
    table.add_row("Images skipped", str(summary.skipped))
    table.add_row("Total fish", f"[bold green]{summary.total_fish}[/]")
    table.add_row("Elapsed", f"{summary.seconds:.1f} s")
    table.add_row("Per image", f"{summary.seconds_per_image:.2f} s")
    table.add_row("Output folder", str(summary.out_dir))
    if wrote_images:
        table.add_row("Annotated images", str(summary.out_dir / "annotated"))
    table.add_row("Counts CSV", str(summary.out_dir / "counts.csv"))
    table.add_row("Results JSON", str(summary.out_dir / "results.json"))
    console.print(table)


def _open_folder(path: Path) -> None:
    if sys.platform == "win32":
        os.startfile(path)
    elif sys.platform == "darwin":
        subprocess.run(["open", str(path)], check=False)
    else:
        subprocess.run(["xdg-open", str(path)], check=False)
