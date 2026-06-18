"""CLI subcommand: fast folder-in, results-out batch detection."""

from __future__ import annotations

import argparse
import logging

from rich.console import Console

console = Console()
logger = logging.getLogger(__name__)


def add_batch_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the batch subcommand."""
    parser = subparsers.add_parser(
        "batch",
        help="Detect fish across a whole folder, fast (no SAHI)",
    )
    parser.add_argument("input", type=str, help="Folder of images")
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output folder (default: outputs/<folder name>)",
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Path to config YAML"
    )
    parser.add_argument(
        "--conf", type=float, default=None,
        help="Detection confidence threshold (default: from config)",
    )
    parser.add_argument(
        "--imgsz", type=int, default=None,
        help="Inference image size; lower is faster (e.g. 640)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Images per model pass",
    )
    parser.add_argument(
        "--no-images", action="store_true",
        help="Skip writing annotated images (counts only)",
    )
    parser.add_argument(
        "--open", action="store_true",
        help="Open the output folder when finished",
    )
    parser.set_defaults(func=run_batch_cmd)


def run_batch_cmd(args: argparse.Namespace) -> None:
    """Execute the batch command."""
    from herringnet.config import load_config
    from herringnet.inference.batch import run_batch

    # Force SAHI off for speed; apply any overrides.
    det_overrides: dict = {"use_sahi": False}
    if args.conf is not None:
        det_overrides["confidence_threshold"] = args.conf
    if args.imgsz is not None:
        det_overrides["image_size"] = args.imgsz

    config = load_config(
        config_path=args.config, overrides={"detector": det_overrides}
    )

    console.print(f"Processing folder: [bold]{args.input}[/bold]")
    console.print(
        f"Model: {config.detector.model_name}  |  "
        f"image size: {config.detector.image_size}  |  SAHI: off"
    )

    with console.status("Detecting fish..."):
        def _progress(done: int, total: int) -> None:
            console.print(f"  {done}/{total} images", end="\r")

        summary = run_batch(
            config,
            args.input,
            output_dir=args.output,
            batch_size=args.batch_size,
            save_annotated=not args.no_images,
            progress=_progress,
        )

    console.print()
    console.print(
        f"[green]Done.[/green] {summary['images']} images, "
        f"{summary['total_fish']} fish in {summary['images_with_fish']} images."
    )
    console.print(
        f"  Time: {summary['seconds']}s "
        f"({summary['seconds_per_image']}s per image)"
    )
    console.print(f"  Results: [bold]{summary['output_dir']}[/bold]")
    console.print(f"  Annotated images: {summary['output_dir']}\\annotated")

    if args.open:
        _open_folder(summary["output_dir"])


def _open_folder(path: str) -> None:
    """Open a folder in the OS file browser (best effort)."""
    import os
    import subprocess
    import sys

    try:
        if sys.platform == "win32":
            os.startfile(path)  # noqa: S606
        elif sys.platform == "darwin":
            subprocess.run(["open", path], check=False)
        else:
            subprocess.run(["xdg-open", path], check=False)
    except Exception as e:  # noqa: BLE001
        logger.debug("Could not open folder %s: %s", path, e)
