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
from fishcount.config import AppConfig, ConfigError, load_config, merge_overrides
from fishcount.detector import Detector, ModelNotFoundError
from fishcount.pipeline import PipelineSummary, run_pipeline

# Recall-first preset for dense schools of small fish, chosen from an empirical
# sweep on frames the default settings scored 0. Trades precision (and speed)
# for recall. --thorough additionally switches to SAHI tiling at slice_size.
_DENSE_PRESET: dict[str, float | int] = {
    "conf": 0.10,
    "iou": 0.7,
    "imgsz": 1536,
    "max_det": 3000,
    "slice_size": 640,
    "overlap_ratio": 0.25,
    # Drop oversized boxes (empty/murky water misread as one giant fish). Real
    # fish here are <5% of the frame and the false positives are >=10%, with a
    # clean gap between, so 0.10 removes them without touching real detections.
    "max_box_frac": 0.10,
}


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
        help="Max detections per image (default 1000). Raise for dense schools.",
    )
    detect.add_argument(
        "--batch-size", type=int, default=None, help="Images per model pass (default 8)."
    )
    detect.add_argument(
        "--slice-size",
        type=int,
        default=None,
        dest="slice_size",
        help="SAHI tile size in px (default 640; --thorough only). Smaller finds smaller fish.",
    )
    detect.add_argument(
        "--overlap",
        type=float,
        default=None,
        help="SAHI tile overlap ratio, 0-0.9 (default 0.2; --thorough only).",
    )
    detect.add_argument(
        "--dense",
        action="store_true",
        help="Recall-first preset for dense schools of small fish (see README).",
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

    pipeline = subcommands.add_parser(
        "pipeline",
        help="Detection first: flag every frame with a fish, then count the flagged ones.",
    )
    pipeline.add_argument("folder", type=Path, help="Folder of images (subfolders included).")
    pipeline.add_argument(
        "--name",
        default=None,
        help="Output name under output/ (default: the input folder's name).",
    )
    pipeline.add_argument(
        "--out", type=Path, default=None, metavar="DIR", help="Output folder (overrides --name)."
    )
    pipeline.add_argument(
        "--min-count",
        type=int,
        default=1,
        dest="min_count",
        help="Detections (at or above --detect-conf) a frame needs to be counted by the "
        "dense/thorough passes (default 1: any real detection gets counted).",
    )
    pipeline.add_argument(
        "--base-conf",
        type=float,
        default=0.10,
        dest="base_conf",
        help="Confidence floor for the base gate (default 0.10). Everything above it is "
        "recorded; nothing is silently dropped. Costs no extra inference time.",
    )
    pipeline.add_argument(
        "--detect-conf",
        type=float,
        default=0.25,
        dest="detect_conf",
        help="Confidence at which a detection counts as a real fish (default 0.25). "
        "Detections between --base-conf and this are tiered 'possible'.",
    )
    pipeline.add_argument(
        "--static-min-frames",
        type=int,
        default=8,
        dest="static_min_frames",
        help="A box that recurs at the same pixels in this many distinct frames is a "
        "stationary object (rock/debris), not a fish; it is demoted to the 'static' "
        "tier (default 8). 0 disables the filter.",
    )
    pipeline.add_argument(
        "--count-possible",
        action="store_true",
        dest="count_possible",
        help="Also run the counting passes on 'possible' frames (weak detections only).",
    )
    pipeline.add_argument(
        "--base-imgsz",
        type=int,
        default=1536,
        dest="base_imgsz",
        help="Inference size for the base gating pass (default 1536). "
        "1024 is faster but misses dense schools, so they never reach the dense passes.",
    )
    pipeline.add_argument(
        "--base-max-box-frac",
        type=float,
        default=0.10,
        dest="base_max_box_frac",
        help="Base gate drops any detection bigger than this fraction of the frame "
        "(default 0.10); removes empty/murky water misread as one giant fish.",
    )
    pipeline.add_argument(
        "--no-images",
        action="store_true",
        help="Write counts/JSON only; skip annotated images.",
    )
    pipeline.add_argument(
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
    if args.command == "pipeline":
        return _pipeline_command(args)
    return 2  # unreachable: subparsers are required


def _detect_command(args: argparse.Namespace) -> int:
    console = Console()
    errors = Console(stderr=True, style="red", soft_wrap=True)

    input_dir: Path = args.folder.expanduser()
    if not input_dir.is_dir():
        errors.print(f"Not a folder: {input_dir}")
        return 1
    input_dir = input_dir.resolve()

    preset = _DENSE_PRESET if args.dense else {}

    def pick(name: str, cli_value: object) -> object:
        # CLI flag wins; otherwise fall back to the --dense preset (if any);
        # otherwise None, so merge_overrides leaves config.yaml/defaults in place.
        return cli_value if cli_value is not None else preset.get(name)

    try:
        config = merge_overrides(
            load_config(),
            conf=pick("conf", args.conf),
            iou=pick("iou", args.iou),
            imgsz=pick("imgsz", args.imgsz),
            max_det=pick("max_det", args.max_det),
            slice_size=pick("slice_size", args.slice_size),
            overlap_ratio=pick("overlap_ratio", args.overlap),
            batch_size=args.batch_size,
        )
    except ConfigError as exc:
        errors.print(str(exc))
        return 1

    if args.dense:
        console.print(
            "[dim]--dense: recall-first (finds more small fish, more false positives; slower).[/]"
        )

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


def _pipeline_command(args: argparse.Namespace) -> int:
    console = Console()
    errors = Console(stderr=True, style="red", soft_wrap=True)

    input_dir: Path = args.folder.expanduser()
    if not input_dir.is_dir():
        errors.print(f"Not a folder: {input_dir}")
        return 1
    input_dir = input_dir.resolve()

    try:
        # The base pass is a presence gate: it decides which frames hold fish,
        # not how many. It runs at a higher imgsz so dense schools register (at
        # 1024 they read as 0 and never reach the dense passes), and drops
        # oversized boxes so empty/murky water is not misread as a giant fish.
        if not args.base_conf < args.detect_conf:
            errors.print("--base-conf must be below --detect-conf")
            return 1
        base_config = merge_overrides(
            load_config(),
            imgsz=args.base_imgsz,
            conf=args.base_conf,
            max_box_frac=args.base_max_box_frac,
        )
        dense_config = merge_overrides(load_config(), **_DENSE_PRESET)
    except ConfigError as exc:
        errors.print(str(exc))
        return 1

    name = args.name if args.name is not None else input_dir.name
    out_dir: Path = args.out if args.out is not None else Path("output") / name
    out_dir = out_dir.expanduser().resolve()

    try:
        weights = detector_module.resolve_model_path(base_config.model_path)
    except ModelNotFoundError as exc:
        errors.print(str(exc))
        return 1

    def make_detector(config: AppConfig, thorough: bool) -> Detector:
        return detector_module.create_detector(weights, config, thorough=thorough)

    console.print(f"[dim]Model:[/] {weights}")
    console.print(
        "[dim]Pipeline (detection first):[/] flag every frame with a fish, then count "
        f"the flagged frames (>= {args.min_count} detection(s) at >= {args.detect_conf})."
    )
    try:
        summary = run_pipeline(
            input_dir,
            out_dir,
            base_config=base_config,
            dense_config=dense_config,
            make_detector=make_detector,
            min_count=args.min_count,
            detect_conf=args.detect_conf,
            count_possible=args.count_possible,
            static_min_frames=args.static_min_frames or None,
            write_images=not args.no_images,
            model_path=weights,
        )
    except (NoImagesFoundError, ModelNotFoundError, RuntimeError) as exc:
        errors.print(str(exc))
        return 1

    _print_pipeline_summary(console, summary)
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


def _print_pipeline_summary(console: Console, summary: PipelineSummary) -> None:
    # Detection first: which frames have fish is the headline; counts come after.
    total = summary.base.processed
    detect = Table(title="fishcount pipeline: detection", show_header=True, title_justify="left")
    detect.add_column("Tier", style="dim")
    detect.add_column("Frames", justify="right")
    detect.add_column("Meaning")
    detect.add_row(
        "confident",
        f"[bold green]{summary.tier_count('confident')}[/]",
        "2+ real detections, or one at >= 0.50",
    )
    detect.add_row(
        "review",
        f"[bold yellow]{summary.tier_count('review')}[/]",
        "one moderate detection; glance to rule out surface ripple",
    )
    detect.add_row(
        "possible",
        str(summary.tier_count("possible")),
        "weak detections only (below --detect-conf)",
    )
    detect.add_row(
        "static",
        str(summary.tier_count("static")),
        "only stationary objects (same box across many frames): not fish",
    )
    detect.add_row("none", str(summary.tier_count("none")), "no detection at all")
    console.print(detect)
    console.print(
        f"[bold]Frames with fish: {summary.flagged} of {total}[/] "
        f"(counted {summary.counted}; base gate {summary.base.seconds:.0f} s)"
    )

    counts = Table(title="counts (flagged frames)", show_header=True, title_justify="left")
    counts.add_column("Stage", style="dim")
    counts.add_column("Frames", justify="right")
    counts.add_column("Total fish", justify="right")
    counts.add_column("Time", justify="right")
    counts.add_row(
        "base gate",
        str(summary.base.processed),
        f"{summary.base.total_fish}",
        f"{summary.base.seconds:.0f} s",
    )
    if summary.dense is not None:
        counts.add_row(
            "dense",
            str(summary.dense.processed),
            f"[bold green]{summary.dense.total_fish}[/]",
            f"{summary.dense.seconds:.0f} s",
        )
    if summary.thorough is not None:
        counts.add_row(
            "thorough",
            str(summary.thorough.processed),
            f"[bold green]{summary.thorough.total_fish}[/]",
            f"{summary.thorough.seconds:.0f} s",
        )
    console.print(counts)
    if summary.counted == 0:
        console.print(
            "[yellow]No frame qualified for counting; dense and thorough were skipped.[/]"
        )
    console.print(f"[dim]Detections:[/] {summary.out_dir / 'detections.csv'}")
    console.print(f"[dim]Flagged frames:[/] {summary.out_dir / 'detected'}")
    console.print(f"[dim]Counts:[/] {summary.out_dir / 'summary.csv'}")


def _open_folder(path: Path) -> None:
    if sys.platform == "win32":
        os.startfile(path)
    elif sys.platform == "darwin":
        subprocess.run(["open", str(path)], check=False)
    else:
        subprocess.run(["xdg-open", str(path)], check=False)
