"""CLI subcommand for extracting frames from video files."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from rich.console import Console

console = Console()
logger = logging.getLogger(__name__)


def add_extract_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the extract subcommand."""
    parser = subparsers.add_parser(
        "extract",
        help="Extract frames from video files",
    )
    parser.add_argument(
        "video",
        type=str,
        help="Path to video file or directory of videos",
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/frames",
        help="Output directory for extracted frames (default: data/frames)",
    )
    parser.add_argument(
        "--interval", type=int, default=None,
        help="Frame extraction interval (default: from config)",
    )
    parser.add_argument(
        "--frr", type=float, default=None,
        help="Mean frame residence rate (default: 4.55)",
    )
    parser.add_argument(
        "--max-frames", type=int, default=None,
        help="Maximum frames to extract",
    )
    parser.add_argument(
        "--format", type=str, default="jpg", choices=["jpg", "png"],
        help="Output image format (default: jpg)",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config YAML",
    )
    parser.set_defaults(func=run_extract)


def run_extract(args: argparse.Namespace) -> None:
    """Execute the extract command."""
    from herringnet.config import load_config
    from herringnet.data.frame_extractor import FrameExtractor

    overrides = {}
    if args.interval is not None or args.frr is not None:
        fe_overrides: dict = {}
        if args.interval is not None:
            fe_overrides["frame_interval"] = args.interval
        if args.frr is not None:
            fe_overrides["mean_frr"] = args.frr
        overrides["frame_extraction"] = fe_overrides

    config = load_config(config_path=args.config, overrides=overrides)
    extractor = FrameExtractor(config.frame_extraction)

    video_path = Path(args.video)
    if not video_path.exists():
        console.print(f"[red]Error: Not found: {video_path}[/red]")
        return

    output_dir = Path(args.output_dir)

    if video_path.is_dir():
        video_files = sorted(
            p for p in video_path.iterdir()
            if p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}
        )
        if not video_files:
            console.print(f"[yellow]No video files found in {video_path}[/yellow]")
            return

        total_frames = 0
        for vf in video_files:
            console.print(f"Extracting from: [bold]{vf.name}[/bold]")
            vf_output = output_dir / vf.stem
            frames = extractor.extract_frames(
                vf, vf_output,
                max_frames=args.max_frames,
                image_format=args.format,
            )
            total_frames += len(frames)
            console.print(f"  Extracted {len(frames)} frames to {vf_output}")

        console.print(
            f"\nTotal: {total_frames} frames from {len(video_files)} videos"
        )
    else:
        console.print(f"Extracting from: [bold]{video_path.name}[/bold]")
        metadata = extractor.get_video_metadata(video_path)
        console.print(
            f"  Video: {metadata.fps:.1f} fps, "
            f"{metadata.total_frames} frames, "
            f"{metadata.duration_seconds:.1f}s"
        )

        frames = extractor.extract_frames(
            video_path, output_dir,
            max_frames=args.max_frames,
            image_format=args.format,
        )
        console.print(f"Extracted {len(frames)} frames to {output_dir}")
