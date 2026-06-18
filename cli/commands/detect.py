"""CLI subcommand for running fish detection on images and videos."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()
logger = logging.getLogger(__name__)

# File extensions recognized as videos
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"}


def add_detect_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the detect subcommand."""
    parser = subparsers.add_parser(
        "detect",
        help="Run fish detection on images or video",
    )
    parser.add_argument(
        "source",
        type=str,
        help="Path to image, video, or directory",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config YAML (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--site", type=str, default=None,
        help="Site name for site-specific config override",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: outputs/)",
    )
    parser.add_argument(
        "--save-images", action="store_true",
        help="Save annotated images with bounding boxes",
    )
    parser.add_argument(
        "--save-crops", action="store_true",
        help="Save cropped detection regions",
    )
    parser.add_argument(
        "--save-csv", action="store_true",
        help="Save results summary as CSV",
    )
    parser.add_argument(
        "--save-json", action="store_true", default=True,
        help="Save results as JSON (default: True)",
    )
    parser.add_argument(
        "--confidence", type=float, default=None,
        help="Override detection confidence threshold",
    )
    parser.add_argument(
        "--max-frames", type=int, default=None,
        help="Maximum frames to process (for video)",
    )
    parser.add_argument(
        "--sahi", action="store_true", default=None,
        help="Enable SAHI sliced inference for small fish detection",
    )
    parser.add_argument(
        "--no-sahi", action="store_true", default=False,
        help="Disable SAHI sliced inference (faster, misses small fish)",
    )
    parser.add_argument(
        "--slice-size", type=int, default=None,
        help="SAHI slice size in pixels (default: 512)",
    )
    parser.set_defaults(func=run_detect)


def run_detect(args: argparse.Namespace) -> None:
    """Execute the detect command."""
    from herringnet.config import load_config

    # Build config overrides from CLI args
    overrides = {}
    det_overrides: dict = {}
    if args.confidence is not None:
        det_overrides["confidence_threshold"] = args.confidence
    if args.sahi:
        det_overrides["use_sahi"] = True
    if args.no_sahi:
        det_overrides["use_sahi"] = False
    if args.slice_size is not None:
        det_overrides["sahi_slice_size"] = args.slice_size
    if det_overrides:
        overrides["detector"] = det_overrides
    if args.output_dir is not None:
        overrides["output"] = {"output_dir": args.output_dir}

    config = load_config(
        config_path=args.config, site=args.site, overrides=overrides
    )

    source = Path(args.source)
    if not source.exists():
        console.print(f"[red]Error: Source not found: {source}[/red]")
        return

    output_dir = Path(args.output_dir or config.output.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine if source is an image, video, or directory
    if source.is_dir():
        _detect_directory(config, source, output_dir, args)
    elif source.suffix.lower() in VIDEO_EXTENSIONS:
        _detect_video(config, source, output_dir, args)
    else:
        _detect_image(config, source, output_dir, args)


def _detect_image(config, source: Path, output_dir: Path, args) -> None:
    """Run detection on a single image."""
    import cv2

    from herringnet.models.pipeline import HerringNetPipeline, save_results_json
    from herringnet.visualization.draw_detections import save_annotated_image

    console.print(f"Processing image: [bold]{source}[/bold]")

    pipeline = HerringNetPipeline(config)
    result = pipeline.process_image(source)

    # Display results
    _print_frame_result(result)

    # Save JSON
    if args.save_json:
        json_path = output_dir / f"{source.stem}_results.json"
        save_results_json([result], json_path)
        console.print(f"Results saved: {json_path}")

    # Save annotated image
    if args.save_images:
        image = cv2.imread(str(source))
        if image is not None:
            out_path = output_dir / f"{source.stem}_annotated{source.suffix}"
            save_annotated_image(image, result.detections, out_path)
            console.print(f"Annotated image saved: {out_path}")

    # Save crops
    if args.save_crops and result.detections:
        import cv2 as cv

        from herringnet.models.detector import FishDetector

        image = cv.imread(str(source))
        if image is not None:
            crops_dir = output_dir / "crops" / source.stem
            crops_dir.mkdir(parents=True, exist_ok=True)
            crops = FishDetector.crop_detections(
                image, [d.detection for d in result.detections]
            )
            for i, crop in enumerate(crops):
                crop_path = crops_dir / f"crop_{i:03d}.jpg"
                cv.imwrite(str(crop_path), crop)
            console.print(f"Saved {len(crops)} crops to {crops_dir}")


def _detect_video(config, source: Path, output_dir: Path, args) -> None:
    """Run detection on a video file."""
    from herringnet.inference.video_inference import (
        process_video_file,
        save_counts_csv,
    )
    from herringnet.models.pipeline import save_results_json

    console.print(f"Processing video: [bold]{source}[/bold]")

    result = process_video_file(
        video_path=source,
        config=config,
        output_dir=output_dir / source.stem,
        save_annotated_frames=args.save_images,
        save_csv=args.save_csv,
        max_frames=args.max_frames,
    )

    # Print summary
    _print_video_summary(result)

    # Save JSON
    if args.save_json:
        json_path = output_dir / source.stem / f"{source.stem}_results.json"
        save_results_json(result, json_path)
        console.print(f"Results saved: {json_path}")

    # Save CSV if not already saved by process_video_file
    if args.save_csv:
        csv_path = output_dir / source.stem / f"{source.stem}_counts.csv"
        if not csv_path.exists():
            save_counts_csv(result, csv_path)
        console.print(f"CSV saved: {csv_path}")


def _detect_directory(config, source: Path, output_dir: Path, args) -> None:
    """Run detection on all images in a directory."""
    import cv2

    from herringnet.models.pipeline import HerringNetPipeline, save_results_json
    from herringnet.visualization.draw_detections import save_annotated_image

    console.print(f"Processing directory: [bold]{source}[/bold]")

    pipeline = HerringNetPipeline(config)
    results = pipeline.process_directory(source)

    total_fish = sum(r.fish_count for r in results)
    console.print(
        f"\nProcessed {len(results)} images, "
        f"detected {total_fish} total fish"
    )

    # Save JSON
    if args.save_json:
        json_path = output_dir / "directory_results.json"
        save_results_json(results, json_path)
        console.print(f"Results saved: {json_path}")

    # Save annotated images
    if args.save_images:
        annotated_dir = output_dir / "annotated"
        annotated_dir.mkdir(parents=True, exist_ok=True)
        for result in results:
            if result.fish_count > 0:
                image = cv2.imread(result.source_path)
                if image is not None:
                    name = Path(result.source_path).stem
                    suffix = Path(result.source_path).suffix
                    out_path = annotated_dir / f"{name}_annotated{suffix}"
                    save_annotated_image(image, result.detections, out_path)
        console.print(f"Annotated images saved to {annotated_dir}")


def _print_frame_result(result) -> None:
    """Print detection results for a single frame."""
    if not result.detections:
        console.print("[yellow]No fish detected.[/yellow]")
        return

    table = Table(title=f"Detections: {result.fish_count} fish")
    table.add_column("#", style="dim")
    table.add_column("Species")
    table.add_column("Confidence", justify="right")
    table.add_column("BBox (x1,y1,x2,y2)")
    table.add_column("Uncertain", justify="center")

    for i, det in enumerate(result.detections):
        species = (
            det.classification.species if det.classification else "fish"
        )
        conf = (
            det.classification.confidence
            if det.classification
            else det.detection.confidence
        )
        bbox = det.detection.bbox
        bbox_str = (
            f"({bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f})"
        )
        uncertain = "[red]yes[/red]" if det.is_uncertain else ""

        table.add_row(
            str(i + 1),
            species.replace("_", " ").title(),
            f"{conf:.1%}",
            bbox_str,
            uncertain,
        )

    console.print(table)


def _print_video_summary(result) -> None:
    """Print a summary of video processing results."""
    counts = result.counts

    console.print("\n[bold]Video Processing Summary[/bold]")
    console.print(f"  Frames extracted:  {result.total_frames_extracted}")
    console.print(f"  Frames with fish:  {counts.frames_with_fish}")
    console.print(f"  Raw fish count:    {counts.raw_total}")
    console.print(f"  Corrected count:   {counts.corrected_total:.1f}")
    console.print(f"  Processing time:   {result.processing_time_seconds:.1f}s")

    if counts.species_counts:
        table = Table(title="Species Breakdown")
        table.add_column("Species")
        table.add_column("Raw Count", justify="right")
        table.add_column("Corrected Count", justify="right")

        for species, pair in sorted(counts.species_counts.items()):
            table.add_row(
                species.replace("_", " ").title(),
                str(pair.raw),
                f"{pair.corrected:.1f}",
            )

        console.print(table)
