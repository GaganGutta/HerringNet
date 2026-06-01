"""CLI subcommand for downloading and preparing training datasets."""

from __future__ import annotations

import argparse
import logging
import os

from rich.console import Console

console = Console()
logger = logging.getLogger(__name__)


def add_setup_data_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the setup-data subcommand."""
    parser = subparsers.add_parser(
        "setup-data",
        help="Download and prepare training datasets",
    )
    parser.add_argument(
        "--roboflow-key", type=str, default=None,
        help="Roboflow API key (or set ROBOFLOW_API_KEY env var)",
    )
    parser.add_argument(
        "--dataset", type=str, default="all",
        choices=["mit-herring", "ml-herring", "all"],
        help="Dataset to download (default: all)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/raw",
        help="Directory to save downloaded datasets (default: data/raw)",
    )
    parser.add_argument(
        "--format", type=str, default="yolov8",
        help="Export format (default: yolov8)",
    )
    parser.set_defaults(func=run_setup_data)


def run_setup_data(args: argparse.Namespace) -> None:
    """Execute the setup-data command."""
    from herringnet.data.dataset_manager import DatasetManager

    api_key = args.roboflow_key or os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        console.print(
            "[red]Error: Roboflow API key required. "
            "Set ROBOFLOW_API_KEY env var or use --roboflow-key.[/red]"
        )
        console.print(
            "Get your API key at: https://app.roboflow.com/settings/api"
        )
        return

    manager = DatasetManager(data_dir=args.output_dir, roboflow_api_key=api_key)

    datasets_to_download = []
    if args.dataset in ("mit-herring", "all"):
        datasets_to_download.append({
            "name": "MIT River Herring 2",
            "workspace": "mit-fishery",
            "project": "mit-river-herring-2",
            "version": 1,
        })
    if args.dataset in ("ml-herring", "all"):
        datasets_to_download.append({
            "name": "MLRiverHerring",
            "workspace": "mlriverherring",
            "project": "mlriverherring",
            "version": 1,
        })

    for ds in datasets_to_download:
        console.print(f"Downloading: [bold]{ds['name']}[/bold]")
        try:
            path = manager.download_roboflow_dataset(
                workspace=ds["workspace"],
                project=ds["project"],
                version=ds["version"],
                export_format=args.format,
            )
            console.print(f"  Saved to: {path}")
        except Exception as e:
            console.print(f"  [red]Failed: {e}[/red]")
            logger.exception("Dataset download failed: %s", ds["name"])

    console.print("\n[green]Dataset setup complete.[/green]")
    console.print(
        "Use 'herringnet train classify --data <path>/data.yaml' to train."
    )
