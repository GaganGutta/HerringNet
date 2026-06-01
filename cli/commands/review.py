"""CLI subcommand for managing the active learning review queue."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()
logger = logging.getLogger(__name__)


def add_review_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the review subcommand."""
    parser = subparsers.add_parser(
        "review",
        help="Manage the active learning review queue",
    )
    parser.add_argument(
        "--queue-dir", type=str, default="data/review_queue",
        help="Review queue directory (default: data/review_queue)",
    )
    parser.add_argument(
        "--export", type=str, default=None,
        help="Export review queue as CSV to this path",
    )
    parser.add_argument(
        "--import-corrections", type=str, default=None,
        help="Import corrected labels from CSV",
    )
    parser.add_argument(
        "--dataset-dir", type=str, default="data/processed",
        help="Training dataset directory for importing corrections",
    )
    parser.add_argument(
        "--stats", action="store_true",
        help="Show review queue statistics",
    )
    parser.set_defaults(func=run_review)


def run_review(args: argparse.Namespace) -> None:
    """Execute the review command."""
    from herringnet.training.active_learning import ActiveLearningManager
    from herringnet.config import ActiveLearningConfig

    al_config = ActiveLearningConfig(review_output_dir=args.queue_dir)
    manager = ActiveLearningManager(al_config)

    if args.stats:
        _show_stats(manager, args.queue_dir)
    elif args.export:
        csv_path = manager.export_review_queue(args.export)
        console.print(f"Review queue exported to: {csv_path}")
    elif args.import_corrections:
        count = manager.import_corrections(
            args.import_corrections, args.dataset_dir
        )
        console.print(f"Imported {count} corrections into {args.dataset_dir}")
    else:
        _show_stats(manager, args.queue_dir)


def _show_stats(manager, queue_dir: str) -> None:
    """Display review queue statistics."""
    queue_path = Path(queue_dir)
    if not queue_path.exists():
        console.print("[yellow]Review queue directory does not exist yet.[/yellow]")
        console.print("Run detection with active learning enabled to populate it.")
        return

    # Count items in the queue
    json_files = list(queue_path.glob("*.json"))
    image_files = list(queue_path.glob("*.jpg")) + list(queue_path.glob("*.png"))

    console.print(f"[bold]Review Queue Statistics[/bold]")
    console.print(f"  Queue directory:   {queue_path}")
    console.print(f"  Items in queue:    {len(json_files)}")
    console.print(f"  Image files:       {len(image_files)}")

    if not json_files:
        console.print("\n[dim]No items in the review queue.[/dim]")
        return

    # Summarize reasons
    import json as json_mod

    reasons: dict[str, int] = {}
    for jf in json_files:
        try:
            with open(jf) as f:
                data = json_mod.load(f)
            reason = data.get("uncertainty_reason", "unknown")
            reasons[reason] = reasons.get(reason, 0) + 1
        except Exception:
            pass

    if reasons:
        table = Table(title="Flagging Reasons")
        table.add_column("Reason")
        table.add_column("Count", justify="right")
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            table.add_row(reason, str(count))
        console.print(table)
