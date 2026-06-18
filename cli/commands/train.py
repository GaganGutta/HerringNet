"""CLI subcommand for training and fine-tuning models."""

from __future__ import annotations

import argparse
import logging

from rich.console import Console

console = Console()
logger = logging.getLogger(__name__)


def add_train_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the train subcommand."""
    parser = subparsers.add_parser(
        "train",
        help="Fine-tune detection or classification models",
    )
    parser.add_argument(
        "task",
        type=str,
        choices=["classify", "detect"],
        help="Training task: 'classify' for species classifier, 'detect' for detector",
    )
    parser.add_argument(
        "--data", type=str, required=True,
        help="Path to data.yaml for training",
    )
    parser.add_argument(
        "--base-model", type=str, default=None,
        help="Base model to fine-tune "
             "(default: yolov8n-cls.pt for classify, yolov8s.pt for detect)",
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Number of training epochs (default: 50)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="Batch size (default: 16)",
    )
    parser.add_argument(
        "--image-size", type=int, default=None,
        help="Training image size (default: 224 for classify, 640 for detect)",
    )
    parser.add_argument(
        "--freeze", type=int, default=None,
        help="Number of backbone layers to freeze (default: None)",
    )
    parser.add_argument(
        "--project", type=str, default="runs",
        help="Output directory for training runs (default: runs)",
    )
    parser.add_argument(
        "--name", type=str, default=None,
        help="Experiment name (default: auto-generated)",
    )
    parser.set_defaults(func=run_train)


def run_train(args: argparse.Namespace) -> None:
    """Execute the train command."""
    from herringnet.training.trainer import HerringNetTrainer

    trainer = HerringNetTrainer()

    if args.task == "classify":
        base_model = args.base_model or "yolov8n-cls.pt"
        image_size = args.image_size or 224
        name = args.name or "herringnet-cls"

        console.print("[bold]Training species classifier[/bold]")
        console.print(f"  Base model:  {base_model}")
        console.print(f"  Data:        {args.data}")
        console.print(f"  Epochs:      {args.epochs}")
        console.print(f"  Batch size:  {args.batch_size}")
        console.print(f"  Image size:  {image_size}")

        best_weights = trainer.train_classifier(
            data_path=args.data,
            base_model=base_model,
            epochs=args.epochs,
            image_size=image_size,
            batch_size=args.batch_size,
            freeze_layers=args.freeze,
            project=args.project,
            name=name,
        )
        console.print("\n[green]Training complete![/green]")
        console.print(f"Best weights: {best_weights}")
        console.print(
            "\nTo use this model, set classifier.model_path in your config:"
        )
        console.print(f'  classifier.model_path: "{best_weights}"')

    elif args.task == "detect":
        base_model = args.base_model or "yolov8s.pt"
        image_size = args.image_size or 640
        name = args.name or "herringnet-det"

        console.print("[bold]Training fish detector[/bold]")
        console.print(f"  Base model:  {base_model}")
        console.print(f"  Data:        {args.data}")
        console.print(f"  Epochs:      {args.epochs}")
        console.print(f"  Batch size:  {args.batch_size}")
        console.print(f"  Image size:  {image_size}")

        best_weights = trainer.train_detector(
            data_path=args.data,
            base_model=base_model,
            epochs=args.epochs,
            image_size=image_size,
            batch_size=args.batch_size,
            freeze_layers=args.freeze,
            project=args.project,
            name=name,
        )
        console.print("\n[green]Training complete![/green]")
        console.print(f"Best weights: {best_weights}")
