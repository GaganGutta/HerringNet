"""HerringNet command-line interface entry point.

Provides subcommands for detection, training, frame extraction,
active learning review, data setup, and launching the demo.
"""

from __future__ import annotations

import argparse
import logging
import sys


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the CLI.

    Args:
        verbose: If True, set log level to DEBUG. Otherwise INFO.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> None:
    """Main entry point for the herringnet CLI."""
    parser = argparse.ArgumentParser(
        prog="herringnet",
        description="HerringNet: Fish detection and species classification",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose (debug) logging",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Import and register subcommands
    from cli.commands.batch import add_batch_parser
    from cli.commands.db import add_db_parser
    from cli.commands.detect import add_detect_parser
    from cli.commands.extract import add_extract_parser
    from cli.commands.review import add_review_parser
    from cli.commands.setup_data import add_setup_data_parser
    from cli.commands.train import add_train_parser

    add_detect_parser(subparsers)
    add_batch_parser(subparsers)
    add_extract_parser(subparsers)
    add_train_parser(subparsers)
    add_setup_data_parser(subparsers)
    add_review_parser(subparsers)
    add_db_parser(subparsers)

    # Demo subcommand (registered inline since it is simple)
    demo_parser = subparsers.add_parser("demo", help="Launch the Gradio demo")
    demo_parser.add_argument(
        "--config", type=str, default=None, help="Path to config YAML"
    )
    demo_parser.add_argument(
        "--port", type=int, default=7860, help="Port for the Gradio server"
    )
    demo_parser.add_argument(
        "--share", action="store_true", help="Create a public shareable link"
    )

    args = parser.parse_args()
    setup_logging(verbose=args.verbose)

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "demo":
        _run_demo(args)
    elif hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()
        sys.exit(1)


def _run_demo(args: argparse.Namespace) -> None:
    """Launch the Gradio demo interface."""
    from app.demo import create_demo
    from herringnet.config import load_config

    config = load_config(config_path=args.config)
    demo = create_demo(config)
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
