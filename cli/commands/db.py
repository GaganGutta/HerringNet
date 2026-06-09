"""CLI subcommand for the image + labeling database and FiftyOne review."""

from __future__ import annotations

import argparse
import logging

from rich.console import Console
from rich.table import Table

console = Console()
logger = logging.getLogger(__name__)


def add_db_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the db subcommand and its actions."""
    parser = subparsers.add_parser(
        "db",
        help="Image database and FiftyOne review workflow",
    )
    parser.add_argument(
        "--db", dest="db_path", type=str, default="data/herringnet.db",
        help="Path to the SQLite database (default: data/herringnet.db)",
    )
    actions = parser.add_subparsers(dest="db_action", help="Database actions")

    actions.add_parser("init", help="Create the database schema")

    p_imp = actions.add_parser("import-excel", help="Migrate an Excel workbook")
    p_imp.add_argument("xlsx", type=str, help="Path to the .xlsx workbook")

    p_scan = actions.add_parser("scan", help="Link image files to database rows")
    p_scan.add_argument("image_root", type=str, help="Root folder of the archive")
    p_scan.add_argument("--session", type=str, default=None,
                        help="Assign all scanned files to this session label")
    p_scan.add_argument("--site", type=str, default=None,
                        help="Site id for new sessions")
    p_scan.add_argument("--no-recursive", action="store_true",
                        help="Do not descend into subfolders")

    p_pred = actions.add_parser("ingest-predictions", help="Load model results JSON")
    p_pred.add_argument("json", type=str, help="Path to a HerringNet results JSON")
    p_pred.add_argument("--model", type=str, default="cfd",
                        help="Model name to record")

    actions.add_parser("stats", help="Show database statistics")

    p_fo = actions.add_parser("review", help="Build a FiftyOne dataset and open it")
    p_fo.add_argument("image_root", type=str, help="Root folder of the archive")
    p_fo.add_argument("--name", type=str, default="herringnet",
                      help="FiftyOne dataset name")
    p_fo.add_argument("--session", action="append", default=None,
                      help="Limit to a session label (repeatable)")
    p_fo.add_argument("--port", type=int, default=5151, help="FiftyOne app port")
    p_fo.add_argument("--no-launch", action="store_true",
                      help="Build the dataset but do not open the app")

    p_sync = actions.add_parser("sync", help="Write FiftyOne edits back to the DB")
    p_sync.add_argument("--name", type=str, default="herringnet",
                        help="FiftyOne dataset name")

    p_yolo = actions.add_parser("export-yolo", help="Export detections to YOLO")
    p_yolo.add_argument("export_dir", type=str, help="Output directory")
    p_yolo.add_argument("--name", type=str, default="herringnet",
                        help="FiftyOne dataset name")
    p_yolo.add_argument("--field", type=str, default="model",
                        help="Detections field to export")

    p_watch = actions.add_parser("watch", help="Watch a drop folder and ingest")
    p_watch.add_argument("incoming", type=str, help="Drop folder to watch")
    p_watch.add_argument("--archive", type=str, default="data/archive",
                         help="Image archive root to move folders into")
    p_watch.add_argument("--interval", type=float, default=15.0,
                         help="Seconds between polls")
    p_watch.add_argument("--no-label-studio", action="store_true",
                         help="Do not push ingested images to Label Studio")

    actions.add_parser("ls-push", help="Push new images to Label Studio as tasks")
    actions.add_parser("ls-pull", help="Pull Label Studio annotations into the DB")

    parser.set_defaults(func=run_db)


def run_db(args: argparse.Namespace) -> None:
    """Execute a db action."""
    from herringnet.database.db import Database

    if args.db_action is None:
        console.print("[yellow]Specify an action: init, import-excel, scan, "
                      "ingest-predictions, stats, review, sync, export-yolo, "
                      "watch, ls-push, ls-pull[/yellow]")
        return

    db = Database(args.db_path)

    if args.db_action == "init":
        db.init_schema()
        console.print(f"[green]Initialized[/green] {args.db_path}")

    elif args.db_action == "import-excel":
        from herringnet.database.import_excel import import_workbook
        db.init_schema()
        console.print(f"Importing [bold]{args.xlsx}[/bold] ...")
        summary = import_workbook(args.xlsx, db)
        _print_summary("Import summary", summary)

    elif args.db_action == "scan":
        from herringnet.database.scan_images import scan_directory
        console.print(f"Scanning [bold]{args.image_root}[/bold] ...")
        summary = scan_directory(
            db, args.image_root, session=args.session, site_id=args.site,
            recursive=not args.no_recursive,
        )
        _print_summary("Scan summary", summary)

    elif args.db_action == "ingest-predictions":
        from herringnet.database.predictions import ingest_results_json
        summary = ingest_results_json(db, args.json, model_name=args.model)
        _print_summary("Prediction ingest", summary)

    elif args.db_action == "stats":
        _print_summary("Database statistics", db.stats())

    elif args.db_action == "review":
        from herringnet.database.fiftyone_io import build_dataset, launch
        console.print("Building FiftyOne dataset ...")
        dataset = build_dataset(
            db, args.image_root, name=args.name, sessions=args.session,
        )
        console.print(f"[green]{len(dataset)} samples[/green] in dataset '{args.name}'")
        if not args.no_launch:
            console.print(f"Launching app on port {args.port} (Ctrl-C to stop) ...")
            launch(dataset, port=args.port, wait=True)

    elif args.db_action == "sync":
        from herringnet.database.fiftyone_io import sync_back
        summary = sync_back(db, name=args.name)
        _print_summary("Sync summary", summary)

    elif args.db_action == "export-yolo":
        from herringnet.database.fiftyone_io import export_yolo
        out = export_yolo(args.name, args.export_dir, label_field=args.field)
        console.print(f"[green]Exported[/green] to {out}")

    elif args.db_action == "watch":
        from herringnet.database.watch import watch_incoming
        ls = None
        if not args.no_label_studio:
            from herringnet.database.label_studio import LabelStudioClient
            ls = LabelStudioClient.from_env()
            if ls is None:
                console.print("[dim]Label Studio not configured (LS_URL/LS_TOKEN/"
                              "LS_PROJECT_ID); ingesting without it.[/dim]")
        console.print(f"Watching [bold]{args.incoming}[/bold] (Ctrl-C to stop) ...")
        watch_incoming(db, args.incoming, args.archive,
                       interval=args.interval, label_studio=ls)

    elif args.db_action == "ls-push":
        from herringnet.database.label_studio import LabelStudioClient
        ls = LabelStudioClient.from_env()
        if ls is None:
            console.print("[red]Set LS_URL, LS_TOKEN, LS_PROJECT_ID first.[/red]")
        else:
            _print_summary("Label Studio push", ls.push_new_tasks(db))

    elif args.db_action == "ls-pull":
        from herringnet.database.label_studio import LabelStudioClient
        ls = LabelStudioClient.from_env()
        if ls is None:
            console.print("[red]Set LS_URL, LS_TOKEN, LS_PROJECT_ID first.[/red]")
        else:
            _print_summary("Label Studio pull", ls.pull_annotations(db))

    db.close()


def _print_summary(title: str, summary: dict) -> None:
    table = Table(title=title)
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    for k, v in summary.items():
        table.add_row(str(k).replace("_", " "), str(v))
    console.print(table)
