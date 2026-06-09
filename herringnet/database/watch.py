"""Folder watcher: the smooth "drop a folder of images" ingest path.

Watches an ``incoming`` directory. When a subfolder appears and stops
changing (so we do not ingest a half-finished upload), the watcher:

  1. moves it into the image archive under its own name (the session),
  2. scans the moved folder into the database (hash, EXIF, file link),
  3. optionally pushes the new images to Label Studio as tasks.

The subfolder name becomes the session label, so dropping a folder named
"BRIDE_week5" creates/extends that session. Uses only the standard
library for the watch loop (polling), so there is no extra dependency.
"""

from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path

from herringnet.database.db import Database
from herringnet.database.scan_images import IMAGE_EXTENSIONS, scan_directory

logger = logging.getLogger(__name__)


def _folder_is_settled(folder: Path, settle_seconds: float) -> bool:
    """True if no image in the folder was modified within settle_seconds.

    This avoids ingesting a folder that is still being uploaded.
    """
    images = [
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not images:
        return False
    newest = max(p.stat().st_mtime for p in images)
    return (time.time() - newest) >= settle_seconds


def process_folder(
    db: Database,
    folder: Path,
    archive_dir: Path,
    label_studio=None,
) -> dict[str, int]:
    """Ingest a single dropped folder: move into the archive, then scan.

    Args:
        db: Initialized database.
        folder: The dropped folder inside the incoming directory.
        archive_dir: The image archive root (stored paths are relative to it).
        label_studio: Optional LabelStudioClient to push new tasks to.

    Returns:
        The scan summary for this folder.
    """
    session_label = folder.name
    dest = archive_dir / session_label
    dest.mkdir(parents=True, exist_ok=True)

    # Move image files into the archive (merge if the session already exists).
    moved = 0
    for src in folder.rglob("*"):
        if src.is_file() and src.suffix.lower() in IMAGE_EXTENSIONS:
            target = dest / src.name
            shutil.move(str(src), str(target))
            moved += 1

    # Clean up the now-empty dropped folder.
    shutil.rmtree(folder, ignore_errors=True)

    summary = scan_directory(db, archive_dir, only_under=session_label)
    summary["moved"] = moved
    logger.info("Ingested dropped folder '%s': %s", session_label, summary)

    if label_studio is not None:
        try:
            label_studio.push_new_tasks(db, session=session_label)
        except Exception as e:  # noqa: BLE001 - ingest must not fail on LS
            logger.warning("Label Studio push failed for %s: %s", session_label, e)

    return summary


def watch_incoming(
    db: Database,
    incoming_dir: str | Path,
    archive_dir: str | Path,
    interval: float = 15.0,
    settle_seconds: float = 20.0,
    label_studio=None,
) -> None:
    """Poll an incoming directory and ingest settled folders forever.

    Args:
        db: Initialized database.
        incoming_dir: Drop zone watched for new folders.
        archive_dir: Image archive root that folders are moved into.
        interval: Seconds between polls.
        settle_seconds: A folder is ingested only after this many seconds
            with no file changes (so partial uploads are not ingested).
        label_studio: Optional LabelStudioClient to push new tasks to.
    """
    incoming_dir = Path(incoming_dir)
    archive_dir = Path(archive_dir)
    incoming_dir.mkdir(parents=True, exist_ok=True)
    archive_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Watching %s (archive=%s, every %.0fs)",
        incoming_dir, archive_dir, interval,
    )
    while True:
        try:
            for folder in sorted(p for p in incoming_dir.iterdir() if p.is_dir()):
                if _folder_is_settled(folder, settle_seconds):
                    process_folder(db, folder, archive_dir, label_studio)
        except Exception as e:  # noqa: BLE001 - keep the loop alive
            logger.exception("Watch loop error: %s", e)
        time.sleep(interval)
