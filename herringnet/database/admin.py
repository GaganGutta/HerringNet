"""Administrative operations on the image database.

Currently: deleting a session (a site-week folder) and everything tied
to it — image rows, observations, expert reviews, model predictions,
Label Studio bookkeeping, and the archived files on disk. Used by both
the CLI (`herringnet db delete-session`) and the upload page's session
manager. The FiftyOne server prunes deleted images on its next refresh.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from herringnet.database.db import Database

logger = logging.getLogger(__name__)

# Child tables that may reference images, in deletion order.
_CHILD_TABLES = ("model_predictions", "observations", "expert_reviews", "ls_tasks")


def _table_exists(db: Database, name: str) -> bool:
    return bool(db.scalar(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ))


def delete_session(
    db: Database,
    label: str,
    archive_dir: str | Path | None = None,
    keep_files: bool = False,
) -> dict[str, int]:
    """Delete a session and all records (and optionally files) under it.

    Args:
        db: Initialized database.
        label: The session label (folder name) to delete.
        archive_dir: Image archive root; the session's subfolder is removed
            from it unless keep_files is set.
        keep_files: If True, only database records are removed.

    Returns:
        Summary dict: sessions, images, label_rows deleted, files_removed.
    """
    sids = [
        int(r["session_id"]) for r in db.query(
            "SELECT session_id FROM sessions WHERE label = ?", (label,)
        )
    ]
    summary = {"sessions": 0, "images": 0, "label_rows": 0, "files_removed": 0}
    if not sids:
        return summary

    for sid in sids:
        image_ids = [
            int(r["image_id"]) for r in db.query(
                "SELECT image_id FROM images WHERE session_id = ?", (sid,)
            )
        ]
        if image_ids:
            ph = ",".join("?" * len(image_ids))
            for table in _CHILD_TABLES:
                if _table_exists(db, table):
                    cur = db.execute(
                        f"DELETE FROM {table} WHERE image_id IN ({ph})", image_ids
                    )
                    summary["label_rows"] += cur.rowcount
            db.execute(f"DELETE FROM images WHERE image_id IN ({ph})", image_ids)
            summary["images"] += len(image_ids)
        db.execute("DELETE FROM sessions WHERE session_id = ?", (sid,))
        summary["sessions"] += 1

    if archive_dir is not None and not keep_files:
        target = Path(archive_dir) / label
        if target.exists():
            summary["files_removed"] = sum(1 for p in target.rglob("*") if p.is_file())
            shutil.rmtree(target, ignore_errors=True)

    db.commit()
    logger.info("Deleted session '%s': %s", label, summary)
    return summary


def list_sessions(db: Database) -> list[dict]:
    """Return all sessions with image counts, for display/management."""
    return [
        dict(r) for r in db.query(
            """
            SELECT s.label,
                   COUNT(i.image_id)               AS images,
                   SUM(COALESCE(i.file_exists, 0)) AS with_files
              FROM sessions s
              LEFT JOIN images i ON i.session_id = s.session_id
             GROUP BY s.session_id
             ORDER BY s.label
            """
        )
    ]
