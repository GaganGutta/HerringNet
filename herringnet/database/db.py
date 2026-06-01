"""Thin SQLite wrapper for the HerringNet image database.

Provides connection management (with foreign keys enabled), schema
initialization, and small helpers used by the import/scan/export
modules. Keeping this minimal on purpose: SQL lives close to where it
is used, this class just manages the connection and common operations.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from herringnet.database.schema import SCHEMA_SQL, SCHEMA_VERSION

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = "data/herringnet.db"


class Database:
    """Connection holder for the HerringNet SQLite database.

    Args:
        db_path: Path to the SQLite file. Created if it does not exist.
    """

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        # Foreign keys are off by default in SQLite; turn them on per connection.
        self.conn.execute("PRAGMA foreign_keys = ON")

    # -- lifecycle ---------------------------------------------------------

    def init_schema(self) -> None:
        """Create all tables and indexes if they do not already exist."""
        self.conn.executescript(SCHEMA_SQL)
        self.conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        self.conn.commit()
        logger.info("Initialized schema (v%d) at %s", SCHEMA_VERSION, self.db_path)

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> Database:
        return self

    def __exit__(self, *exc: object) -> None:
        self.conn.commit()
        self.close()

    # -- low-level helpers -------------------------------------------------

    def execute(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Cursor:
        return self.conn.execute(sql, params)

    def executemany(self, sql: str, rows: Iterable[Sequence[Any]]) -> sqlite3.Cursor:
        return self.conn.executemany(sql, rows)

    def commit(self) -> None:
        self.conn.commit()

    def query(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        return list(self.conn.execute(sql, params).fetchall())

    def scalar(self, sql: str, params: Sequence[Any] = ()) -> Any:
        row = self.conn.execute(sql, params).fetchone()
        return row[0] if row else None

    # -- common operations -------------------------------------------------

    def upsert(
        self, table: str, data: dict[str, Any], conflict: str = "REPLACE"
    ) -> None:
        """Insert a row, replacing on primary-key/unique conflict.

        Args:
            table: Target table name.
            data: Column name -> value mapping.
            conflict: SQLite conflict resolution (REPLACE, IGNORE, ...).
        """
        cols = ", ".join(data)
        placeholders = ", ".join("?" for _ in data)
        sql = (
            f"INSERT OR {conflict} INTO {table} ({cols}) VALUES ({placeholders})"
        )
        self.conn.execute(sql, list(data.values()))

    def get_or_create_session(
        self,
        label: str,
        site_id: str | None = None,
        **fields: Any,
    ) -> int:
        """Return the session_id for a session label, creating it if needed."""
        existing = self.scalar(
            "SELECT session_id FROM sessions WHERE label = ?", (label,)
        )
        if existing is not None:
            return int(existing)

        cols = {"label": label, "site_id": site_id, **fields}
        col_names = ", ".join(cols)
        placeholders = ", ".join("?" for _ in cols)
        cur = self.conn.execute(
            f"INSERT INTO sessions ({col_names}) VALUES ({placeholders})",
            list(cols.values()),
        )
        return int(cur.lastrowid)

    def get_or_create_image(
        self,
        session_id: int,
        original_name: str,
        **fields: Any,
    ) -> int:
        """Return the image_id for (session, original_name), creating if needed.

        Existing rows are updated with any non-null fields provided, so this
        is safe to call from both the Excel migration and the folder scan.
        """
        existing = self.scalar(
            "SELECT image_id FROM images WHERE session_id = ? AND original_name = ?",
            (session_id, original_name),
        )
        if existing is not None:
            image_id = int(existing)
            updates = {k: v for k, v in fields.items() if v is not None}
            if updates:
                assignments = ", ".join(f"{k} = ?" for k in updates)
                self.conn.execute(
                    f"UPDATE images SET {assignments} WHERE image_id = ?",
                    [*updates.values(), image_id],
                )
            return image_id

        cols = {"session_id": session_id, "original_name": original_name, **fields}
        col_names = ", ".join(cols)
        placeholders = ", ".join("?" for _ in cols)
        cur = self.conn.execute(
            f"INSERT INTO images ({col_names}) VALUES ({placeholders})",
            list(cols.values()),
        )
        return int(cur.lastrowid)

    def stats(self) -> dict[str, Any]:
        """Return a summary of row counts and key data-quality figures."""
        counts = {
            t: self.scalar(f"SELECT COUNT(*) FROM {t}")
            for t in (
                "sites",
                "observers",
                "sessions",
                "images",
                "observations",
                "expert_reviews",
                "model_predictions",
            )
        }
        counts["images_with_files"] = self.scalar(
            "SELECT COUNT(*) FROM images WHERE file_exists = 1"
        )
        counts["images_rh_present"] = self.scalar(
            "SELECT COUNT(DISTINCT image_id) FROM observations WHERE rh_present = 1"
        )
        counts["images_expert_reviewed"] = self.scalar(
            "SELECT COUNT(DISTINCT image_id) FROM expert_reviews "
            "WHERE expert_reviewed = 1"
        )
        return counts
