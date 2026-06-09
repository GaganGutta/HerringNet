"""Tests for the image + labeling database: cleaning, migration, scan."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from herringnet.database.db import Database
from herringnet.database.import_excel import (
    _clean_reason,
    _normalize_observer,
    _to_int,
    _yn_to_int,
    import_workbook,
)


class TestCleaningHelpers:
    """Unit tests for the value-cleaning functions."""

    def test_yn_to_int(self):
        assert _yn_to_int("Y") == 1
        assert _yn_to_int("n") == 0
        assert _yn_to_int("yes") == 1
        assert _yn_to_int("") is None
        assert _yn_to_int(None) is None
        assert _yn_to_int("maybe") is None

    def test_to_int(self):
        assert _to_int("2") == 2
        assert _to_int("2.0") == 2          # float artifact from Excel
        assert _to_int("?") is None
        assert _to_int(None) is None

    def test_normalize_observer(self):
        assert _normalize_observer("OBS_002") == "OBS002"
        assert _normalize_observer("obs002") == "OBS002"
        assert _normalize_observer(None) is None

    def test_clean_reason_valid(self):
        code, note = _clean_reason("2", {1, 2, 3})
        assert code == 2 and note is None

    def test_clean_reason_invalid_is_quarantined(self):
        code, note = _clean_reason("6", {1, 2, 3})       # out of range
        assert code is None and "6" in note
        code, note = _clean_reason("2,3", {1, 2, 3})     # multi-value
        assert code is None and note is not None


def _make_workbook(path) -> str:
    """Write a tiny workbook mimicking the real one's structure."""
    with pd.ExcelWriter(path) as xw:
        pd.DataFrame({
            "site_ID": ["BRIDE"],
            "site_name": ["Bride Lake"],
            "location_description": ["spillway"],
            "camera_type": ["Wireless"],
            "installation_date": ["?"],
            "notes": [None],
        }).to_excel(xw, sheet_name="Sites", index=False)

        pd.DataFrame({
            "observer_id": ["OBS002"],
            "observer_name": ["Evan"],
            "role": ["General Observer"],
            "email": ["e@umass.edu"],
            "training_date": [None],
            "active": ["Yes"],
            "notes": [None],
        }).to_excel(xw, sheet_name="Observers", index=False)

        pd.DataFrame({
            "Category_shortcut": [1, 2],
            "category": ["Technical Malfunction", "Lighting Malfunction"],
            "description": ["x", "y"],
            "common_causes": ["a", "b"],
        }).to_excel(xw, sheet_name="Uncountable_Categories", index=False)

        # Data sheet with messy values to exercise cleaning.
        pd.DataFrame({
            "image_id": ["GOPRO001", "GOPRO002", "GOPRO003"],
            "site_name": ["Bride Lake"] * 3,
            "camera_id": ["BRIDE"] * 3,
            "date_captured": ["2025-06-13"] * 3,
            "time_captured": ["10:00:00"] * 3,
            "countable (Y/N)": ["Y", "N", "Y"],
            "uncountable_reason": [np.nan, 2, 6],      # 6 is invalid -> quarantine
            "rh_present (Y/N)": ["Y", "N", "N"],
            "rh_count": [5, "?", 0],                   # '?' -> None + note
            "count_bin": [1, np.nan, np.nan],
            "observer_id": ["OBS_002", "OBS002", np.nan],
            "expert_id": ["EXP001", np.nan, np.nan],
            "expert_review_date": ["2026-02-02", np.nan, np.nan],
            "notes": [None, None, None],
        }).to_excel(xw, sheet_name="BRIDE testweek", index=False)
    return str(path)


class TestMigration:
    """Round-trip migration into a temp database."""

    def test_import(self, tmp_path):
        xlsx = _make_workbook(tmp_path / "wb.xlsx")
        db = Database(tmp_path / "test.db")
        db.init_schema()
        summary = import_workbook(xlsx, db)

        assert summary["sessions"] == 1
        assert summary["images"] == 3
        assert summary["observations"] == 3
        assert summary["quarantined_reasons"] == 1   # the '6'
        assert summary["unparsed_counts"] == 1        # the '?'

        # Y/N columns parsed despite the "(Y/N)" suffix.
        countable = {r[0]: r[1] for r in db.query(
            "SELECT countable, COUNT(*) FROM observations GROUP BY countable"
        )}
        assert countable.get(1) == 2 and countable.get(0) == 1

        # Observer id normalized.
        assert db.scalar(
            "SELECT COUNT(*) FROM observations WHERE observer_id = 'OBS002'"
        ) == 2

        # Foreign keys hold.
        assert list(db.execute("PRAGMA foreign_key_check").fetchall()) == []

        # Expert review recorded for the one row with expert fields.
        assert db.scalar("SELECT COUNT(*) FROM expert_reviews") == 1
        db.close()


class TestScan:
    """Folder scan links files to rows and creates new ones."""

    def test_scan_creates_rows(self, tmp_path):
        pytest.importorskip("PIL")
        from PIL import Image

        root = tmp_path / "archive"
        week = root / "BRIDE testweek"
        week.mkdir(parents=True)
        for name in ("GOPRO001", "GOPRO999"):
            Image.new("RGB", (64, 48), (10, 20, 30)).save(week / f"{name}.jpg")

        xlsx = _make_workbook(tmp_path / "wb.xlsx")
        db = Database(tmp_path / "test.db")
        db.init_schema()
        import_workbook(xlsx, db)

        from herringnet.database.scan_images import scan_directory
        summary = scan_directory(db, root)

        assert summary["files"] == 2
        # GOPRO001 matches the migrated row; GOPRO999 is new.
        assert summary["updated"] == 1
        assert summary["created"] == 1
        assert db.scalar("SELECT COUNT(*) FROM images WHERE file_exists = 1") == 2
        db.close()


class TestWatcher:
    """The folder-drop ingest moves files into the archive and records them."""

    def test_process_folder(self, tmp_path):
        pytest.importorskip("PIL")
        from PIL import Image

        from herringnet.database.watch import process_folder

        incoming = tmp_path / "incoming"
        archive = tmp_path / "archive"
        dropped = incoming / "BRIDE_week9"
        dropped.mkdir(parents=True)
        for name in ("GOPRO100", "GOPRO101", "GOPRO102"):
            Image.new("RGB", (32, 24), (5, 5, 5)).save(dropped / f"{name}.jpg")

        db = Database(tmp_path / "test.db")
        db.init_schema()
        summary = process_folder(db, dropped, archive)

        assert summary["moved"] == 3
        assert summary["created"] == 3
        # Files now live under the archive, organized by session.
        assert (archive / "BRIDE_week9" / "GOPRO100.jpg").exists()
        # The dropped folder is cleaned up after ingest.
        assert not dropped.exists()
        # A session was created from the folder name, with linked images.
        assert db.scalar("SELECT COUNT(*) FROM sessions WHERE label='BRIDE_week9'") == 1
        assert db.scalar("SELECT COUNT(*) FROM images WHERE file_exists=1") == 3
        # Paths are stored relative to the archive root.
        path = db.scalar("SELECT file_path FROM images LIMIT 1")
        assert path.startswith("BRIDE_week9/")
        db.close()
