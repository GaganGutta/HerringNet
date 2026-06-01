"""Migrate the legacy River Herring Excel workbook into the SQLite database.

The workbook stores one sheet per site-week plus several lookup sheets.
This module loads the lookups, then walks every data sheet and writes
one `images` row and one `observations` row per image, cleaning the
values as it goes:

  - duplicate image ids across weeks are fine: images are keyed by
    (session, original_name), so each week's GoPro numbering is isolated,
  - inconsistent observer ids (OBS_002 vs OBS002) are normalized,
  - invalid `uncountable_reason` values (codes outside the lookup,
    date-corrupted cells, "2,3" multi-values) are quarantined into the
    observation `notes` instead of being dropped or silently kept,
  - Y/N strings become 1/0/NULL integers, numeric-looking floats ("2.0")
    become ints.

Nothing is discarded silently. Every value that cannot be mapped cleanly
is preserved verbatim in a note.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd

from herringnet.database.db import Database

logger = logging.getLogger(__name__)

# Sheets that are reference tables rather than per-week image logs.
LOOKUP_SHEETS = {
    "Dashboard",
    "Sites",
    "Observers",
    "Count_Bins",
    "Uncountable_Categories",
    "Important Dates",
}

# The Count_Bins sheet is corrupted by Excel auto-formatting ("1-10" became
# a date), so the correct mapping is hard-coded here from the project legend.
COUNT_BINS = [
    (1, "1-10", 1, 10),
    (2, "10-100", 10, 100),
    (3, "100-500", 100, 500),
    (4, "500+", 500, None),
]


def _clean_str(value: Any) -> str | None:
    """Return a stripped string, or None for blanks/NaN."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    s = str(value).strip()
    if s == "" or s.lower() in {"nan", "na", "none"}:
        return None
    return s


def _yn_to_int(value: Any) -> int | None:
    """Map Y/N (and yes/no/true/false) to 1/0, anything else to None."""
    s = _clean_str(value)
    if s is None:
        return None
    s = s.lower()
    if s in {"y", "yes", "true", "1"}:
        return 1
    if s in {"n", "no", "false", "0"}:
        return 0
    return None


def _to_int(value: Any) -> int | None:
    """Coerce ints, int-like floats ('2.0'), and digit strings to int."""
    s = _clean_str(value)
    if s is None:
        return None
    try:
        return int(float(s))
    except (ValueError, TypeError):
        return None


def _normalize_observer(value: Any) -> str | None:
    """Normalize observer ids: strip underscores so OBS_002 -> OBS002."""
    s = _clean_str(value)
    if s is None:
        return None
    return s.replace("_", "").upper()


def _clean_reason(value: Any, valid: set[int]) -> tuple[int | None, str | None]:
    """Clean an uncountable_reason cell.

    Returns (reason_code_or_None, note_or_None). A note is produced
    whenever the raw value is not a single valid code, so the original
    is never lost.
    """
    s = _clean_str(value)
    if s is None:
        return None, None

    # Single clean integer code.
    code = _to_int(s)
    if code is not None and code in valid:
        return code, None

    # Anything else (multi-value "2,3", date-corrupted, out-of-range "6")
    # is preserved verbatim as a note for later manual repair.
    return None, f"raw_uncountable_reason={s!r}"


def _parse_date(value: Any) -> str | None:
    """Return an ISO date string for date-like cells, else None."""
    s = _clean_str(value)
    if s is None:
        return None
    try:
        ts = pd.to_datetime(value)
        return ts.date().isoformat()
    except (ValueError, TypeError):
        return s


def _parse_time(value: Any) -> str | None:
    s = _clean_str(value)
    if s is None:
        return None
    # Excel times often arrive as 'HH:MM:SS'; keep as-is if so.
    return s


def _site_from_sheet(sheet_name: str, data_site_name: str | None) -> str:
    """Resolve a site_id from the sheet name prefix or the row's site_name."""
    prefix = sheet_name.split()[0].upper() if sheet_name.split() else ""
    # Both "BRIDE ..." and "BL ..." sheets are Bride Lake in this workbook.
    if prefix in {"BRIDE", "BL"}:
        return "BRIDE"
    if data_site_name:
        return data_site_name.strip().upper().replace(" ", "_")
    return prefix or "UNKNOWN"


def _append_note(base: str | None, extra: str | None) -> str | None:
    parts = [p for p in (base, extra) if p]
    return " | ".join(parts) if parts else None


def _ensure_observer(db: Database, observer_id: str | None) -> None:
    """Create a stub observer row if this id is not already known.

    Data sheets reference observer/expert ids that are sometimes missing
    from the Observers sheet. Rather than fail the foreign key or drop the
    id, record a stub so the audit trail (who labeled it) survives.
    """
    if not observer_id:
        return
    db.execute(
        "INSERT OR IGNORE INTO observers (observer_id, notes) VALUES (?, ?)",
        (observer_id, "auto-created during import (absent from Observers sheet)"),
    )


def import_workbook(xlsx_path: str | Path, db: Database) -> dict[str, int]:
    """Load an Excel workbook into the database.

    Args:
        xlsx_path: Path to the .xlsx workbook.
        db: An initialized Database (schema already created).

    Returns:
        Summary dict with counts of rows written and values quarantined.
    """
    xlsx_path = Path(xlsx_path)
    xl = pd.ExcelFile(xlsx_path)

    _load_lookups(xl, db)
    valid_reasons = {
        int(r["category_id"]) for r in db.query(
            "SELECT category_id FROM uncountable_categories"
        )
    }
    valid_bins = {
        int(r["bin_code"]) for r in db.query("SELECT bin_code FROM count_bins")
    }

    summary = {
        "sessions": 0,
        "images": 0,
        "observations": 0,
        "expert_reviews": 0,
        "quarantined_reasons": 0,
        "unparsed_counts": 0,
    }

    data_sheets = [s for s in xl.sheet_names if s not in LOOKUP_SHEETS]
    for sheet in data_sheets:
        df = pd.read_excel(xl, sheet_name=sheet)
        df = df.dropna(how="all")
        # Drop fully-unnamed trailing columns from ragged sheets.
        df = df[[c for c in df.columns if not str(c).startswith("Unnamed")]]
        # Normalize headers: strip whitespace and drop "(Y/N)" suffixes so
        # "countable (Y/N)" and "rh_present (Y/N)" resolve to clean names.
        df = df.rename(
            columns=lambda c: re.sub(r"\s*\(y/n\)\s*", "", str(c), flags=re.I).strip()
        )
        if "image_id" not in df.columns or df.empty:
            logger.warning("Skipping sheet with no image_id column: %s", sheet)
            continue

        first_site = _clean_str(
            df["site_name"].dropna().iloc[0] if "site_name" in df else None
        )
        site_id = _site_from_sheet(sheet, first_site)
        session_id = db.get_or_create_session(label=sheet, site_id=site_id)
        summary["sessions"] += 1

        for _, row in df.iterrows():
            original_name = _clean_str(row.get("image_id"))
            if original_name is None:
                continue

            image_id = db.get_or_create_image(
                session_id=session_id,
                original_name=original_name,
                site_id=site_id,
                camera_id=_clean_str(row.get("camera_id")),
                date_captured=_parse_date(row.get("date_captured")),
                time_captured=_parse_time(row.get("time_captured")),
            )
            summary["images"] += 1

            # -- observation --------------------------------------------
            reason, reason_note = _clean_reason(
                row.get("uncountable_reason"), valid_reasons
            )
            if reason_note:
                summary["quarantined_reasons"] += 1

            rh_count_raw = _clean_str(row.get("rh_count"))
            rh_count = _to_int(row.get("rh_count"))
            count_note = None
            if rh_count_raw is not None and rh_count is None:
                count_note = f"raw_rh_count={rh_count_raw!r}"
                summary["unparsed_counts"] += 1

            count_bin = _to_int(row.get("count_bin"))
            bin_note = None
            if count_bin is not None and count_bin not in valid_bins:
                bin_note = f"raw_count_bin={count_bin}"
                count_bin = None

            note = _append_note(_clean_str(row.get("notes")), reason_note)
            note = _append_note(note, count_note)
            note = _append_note(note, bin_note)

            observer = _normalize_observer(row.get("observer_id"))
            _ensure_observer(db, observer)

            db.execute(
                """
                INSERT INTO observations (
                    image_id, observer_id, countable, uncountable_reason,
                    rh_present, rh_count, count_bin, general_count_estimate,
                    bycatch_present, bycatch_id, bycatch_count, count_method,
                    confidence_level, observation_date, source, notes
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    image_id,
                    observer,
                    _yn_to_int(row.get("countable")),
                    reason,
                    _yn_to_int(row.get("rh_present")),
                    rh_count,
                    count_bin,
                    _to_int(row.get("general_count_estimate")),
                    _yn_to_int(row.get("bycatch_present")),
                    _clean_str(row.get("bycatch_id")),
                    _to_int(row.get("bycatch_count")),
                    _clean_str(row.get("count_method")),
                    _clean_str(row.get("confidence_level")),
                    _parse_date(row.get("observation_date")),
                    "excel_migration",
                    note,
                ),
            )
            summary["observations"] += 1

            # -- expert review (only if any expert field is populated) ---
            expert_id = _normalize_observer(row.get("expert_id"))
            expert_reviewed = _yn_to_int(row.get("expert_reviewed"))
            expert_date = _parse_date(row.get("expert_review_date"))
            if expert_id or expert_reviewed is not None or expert_date:
                _ensure_observer(db, expert_id)
                db.execute(
                    """
                    INSERT INTO expert_reviews (
                        image_id, expert_id, expert_reviewed,
                        expert_review_date
                    ) VALUES (?,?,?,?)
                    """,
                    (image_id, expert_id, expert_reviewed, expert_date),
                )
                summary["expert_reviews"] += 1

        db.commit()
        logger.info("Imported sheet %s (%d rows)", sheet, len(df))

    db.commit()
    return summary


def _load_lookups(xl: pd.ExcelFile, db: Database) -> None:
    """Load the Sites, Observers, and category lookup tables."""
    # Sites
    if "Sites" in xl.sheet_names:
        sites = pd.read_excel(xl, sheet_name="Sites")
        for _, r in sites.iterrows():
            sid = _clean_str(r.get("site_ID") or r.get("site_id"))
            if not sid:
                continue
            db.upsert("sites", {
                "site_id": sid.upper(),
                "site_name": _clean_str(r.get("site_name")),
                "location_description": _clean_str(r.get("location_description")),
                "camera_type": _clean_str(r.get("camera_type")),
                "installation_date": _parse_date(r.get("installation_date")),
                "notes": _clean_str(r.get("notes")),
            })

    # Observers
    if "Observers" in xl.sheet_names:
        obs = pd.read_excel(xl, sheet_name="Observers")
        for _, r in obs.iterrows():
            oid = _normalize_observer(r.get("observer_id"))
            if not oid:
                continue
            db.upsert("observers", {
                "observer_id": oid,
                "observer_name": _clean_str(r.get("observer_name")),
                "role": _clean_str(r.get("role")),
                "email": _clean_str(r.get("email")),
                "training_date": _parse_date(r.get("training_date")),
                "active": _yn_to_int(r.get("active")),
                "notes": _clean_str(r.get("notes")),
            })

    # Count bins (hard-coded corrected mapping).
    for code, label, lo, hi in COUNT_BINS:
        db.upsert("count_bins", {
            "bin_code": code, "label": label, "min_count": lo, "max_count": hi,
        })

    # Uncountable categories
    if "Uncountable_Categories" in xl.sheet_names:
        cats = pd.read_excel(xl, sheet_name="Uncountable_Categories")
        for _, r in cats.iterrows():
            cid = _to_int(r.get("Category_shortcut") or r.get("category_id"))
            if cid is None:
                continue
            db.upsert("uncountable_categories", {
                "category_id": cid,
                "category": _clean_str(r.get("category")),
                "description": _clean_str(r.get("description")),
                "common_causes": _clean_str(r.get("common_causes")),
            })

    db.commit()
