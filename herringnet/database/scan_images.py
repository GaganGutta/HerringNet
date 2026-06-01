"""Link database rows to actual image files on disk.

The Excel migration creates image rows with no file path (the original
workbook never recorded one). This module walks an image archive, and
for each file computes a content hash, reads EXIF capture time and
dimensions, and attaches that to the matching image row, or creates a
new row for images that were never in the spreadsheet.

Matching strategy
-----------------
Images are keyed by (session, original_name). A file's session is taken
from the name of the folder it sits in (your "site + week" folders),
matched against existing session labels first exactly, then by a
loosened comparison (alphanumerics only). The original_name is the
filename stem. This means:

  - if a folder matches a migrated session, files attach to those rows,
  - new folders become new sessions and new image rows automatically,
    so going forward you can grow the database by scanning, no Excel.

Paths are stored relative to ``image_root`` so the archive can move
between machines or drives without breaking links; the content hash lets
you detect duplicates and re-locate files that were renamed.
"""

from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from herringnet.database.db import Database

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _read_image_meta(
    path: Path,
) -> tuple[int | None, int | None, str | None, str | None]:
    """Return (width, height, date_captured, time_captured) from EXIF if present."""
    try:
        from PIL import Image
    except ImportError:
        return None, None, None, None

    try:
        with Image.open(path) as img:
            width, height = img.size
            date_captured = time_captured = None
            exif = getattr(img, "_getexif", lambda: None)()
            if exif:
                # 36867 = DateTimeOriginal, 306 = DateTime
                raw = exif.get(36867) or exif.get(306)
                if raw:
                    try:
                        dt = datetime.strptime(str(raw), "%Y:%m:%d %H:%M:%S")
                        date_captured = dt.date().isoformat()
                        time_captured = dt.time().isoformat()
                    except ValueError:
                        pass
            return width, height, date_captured, time_captured
    except Exception as e:  # noqa: BLE001 - never let one bad file stop the scan
        logger.debug("Could not read image meta for %s: %s", path, e)
        return None, None, None, None


def _normalize(label: str) -> str:
    return re.sub(r"[^a-z0-9]", "", label.lower())


def _resolve_session(db: Database, folder_label: str, site_id: str | None) -> int:
    """Find a session whose label matches the folder, else create one."""
    exact = db.scalar(
        "SELECT session_id FROM sessions WHERE label = ?", (folder_label,)
    )
    if exact is not None:
        return int(exact)

    target = _normalize(folder_label)
    for row in db.query("SELECT session_id, label FROM sessions"):
        if _normalize(row["label"]) == target:
            return int(row["session_id"])

    return db.get_or_create_session(label=folder_label, site_id=site_id)


def scan_directory(
    db: Database,
    image_root: str | Path,
    session: str | None = None,
    site_id: str | None = None,
    recursive: bool = True,
) -> dict[str, int]:
    """Scan an image archive and attach files to image rows.

    Args:
        db: Initialized database.
        image_root: Root folder of the image archive. Stored paths are
            relative to this root.
        session: Optional explicit session label to assign every scanned
            file to. If omitted, the file's parent folder name is used.
        site_id: Optional site id for any sessions created during the scan.
        recursive: Whether to descend into subfolders.

    Returns:
        Summary dict: files seen, rows updated, rows created, duplicates.
    """
    image_root = Path(image_root)
    if not image_root.exists():
        raise FileNotFoundError(f"Image root not found: {image_root}")

    pattern = "**/*" if recursive else "*"
    files = sorted(
        p for p in image_root.glob(pattern)
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    summary = {"files": 0, "updated": 0, "created": 0, "duplicate_hashes": 0}
    seen_hashes: set[str] = set()
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    for path in files:
        summary["files"] += 1
        rel_path = path.relative_to(image_root).as_posix()

        folder_label = session or (
            path.parent.name if path.parent != image_root else "unsorted"
        )
        session_id = _resolve_session(db, folder_label, site_id)
        original_name = path.stem

        content_hash = _sha256(path)
        if content_hash in seen_hashes:
            summary["duplicate_hashes"] += 1
        seen_hashes.add(content_hash)

        width, height, exif_date, exif_time = _read_image_meta(path)
        file_size = path.stat().st_size

        existing = db.scalar(
            "SELECT image_id FROM images WHERE session_id = ? AND original_name = ?",
            (session_id, original_name),
        )
        if existing is not None:
            # Attach file info; do not clobber capture date/time from Excel.
            db.execute(
                """
                UPDATE images
                   SET file_path = ?, content_hash = ?, file_exists = 1,
                       width = ?, height = ?, file_size = ?,
                       date_captured = COALESCE(date_captured, ?),
                       time_captured = COALESCE(time_captured, ?)
                 WHERE image_id = ?
                """,
                (rel_path, content_hash, width, height, file_size,
                 exif_date, exif_time, int(existing)),
            )
            summary["updated"] += 1
        else:
            db.execute(
                """
                INSERT INTO images (
                    original_name, site_id, session_id, file_path, content_hash,
                    file_exists, width, height, file_size,
                    date_captured, time_captured, added_at
                ) VALUES (?,?,?,?,?,1,?,?,?,?,?,?)
                """,
                (original_name, site_id, session_id, rel_path, content_hash,
                 width, height, file_size, exif_date, exif_time, now),
            )
            summary["created"] += 1

    db.commit()
    logger.info("Scan complete: %s", summary)
    return summary
