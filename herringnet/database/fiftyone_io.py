"""FiftyOne integration: a visual review surface over the SQLite database.

The database is the source of truth. This module projects it into a
FiftyOne dataset for browsing and labeling, then pulls reviewer edits
back. FiftyOne is an optional dependency; install with::

    pip install -e ".[label]"

Typical loop
------------
1. ``build_dataset`` reads images (that have files on disk), their latest
   human observation, expert-review status, and any model predictions,
   and attaches them to FiftyOne samples (predictions as overlaid boxes,
   labels as fields, plus convenience tags for one-click filtering).
2. ``launch`` opens the app. You filter (e.g. tag ``model_found_fish``
   minus field ``rh_present``), confirm or correct, and tag samples.
3. ``sync_back`` writes the samples you touched into the database as new
   observations with ``source='fiftyone'`` so nothing overwrites the
   original record.

Reviewer conventions read by ``sync_back``:
  - set sample field ``rh_present`` (bool) and/or ``rh_count`` (int),
  - set ``review_notes`` (str) for free text,
  - tag a sample with one of REVIEW_TAGS to mark it for sync.
Only touched samples (tagged or with notes) are written back.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from herringnet.database.db import Database

logger = logging.getLogger(__name__)

DATASET_NAME = "herringnet"
REVIEW_TAGS = {"confirmed", "corrected", "needs_review"}


def _require_fiftyone():
    try:
        import fiftyone as fo  # noqa: F401
        return fo
    except ImportError as e:
        raise ImportError(
            "FiftyOne is required for visual review. Install with: "
            'pip install -e ".[label]"  (or: pip install fiftyone)'
        ) from e


def _latest_observations(db: Database) -> dict[int, Any]:
    obs: dict[int, Any] = {}
    for r in db.query("SELECT * FROM observations ORDER BY obs_id"):
        obs[r["image_id"]] = r  # later rows overwrite -> keep latest
    return obs


def _expert_flags(db: Database) -> dict[int, int]:
    return {
        r["image_id"]: r["er"]
        for r in db.query(
            "SELECT image_id, MAX(expert_reviewed) AS er "
            "FROM expert_reviews GROUP BY image_id"
        )
    }


def _latest_predictions(db: Database) -> dict[int, Any]:
    preds: dict[int, Any] = {}
    for r in db.query("SELECT * FROM model_predictions ORDER BY pred_id"):
        preds[r["image_id"]] = r
    return preds


def build_dataset(
    db: Database,
    image_root: str | Path,
    name: str = DATASET_NAME,
    sessions: list[str] | None = None,
    overwrite: bool = True,
):
    """Build a FiftyOne dataset from the database.

    Args:
        db: Initialized database.
        image_root: Root the stored relative paths are resolved against.
        name: FiftyOne dataset name.
        sessions: Optional list of session labels to include. None = all.
        overwrite: Delete an existing dataset of the same name first.

    Returns:
        The constructed ``fiftyone.Dataset``.
    """
    fo = _require_fiftyone()
    image_root = Path(image_root)

    if overwrite and fo.dataset_exists(name):
        fo.delete_dataset(name)
    dataset = fo.Dataset(name, overwrite=overwrite and fo.dataset_exists(name))

    where = "WHERE i.file_exists = 1"
    params: list[Any] = []
    if sessions:
        placeholders = ",".join("?" for _ in sessions)
        where += f" AND s.label IN ({placeholders})"
        params = list(sessions)

    images = db.query(
        f"""
        SELECT i.image_id, i.original_name, i.file_path, i.width, i.height,
               i.site_id, s.label AS session
          FROM images i JOIN sessions s ON i.session_id = s.session_id
          {where}
        """,
        params,
    )
    obs = _latest_observations(db)
    experts = _expert_flags(db)
    preds = _latest_predictions(db)

    samples = []
    skipped = 0
    for img in images:
        abs_path = (image_root / img["file_path"]).resolve()
        if not abs_path.exists():
            skipped += 1
            continue

        sample = fo.Sample(filepath=str(abs_path))
        sample["db_image_id"] = int(img["image_id"])
        sample["session"] = img["session"]
        sample["site"] = img["site_id"]
        sample["original_name"] = img["original_name"]

        tags: list[str] = []
        o = obs.get(img["image_id"])
        if o is not None:
            sample["countable"] = _as_bool(o["countable"])
            sample["rh_present"] = _as_bool(o["rh_present"])
            sample["rh_count"] = o["rh_count"]
            sample["uncountable_reason"] = o["uncountable_reason"]
            if o["rh_present"] == 1:
                tags.append("rh_present")
            if o["countable"] == 0:
                tags.append("uncountable")

        if experts.get(img["image_id"]) == 1:
            sample["expert_reviewed"] = True
            tags.append("expert_reviewed")

        pred = preds.get(img["image_id"])
        if pred is not None:
            img_w = img["width"] or _img_width(abs_path)
            img_h = img["height"] or _img_height(abs_path)
            detections = _to_fo_detections(
                fo, pred["detections_json"], img_w, img_h
            )
            if detections is not None:
                sample["model"] = detections
            sample["model_fish_count"] = pred["fish_count"]
            tags.append("has_prediction")
            if (pred["fish_count"] or 0) > 0:
                tags.append("model_found_fish")

        sample.tags = tags
        samples.append(sample)

    dataset.add_samples(samples)
    dataset.persistent = True
    logger.info(
        "Built FiftyOne dataset '%s': %d samples (%d files missing)",
        name, len(samples), skipped,
    )
    return dataset


def launch(dataset, port: int = 5151, wait: bool = True):
    """Launch the FiftyOne app on a dataset.

    Args:
        dataset: A FiftyOne dataset.
        port: Port for the app server.
        wait: Block until the app is closed (Ctrl-C). Set False in notebooks.
    """
    fo = _require_fiftyone()
    session = fo.launch_app(dataset, port=port)
    if wait:
        session.wait()
    return session


def sync_back(db: Database, name: str = DATASET_NAME) -> dict[str, int]:
    """Write reviewer edits from FiftyOne back into the database.

    Only samples that were tagged (REVIEW_TAGS) or given review_notes are
    written, each as a new observation with ``source='fiftyone'``.

    Returns:
        Summary dict with the number of observations written.
    """
    fo = _require_fiftyone()
    dataset = fo.load_dataset(name)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    written = 0

    for sample in dataset:
        review_tags = [t for t in (sample.tags or []) if t in REVIEW_TAGS]
        notes = _opt_field(sample, "review_notes")
        if not review_tags and not notes:
            continue

        image_id = sample.get_field("db_image_id")
        if image_id is None:
            continue

        note_parts = []
        if review_tags:
            note_parts.append("tags=" + ",".join(review_tags))
        if notes:
            note_parts.append(str(notes))

        db.execute(
            """
            INSERT INTO observations (
                image_id, countable, rh_present, rh_count, source, notes,
                observation_date
            ) VALUES (?,?,?,?,?,?,?)
            """,
            (
                int(image_id),
                _bool_to_int(_opt_field(sample, "countable")),
                _bool_to_int(_opt_field(sample, "rh_present")),
                _opt_field(sample, "rh_count"),
                "fiftyone",
                " | ".join(note_parts) or None,
                now,
            ),
        )
        written += 1

    db.commit()
    logger.info("Synced %d reviewed samples back to the database", written)
    return {"written": written}


def export_yolo(
    name: str,
    export_dir: str | Path,
    label_field: str = "model",
) -> str:
    """Export a FiftyOne dataset's detections to YOLO format for training.

    Args:
        name: FiftyOne dataset name.
        export_dir: Output directory.
        label_field: Sample detections field to export (default 'model').

    Returns:
        The export directory path.
    """
    fo = _require_fiftyone()
    dataset = fo.load_dataset(name)
    dataset.export(
        export_dir=str(export_dir),
        dataset_type=fo.types.YOLOv5Dataset,
        label_field=label_field,
    )
    return str(export_dir)


# -- small helpers ---------------------------------------------------------

def _opt_field(sample, name: str) -> Any:
    """Return a sample field value, or None if the field is not set."""
    return sample.get_field(name) if sample.has_field(name) else None


def _as_bool(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)


def _bool_to_int(value: Any) -> int | None:
    if value is None:
        return None
    return 1 if value else 0


def _to_fo_detections(
    fo, detections_json: str | None, width: int | None, height: int | None
):
    if not detections_json or not width or not height:
        return None
    try:
        boxes = json.loads(detections_json)
    except (ValueError, TypeError):
        return None

    dets = []
    for b in boxes:
        bbox = b.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = bbox
        dets.append(
            fo.Detection(
                label=b.get("class_name", "fish"),
                bounding_box=[x1 / width, y1 / height,
                              (x2 - x1) / width, (y2 - y1) / height],
                confidence=b.get("confidence"),
            )
        )
    return fo.Detections(detections=dets)


def _img_width(path: Path) -> int | None:
    return _img_size(path)[0]


def _img_height(path: Path) -> int | None:
    return _img_size(path)[1]


def _img_size(path: Path) -> tuple[int | None, int | None]:
    try:
        from PIL import Image
        with Image.open(path) as img:
            return img.size
    except Exception:  # noqa: BLE001
        return None, None
