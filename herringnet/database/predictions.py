"""Ingest HerringNet model predictions into the database.

The detector/pipeline writes results as JSON (see
``herringnet.models.pipeline.save_results_json``). This module loads that
JSON and stores one ``model_predictions`` row per image, so the model's
boxes and counts sit alongside the human observations for the same image.
FiftyOne then overlays these predictions during review, which is exactly
the "show me where the model and the human disagree" workflow.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from herringnet.database.db import Database
from herringnet.database.scan_images import _normalize

logger = logging.getLogger(__name__)


def _find_image(db: Database, source_path: str) -> int | None:
    """Resolve a model result's source path to a database image_id."""
    p = Path(str(source_path).replace("\\", "/"))
    stem = p.stem
    folder = p.parent.name

    # Prefer a match within the session matching the source folder.
    target = _normalize(folder)
    for row in db.query(
        "SELECT i.image_id, s.label FROM images i "
        "JOIN sessions s ON i.session_id = s.session_id "
        "WHERE i.original_name = ?",
        (stem,),
    ):
        if _normalize(row["label"]) == target:
            return int(row["image_id"])

    # Fall back to a unique stem match anywhere in the database.
    rows = db.query("SELECT image_id FROM images WHERE original_name = ?", (stem,))
    if len(rows) == 1:
        return int(rows[0]["image_id"])
    return None


def ingest_results_json(
    db: Database,
    json_path: str | Path,
    model_name: str = "cfd",
) -> dict[str, int]:
    """Load a HerringNet results JSON file into ``model_predictions``.

    Args:
        db: Initialized database.
        json_path: Path to a results JSON (directory or video output).
        model_name: Name to record for this model run.

    Returns:
        Summary dict: frames seen, predictions written, unmatched frames.
    """
    json_path = Path(json_path)
    with open(json_path) as f:
        data = json.load(f)

    if "frames" in data:
        frames = data["frames"]
    elif "per_frame_results" in data:
        frames = data["per_frame_results"]
    else:
        frames = data if isinstance(data, list) else []

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    summary = {"frames": 0, "written": 0, "unmatched": 0}

    for frame in frames:
        summary["frames"] += 1
        source = frame.get("source_path", "")
        image_id = _find_image(db, source)
        if image_id is None:
            summary["unmatched"] += 1
            continue

        dets = frame.get("detections", [])
        boxes = []
        max_conf = 0.0
        for d in dets:
            det = d.get("detection", d)
            conf = float(det.get("confidence", 0.0))
            max_conf = max(max_conf, conf)
            boxes.append({
                "bbox": det.get("bbox"),
                "confidence": conf,
                "class_name": det.get("class_name", "fish"),
            })

        db.execute(
            """
            INSERT INTO model_predictions (
                image_id, model_name, fish_count, max_confidence,
                detections_json, run_at
            ) VALUES (?,?,?,?,?,?)
            """,
            (
                image_id,
                model_name,
                int(frame.get("fish_count", len(boxes))),
                max_conf,
                json.dumps(boxes),
                now,
            ),
        )
        summary["written"] += 1

    db.commit()
    logger.info("Ingested predictions from %s: %s", json_path, summary)
    return summary
