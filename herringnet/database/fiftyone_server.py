"""Long-running FiftyOne app server, kept in sync with the SQLite database.

Serves the FiftyOne app on localhost (fronted by Caddy's password gate)
and refreshes continuously: any image the watcher ingests shows up in
the app within about a minute, with its labels and any model boxes.

Run via systemd (see deploy/server_setup.sh):

    HN_DB=/srv/herring-data/herringnet.db \
    HN_ARCHIVE=/srv/herring-data/archive \
    HN_FO_PORT=5152 \
    python -m herringnet.database.fiftyone_server

The SQLite database remains the source of truth; this process only
projects it into FiftyOne for viewing.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

from herringnet.database.db import Database
from herringnet.database.fiftyone_io import (
    DATASET_NAME,
    _as_bool,
    _expert_flags,
    _latest_observations,
    _latest_predictions,
    _to_fo_detections,
)

logger = logging.getLogger(__name__)

REFRESH_SECONDS = 60


def _new_samples(fo, db: Database, image_root: Path, known_ids: set[int]) -> list:
    """Build FiftyOne samples for DB images with files not yet in the dataset."""
    obs = _latest_observations(db)
    experts = _expert_flags(db)
    preds = _latest_predictions(db)

    samples = []
    rows = db.query(
        """
        SELECT i.image_id, i.original_name, i.file_path, i.width, i.height,
               i.site_id, s.label AS session
          FROM images i JOIN sessions s ON i.session_id = s.session_id
         WHERE i.file_exists = 1
        """
    )
    for img in rows:
        image_id = int(img["image_id"])
        if image_id in known_ids:
            continue
        abs_path = (image_root / img["file_path"]).resolve()
        if not abs_path.exists():
            continue

        sample = fo.Sample(filepath=str(abs_path))
        sample["db_image_id"] = image_id
        sample["session"] = img["session"]
        sample["site"] = img["site_id"]
        sample["original_name"] = img["original_name"]

        tags = []
        o = obs.get(image_id)
        if o is not None:
            sample["countable"] = _as_bool(o["countable"])
            sample["rh_present"] = _as_bool(o["rh_present"])
            sample["rh_count"] = o["rh_count"]
            if o["rh_present"] == 1:
                tags.append("rh_present")
            if o["countable"] == 0:
                tags.append("uncountable")
        if experts.get(image_id) == 1:
            sample["expert_reviewed"] = True
            tags.append("expert_reviewed")

        pred = preds.get(image_id)
        if pred is not None and img["width"] and img["height"]:
            detections = _to_fo_detections(
                fo, pred["detections_json"], img["width"], img["height"]
            )
            if detections is not None:
                sample["model"] = detections
            sample["model_fish_count"] = pred["fish_count"]
            tags.append("has_prediction")
            if (pred["fish_count"] or 0) > 0:
                tags.append("model_found_fish")

        sample.tags = tags
        samples.append(sample)

    return samples


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    import fiftyone as fo

    db_path = os.environ.get("HN_DB", "data/herringnet.db")
    image_root = Path(os.environ.get("HN_ARCHIVE", "data/archive"))
    port = int(os.environ.get("HN_FO_PORT", "5152"))

    if fo.dataset_exists(DATASET_NAME):
        dataset = fo.load_dataset(DATASET_NAME)
    else:
        dataset = fo.Dataset(DATASET_NAME)
    dataset.persistent = True

    known_ids: set[int] = set()
    if len(dataset) > 0:
        known_ids = {int(v) for v in dataset.values("db_image_id") if v is not None}

    logger.info(
        "Starting FiftyOne app on 127.0.0.1:%d (%d samples, refresh every %ds)",
        port, len(dataset), REFRESH_SECONDS,
    )
    session = fo.launch_app(
        dataset, address="127.0.0.1", port=port, remote=True, auto=False
    )

    from fiftyone import ViewField as F

    while True:
        try:
            db = Database(db_path)
            samples = _new_samples(fo, db, image_root, known_ids)
            valid_ids = {
                int(r["image_id"]) for r in db.query(
                    "SELECT image_id FROM images WHERE file_exists = 1"
                )
            }
            db.close()

            changed = False
            if samples:
                dataset.add_samples(samples, progress=False)
                known_ids.update(int(s["db_image_id"]) for s in samples)
                changed = True
                logger.info(
                    "Added %d new samples (total %d)", len(samples), len(dataset)
                )

            # Prune samples whose images were deleted from the database.
            stale = known_ids - valid_ids
            if stale:
                view = dataset.match(F("db_image_id").is_in(list(stale)))
                n = len(view)
                if n:
                    dataset.delete_samples(view)
                known_ids -= stale
                changed = True
                logger.info("Pruned %d deleted samples (total %d)", n, len(dataset))

            if changed:
                session.refresh()
        except Exception:  # noqa: BLE001 - keep serving even if a refresh fails
            logger.exception("Refresh failed; will retry")
        time.sleep(REFRESH_SECONDS)


if __name__ == "__main__":
    main()
