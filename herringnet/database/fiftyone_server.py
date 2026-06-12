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

# Tags applied automatically when samples are added; never synced back.
AUTO_TAGS = {"has_prediction", "model_found_fish", "expert_reviewed"}


def _tags_to_label(
    tags: list[str],
) -> tuple[int | None, int | None, int | None, list[str]]:
    """Map reviewer tags to (rh_present, countable, count_bin, extra_tags).

    Reviewer vocabulary (case-insensitive):
        herring / fish    -> river herring present
        no-fish / empty   -> no river herring
        countable         -> image is countable
        uncountable       -> image is not countable
        bin1..bin4        -> count bin (1-10, 10-100, 100-500, 500+);
                             implies herring present unless tagged no-fish
    Anything else is preserved verbatim in the observation notes.
    """
    rh_present = countable = count_bin = None
    extras: list[str] = []
    for tag in tags:
        t = tag.lower()
        if t in AUTO_TAGS:
            continue
        if t in ("herring", "fish", "rh", "rh-yes", "rh_present"):
            rh_present = 1
        elif t in ("no-fish", "nofish", "empty", "rh-no"):
            rh_present = 0
        elif t == "countable":
            countable = 1
        elif t == "uncountable":
            countable = 0
        elif t in ("bin1", "bin2", "bin3", "bin4"):
            count_bin = int(t[3])
        else:
            extras.append(tag)
    if count_bin is not None and rh_present is None:
        rh_present = 1
    return rh_present, countable, count_bin, extras


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


def _sync_tags(db_path: str, dataset, tag_cache: dict[int, frozenset]) -> int:
    """Write changed sample tags to the database as new observations.

    Compares each sample's tags against the cache; for any change, maps
    the tags to label fields and appends an observation row with
    source='fiftyone'. The original records are never overwritten.
    """
    from datetime import datetime, timezone

    ids, tags_list = dataset.values(["db_image_id", "tags"])
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    synced = 0
    db = None
    try:
        for sid, tags in zip(ids, tags_list):
            if sid is None:
                continue
            sid = int(sid)
            current = frozenset(tags or [])
            if current == tag_cache.get(sid, frozenset()):
                continue
            rh, countable, count_bin, extras = _tags_to_label(list(current))
            note_tags = sorted(t for t in current if t not in AUTO_TAGS)
            note = "fiftyone tags: " + ",".join(note_tags) if note_tags else None
            if extras and note is None:
                note = "fiftyone tags: " + ",".join(extras)
            if db is None:
                db = Database(db_path)
            db.execute(
                """
                INSERT INTO observations (
                    image_id, rh_present, countable, count_bin,
                    source, observation_date, notes
                ) VALUES (?,?,?,?,?,?,?)
                """,
                (sid, rh, countable, count_bin, "fiftyone", now, note),
            )
            tag_cache[sid] = current
            synced += 1
        if db is not None:
            db.commit()
    finally:
        if db is not None:
            db.close()
    return synced


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
    tag_cache: dict[int, frozenset] = {}
    if len(dataset) > 0:
        for sid, tags in zip(*dataset.values(["db_image_id", "tags"])):
            if sid is None:
                continue
            known_ids.add(int(sid))
            # Seed the cache so existing tags are not re-synced on restart.
            tag_cache[int(sid)] = frozenset(tags or [])

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
                for s in samples:
                    sid = int(s["db_image_id"])
                    known_ids.add(sid)
                    tag_cache[sid] = frozenset(s.tags or [])
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
                for sid in stale:
                    tag_cache.pop(sid, None)
                changed = True
                logger.info("Pruned %d deleted samples (total %d)", n, len(dataset))

            # Sync reviewer tags back into the database as observations.
            if len(dataset) > 0:
                synced = _sync_tags(db_path, dataset, tag_cache)
                if synced:
                    logger.info("Synced %d hand-labeled images to the DB", synced)

            if changed:
                session.refresh()
        except Exception:  # noqa: BLE001 - keep serving even if a refresh fails
            logger.exception("Refresh failed; will retry")
        time.sleep(REFRESH_SECONDS)


if __name__ == "__main__":
    main()
