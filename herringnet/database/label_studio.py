"""Label Studio integration: push images as tasks, pull labels back.

Mirrors the FiftyOne integration but for the hosted, multi-user Label
Studio app. The database stays the source of truth:

  - ``push_new_tasks`` sends images (that have files) to a Label Studio
    project, with the model's detections attached as pre-annotations so
    reviewers confirm/correct instead of starting from scratch,
  - ``pull_annotations`` reads completed annotations back into the
    ``observations`` table with ``source='label_studio'``.

A small bookkeeping table tracks which images were pushed and which
annotations were pulled, so both are safe to re-run.

NOTE: this talks to a live Label Studio server over its REST API. The
logic is straightforward but should be verified against the running
server during the first deploy (see deploy/README.md).
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone

from herringnet.database.db import Database

logger = logging.getLogger(__name__)

# Default local-files URL prefix (matches LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT
# mounted at the image archive). The task image becomes prefix + file_path.
LOCAL_FILES_PREFIX = "/data/local-files/?d="

_BOOKKEEPING = """
CREATE TABLE IF NOT EXISTS ls_tasks (
    image_id   INTEGER PRIMARY KEY REFERENCES images(image_id),
    project_id INTEGER,
    task_id    INTEGER
);
CREATE TABLE IF NOT EXISTS ls_pulled (
    annotation_id INTEGER PRIMARY KEY
);
"""


class LabelStudioClient:
    """Thin REST client for a Label Studio server.

    Args:
        url: Base URL of the Label Studio server (e.g. http://label-studio:8080).
        token: API token for a Label Studio account.
        project_id: Target project id.
        local_files_prefix: URL prefix used to reference archived images.
    """

    def __init__(
        self,
        url: str,
        token: str,
        project_id: int,
        local_files_prefix: str = LOCAL_FILES_PREFIX,
    ):
        import requests  # local import keeps this optional for the base package

        self._requests = requests
        self.url = url.rstrip("/")
        self.token = token
        self.project_id = int(project_id)
        self.local_files_prefix = local_files_prefix

    @classmethod
    def from_env(cls) -> LabelStudioClient | None:
        """Build a client from LS_URL / LS_TOKEN / LS_PROJECT_ID env vars.

        Returns None if the environment is not configured, so callers can
        treat Label Studio as optional.
        """
        url = os.environ.get("LS_URL")
        token = os.environ.get("LS_TOKEN")
        project = os.environ.get("LS_PROJECT_ID")
        if not (url and token and project):
            return None
        return cls(url, token, int(project))

    @property
    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Token {self.token}"}

    # -- pushing tasks -----------------------------------------------------

    def push_new_tasks(
        self,
        db: Database,
        session: str | None = None,
        batch_size: int = 100,
    ) -> dict[str, int]:
        """Send images that have files and were not yet pushed as tasks.

        Args:
            db: Initialized database.
            session: Optional session label to limit the push to.
            batch_size: Tasks per import request.

        Returns:
            Summary dict: tasks created.
        """
        db.conn.executescript(_BOOKKEEPING)
        db.commit()

        where = "WHERE i.file_exists = 1 AND t.image_id IS NULL"
        params: list = []
        if session:
            where += " AND s.label = ?"
            params.append(session)

        rows = db.query(
            f"""
            SELECT i.image_id, i.file_path, i.width, i.height,
                   i.original_name, s.label AS session
              FROM images i
              JOIN sessions s ON i.session_id = s.session_id
              LEFT JOIN ls_tasks t ON t.image_id = i.image_id
              {where}
            """,
            params,
        )
        preds = {
            r["image_id"]: r
            for r in db.query("SELECT * FROM model_predictions ORDER BY pred_id")
        }

        created = 0
        batch: list[tuple[int, dict]] = []
        for r in rows:
            task = {
                "data": {
                    "image": self.local_files_prefix + r["file_path"],
                    "db_image_id": r["image_id"],
                    "original_name": r["original_name"],
                    "session": r["session"],
                },
            }
            pred = preds.get(r["image_id"])
            if pred is not None and r["width"] and r["height"]:
                result = self._predictions_to_result(
                    pred["detections_json"], r["width"], r["height"]
                )
                if result:
                    task["predictions"] = [
                        {"model_version": pred["model_name"], "result": result}
                    ]
            batch.append((r["image_id"], task))

            if len(batch) >= batch_size:
                created += self._import_batch(db, batch)
                batch = []

        if batch:
            created += self._import_batch(db, batch)

        logger.info("Pushed %d new tasks to Label Studio", created)
        return {"created": created}

    def _import_batch(self, db: Database, batch: list[tuple[int, dict]]) -> int:
        """Import a batch of tasks and record their task ids."""
        tasks = [t for _, t in batch]
        resp = self._requests.post(
            f"{self.url}/api/projects/{self.project_id}/import",
            headers=self._headers,
            json=tasks,
            timeout=120,
        )
        resp.raise_for_status()
        # Label Studio returns created task ids in order for a JSON import.
        task_ids = resp.json().get("task_ids") or []
        for (image_id, _), task_id in zip(batch, task_ids):
            db.execute(
                "INSERT OR REPLACE INTO ls_tasks (image_id, project_id, task_id) "
                "VALUES (?,?,?)",
                (image_id, self.project_id, task_id),
            )
        db.commit()
        return len(tasks)

    def _predictions_to_result(
        self, detections_json: str | None, width: int, height: int
    ) -> list[dict]:
        """Convert stored detections to Label Studio rectangle results."""
        if not detections_json:
            return []
        try:
            boxes = json.loads(detections_json)
        except (ValueError, TypeError):
            return []

        result = []
        for b in boxes:
            bbox = b.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = bbox
            result.append({
                "from_name": "box",
                "to_name": "image",
                "type": "rectanglelabels",
                "value": {
                    "x": 100.0 * x1 / width,
                    "y": 100.0 * y1 / height,
                    "width": 100.0 * (x2 - x1) / width,
                    "height": 100.0 * (y2 - y1) / height,
                    "rectanglelabels": ["fish"],
                },
                "score": b.get("confidence"),
            })
        return result

    # -- pulling annotations ----------------------------------------------

    def pull_annotations(self, db: Database) -> dict[str, int]:
        """Read completed annotations back into the observations table.

        Returns:
            Summary dict: observations written.
        """
        db.conn.executescript(_BOOKKEEPING)
        db.commit()

        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        written = 0
        page = 1
        while True:
            resp = self._requests.get(
                f"{self.url}/api/projects/{self.project_id}/tasks",
                headers=self._headers,
                params={"page": page, "page_size": 200, "fields": "all"},
                timeout=120,
            )
            resp.raise_for_status()
            tasks = resp.json()
            if isinstance(tasks, dict):
                tasks = tasks.get("tasks", [])
            if not tasks:
                break

            for task in tasks:
                image_id = (task.get("data") or {}).get("db_image_id")
                for ann in task.get("annotations", []):
                    ann_id = ann.get("id")
                    if image_id is None or ann_id is None:
                        continue
                    already = db.scalar(
                        "SELECT 1 FROM ls_pulled WHERE annotation_id = ?", (ann_id,)
                    )
                    if already:
                        continue
                    rh_present, fish_boxes = self._parse_annotation(ann)
                    db.execute(
                        """
                        INSERT INTO observations (
                            image_id, rh_present, rh_count, source,
                            observation_date, notes
                        ) VALUES (?,?,?,?,?,?)
                        """,
                        (
                            int(image_id),
                            rh_present,
                            fish_boxes,
                            "label_studio",
                            now,
                            f"label_studio_annotation={ann_id}",
                        ),
                    )
                    db.execute(
                        "INSERT OR IGNORE INTO ls_pulled (annotation_id) VALUES (?)",
                        (ann_id,),
                    )
                    written += 1
            db.commit()
            page += 1

        logger.info("Pulled %d annotations from Label Studio", written)
        return {"written": written}

    @staticmethod
    def _parse_annotation(ann: dict) -> tuple[int | None, int]:
        """Extract (rh_present, fish_box_count) from an annotation."""
        rh_present: int | None = None
        fish_boxes = 0
        for item in ann.get("result", []):
            itype = item.get("type")
            value = item.get("value", {})
            if itype == "choices":
                choices = value.get("choices", [])
                if "Yes" in choices:
                    rh_present = 1
                elif "No" in choices:
                    rh_present = 0
            elif itype == "rectanglelabels":
                fish_boxes += 1
        if rh_present is None and fish_boxes > 0:
            rh_present = 1
        return rh_present, fish_boxes
