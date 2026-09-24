"""A small local labeling tool: the ground truth this project has never had.

Every filtering rule the previous version shipped was invented rather than
measured, because there was nothing to measure against. This produces that
something. It serves one frame at a time in a browser, draws the boxes the
detector produced, and records a human verdict on each one plus any fish the
detector missed entirely. Nothing leaves the machine: the server binds to
localhost and reads only the sampled frames.

Two files are written, and both are rewritten in full after every frame, so
quitting at any point loses nothing and reopening resumes at the first frame
not yet done:

    labels.csv          one row per box: the detector's boxes with a verdict,
                        plus boxes drawn by hand around missed fish
    labeled_frames.csv  one row per frame reviewed

The second file is not redundant. A frame with no rows in labels.csv could
mean "reviewed, nothing here" or "not looked at yet", and those are opposite
facts: the first is a true negative, the second is missing data. Recall
measured without that distinction would be wrong.
"""

from __future__ import annotations

import csv
import json
import os
import threading
import webbrowser
from collections import defaultdict
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

LABELS_FILENAME = "labels.csv"
FRAMES_FILENAME = "labeled_frames.csv"

LABELS_HEADER = [
    "frame_id",
    "source",
    "frame",
    "box_index",
    "origin",  # "model" (the detector drew it) or "missed" (the human did)
    "label",  # "fish" or "not_fish"
    "x1",
    "y1",
    "x2",
    "y2",
    "confidence",  # blank for a hand-drawn box: the model never saw it
]

FRAMES_HEADER = [
    "frame_id",
    "source",
    "frame",
    "n_model_boxes",
    "n_fish",
    "n_not_fish",
    "n_missed",
    "has_fish",  # the frame-level truth: any fish at all, model-found or not
    "excluded",  # "yes": deliberately left out, e.g. a school too dense to box fully
]

_MIME = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}


@dataclass(slots=True)
class Task:
    """One frame to label: where it is on disk and what the detector said."""

    frame_id: str
    source: str
    frame: str
    path: Path
    camera: str
    folder: str
    conf_band: str
    blur: str
    brightness: str
    boxes: list[dict[str, Any]] = field(default_factory=list)


def load_tasks(sample_csv: Path, roots: dict[str, Path], runs: dict[str, Path]) -> list[Task]:
    """Build the work list: every sampled frame, with its recorded boxes.

    `roots` maps a source name to the folder its frames live in; `runs` maps it
    to that run's output folder, where detections.csv is.
    """
    boxes_by_frame = _load_boxes(runs)
    tasks: list[Task] = []
    with sample_csv.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            source = row["source"]
            root = roots.get(source)
            if root is None:
                raise KeyError(f"No image folder given for source {source!r}")
            tasks.append(
                Task(
                    frame_id=row["frame_id"],
                    source=source,
                    frame=row["frame"],
                    path=root / row["frame"],
                    camera=row["camera"],
                    folder=row["folder"],
                    conf_band=row["conf_band"],
                    blur=row["blur"],
                    brightness=row["brightness"],
                    boxes=boxes_by_frame.get((source, row["frame"]), []),
                )
            )
    return tasks


def _load_boxes(runs: dict[str, Path]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    result: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for source, out_dir in runs.items():
        path = out_dir / "detections.csv"
        if not path.is_file():
            continue
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                result[(source, row["frame"])].append(
                    {
                        "x1": int(row["x1"]),
                        "y1": int(row["y1"]),
                        "x2": int(row["x2"]),
                        "y2": int(row["y2"]),
                        "confidence": float(row["confidence"]),
                        "area_frac": float(row["area_frac"]),
                    }
                )
    return result


class LabelStore:
    """Holds the verdicts and rewrites both CSVs after every frame.

    Rewriting rather than appending means going back to a frame and changing
    your mind produces one row set, not two. At a few hundred frames the whole
    file is a few tens of kilobytes, so the cost of rewriting is nothing next
    to the cost of an ambiguous record.
    """

    def __init__(self, out_dir: Path) -> None:
        self.dir = out_dir
        self.labels_path = out_dir / LABELS_FILENAME
        self.frames_path = out_dir / FRAMES_FILENAME
        self._lock = threading.Lock()
        self.frames: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        """Pick up an earlier session's work, so quitting is always safe."""
        if not self.frames_path.is_file():
            return
        boxes: dict[str, list[dict[str, Any]]] = defaultdict(list)
        if self.labels_path.is_file():
            with self.labels_path.open(encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    boxes[row["frame_id"]].append(
                        {
                            "box_index": int(row["box_index"]),
                            "origin": row["origin"],
                            "label": row["label"],
                            "x1": int(row["x1"]),
                            "y1": int(row["y1"]),
                            "x2": int(row["x2"]),
                            "y2": int(row["y2"]),
                            "confidence": row["confidence"],
                        }
                    )
        with self.frames_path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                self.frames[row["frame_id"]] = {
                    "source": row["source"],
                    "frame": row["frame"],
                    "boxes": boxes.get(row["frame_id"], []),
                    # files written before this column existed have no exclusions
                    "excluded": row.get("excluded") == "yes",
                }

    def save_frame(
        self,
        frame_id: str,
        source: str,
        frame: str,
        boxes: list[dict[str, Any]],
        *,
        excluded: bool = False,
    ) -> None:
        """Record one frame. An excluded frame keeps no boxes.

        Exclusion exists for frames that cannot be labelled completely, most
        often a dense school. A half-boxed frame is worse than none: every fish
        left unboxed is trained as background, which teaches the model to miss
        exactly the fish it already struggles with. Excluding it records that
        the frame was seen, so it does not come back, while making sure nothing
        downstream reads it as either a positive or a negative.
        """
        with self._lock:
            self.frames[frame_id] = {
                "source": source,
                "frame": frame,
                "boxes": [] if excluded else boxes,
                "excluded": excluded,
            }
            self._write()

    def _write(self) -> None:
        _atomic_csv(
            self.labels_path,
            LABELS_HEADER,
            [
                [
                    frame_id,
                    record["source"],
                    record["frame"],
                    box["box_index"],
                    box["origin"],
                    box["label"],
                    box["x1"],
                    box["y1"],
                    box["x2"],
                    box["y2"],
                    box.get("confidence", ""),
                ]
                for frame_id, record in sorted(self.frames.items())
                for box in record["boxes"]
            ],
        )
        rows = []
        for frame_id, record in sorted(self.frames.items()):
            if record.get("excluded"):
                # No verdict at all: blank rather than "no", which would be a claim.
                rows.append(
                    [frame_id, record["source"], record["frame"], "", "", "", "", "", "yes"]
                )
                continue
            boxes = record["boxes"]
            model = [b for b in boxes if b["origin"] == "model"]
            fish = [b for b in model if b["label"] == "fish"]
            missed = [b for b in boxes if b["origin"] == "missed"]
            rows.append(
                [
                    frame_id,
                    record["source"],
                    record["frame"],
                    len(model),
                    len(fish),
                    len(model) - len(fish),
                    len(missed),
                    "yes" if (fish or missed) else "no",
                    "no",
                ]
            )
        _atomic_csv(self.frames_path, FRAMES_HEADER, rows)


def _atomic_csv(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    """Write via a temp file and replace, so a crash cannot truncate the record."""
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


def _page() -> bytes:
    return (Path(__file__).parent / "label_ui.html").read_bytes()


class _Handler(BaseHTTPRequestHandler):
    tasks: list[Task]
    store: LabelStore
    training: bool = False

    def log_message(self, format: str, *args: Any) -> None:
        pass  # the console belongs to the progress line, not to request logs

    # Method names are fixed by BaseHTTPRequestHandler.
    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self._send(200, "text/html; charset=utf-8", _page())
        elif self.path == "/api/tasks":
            self._send(200, "application/json", json.dumps(self._task_payload()).encode())
        elif self.path.startswith("/img/"):
            self._send_image(int(self.path.removeprefix("/img/")))
        else:
            self._send(404, "text/plain", b"not found")

    def do_POST(self) -> None:
        if self.path != "/api/label":
            self._send(404, "text/plain", b"not found")
            return
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length))
        task = self.tasks[int(payload["index"])]
        self.store.save_frame(
            task.frame_id,
            task.source,
            task.frame,
            payload["boxes"],
            excluded=bool(payload.get("excluded", False)),
        )
        done = len(self.store.frames)
        print(f"\r  labeled {done}/{len(self.tasks)} frames", end="", flush=True)
        self._send(200, "application/json", json.dumps({"done": done}).encode())

    def _task_payload(self) -> dict[str, Any]:
        return {
            "training": self.training,
            "tasks": [
                {
                    "index": index,
                    "frame_id": task.frame_id,
                    "frame": task.frame,
                    "camera": task.camera,
                    "folder": task.folder,
                    "conf_band": task.conf_band,
                    "blur": task.blur,
                    "brightness": task.brightness,
                    "boxes": task.boxes,
                    "saved": self.store.frames.get(task.frame_id, {}).get("boxes"),
                    "excluded": self.store.frames.get(task.frame_id, {}).get("excluded", False),
                    "done": task.frame_id in self.store.frames,
                }
                for index, task in enumerate(self.tasks)
            ],
        }

    def _send_image(self, index: int) -> None:
        if not 0 <= index < len(self.tasks):
            self._send(404, "text/plain", b"no such frame")
            return
        path = self.tasks[index].path
        try:
            data = path.read_bytes()
        except OSError:
            self._send(404, "text/plain", b"image unreadable")
            return
        self._send(200, _MIME.get(path.suffix.lower(), "application/octet-stream"), data)

    def _send(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def serve(
    tasks: list[Task],
    store: LabelStore,
    *,
    port: int = 8765,
    open_browser: bool = True,
    training: bool = False,
) -> None:
    """Run the labeling server until Ctrl-C. Binds to localhost only.

    `training` puts a standing reminder on every frame that the labels will be
    used to train, where an unboxed fish is not a missed count but a lesson in
    ignoring fish.
    """
    handler = type("Handler", (_Handler,), {"tasks": tasks, "store": store, "training": training})
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    url = f"http://127.0.0.1:{port}/"
    print(f"Labeling {len(tasks)} frames; {len(store.frames)} already done.")
    print(f"Open {url} (Ctrl-C here when you are finished).")
    print(f"Saving to {store.labels_path}")
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped. Re-run the same command to carry on where you left off.")
    finally:
        server.server_close()
