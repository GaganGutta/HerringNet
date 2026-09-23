"""The durable, resumable record of what the detector has already seen.

A pass over thousands of 4000x3000 frames runs for hours, so it has to survive
being killed: a closed laptop, a power cut, Ctrl-C. A single JSON array cannot,
because it is only valid once the closing bracket is written. The journal is
therefore line-delimited JSON, one frame per line, appended and flushed as the
run goes. Killing the process costs at most the batch in flight.

Two files live next to each other in the output folder:

    results.jsonl   one frame per line, appended as each batch finishes
    run.json        the detection settings the journal was produced under

`run.json` is what makes resuming safe. Frames already in the journal are
skipped on a rerun, which is only correct if they would have been detected the
same way today, so a rerun with different detection settings is refused rather
than silently mixed. The reporting threshold is deliberately not among those
settings: it is applied when the CSVs are written, never during detection, so
it can be changed and re-reported without re-running the model.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from types import TracebackType
from typing import Any

JOURNAL_FILENAME = "results.jsonl"
PARAMS_FILENAME = "run.json"


class ResumeMismatchError(RuntimeError):
    """The output folder holds a journal produced under different settings."""


@dataclass(frozen=True, slots=True)
class RunParams:
    """The settings that decide what the detector would output for a frame.

    Two runs sharing these can share a journal. Anything applied after
    inference (the reporting threshold, whether images are annotated) is
    absent on purpose, so changing it never forces a re-run.
    """

    model: str
    conf: float
    iou: float
    imgsz: int
    max_det: int


def read_params(out_dir: Path) -> RunParams | None:
    """The settings an existing journal was produced under, or None."""
    path = out_dir / PARAMS_FILENAME
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return RunParams(**raw)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None  # unreadable params mean "cannot vouch for the journal"


def write_params(out_dir: Path, params: RunParams) -> None:
    (out_dir / PARAMS_FILENAME).write_text(json.dumps(asdict(params), indent=2), encoding="utf-8")


def describe_mismatch(stored: RunParams, current: RunParams) -> str:
    """A human-readable account of why a journal cannot be resumed."""
    changed = [
        f"  {field}: journal has {getattr(stored, field)!r}, "
        f"this run wants {getattr(current, field)!r}"
        for field in ("model", "conf", "iou", "imgsz", "max_det")
        if getattr(stored, field) != getattr(current, field)
    ]
    return (
        "This output folder holds results from a run with different detection settings:\n"
        + "\n".join(changed)
        + "\nResuming would mix them. Re-run with --restart to detect the folder again,"
        "\nor point --out at a different folder."
    )


def completed_frames(journal: Path) -> set[str]:
    """The frames already recorded, by path relative to the input folder."""
    return set(_offsets(journal))


def stream_sorted(journal: Path) -> Iterator[dict[str, Any]]:
    """Every recorded frame, in sorted path order, one at a time.

    Frames are read back by seeking to each one in turn rather than loading the
    journal, so memory stays flat no matter how long the run was. A frame
    recorded more than once (possible only if a journal is hand-edited) yields
    its last entry.
    """
    offsets = _offsets(journal)
    if not offsets:
        return
    with journal.open("rb") as handle:
        for file in sorted(offsets):
            handle.seek(offsets[file])
            entry: dict[str, Any] = json.loads(handle.readline())
            yield entry


def _offsets(journal: Path) -> dict[str, int]:
    """Map each recorded frame to the byte offset of its line.

    A run killed mid-write leaves a torn final line. Parsing stops there: that
    frame is simply not recorded yet, and the next run redoes it.
    """
    if not journal.is_file():
        return {}
    offsets: dict[str, int] = {}
    with journal.open("rb") as handle:
        offset = 0
        for line in handle:
            stripped = line.strip()
            if stripped:
                try:
                    entry = json.loads(stripped)
                except json.JSONDecodeError:
                    break  # torn tail from a kill: ignore it and anything after
                file = entry.get("file")
                if isinstance(file, str):
                    offsets[file] = offset
            offset += len(line)
    return offsets


class JournalWriter:
    """Appends frames to the journal, flushing each batch to disk.

    `flush` pushes the batch through Python's buffer and asks the OS to commit
    it, so a kill between batches loses nothing already reported as done.
    """

    def __init__(self, journal: Path) -> None:
        self._path = journal
        self._handle = journal.open("a", encoding="utf-8", newline="\n")

    def append(self, entry: dict[str, Any]) -> None:
        self._handle.write(json.dumps(entry, separators=(",", ":")) + "\n")

    def flush(self) -> None:
        self._handle.flush()
        os.fsync(self._handle.fileno())

    def close(self) -> None:
        self.flush()
        self._handle.close()

    def __enter__(self) -> JournalWriter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
