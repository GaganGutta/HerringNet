"""The journal is what makes a multi-hour run survivable. These are its rules."""

import json
from pathlib import Path

from fishcount.journal import (
    JournalWriter,
    RunParams,
    completed_frames,
    describe_mismatch,
    read_params,
    stream_sorted,
    write_params,
)

PARAMS = RunParams(model="m.pt", conf=0.1, iou=0.7, imgsz=1536, max_det=3000)


def _write(path: Path, files: list[str]) -> None:
    with JournalWriter(path) as writer:
        for name in files:
            writer.append({"file": name, "width": 10, "height": 10, "detections": []})


def test_frames_are_readable_the_moment_their_batch_is_flushed(tmp_path: Path) -> None:
    """The point of the journal: what is written is durable before the run ends."""
    path = tmp_path / "results.jsonl"
    writer = JournalWriter(path)
    writer.append({"file": "a.jpg", "width": 1, "height": 1, "detections": []})
    writer.flush()

    assert completed_frames(path) == {"a.jpg"}  # readable without closing the writer
    writer.close()


def test_a_torn_final_line_from_a_kill_is_ignored(tmp_path: Path) -> None:
    """A process killed mid-write leaves half a line. That frame is simply not done."""
    path = tmp_path / "results.jsonl"
    _write(path, ["a.jpg", "b.jpg"])
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"file": "c.jpg", "width": 10, "hei')  # killed here

    assert completed_frames(path) == {"a.jpg", "b.jpg"}
    assert [entry["file"] for entry in stream_sorted(path)] == ["a.jpg", "b.jpg"]


def test_frames_stream_back_in_sorted_order_whatever_order_they_were_written(
    tmp_path: Path,
) -> None:
    """A resumed run appends out of order; the CSVs must still come out sorted."""
    path = tmp_path / "results.jsonl"
    _write(path, ["c.jpg", "a.jpg"])  # first run
    _write(path, ["b.jpg"])  # resumed run appends later

    assert [entry["file"] for entry in stream_sorted(path)] == ["a.jpg", "b.jpg", "c.jpg"]


def test_missing_journal_reads_as_nothing_done(tmp_path: Path) -> None:
    path = tmp_path / "results.jsonl"
    assert completed_frames(path) == set()
    assert list(stream_sorted(path)) == []


def test_params_round_trip_and_mismatch_is_explained(tmp_path: Path) -> None:
    write_params(tmp_path, PARAMS)
    assert read_params(tmp_path) == PARAMS

    other = RunParams(model="m.pt", conf=0.3, iou=0.7, imgsz=1024, max_det=3000)
    message = describe_mismatch(PARAMS, other)
    assert "conf" in message and "imgsz" in message
    assert "iou" not in message  # only what actually differs is reported
    assert "--restart" in message


def test_unreadable_params_do_not_vouch_for_the_journal(tmp_path: Path) -> None:
    (tmp_path / "run.json").write_text("{not json", encoding="utf-8")
    assert read_params(tmp_path) is None


def test_entries_are_one_line_each(tmp_path: Path) -> None:
    """One frame per line is the whole contract; indented JSON would break resume."""
    path = tmp_path / "results.jsonl"
    _write(path, ["a.jpg", "b.jpg"])

    lines = path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
    assert all(json.loads(line)["file"] for line in lines)
