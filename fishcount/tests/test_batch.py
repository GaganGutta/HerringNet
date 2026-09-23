import json
from pathlib import Path

import pytest

from fishcount.batch import NoImagesFoundError, run_batch
from fishcount.config import AppConfig
from fishcount.journal import JOURNAL_FILENAME, ResumeMismatchError, completed_frames
from helpers import FakeDetector, fish, write_image


def _input_folder(tmp_path: Path) -> Path:
    folder = tmp_path / "dive1"
    write_image(folder / "a.jpg")
    write_image(folder / "c.jpg")
    write_image(folder / "sub" / "b.png")
    return folder  # discovery order: a.jpg, c.jpg, sub/b.png


def _journal(out: Path) -> list[dict]:
    text = (out / JOURNAL_FILENAME).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def test_end_to_end_records_detections_and_frame_stats(tmp_path: Path) -> None:
    folder = _input_folder(tmp_path)
    originals = {path: path.read_bytes() for path in folder.rglob("*") if path.is_file()}
    detector = FakeDetector([[fish(), fish(conf=0.5)], [], [fish()]])
    out = tmp_path / "out"

    summary = run_batch(folder, out, AppConfig(batch_size=2), detector, show_progress=False)

    assert summary.processed == 3
    assert summary.skipped == 0
    assert summary.already_done == 0
    assert summary.total == 3
    assert summary.conf == 0.10
    assert detector.batch_sizes == [2, 1]  # batched inference, not one call per image

    entries = {entry["file"]: entry for entry in _journal(out)}
    assert entries["a.jpg"]["detections"][0]["box"] == [4, 6, 30, 24]
    assert entries["a.jpg"]["detections"][1]["confidence"] == 0.5
    assert entries["c.jpg"]["detections"] == []
    for entry in entries.values():
        # blur and brightness recorded for every readable frame, acted on by nothing
        assert isinstance(entry["blur"], float)
        assert isinstance(entry["brightness"], float)

    # input files are never modified
    assert {path: path.read_bytes() for path in folder.rglob("*") if path.is_file()} == originals


def test_every_box_above_the_floor_is_recorded(tmp_path: Path) -> None:
    """Sub-threshold boxes stay in the record: no rule drops a detection."""
    folder = tmp_path / "dive"
    write_image(folder / "a.jpg")
    weak = [fish(conf=0.11), fish(0, 0, 60, 46, conf=0.12)]  # weak, and nearly frame-filling
    out = tmp_path / "out"

    run_batch(folder, out, AppConfig(threshold=0.9), FakeDetector([weak]), show_progress=False)

    assert len(_journal(out)[0]["detections"]) == 2


def test_rerun_skips_frames_already_done(tmp_path: Path) -> None:
    folder = _input_folder(tmp_path)
    out = tmp_path / "out"
    first = FakeDetector([[fish()], [], [fish()]])
    run_batch(folder, out, AppConfig(), first, show_progress=False)

    second = FakeDetector([[fish()], [], [fish()]])
    summary = run_batch(folder, out, AppConfig(), second, show_progress=False)

    assert summary.already_done == 3
    assert summary.processed == 0
    assert second.batch_sizes == []  # the model was never asked to do anything
    assert len(_journal(out)) == 3  # and nothing was appended twice


def test_run_interrupted_partway_resumes_without_gaps_or_duplicates(tmp_path: Path) -> None:
    """The crash case: the detector dies mid-run, and a rerun finishes the job."""
    folder = tmp_path / "dive"
    for name in "abcde":
        write_image(folder / f"{name}.jpg")
    out = tmp_path / "out"

    class DyingDetector(FakeDetector):
        def detect_batch(self, images):  # type: ignore[no-untyped-def]
            if self.batch_sizes:  # succeed on the first batch, die on the second
                raise RuntimeError("simulated crash")
            return super().detect_batch(images)

    with pytest.raises(RuntimeError, match="simulated crash"):
        run_batch(folder, out, AppConfig(batch_size=2), DyingDetector(), show_progress=False)

    assert completed_frames(out / JOURNAL_FILENAME) == {"a.jpg", "b.jpg"}  # batch 1 survived

    summary = run_batch(folder, out, AppConfig(batch_size=2), FakeDetector(), show_progress=False)

    assert summary.already_done == 2
    assert summary.processed == 3
    recorded = [entry["file"] for entry in _journal(out)]
    assert sorted(recorded) == ["a.jpg", "b.jpg", "c.jpg", "d.jpg", "e.jpg"]
    assert len(recorded) == len(set(recorded))  # no frame recorded twice


def test_restart_discards_the_previous_journal(tmp_path: Path) -> None:
    folder = _input_folder(tmp_path)
    out = tmp_path / "out"
    run_batch(folder, out, AppConfig(), FakeDetector(), show_progress=False)

    summary = run_batch(folder, out, AppConfig(), FakeDetector(), show_progress=False, resume=False)

    assert summary.already_done == 0
    assert summary.processed == 3
    assert len(_journal(out)) == 3  # rebuilt, not appended to


def test_resuming_with_different_detection_settings_is_refused(tmp_path: Path) -> None:
    """Skipping a frame is only sound if today's run would have detected it the same."""
    folder = _input_folder(tmp_path)
    out = tmp_path / "out"
    run_batch(folder, out, AppConfig(), FakeDetector(), show_progress=False)

    with pytest.raises(ResumeMismatchError, match="conf"):
        run_batch(folder, out, AppConfig(conf=0.5), FakeDetector(), show_progress=False)


def test_changing_only_the_threshold_still_resumes(tmp_path: Path) -> None:
    """The threshold is applied when reporting, so it cannot invalidate a journal."""
    folder = _input_folder(tmp_path)
    out = tmp_path / "out"
    run_batch(folder, out, AppConfig(threshold=0.25), FakeDetector(), show_progress=False)

    summary = run_batch(folder, out, AppConfig(threshold=0.8), FakeDetector(), show_progress=False)

    assert summary.already_done == 3


def test_corrupt_image_is_skipped_with_run_continuing(tmp_path: Path) -> None:
    folder = tmp_path / "dive2"
    write_image(folder / "good.jpg")
    (folder / "bad.jpg").write_bytes(b"this is not an image")
    out = tmp_path / "out"

    summary = run_batch(folder, out, AppConfig(), FakeDetector([[fish()]]), show_progress=False)

    assert summary.processed == 1
    assert summary.skipped == 1
    entries = {entry["file"]: entry for entry in _journal(out)}
    assert entries["bad.jpg"]["error"] == "unreadable image"
    assert len(entries["good.jpg"]["detections"]) == 1


def test_unreadable_frames_are_not_retried_on_resume(tmp_path: Path) -> None:
    folder = tmp_path / "dive"
    write_image(folder / "good.jpg")
    (folder / "bad.jpg").write_bytes(b"not an image")
    out = tmp_path / "out"
    run_batch(folder, out, AppConfig(), FakeDetector([[fish()]]), show_progress=False)

    summary = run_batch(folder, out, AppConfig(), FakeDetector(), show_progress=False)

    assert summary.already_done == 2
    assert summary.skipped == 0


def test_empty_folder_raises_and_writes_nothing(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()

    with pytest.raises(NoImagesFoundError):
        run_batch(empty, tmp_path / "out", AppConfig(), FakeDetector(), show_progress=False)
    assert not (tmp_path / "out").exists()


def test_batch_size_defaults_to_the_device_and_an_explicit_one_wins(tmp_path: Path) -> None:
    """A GPU default of 8 would spill this model into system RAM; 1 is measured."""
    folder = tmp_path / "dive"
    for name in "abcd":
        write_image(folder / f"{name}.jpg")

    on_gpu = FakeDetector()
    on_gpu.device = "cuda"
    run_batch(folder, tmp_path / "gpu", AppConfig(), on_gpu, show_progress=False)
    assert on_gpu.batch_sizes == [1, 1, 1, 1]

    on_cpu = FakeDetector()
    run_batch(folder, tmp_path / "cpu", AppConfig(), on_cpu, show_progress=False)
    assert on_cpu.batch_sizes == [4]  # the CPU default of 8 covers all four at once

    explicit = FakeDetector()
    explicit.device = "cuda"
    run_batch(folder, tmp_path / "set", AppConfig(batch_size=2), explicit, show_progress=False)
    assert explicit.batch_sizes == [2, 2]
