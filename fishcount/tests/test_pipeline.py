import csv
from pathlib import Path

from fishcount.config import AppConfig
from fishcount.detector import Detector
from fishcount.pipeline import run_pipeline
from helpers import FakeDetector, fish, write_image


def _read(path: Path) -> dict[str, list[str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    return {"header": rows[0], "rows": rows[1:]}


def test_pipeline_reruns_only_frames_with_enough_fish(tmp_path: Path) -> None:
    folder = tmp_path / "dive"
    for name in ("a.jpg", "b.jpg", "c.jpg"):
        write_image(folder / name)  # sorted order: a, b, c

    # base gives a=2, b=0, c=3; only a and c clear min_count=2.
    base = FakeDetector([[fish(), fish()], [], [fish(), fish(), fish()]])
    # dense/thorough are only fed the 2 selected frames, so 2-length scripts.
    dense = FakeDetector([[fish()] * 5, [fish()] * 6])
    thorough = FakeDetector([[fish()] * 9, [fish()] * 10])
    calls = {"non_thorough": 0}

    def make_detector(config: AppConfig, thorough_flag: bool) -> Detector:
        if thorough_flag:
            return thorough
        key = calls["non_thorough"]
        calls["non_thorough"] += 1
        return base if key == 0 else dense

    out = tmp_path / "out"
    summary = run_pipeline(
        folder,
        out,
        base_config=AppConfig(),
        dense_config=AppConfig(conf=0.10),
        make_detector=make_detector,
        min_count=2,
        show_progress=False,
    )

    assert summary.base.processed == 3
    assert summary.subset_size == 2
    assert summary.dense is not None and summary.dense.processed == 2
    assert summary.thorough is not None and summary.thorough.processed == 2

    # Only the selected frames appear in the dense/thorough outputs.
    dense_rows = {r[0] for r in _read(out / "dense" / "counts.csv")["rows"]}
    assert dense_rows == {"a.jpg", "c.jpg", "TOTAL"}

    # summary.csv lines up base/dense/thorough per selected frame.
    summ = _read(out / "summary.csv")
    assert summ["header"] == ["filename", "base_count", "dense_count", "thorough_count"]
    body = {r[0]: r[1:] for r in summ["rows"]}
    assert body["a.jpg"] == ["2", "5", "9"]
    assert body["c.jpg"] == ["3", "6", "10"]
    assert body["TOTAL"] == ["5", "11", "19"]


def test_pipeline_skips_dense_when_nothing_qualifies(tmp_path: Path) -> None:
    folder = tmp_path / "dive"
    write_image(folder / "a.jpg")
    write_image(folder / "b.jpg")
    base = FakeDetector([[fish()], []])  # a=1, b=0; nothing reaches min_count=2

    def make_detector(config: AppConfig, thorough_flag: bool) -> Detector:
        return base

    out = tmp_path / "out"
    summary = run_pipeline(
        folder,
        out,
        base_config=AppConfig(),
        dense_config=AppConfig(),
        make_detector=make_detector,
        min_count=2,
        show_progress=False,
    )

    assert summary.subset_size == 0
    assert summary.dense is None
    assert summary.thorough is None
    assert not (out / "dense").exists()
    assert (out / "base" / "counts.csv").is_file()
    # summary.csv exists with only the TOTAL row (zeros).
    rows = _read(out / "summary.csv")["rows"]
    assert rows == [["TOTAL", "0", "0", "0"]]
