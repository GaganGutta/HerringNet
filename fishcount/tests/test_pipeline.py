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


def _factory(base: FakeDetector, dense: FakeDetector, thorough: FakeDetector) -> object:
    calls = {"plain": 0}

    def make_detector(config: AppConfig, thorough_flag: bool) -> Detector:
        if thorough_flag:
            return thorough
        calls["plain"] += 1
        return base if calls["plain"] == 1 else dense

    return make_detector


def test_pipeline_counts_only_frames_with_enough_detections(tmp_path: Path) -> None:
    folder = tmp_path / "dive"
    for name in ("a.jpg", "b.jpg", "c.jpg"):
        write_image(folder / name)  # sorted order: a, b, c

    # base gives a=2, b=0, c=3; only a and c clear min_count=2.
    base = FakeDetector([[fish(), fish()], [], [fish(), fish(), fish()]])
    # dense/thorough are only fed the 2 selected frames, so 2-length scripts.
    dense = FakeDetector([[fish()] * 5, [fish()] * 6])
    thorough = FakeDetector([[fish()] * 9, [fish()] * 10])

    out = tmp_path / "out"
    summary = run_pipeline(
        folder,
        out,
        base_config=AppConfig(conf=0.10),
        dense_config=AppConfig(conf=0.10),
        make_detector=_factory(base, dense, thorough),  # type: ignore[arg-type]
        min_count=2,
        show_progress=False,
    )

    assert summary.base.processed == 3
    assert summary.flagged == 2
    assert summary.counted == 2
    assert summary.dense is not None and summary.dense.processed == 2
    assert summary.thorough is not None and summary.thorough.processed == 2

    # Only the selected frames appear in the dense/thorough outputs.
    dense_rows = {r[0] for r in _read(out / "dense" / "counts.csv")["rows"]}
    assert dense_rows == {"a.jpg", "c.jpg", "TOTAL"}

    # summary.csv lines up tier + base/dense/thorough per counted frame.
    summ = _read(out / "summary.csv")
    assert summ["header"] == ["filename", "tier", "base_count", "dense_count", "thorough_count"]
    body = {r[0]: r[1:] for r in summ["rows"]}
    assert body["a.jpg"] == ["confident", "2", "5", "9"]
    assert body["c.jpg"] == ["confident", "3", "6", "10"]
    assert body["TOTAL"] == ["", "5", "11", "19"]


def test_detections_csv_and_detected_folders_are_the_headline_output(tmp_path: Path) -> None:
    folder = tmp_path / "dive"
    for name in ("conf.jpg", "many.jpg", "review.jpg", "weak.jpg", "empty.jpg"):
        write_image(folder / name)
    # sorted order: conf, empty, many, review, weak
    base = FakeDetector(
        [
            [fish(conf=0.9)],  # conf.jpg: one strong detection -> confident
            [],  # empty.jpg -> none
            [fish(conf=0.3), fish(conf=0.3)],  # many.jpg: 2 real detections -> confident
            [fish(conf=0.3)],  # review.jpg: one moderate detection -> review
            [fish(conf=0.15)],  # weak.jpg: only a weak detection -> possible
        ]
    )
    dense = FakeDetector()
    thorough = FakeDetector()

    out = tmp_path / "out"
    summary = run_pipeline(
        folder,
        out,
        base_config=AppConfig(conf=0.10),
        dense_config=AppConfig(conf=0.10),
        make_detector=_factory(base, dense, thorough),  # type: ignore[arg-type]
        min_count=1,
        detect_conf=0.25,
        show_progress=False,
    )

    tiers = {f.file: f.tier for f in summary.frames}
    assert tiers == {
        "conf.jpg": "confident",
        "many.jpg": "confident",
        "review.jpg": "review",
        "weak.jpg": "possible",
        "empty.jpg": "none",
    }
    assert summary.flagged == 4  # everything with any detection is flagged
    # Default counting = frames with >= 1 real detection; the weak-only frame is not counted.
    assert summary.counted == 3
    assert summary.dense is not None and summary.dense.processed == 3

    det = _read(out / "detections.csv")
    assert det["header"] == [
        "filename",
        "has_fish",
        "tier",
        "max_confidence",
        "detections",
        "weak",
        "static",
    ]
    by_file = {r[0]: r[1:] for r in det["rows"]}
    assert by_file["conf.jpg"] == ["yes", "confident", "0.900", "1", "0", "0"]
    assert by_file["review.jpg"] == ["yes", "review", "0.300", "1", "0", "0"]
    assert by_file["weak.jpg"] == ["yes", "possible", "0.150", "0", "1", "0"]
    assert by_file["empty.jpg"] == ["no", "none", "0.000", "0", "0", "0"]
    # flagged frames sort first, confident before review before possible before none
    assert [r[0] for r in det["rows"]] == [
        "conf.jpg",
        "many.jpg",
        "review.jpg",
        "weak.jpg",
        "empty.jpg",
    ]

    # detected/<tier>/ holds annotated copies of exactly the flagged frames
    assert (out / "detected" / "confident" / "conf.jpg").is_file()
    assert (out / "detected" / "confident" / "many.jpg").is_file()
    assert (out / "detected" / "review" / "review.jpg").is_file()
    assert (out / "detected" / "possible" / "weak.jpg").is_file()
    assert not (out / "detected" / "none").exists()


def test_count_possible_includes_weak_only_frames(tmp_path: Path) -> None:
    folder = tmp_path / "dive"
    write_image(folder / "weak.jpg")
    base = FakeDetector([[fish(conf=0.15)]])
    dense = FakeDetector([[fish()] * 3])
    thorough = FakeDetector([[fish()] * 4])

    out = tmp_path / "out"
    summary = run_pipeline(
        folder,
        out,
        base_config=AppConfig(conf=0.10),
        dense_config=AppConfig(conf=0.10),
        make_detector=_factory(base, dense, thorough),  # type: ignore[arg-type]
        count_possible=True,
        show_progress=False,
    )

    assert summary.tier_count("possible") == 1
    assert summary.counted == 1
    assert summary.dense is not None and summary.dense.total_fish == 3


def test_pipeline_skips_counting_when_nothing_qualifies(tmp_path: Path) -> None:
    folder = tmp_path / "dive"
    write_image(folder / "a.jpg")
    write_image(folder / "b.jpg")
    base = FakeDetector([[fish(conf=0.15)], []])  # weak-only and empty; nothing to count

    out = tmp_path / "out"
    summary = run_pipeline(
        folder,
        out,
        base_config=AppConfig(conf=0.10),
        dense_config=AppConfig(conf=0.10),
        make_detector=lambda config, thorough_flag: base,
        show_progress=False,
    )

    assert summary.flagged == 1  # the weak frame is still flagged as possible
    assert summary.counted == 0
    assert summary.dense is None
    assert summary.thorough is None
    assert not (out / "dense").exists()
    assert (out / "detections.csv").is_file()
    rows = _read(out / "summary.csv")["rows"]
    assert rows == [["TOTAL", "", "0", "0", "0"]]


def test_static_objects_are_demoted_not_dropped(tmp_path: Path) -> None:
    folder = tmp_path / "dive"
    names = [f"f{i:02d}.jpg" for i in range(10)]
    for name in names:
        write_image(folder / name, width=200, height=200)
    # A "rock": the identical confident box in every one of the 10 frames.
    rock = fish(50, 50, 70, 70, conf=0.6)
    # A real fish moves: it appears at different places in two frames only.
    moving_a = fish(120, 120, 150, 140, conf=0.9)
    moving_b = fish(20, 150, 50, 170, conf=0.9)
    script = [[rock] for _ in names]
    script[3] = [rock, moving_a]
    script[7] = [rock, moving_b]
    base = FakeDetector(script)

    out = tmp_path / "out"
    summary = run_pipeline(
        folder,
        out,
        base_config=AppConfig(conf=0.10),
        dense_config=AppConfig(conf=0.10),
        make_detector=lambda config, thorough_flag: base,
        static_min_frames=8,
        show_progress=False,
    )

    tiers = {f.file: f.tier for f in summary.frames}
    # Only the two frames with a moving fish are flagged; the rock frames are static.
    assert tiers["f03.jpg"] == "confident"
    assert tiers["f07.jpg"] == "confident"
    assert sum(1 for t in tiers.values() if t == "static") == 8
    assert summary.flagged == 2
    # The rock is still recorded, just not counted as a fish.
    assert all(f.n_static == 1 for f in summary.frames)
    assert (out / "detected" / "static" / "f00.jpg").is_file()
    assert (out / "detected" / "confident" / "f03.jpg").is_file()
    det = _read(out / "detections.csv")
    assert det["header"][-1] == "static"
    row = {r[0]: r for r in det["rows"]}["f00.jpg"]
    assert row[1:3] == ["no", "static"]


def test_static_filter_can_be_disabled(tmp_path: Path) -> None:
    folder = tmp_path / "dive"
    names = [f"f{i:02d}.jpg" for i in range(10)]
    for name in names:
        write_image(folder / name, width=200, height=200)
    base = FakeDetector([[fish(50, 50, 70, 70, conf=0.6)] for _ in names])

    summary = run_pipeline(
        folder,
        tmp_path / "out",
        base_config=AppConfig(conf=0.10),
        dense_config=AppConfig(conf=0.10),
        make_detector=lambda config, thorough_flag: base,
        static_min_frames=None,
        show_progress=False,
    )
    assert summary.tier_count("static") == 0
    assert summary.flagged == 10
