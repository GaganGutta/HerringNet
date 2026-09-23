"""The eval arithmetic decides the threshold, so its definitions are pinned down."""

from fishcount.evaluate import (
    Box,
    Frame,
    band_of,
    bootstrap,
    box_counts,
    box_precision,
    box_recall,
    difference,
    frame_counts,
    frame_precision,
    overlaps,
    wilson,
)


def _frame(
    frame_id: str,
    boxes: list[Box],
    *,
    band: str = "mid",
    weight: float = 1.0,
    camera: str = "Primary",
) -> Frame:
    return Frame(
        frame_id=frame_id,
        band=band,
        camera=camera,
        folder="f",
        brightness_bin="mid",
        blur_bin="mid",
        brightness=100.0,
        blur=50.0,
        boxes=boxes,
        weight=weight,
    )


def model(conf: float, *, fish: bool, area: float = 0.01) -> Box:
    return Box(conf=conf, area_frac=area, is_fish=fish)


def missed(area: float = 0.01) -> Box:
    return Box(conf=None, area_frac=area, is_fish=True)


def test_a_hand_drawn_box_is_a_false_negative_at_every_threshold() -> None:
    """It is a fish the detector never proposed, so no threshold can recover it."""
    frames = [_frame("a", [missed()])]

    for threshold in (0.10, 0.50, 0.90):
        counts = box_counts(frames, threshold)
        assert counts.fn == 1
        assert counts.tp == 0
        assert counts.recall == 0.0


def test_box_truth_is_the_human_verdict_with_no_iou_matching() -> None:
    frames = [_frame("a", [model(0.9, fish=True), model(0.8, fish=False)])]

    counts = box_counts(frames, 0.5)

    assert (counts.tp, counts.fp, counts.fn) == (1, 1, 0)
    assert counts.precision == 0.5
    assert counts.recall == 1.0


def test_raising_the_threshold_turns_a_weak_fish_into_a_false_negative() -> None:
    frames = [_frame("a", [model(0.3, fish=True)])]

    assert box_counts(frames, 0.25).recall == 1.0
    assert box_counts(frames, 0.25).fn == 0
    assert box_counts(frames, 0.40).recall == 0.0
    assert box_counts(frames, 0.40).fn == 1


def test_frame_is_caught_if_flagged_and_it_holds_a_fish() -> None:
    caught = _frame("a", [model(0.8, fish=True)])
    false_alarm = _frame("b", [model(0.8, fish=False)])
    missed_frame = _frame("c", [missed()])
    quiet = _frame("d", [])

    counts = frame_counts([caught, false_alarm, missed_frame, quiet], 0.5)

    assert (counts.tp, counts.fp, counts.fn, counts.tn) == (1, 1, 1, 1)
    assert counts.precision == 0.5
    assert counts.recall == 0.5


def test_a_frame_flagged_for_the_wrong_box_still_counts_as_caught() -> None:
    """It reached a human, which is what frame-level detection is measuring."""
    frame = _frame("a", [model(0.9, fish=False), model(0.15, fish=True)])

    counts = frame_counts([frame], 0.5)

    assert counts.tp == 1
    assert counts.fp == 0


def test_weights_scale_counts_to_the_population() -> None:
    """A band sampled one in eighty must count for eighty when estimating a run."""
    rare = _frame("a", [model(0.9, fish=True)], band="high", weight=40.0)
    common = _frame("b", [model(0.9, fish=False)], band="none", weight=80.0)

    counts = frame_counts([rare, common], 0.5)

    assert counts.tp == 40.0
    assert counts.fp == 80.0
    assert counts.raw_tp == 1 and counts.raw_fp == 1  # unweighted still visible
    assert counts.precision == 40.0 / 120.0  # not 0.5


def test_a_keep_rule_trades_recall_for_precision_and_the_counts_show_both() -> None:
    """Dropping large boxes must cost the fish among them, not just the murk."""
    frames = [
        _frame("a", [model(0.9, fish=False, area=0.30)]),  # large murk
        _frame("b", [model(0.9, fish=True, area=0.30)]),  # large fish
        _frame("c", [model(0.9, fish=True, area=0.01)]),  # small fish
    ]

    without = box_counts(frames, 0.5)
    with_rule = box_counts(frames, 0.5, keep=lambda b: not b.large)

    assert (without.tp, without.fp, without.fn) == (2, 1, 0)
    assert (with_rule.tp, with_rule.fp, with_rule.fn) == (1, 0, 1)
    assert with_rule.precision == 1.0  # precision bought
    assert with_rule.recall == 0.5  # and a real fish paid for it


def test_bootstrap_keeps_boxes_with_their_frame() -> None:
    """Forty fish on one frame is one observation, not forty independent ones."""
    school = _frame("school", [model(0.9, fish=True) for _ in range(40)], band="high")
    singles = [_frame(f"s{i}", [model(0.9, fish=False)], band="high") for i in range(10)]

    interval = bootstrap([school, *singles], box_precision(0.5), rounds=400, seed=1)

    assert interval.point is not None
    # The school frame is in or out of a resample as a unit, so precision swings
    # widely. A box-level resample would report a spuriously tight interval.
    assert interval.width is not None
    assert interval.width > 0.20


def test_bootstrap_interval_contains_the_point_estimate() -> None:
    frames = [_frame(f"f{i}", [model(0.9, fish=i % 3 == 0)], band="mid") for i in range(60)]

    interval = bootstrap(frames, box_precision(0.5), rounds=500, seed=2)

    assert interval.point is not None and interval.low is not None and interval.high is not None
    assert interval.low <= interval.point <= interval.high
    assert interval.n == 60


def test_bootstrap_resamples_within_bands_not_across_them() -> None:
    """Every resample must keep each band's size, or the weights stop meaning anything."""
    seen: list[dict[str, int]] = []

    def statistic(frames):  # type: ignore[no-untyped-def]
        counts: dict[str, int] = {}
        for frame in frames:
            counts[frame.band] = counts.get(frame.band, 0) + 1
        seen.append(counts)
        return 1.0

    frames = [_frame(f"n{i}", [], band="none") for i in range(7)]
    frames += [_frame(f"h{i}", [model(0.9, fish=True)], band="high") for i in range(3)]

    bootstrap(frames, statistic, rounds=50, seed=3)

    assert all(counts == {"none": 7, "high": 3} for counts in seen[1:])


def test_a_statistic_that_is_undefined_gives_no_interval() -> None:
    """No boxes above the threshold means precision has no denominator."""
    frames = [_frame("a", [model(0.2, fish=True)])]

    interval = bootstrap(frames, box_precision(0.9), rounds=100, seed=4)

    assert interval.point is None
    assert interval.low is None


def test_difference_is_stricter_than_comparing_two_intervals() -> None:
    """A consistent gap can hide inside two overlapping intervals."""
    left = [_frame(f"l{i}", [model(0.9, fish=i % 10 != 0)], band="mid") for i in range(60)]
    right = [_frame(f"r{i}", [model(0.9, fish=i % 10 < 3)], band="mid") for i in range(60)]

    gap = difference(left, right, box_precision(0.5), rounds=800, seed=5)

    assert gap.point is not None and gap.point > 0
    assert gap.low is not None and gap.low > 0  # the difference is real


def test_overlaps_is_conservative_about_calling_a_difference_real() -> None:
    from fishcount.evaluate import Interval

    assert overlaps(Interval(0.5, 0.4, 0.6, 10), Interval(0.55, 0.45, 0.65, 10))
    assert not overlaps(Interval(0.5, 0.4, 0.45, 10), Interval(0.8, 0.7, 0.9, 10))
    assert overlaps(Interval(0.5, None, None, 10), Interval(0.8, 0.7, 0.9, 10))  # unknown


def test_band_of_separates_never_fired_from_fired_weakly() -> None:
    assert band_of(0, 0.0) == "none"
    assert band_of(1, 0.15) == "low"
    assert band_of(1, 0.25) == "mid"
    assert band_of(1, 0.80) == "high"


def test_wilson_behaves_at_the_extremes() -> None:
    point, low, high = wilson(0, 20)
    assert point == 0.0 and low == 0.0 and 0.0 < high < 0.25  # not a zero-width interval

    point, low, high = wilson(20, 20)
    assert point == 1.0 and high == 1.0 and 0.75 < low < 1.0

    point, low, high = wilson(0, 0)
    assert (point, low, high) == (0.0, 0.0, 1.0)


def test_recall_denominator_includes_never_proposed_fish() -> None:
    """Recall that ignores hand-drawn boxes would flatter the model badly."""
    frames = [_frame("a", [model(0.9, fish=True), missed(), missed()])]

    counts = box_counts(frames, 0.5)

    assert counts.tp == 1
    assert counts.fn == 2
    assert counts.recall is not None
    assert abs(counts.recall - 1 / 3) < 1e-9


def test_precision_and_recall_helpers_match_the_counts() -> None:
    frames = [
        _frame("a", [model(0.9, fish=True)]),
        _frame("b", [model(0.6, fish=False)]),
        _frame("c", [missed()]),
    ]

    assert box_precision(0.5)(frames) == box_counts(frames, 0.5).precision
    assert box_recall(0.5)(frames) == box_counts(frames, 0.5).recall
    assert frame_precision(0.5)(frames) == frame_counts(frames, 0.5).precision


def _pop(band: str, folder: str, brightness: str = "mid", blur: str = "mid", large: bool = False):
    from fishcount.evaluate import PopulationFrame

    return PopulationFrame(
        band=band, folder=folder, brightness_bin=brightness, blur_bin=blur, large_box=large
    )


def test_calibrated_weights_reproduce_known_population_totals() -> None:
    """The one check that cannot be fooled: the run is fully enumerated."""
    from fishcount.evaluate import calibrate_weights

    # Two folders of very different size, sampled equally, which is what the
    # round-robin sampler actually does and what a flat band weight gets wrong.
    population = [_pop("mid", "big") for _ in range(900)] + [
        _pop("mid", "small") for _ in range(100)
    ]
    frames = [_frame(f"b{i}", [], band="mid") for i in range(5)]
    for f in frames:
        f.folder = "big"
    small = [_frame(f"s{i}", [], band="mid") for i in range(5)]
    for f in small:
        f.folder = "small"
    frames += small

    weights = calibrate_weights(frames, population)

    big_total = sum(w for f, w in zip(frames, weights, strict=True) if f.folder == "big")
    small_total = sum(w for f, w in zip(frames, weights, strict=True) if f.folder == "small")
    assert abs(big_total - 900) < 1e-6  # not 500, which a flat per-band weight would give
    assert abs(small_total - 100) < 1e-6
    assert abs(sum(weights) - 1000) < 1e-6


def test_a_flat_band_weight_would_get_those_totals_wrong() -> None:
    """Pins the bug the calibration exists to fix, so it cannot come back."""
    population = [_pop("mid", "big") for _ in range(900)] + [
        _pop("mid", "small") for _ in range(100)
    ]
    flat = 1000 / 10  # N_band / n_band, the naive weight
    assert flat * 5 == 500  # big folder estimated at 500 against a true 900
    assert len(population) == 1000


def test_calibration_weights_are_positional_so_duplicates_each_count() -> None:
    """A bootstrap resample holds the same frame many times; each is its own unit."""
    from fishcount.evaluate import calibrate_weights

    population = [_pop("mid", "f") for _ in range(100)]
    one = _frame("a", [], band="mid")
    one.folder = "f"
    resample = [one, one, one, one]  # the same object four times

    weights = calibrate_weights(resample, population)

    assert len(weights) == 4
    assert abs(sum(weights) - 100) < 1e-6  # not 100 per copy, 100 in total


def test_calibration_report_flags_a_margin_it_was_not_calibrated_on() -> None:
    from fishcount.evaluate import calibration_report

    population = [_pop("mid", "f", brightness="low") for _ in range(50)]
    population += [_pop("mid", "f", brightness="high") for _ in range(50)]
    frames = [_frame(f"f{i}", [], band="mid") for i in range(4)]
    for i, f in enumerate(frames):
        f.folder = "f"
        f.brightness_bin = "low" if i < 3 else "high"
        f.weight = 25.0

    rows = {name: (got, want) for name, got, want, _ in calibration_report(frames, population)}

    assert rows["brightness=low"] == (75.0, 50)  # over-represented, and visibly so
    assert rows["brightness=high"] == (25.0, 50)


def test_restrict_reweights_to_the_subset_rather_than_keeping_stale_weights() -> None:
    """Dropping rows but keeping whole-run weights would describe the wrong run."""
    from fishcount.evaluate import calibrate_weights, restrict

    population = [_pop("mid", "keep") for _ in range(300)]
    population += [_pop("mid", "drop") for _ in range(700)]
    frames = []
    for name in ("keep", "drop"):
        for i in range(5):
            f = _frame(f"{name}{i}", [], band="mid")
            f.folder = name
            frames.append(f)
    for frame, weight in zip(frames, calibrate_weights(frames, population), strict=True):
        frame.weight = weight
    assert abs(sum(f.weight for f in frames) - 1000) < 1e-6

    kept, sub = restrict(frames, population, lambda f: f.folder == "keep")

    assert len(kept) == 5
    assert len(sub) == 300
    # totals now describe the subset: 300, not the 60 a stale weight would give
    assert abs(sum(f.weight for f in kept) - 300) < 1e-6


def test_effective_sample_size_falls_when_weights_are_uneven() -> None:
    from fishcount.evaluate import effective_sample_size

    even = [_frame(f"e{i}", [], weight=10.0) for i in range(10)]
    uneven = [_frame("a", [], weight=91.0)] + [_frame(f"u{i}", [], weight=1.0) for i in range(9)]

    assert abs(effective_sample_size(even) - 10.0) < 1e-9
    assert effective_sample_size(uneven) < 2.0  # one frame carries almost everything
