"""Measure the detector against the hand-labelled sample.

Everything here is computed from labels.csv, labeled_frames.csv and the run's
detections.csv. The model is never re-run: a threshold sweep is a pass over
text files, which is the whole reason the reporting threshold was kept out of
detection in the first place.

Two things about the arithmetic are not optional.

**The sample is stratified, so raw counts are biased.** Frames were drawn in
four confidence bands with very different sampling rates (roughly 1 in 79 of
the frames the detector found nothing in, against 1 in 35 of the weak ones).
Counting the sample directly would answer "what does the sample look like",
not "what would this threshold do to the run". Every frame therefore carries
an inverse-probability weight N_h / n_h for its band h, and every count below
is a weighted count. Unweighted numbers are reported alongside, clearly
labelled, because they are what the bootstrap resamples and what a reader
checking the work by hand would count.

**Confidence intervals come from a stratified bootstrap over frames.** Frames
are the sampling unit and boxes are nested inside them, so resampling boxes
independently would pretend a dense school frame is many independent
observations when it is one. Resampling frames within each band preserves both
the design and the clustering. Intervals are percentile intervals; with 24
fish frames in 300 they are wide, and saying so is the point.

Box-level truth needs no IoU matching. Every box the detector produced was
judged individually by a human, so a model box is a true positive exactly when
it was marked fish. A hand-drawn box is a fish the detector never proposed at
all, at any confidence down to the recording floor, which is a different and
worse failure than one it proposed too weakly.
"""

from __future__ import annotations

import csv
import json
import math
import random
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

from fishcount.sample import bin_of

BANDS = ("none", "low", "mid", "high")

# The sweep. 0.10 is the recording floor, so nothing below it can be measured.
DEFAULT_THRESHOLDS = tuple(round(0.10 + 0.05 * i, 2) for i in range(17))  # 0.10 .. 0.90

LARGE_BOX_FRAC = 0.10  # the old OVERSIZE_FRAC, kept so the removed rule can be judged


@dataclass(frozen=True, slots=True)
class Box:
    """One box: either the detector's (with a verdict) or a human's (a miss)."""

    conf: float | None  # None for a hand-drawn box: the detector never proposed it
    area_frac: float
    is_fish: bool

    @property
    def from_model(self) -> bool:
        return self.conf is not None

    @property
    def large(self) -> bool:
        return self.area_frac >= LARGE_BOX_FRAC


@dataclass(slots=True)
class Frame:
    """One labelled frame, with its stratum and everything measured on it."""

    frame_id: str
    band: str
    camera: str
    folder: str
    brightness_bin: str
    blur_bin: str
    brightness: float
    blur: float
    boxes: list[Box] = field(default_factory=list)
    weight: float = 1.0

    @property
    def model_boxes(self) -> list[Box]:
        return [b for b in self.boxes if b.from_model]

    @property
    def missed_boxes(self) -> list[Box]:
        return [b for b in self.boxes if not b.from_model]

    @property
    def has_fish(self) -> bool:
        """Truth at frame level: any fish at all, whether the detector saw it."""
        return any(b.is_fish for b in self.boxes)

    @property
    def max_conf(self) -> float:
        return max((b.conf or 0.0 for b in self.model_boxes), default=0.0)

    @property
    def max_area_frac(self) -> float:
        return max((b.area_frac for b in self.model_boxes), default=0.0)

    def flagged(self, threshold: float) -> bool:
        return self.max_conf >= threshold


@dataclass(frozen=True, slots=True)
class Counts:
    """Weighted confusion counts, plus the unweighted ones behind them."""

    tp: float
    fp: float
    fn: float
    tn: float
    raw_tp: int = 0
    raw_fp: int = 0
    raw_fn: int = 0
    raw_tn: int = 0

    @property
    def precision(self) -> float | None:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp > 0 else None

    @property
    def recall(self) -> float | None:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn > 0 else None

    @property
    def f1(self) -> float | None:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p and r and p + r > 0 else None


def load_dataset(
    labels_dir: Path,
    runs: dict[str, Path],
    *,
    calibrate: bool = True,
) -> list[Frame]:
    """Build the labelled frames, weighted so they stand in for the whole run.

    With `calibrate` (the default) the weights are raked until the weighted
    sample reproduces the run's known band x folder, brightness and blur
    totals. A flat per-band weight is wrong here because the sampler allocated
    equally across folders and conditions rather than proportionally, so frames
    from small folders and rare conditions were far likelier to be drawn.
    Passing calibrate=False gives the naive per-band weight, which is kept only
    so the difference can be shown.
    """
    sample = {r["frame_id"]: r for r in _rows(labels_dir / "sample.csv")}
    # A frame left out on purpose has no verdict; counting it would turn "could
    # not label" into a confident true negative.
    reviewed = {
        r["frame_id"]: r
        for r in _rows(labels_dir / "labeled_frames.csv")
        if r.get("excluded") != "yes"
    }
    frame_area = frame_areas(runs)

    boxes: dict[str, list[Box]] = defaultdict(list)
    for row in _rows(labels_dir / "labels.csv"):
        conf = float(row["confidence"]) if row["confidence"] else None
        area = frame_area.get(row["frame_id"])
        if area is None:
            raise KeyError(f"No recorded frame size for {row['frame_id']}; is the run present?")
        boxes[row["frame_id"]].append(
            Box(
                conf=conf,
                area_frac=_box_area(row) / area,
                is_fish=row["label"] == "fish",
            )
        )

    frames: list[Frame] = []
    for frame_id in reviewed:
        meta = sample[frame_id]
        frames.append(
            Frame(
                frame_id=frame_id,
                band=meta["conf_band"],
                camera=meta["camera"],
                folder=meta["folder"],
                brightness_bin=meta["brightness_bin"],
                blur_bin=meta["blur_bin"],
                brightness=float(meta["brightness"] or 0.0),
                blur=float(meta["blur"] or 0.0),
                boxes=boxes.get(frame_id, []),
            )
        )

    bands = population_bands(runs)
    sampled = _counter(f.band for f in frames)
    for frame in frames:
        frame.weight = bands[frame.band] / sampled[frame.band]
    if calibrate:
        cuts = json.loads((labels_dir / "sample_meta.json").read_text(encoding="utf-8"))
        population = population_frames(runs, cuts["brightness_cuts"], cuts["blur_cuts"])
        for frame, weight in zip(frames, calibrate_weights(frames, population), strict=True):
            frame.weight = weight
    return frames


def _box_area(row: dict[str, str]) -> float:
    """Box area in pixels, from its corners."""
    return float(max(0, int(row["x2"]) - int(row["x1"])) * max(0, int(row["y2"]) - int(row["y1"])))


def frame_areas(runs: dict[str, Path]) -> dict[str, float]:
    """Pixel area of every frame, keyed by frame_id, read from the journal.

    Taken from what was recorded rather than assumed, so a deployment shot at
    a different resolution cannot quietly corrupt every area fraction.
    """
    areas: dict[str, float] = {}
    for source, out_dir in runs.items():
        journal = out_dir / "results.jsonl"
        if not journal.is_file():
            continue
        with journal.open("rb") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    entry = json.loads(stripped)
                except json.JSONDecodeError:
                    break
                if "width" in entry and "height" in entry:
                    areas[f"{source}/{entry['file']}"] = float(entry["width"]) * float(
                        entry["height"]
                    )
    return areas


def population_bands(runs: dict[str, Path]) -> dict[str, int]:
    """How many frames of each confidence band the full run holds."""
    counts: dict[str, int] = dict.fromkeys(BANDS, 0)
    for out_dir in runs.values():
        boxes = _counter(row["frame"] for row in _rows(out_dir / "detections.csv"))
        for row in _rows(out_dir / "frames.csv"):
            if row["error"]:
                continue
            counts[band_of(boxes.get(row["frame"], 0), float(row["max_conf"] or 0.0))] += 1
    return counts


def band_of(n_boxes: int, max_conf: float) -> str:
    if n_boxes == 0:
        return "none"
    if max_conf < 0.25:
        return "low"
    if max_conf < 0.50:
        return "mid"
    return "high"


def frame_counts(frames: Sequence[Frame], threshold: float) -> Counts:
    """Frame level: did we flag the frame, and does it hold a fish.

    A frame flagged because of a box that turned out not to be a fish, while
    holding a different fish the detector found weakly, still counts as caught.
    That is the honest reading of "this frame reached a human".
    """
    tp = fp = fn = tn = 0.0
    raw = [0, 0, 0, 0]
    for frame in frames:
        flagged, truth = frame.flagged(threshold), frame.has_fish
        if flagged and truth:
            tp += frame.weight
            raw[0] += 1
        elif flagged:
            fp += frame.weight
            raw[1] += 1
        elif truth:
            fn += frame.weight
            raw[2] += 1
        else:
            tn += frame.weight
            raw[3] += 1
    return Counts(tp, fp, fn, tn, *raw)


def box_counts(
    frames: Sequence[Frame],
    threshold: float,
    *,
    keep: Callable[[Box], bool] | None = None,
) -> Counts:
    """Box level. Hand-drawn boxes are false negatives at every threshold.

    `keep` is an optional extra filter standing for a candidate rule, such as
    "drop boxes covering more than 10% of the frame". A box the rule rejects
    is not predicted, so a real fish it rejects becomes a false negative and a
    non-fish it rejects stops being a false positive. That is exactly the
    trade a rule has to win before it earns a place.
    """
    tp = fp = fn = 0.0
    raw = [0, 0, 0, 0]
    for frame in frames:
        for box in frame.boxes:
            if not box.from_model:
                fn += frame.weight  # never proposed at all, at any threshold
                raw[2] += 1
                continue
            predicted = (box.conf or 0.0) >= threshold and (keep is None or keep(box))
            if predicted and box.is_fish:
                tp += frame.weight
                raw[0] += 1
            elif predicted:
                fp += frame.weight
                raw[1] += 1
            elif box.is_fish:
                fn += frame.weight
                raw[2] += 1
            else:
                raw[3] += 1  # correctly not predicted; no true-negative box population
    return Counts(tp, fp, fn, 0.0, *raw)


@dataclass(frozen=True, slots=True)
class Interval:
    """A point estimate with a percentile bootstrap interval."""

    point: float | None
    low: float | None
    high: float | None
    n: int  # unweighted observations behind it

    def __str__(self) -> str:
        if self.point is None:
            return "n/a"
        if self.low is None or self.high is None:
            return f"{self.point:.3f}"
        return f"{self.point:.3f} [{self.low:.3f}, {self.high:.3f}]"

    @property
    def width(self) -> float | None:
        if self.low is None or self.high is None:
            return None
        return self.high - self.low


def bootstrap(
    frames: Sequence[Frame],
    statistic: Callable[[Sequence[Frame]], float | None],
    *,
    rounds: int = 2000,
    seed: int = 20260923,
    alpha: float = 0.05,
    population: Sequence[PopulationFrame] | None = None,
    margins: MarginSet | None = None,
) -> Interval:
    """Stratified percentile bootstrap over frames.

    Frames are resampled within their own confidence band, which is how they
    were drawn. Boxes travel with their frame, so a frame holding forty fish
    counts once, not forty times: treating them as independent would shrink
    the interval to a width the data does not support.

    With `population`, the weights are re-raked inside every resample. That is
    the honest thing to do when the point estimate uses calibrated weights: the
    calibration is part of the estimator, so its variability belongs in the
    interval. Holding the weights fixed instead gives intervals that are too
    narrow, which is the direction that matters when the question is whether a
    difference is real.
    """
    point = statistic(frames)
    if point is None or not frames:
        return Interval(point, None, None, len(frames))

    by_band: dict[str, list[Frame]] = defaultdict(list)
    for frame in frames:
        by_band[frame.band].append(frame)

    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(rounds):
        resample: list[Frame] = []
        for band_frames in by_band.values():
            size = len(band_frames)
            resample.extend(band_frames[rng.randrange(size)] for _ in range(size))
        if population is not None:
            resample = _reweighted(resample, population, margins)
        value = statistic(resample)
        if value is not None:
            draws.append(value)
    if len(draws) < rounds // 4:
        return Interval(point, None, None, len(frames))  # too degenerate to trust
    draws.sort()
    return Interval(
        point,
        draws[int(alpha / 2 * len(draws))],
        draws[min(len(draws) - 1, int((1 - alpha / 2) * len(draws)))],
        len(frames),
    )


def overlaps(a: Interval, b: Interval) -> bool:
    """Whether two intervals overlap, i.e. the difference could be noise.

    Non-overlap is a sufficient condition for a real difference, not a
    necessary one, so this is the conservative direction: it will call some
    real differences unproven, and will not call noise real.
    """
    if a.low is None or a.high is None or b.low is None or b.high is None:
        return True
    return a.low <= b.high and b.low <= a.high


def difference(
    left: Sequence[Frame],
    right: Sequence[Frame],
    statistic: Callable[[Sequence[Frame]], float | None],
    *,
    rounds: int = 2000,
    seed: int = 20260923,
    alpha: float = 0.05,
) -> Interval:
    """Bootstrap interval on `statistic(left) - statistic(right)`.

    Resampling both groups together in one pass is stricter than comparing two
    separate intervals: two overlapping intervals can still hide a difference
    that is consistently in one direction across resamples.
    """
    base_left, base_right = statistic(left), statistic(right)
    point = None if base_left is None or base_right is None else base_left - base_right
    if point is None:
        return Interval(None, None, None, len(left) + len(right))

    def banded(frames: Sequence[Frame]) -> dict[str, list[Frame]]:
        out: dict[str, list[Frame]] = defaultdict(list)
        for frame in frames:
            out[frame.band].append(frame)
        return out

    left_bands, right_bands = banded(left), banded(right)
    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(rounds):
        a = _resample(left_bands, rng)
        b = _resample(right_bands, rng)
        va, vb = statistic(a), statistic(b)
        if va is not None and vb is not None:
            draws.append(va - vb)
    if len(draws) < rounds // 4:
        return Interval(point, None, None, len(left) + len(right))
    draws.sort()
    return Interval(
        point,
        draws[int(alpha / 2 * len(draws))],
        draws[min(len(draws) - 1, int((1 - alpha / 2) * len(draws)))],
        len(left) + len(right),
    )


def _reweighted(
    resample: Sequence[Frame],
    population: Sequence[PopulationFrame],
    margins: MarginSet | None = None,
) -> list[Frame]:
    """Copy a resample with freshly raked weights, leaving the originals alone.

    Copies rather than mutation because a resample holds the same Frame object
    several times; writing to frame.weight would have every copy overwrite the
    others and quietly corrupt the estimate.
    """
    weights = calibrate_weights(
        resample, population, iterations=25, tolerance=1e-5, margins=margins
    )
    return [replace(frame, weight=weight) for frame, weight in zip(resample, weights, strict=True)]


def _resample(bands: dict[str, list[Frame]], rng: random.Random) -> list[Frame]:
    out: list[Frame] = []
    for band_frames in bands.values():
        size = len(band_frames)
        out.extend(band_frames[rng.randrange(size)] for _ in range(size))
    return out


def frame_precision(threshold: float) -> Callable[[Sequence[Frame]], float | None]:
    return lambda frames: frame_counts(frames, threshold).precision


def frame_recall(threshold: float) -> Callable[[Sequence[Frame]], float | None]:
    return lambda frames: frame_counts(frames, threshold).recall


def box_precision(
    threshold: float, keep: Callable[[Box], bool] | None = None
) -> Callable[[Sequence[Frame]], float | None]:
    return lambda frames: box_counts(frames, threshold, keep=keep).precision


def box_recall(
    threshold: float, keep: Callable[[Box], bool] | None = None
) -> Callable[[Sequence[Frame]], float | None]:
    return lambda frames: box_counts(frames, threshold, keep=keep).recall


def _rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _counter(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for value in values:
        counts[value] += 1
    return counts


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float, float]:
    """Wilson interval, for plain unweighted proportions like "how many of
    these boxes are fish". Behaves sensibly at 0 and 100%, unlike the normal
    approximation, which is the case this project keeps running into."""
    if total == 0:
        return (0.0, 0.0, 1.0)
    p = successes / total
    denominator = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denominator
    return (p, max(0.0, centre - margin), min(1.0, centre + margin))


@dataclass(frozen=True, slots=True)
class SweepRow:
    """One (threshold, level, metric) result with its interval and raw counts."""

    threshold: float
    level: str  # "frame" or "box"
    metric: str  # "precision" or "recall"
    interval: Interval
    counts: Counts


def sweep(
    frames: Sequence[Frame],
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    *,
    rounds: int = 1000,
    seed: int = 20260923,
) -> list[SweepRow]:
    """Precision and recall at every threshold, both levels, with intervals."""
    out: list[SweepRow] = []
    for threshold in thresholds:
        for level, counter, prec, rec in (
            ("frame", frame_counts, frame_precision, frame_recall),
            ("box", box_counts, box_precision, box_recall),
        ):
            counts = counter(frames, threshold)
            for metric, statistic in (("precision", prec), ("recall", rec)):
                out.append(
                    SweepRow(
                        threshold=threshold,
                        level=level,
                        metric=metric,
                        interval=bootstrap(frames, statistic(threshold), rounds=rounds, seed=seed),
                        counts=counts,
                    )
                )
    return out


def groups_of(frames: Sequence[Frame], dimension: str) -> dict[str, list[Frame]]:
    """Split frames by one breakdown dimension."""
    getters: dict[str, Callable[[Frame], str]] = {
        "camera": lambda f: f.camera,
        "folder": lambda f: f.folder,
        "brightness_bin": lambda f: f.brightness_bin,
        "blur_bin": lambda f: f.blur_bin,
        "box_size": lambda f: "large" if f.max_area_frac >= LARGE_BOX_FRAC else "normal",
    }
    getter = getters[dimension]
    out: dict[str, list[Frame]] = defaultdict(list)
    for frame in frames:
        out[getter(frame)].append(frame)
    return dict(out)


DIMENSIONS = ("camera", "folder", "brightness_bin", "blur_bin", "box_size")


def write_sweep(rows: Sequence[SweepRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "threshold",
                "level",
                "metric",
                "estimate",
                "ci_low",
                "ci_high",
                "weighted_tp",
                "weighted_fp",
                "weighted_fn",
                "raw_tp",
                "raw_fp",
                "raw_fn",
                "n_frames",
            ]
        )
        for row in rows:
            i, c = row.interval, row.counts
            writer.writerow(
                [
                    f"{row.threshold:.2f}",
                    row.level,
                    row.metric,
                    "" if i.point is None else f"{i.point:.4f}",
                    "" if i.low is None else f"{i.low:.4f}",
                    "" if i.high is None else f"{i.high:.4f}",
                    f"{c.tp:.1f}",
                    f"{c.fp:.1f}",
                    f"{c.fn:.1f}",
                    c.raw_tp,
                    c.raw_fp,
                    c.raw_fn,
                    i.n,
                ]
            )


@dataclass(frozen=True, slots=True)
class PopulationFrame:
    """One frame of the full run, reduced to the attributes weights calibrate on."""

    band: str
    folder: str
    brightness_bin: str
    blur_bin: str
    large_box: bool


def population_frames(
    runs: dict[str, Path], brightness_cuts: Sequence[float], blur_cuts: Sequence[float]
) -> list[PopulationFrame]:
    """Every frame in the run, binned exactly as the sample was binned.

    The cut points come from sample_meta.json rather than being recomputed, so
    a population frame and a sampled frame with the same brightness always land
    in the same bin. Recomputing them would silently shift the boundaries.
    """
    out: list[PopulationFrame] = []
    for source, out_dir in runs.items():
        boxes: dict[str, tuple[int, float]] = {}
        for row in _rows(out_dir / "detections.csv"):
            count, largest = boxes.get(row["frame"], (0, 0.0))
            boxes[row["frame"]] = (count + 1, max(largest, float(row["area_frac"])))
        for row in _rows(out_dir / "frames.csv"):
            if row["error"]:
                continue
            count, largest = boxes.get(row["frame"], (0, 0.0))
            parts = row["frame"].split("/")
            folder = f"{source}/{'/'.join(parts[:-1])}" if len(parts) > 1 else source
            out.append(
                PopulationFrame(
                    band=band_of(count, float(row["max_conf"] or 0.0)),
                    folder=folder,
                    brightness_bin=bin_of(_maybe(row["brightness"]), brightness_cuts),
                    blur_bin=bin_of(_maybe(row["blur"]), blur_cuts),
                    large_box=largest >= LARGE_BOX_FRAC,
                )
            )
    return out


def _maybe(value: str) -> float | None:
    return float(value) if value else None


# The margins weights are calibrated against. Band x folder is the design's own
# joint, because the sampler allocated equally across folders inside each band;
# brightness and blur are marginals because the sampler also round-robined over
# condition cells, which flattens their true proportions. Whether a frame holds
# a large box is included because the large-box quota over-sampled those frames
# on purpose, and they are the denominator of the size-rule question.
MarginSet = tuple[Callable[["PopulationFrame | Frame"], str], ...]

# Named margin sets, from most constrained to least. More margins remove more
# bias but cost effective sample size, and on a small subset they can collapse
# it: raking 40 constraints onto 147 frames drives some weights to zero and
# leaves the estimate resting on one or two frames. `choose_margins` picks the
# richest set the data can actually carry.
MARGIN_SETS: tuple[tuple[str, MarginSet], ...] = (
    (
        "band x folder + brightness + blur + large",
        (
            lambda f: f"{f.band}|{f.folder}",
            lambda f: f"b:{f.brightness_bin}",
            lambda f: f"z:{f.blur_bin}",
            lambda f: f"L:{_is_large(f)}",
        ),
    ),
    (
        "band x folder + brightness + blur",
        (
            lambda f: f"{f.band}|{f.folder}",
            lambda f: f"b:{f.brightness_bin}",
            lambda f: f"z:{f.blur_bin}",
        ),
    ),
    (
        "band x folder + large",
        (lambda f: f"{f.band}|{f.folder}", lambda f: f"L:{_is_large(f)}"),
    ),
    ("band x folder", (lambda f: f"{f.band}|{f.folder}",)),
    ("band + folder", (lambda f: f"band:{f.band}", lambda f: f"folder:{f.folder}")),
    ("band", (lambda f: f"band:{f.band}",)),
)

_MARGINS: MarginSet = MARGIN_SETS[0][1]


def calibrate_weights(
    frames: Sequence[Frame],
    population: Sequence[PopulationFrame],
    *,
    iterations: int = 200,
    tolerance: float = 1e-9,
    margins: MarginSet | None = None,
) -> list[float]:
    """Rake the sample weights until they reproduce known population totals.

    A flat N_h/n_h weight is only the inverse inclusion probability when every
    frame in a band had the same chance of being drawn. This sample's frames
    did not: the sampler gave each folder an equal turn inside each band, and
    each brightness/blur cell an equal turn inside each folder, so a frame from
    a small folder or a rare condition was far likelier to be picked than one
    from a large folder or a common condition. Weighting as though it were flat
    misses known population totals by up to 90%.

    Iterative proportional fitting fixes that using the one thing that is not
    in doubt: the run itself is fully enumerated, so the true size of every
    band x folder cell and every brightness and blur bin is known exactly. The
    weights are scaled repeatedly until the weighted sample reproduces all of
    them, which is the only check that can be verified rather than assumed.

    Returns one weight per frame, positionally. Positional rather than keyed by
    frame_id because a bootstrap resample contains the same frame several times
    and each copy is its own unit of weight; keying would silently merge them.
    """
    margins = margins if margins is not None else _MARGINS
    targets: list[dict[str, float]] = []
    for key in margins:
        counts: dict[str, float] = defaultdict(float)
        for entry in population:
            counts[key(entry)] += 1.0
        targets.append(dict(counts))

    # Precompute each frame's cell on every margin: the inner loop runs
    # thousands of times inside the bootstrap and string formatting dominates.
    cells = [[key(frame) for frame in frames] for key in margins]

    nominal: dict[str, float] = defaultdict(float)
    for cell in cells[0]:
        nominal[cell] += 1.0
    weights = [targets[0].get(cell, 0.0) / nominal[cell] for cell in cells[0]]

    for _ in range(iterations):
        shift = 0.0
        for margin, target in zip(cells, targets, strict=True):
            current: dict[str, float] = defaultdict(float)
            for index, cell in enumerate(margin):
                current[cell] += weights[index]
            for index, cell in enumerate(margin):
                want, have = target.get(cell, 0.0), current[cell]
                if have > 0 and want > 0:
                    factor = want / have
                    shift = max(shift, abs(factor - 1.0))
                    weights[index] *= factor
        if shift < tolerance:
            break
    return weights


def choose_margins(
    frames: Sequence[Frame],
    population: Sequence[PopulationFrame],
    *,
    min_effective_fraction: float = 0.35,
) -> tuple[str, MarginSet, float]:
    """Pick the richest calibration the sample can carry without collapsing.

    Each extra margin removes more of the sampler's design bias, but raking
    too many constraints onto too few frames does real damage: weights spread
    without limit, some frames fall to zero, and the estimate ends up resting
    on a handful of observations while still reporting an interval. The guard
    is the Kish effective sample size, which is exactly the quantity that
    collapses when that happens. The first margin set keeping a reasonable
    share of the frames' worth of information wins; if none does, the plainest
    set is used, and the caller should say so rather than quote the number as
    though it were solid.
    """
    best: tuple[str, MarginSet, float] | None = None
    for name, margins in MARGIN_SETS:
        weights = calibrate_weights(frames, population, margins=margins)
        total = sum(weights)
        squares = sum(w * w for w in weights)
        effective = (total * total / squares) if squares > 0 else 0.0
        if best is None or effective > best[2]:
            best = (name, margins, effective)
        if effective >= min_effective_fraction * len(frames):
            return (name, margins, effective)
    assert best is not None
    return best


def calibration_report(
    frames: Sequence[Frame], population: Sequence[PopulationFrame]
) -> list[tuple[str, float, int, float]]:
    """Weighted estimate against true total for every checkable group.

    This is the test that the weights are right. Every row compares a number
    the weighted sample produces with a number counted directly from all
    15,381 frames, so a broken weight cannot hide.
    """
    checks: list[tuple[str, Callable[[PopulationFrame | Frame], str]]] = [
        ("band", lambda f: f.band),
        ("folder", lambda f: f.folder),
        ("brightness", lambda f: f.brightness_bin),
        ("blur", lambda f: f.blur_bin),
        ("large_box", lambda f: "large" if _is_large(f) else "normal"),
    ]
    rows: list[tuple[str, float, int, float]] = []
    for name, key in checks:
        truth: dict[str, int] = defaultdict(int)
        for entry in population:
            truth[key(entry)] += 1
        estimate: dict[str, float] = defaultdict(float)
        for frame in frames:
            estimate[key(frame)] += frame.weight
        for group in sorted(truth):
            got, want = estimate.get(group, 0.0), truth[group]
            rows.append((f"{name}={group}", got, want, 100 * (got - want) / want if want else 0.0))
    return rows


def _is_large(frame: PopulationFrame | Frame) -> bool:
    if isinstance(frame, Frame):
        return frame.max_area_frac >= LARGE_BOX_FRAC
    return frame.large_box


def bootstrap_many(
    frames: Sequence[Frame],
    statistics: dict[str, Callable[[Sequence[Frame]], float | None]],
    *,
    rounds: int = 2000,
    seed: int = 20260923,
    alpha: float = 0.05,
    population: Sequence[PopulationFrame] | None = None,
    margins: MarginSet | None = None,
) -> dict[str, Interval]:
    """Intervals for many statistics from a single resampling pass.

    Every statistic sees the same resamples, which is both far cheaper than
    bootstrapping each one separately and better behaved: a precision/recall
    sweep computed on common random numbers moves smoothly with the threshold
    instead of jittering because each point drew its own resamples.
    """
    points = {name: statistic(frames) for name, statistic in statistics.items()}
    by_band: dict[str, list[Frame]] = defaultdict(list)
    for frame in frames:
        by_band[frame.band].append(frame)

    rng = random.Random(seed)
    draws: dict[str, list[float]] = {name: [] for name in statistics}
    for _ in range(rounds):
        resample: list[Frame] = []
        for band_frames in by_band.values():
            size = len(band_frames)
            resample.extend(band_frames[rng.randrange(size)] for _ in range(size))
        if population is not None:
            resample = _reweighted(resample, population, margins)
        for name, statistic in statistics.items():
            value = statistic(resample)
            if value is not None:
                draws[name].append(value)

    out: dict[str, Interval] = {}
    for name, values in draws.items():
        point = points[name]
        if point is None or len(values) < rounds // 4:
            out[name] = Interval(point, None, None, len(frames))
            continue
        values.sort()
        out[name] = Interval(
            point,
            values[int(alpha / 2 * len(values))],
            values[min(len(values) - 1, int((1 - alpha / 2) * len(values)))],
            len(frames),
        )
    return out


def effective_sample_size(frames: Sequence[Frame]) -> float:
    """Kish effective sample size: how many equally-weighted frames this is worth.

    Unequal weights cost precision. 147 frames raked to a population can carry
    the information of far fewer, and quoting the raw count would overstate
    what the sample supports.
    """
    weights = [f.weight for f in frames]
    total = sum(weights)
    squares = sum(w * w for w in weights)
    return (total * total / squares) if squares > 0 else 0.0


def restrict(
    frames: Sequence[Frame],
    population: Sequence[PopulationFrame],
    predicate: Callable[[Frame | PopulationFrame], bool],
    *,
    margins: MarginSet | None = None,
) -> tuple[list[Frame], list[PopulationFrame]]:
    """Narrow the evaluation to part of the run, re-raking weights to that part.

    Filtering the labelled frames alone would leave them carrying weights built
    to reproduce the whole run's totals, so they would still be standing in for
    frames that are no longer in scope and every estimate would inherit the
    excluded folders' shape. Both sides are filtered and the weights are
    recomputed against the narrowed population, so the result describes the
    subset and nothing else.
    """
    kept = [f for f in frames if predicate(f)]
    sub_population = [p for p in population if predicate(p)]
    if kept and sub_population:
        weights = calibrate_weights(kept, sub_population, margins=margins)
        for frame, weight in zip(kept, weights, strict=True):
            frame.weight = weight
    return kept, sub_population
