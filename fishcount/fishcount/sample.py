"""Draw a stratified sample of frames to label by hand.

Evaluation is only as good as the frames it is measured on, and the frames a
detector gets wrong are not the ones it is confident about. So the sample is
not drawn at random from the run, and it is emphatically not drawn from the
top of the confidence list. It is spread deliberately across:

- **camera and folder**, because each deployment has its own water, light and
  seabed, and a detector can fail at one site while looking fine overall;
- **brightness and blur**, the two recorded frame statistics, so that per
  condition recall can be measured rather than guessed at;
- **max confidence**, in four bands that include frames the detector found
  *nothing* in and frames where it only fired weakly. Those two bands cost
  nothing to skip and are exactly where false negatives hide.

Frames holding a very large box get their own quota. Under the old heuristics
those were demoted automatically as "oversized", which cost at least one real
fish (GOPR7891); whether that rule was worth its cost is a question only
labelled large-box frames can answer.

Sampling is seeded, so the same run and seed give the same 300 frames.
"""

from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

# Confidence bands. The first two are the ones a lazy sample would miss.
CONF_BANDS: tuple[tuple[str, float, float], ...] = (
    ("none", 0.0, 0.0),  # the detector found nothing at all
    ("low", 0.0, 0.25),  # it fired, but only below the reporting threshold
    ("mid", 0.25, 0.50),
    ("high", 0.50, 1.01),
)

# Relative weight of each band in the sample, scaled to whatever size is asked
# for. Deliberately flat rather than proportional to the run: "none" frames are
# the overwhelming majority of a run but would say nothing about precision, and
# "high" frames are rare but decide it. At the default size of 300 these come
# out as 90 / 70 / 70 / 70.
DEFAULT_BAND_WEIGHTS: dict[str, int] = {"none": 90, "low": 70, "mid": 70, "high": 70}

# A box covering at least this much of the frame is "large". The value is the
# old OVERSIZE_FRAC, kept only so the removed rule can be judged on evidence.
LARGE_BOX_FRAC = 0.10
DEFAULT_LARGE_BOX_QUOTA = 25

SAMPLE_HEADER = [
    "frame_id",
    "source",
    "frame",
    "camera",
    "folder",
    "blur",
    "brightness",
    "blur_bin",
    "brightness_bin",
    "max_conf",
    "conf_band",
    "n_boxes",
    "max_area_frac",
    "large_box",
]


@dataclass(frozen=True, slots=True)
class FrameRecord:
    """One frame from a run, with everything the sampler stratifies on."""

    frame_id: str  # unique across runs: "<source>/<relative path>"
    source: str  # the run this frame came from
    frame: str  # path relative to that run's input folder
    camera: str
    folder: str
    blur: float | None
    brightness: float | None
    max_conf: float
    n_boxes: int  # boxes recorded above the floor, not above the threshold
    max_area_frac: float

    @property
    def large_box(self) -> bool:
        return self.max_area_frac >= LARGE_BOX_FRAC

    @property
    def conf_band(self) -> str:
        for name, low, high in CONF_BANDS:
            if name == "none":
                if self.n_boxes == 0:
                    return name
            elif low <= self.max_conf < high:
                return name
        return CONF_BANDS[-1][0]


def load_run(out_dir: Path, source: str) -> list[FrameRecord]:
    """Read one run's frames.csv and detections.csv into FrameRecords.

    Box counts come from detections.csv, not from frames.csv: the column there
    counts boxes past the reporting threshold, whereas the sampler needs to
    know whether the detector fired *at all*, which is a different question
    and the one that separates the "none" band from the "low" band.

    Unreadable frames are left out: there is nothing on them to label.
    """
    boxes = _boxes_by_frame(out_dir / "detections.csv")
    records: list[FrameRecord] = []
    with (out_dir / "frames.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["error"]:
                continue
            frame = row["frame"]
            camera, folder = _camera_and_folder(source, frame)
            largest, count = boxes.get(frame, (0.0, 0))
            records.append(
                FrameRecord(
                    frame_id=f"{source}/{frame}",
                    source=source,
                    frame=frame,
                    camera=camera,
                    folder=folder,
                    blur=_maybe_float(row["blur"]),
                    brightness=_maybe_float(row["brightness"]),
                    max_conf=float(row["max_conf"] or 0.0),
                    n_boxes=count,
                    max_area_frac=largest,
                )
            )
    return records


def _boxes_by_frame(path: Path) -> dict[str, tuple[float, int]]:
    """Per frame: the largest box area fraction, and how many boxes it has."""
    result: dict[str, tuple[float, int]] = defaultdict(lambda: (0.0, 0))
    if not path.is_file():
        return result
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            largest, count = result[row["frame"]]
            result[row["frame"]] = (max(largest, float(row["area_frac"])), count + 1)
    return result


def _camera_and_folder(source: str, frame: str) -> tuple[str, str]:
    """Split a frame path into the camera it came from and its folder.

    GoPro card layouts look like "Primary camera/DCIM/104GOPRO/GOPR1234.JPG".
    Both cameras use the same card-folder names, so the folder key keeps the
    full path; only the label shown to a human is shortened.
    """
    parts = frame.split("/")
    if len(parts) >= 2:
        camera = parts[0].removesuffix(" camera")
        folder = f"{source}/{'/'.join(parts[:-1])}"
        return camera, folder
    return source, source


def _maybe_float(value: str) -> float | None:
    return float(value) if value else None


def quantile_bins(values: Sequence[float], count: int) -> list[float]:
    """Cut points splitting `values` into `count` roughly equal groups."""
    ordered = sorted(values)
    if not ordered:
        return []
    return [ordered[int(len(ordered) * i / count)] for i in range(1, count)]


def bin_of(value: float | None, cuts: Sequence[float]) -> str:
    """Which bin a value falls in: low / mid / high for two cut points."""
    if value is None:
        return "unknown"
    if len(cuts) == 2:
        names: tuple[str, ...] = ("low", "mid", "high")
    else:
        names = tuple(f"q{i}" for i in range(len(cuts) + 1))
    index = sum(value >= cut for cut in cuts)
    return names[index]


@dataclass(slots=True)
class SampleResult:
    """The chosen frames, plus everything needed to explain the choice."""

    frames: list[FrameRecord]
    brightness_cuts: list[float]
    blur_cuts: list[float]
    seed: int
    population: int
    band_quotas: dict[str, int]
    band_counts: dict[str, int]
    camera_counts: dict[str, int]
    folder_counts: dict[str, int]
    large_box_count: int
    forced: list[str]


def stratified_sample(
    records: Sequence[FrameRecord],
    *,
    size: int = 300,
    seed: int = 20260922,
    band_weights: dict[str, int] | None = None,
    large_box_quota: int = DEFAULT_LARGE_BOX_QUOTA,
    force_include: Iterable[str] = (),
) -> SampleResult:
    """Pick `size` frames spread across camera, folder, condition and confidence.

    The confidence bands split `size` between them by weight. Within a band,
    the frames are divided by (folder, brightness bin, blur bin) and drawn
    round robin from those cells, so no one folder or lighting condition can
    crowd the others out. A band with too few frames to fill its share hands
    the remainder to the others, so the sample still comes out the size asked
    for rather than quietly short.
    """
    rng = random.Random(seed)
    weights = dict(band_weights if band_weights is not None else DEFAULT_BAND_WEIGHTS)

    brightness_cuts = quantile_bins([r.brightness for r in records if r.brightness is not None], 3)
    blur_cuts = quantile_bins([r.blur for r in records if r.blur is not None], 3)

    by_id = {record.frame_id: record for record in records}
    chosen: dict[str, FrameRecord] = {}
    forced: list[str] = []
    for frame_id in force_include:
        record = by_id.get(frame_id)
        if record is not None:
            chosen[frame_id] = record
            forced.append(frame_id)

    by_band: dict[str, list[FrameRecord]] = defaultdict(list)
    for record in records:
        by_band[record.conf_band].append(record)

    quotas = _apportion(weights, size, {band: len(rows) for band, rows in by_band.items()})

    for band, quota in quotas.items():
        available = [r for r in by_band.get(band, []) if r.frame_id not in chosen]
        already = sum(1 for r in chosen.values() if r.conf_band == band)
        for record in spread(available, quota - already, brightness_cuts, blur_cuts, rng):
            chosen[record.frame_id] = record

    _top_up_large_boxes(chosen, records, large_box_quota, quotas, brightness_cuts, blur_cuts, rng)

    frames = sorted(chosen.values(), key=lambda r: r.frame_id)
    return SampleResult(
        frames=frames,
        brightness_cuts=brightness_cuts,
        blur_cuts=blur_cuts,
        seed=seed,
        population=len(records),
        band_quotas=quotas,
        band_counts=_counts(r.conf_band for r in frames),
        camera_counts=_counts(r.camera for r in frames),
        folder_counts=_counts(r.folder for r in frames),
        large_box_count=sum(1 for r in frames if r.large_box),
        forced=forced,
    )


def _apportion(weights: dict[str, int], size: int, supply: dict[str, int]) -> dict[str, int]:
    """Split `size` between the bands by weight, capped at what each one has.

    Largest-remainder apportionment, so the parts sum to exactly `size`. Any
    band short of frames hands its unfillable share to bands that still have
    room, which is what keeps a sample of 300 at 300 when one band is thin.
    """
    total = sum(weights.values()) or 1
    exact = {band: size * weight / total for band, weight in weights.items()}
    target = {band: int(value) for band, value in exact.items()}
    for band in sorted(exact, key=lambda b: (-(exact[b] - target[b]), b)):
        if sum(target.values()) >= size:
            break
        target[band] += 1

    capped = {band: min(count, supply.get(band, 0)) for band, count in target.items()}
    while sum(capped.values()) < size:
        room = [b for b in capped if supply.get(b, 0) > capped[b]]
        if not room:
            break
        for band in sorted(room, key=lambda b: (-(supply[b] - capped[b]), b)):
            if sum(capped.values()) >= size:
                break
            capped[band] += 1
    return capped


def spread(
    records: Sequence[FrameRecord],
    count: int,
    brightness_cuts: Sequence[float],
    blur_cuts: Sequence[float],
    rng: random.Random,
) -> list[FrameRecord]:
    """Take `count` frames, round robin over folders and then over conditions.

    Folders come first and conditions second, deliberately. A flat round robin
    over every (folder, brightness, blur) cell looks fair but is not: there are
    far more cells than frames to draw, so whichever cells come last in the
    iteration order never get reached at all. Going folder by folder means a
    folder is only skipped once every other folder has had its turn, which is
    what "stratified by folder" has to mean when the sample is smaller than the
    number of strata.

    The folder order is shuffled once and the starting folder rotates every
    pass, so the frames left over when `count` is not a multiple of the folder
    count do not always land on the same folders.
    """
    if count <= 0:
        return []
    by_folder: dict[str, dict[tuple[str, str], list[FrameRecord]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in records:
        condition = (
            bin_of(record.brightness, brightness_cuts),
            bin_of(record.blur, blur_cuts),
        )
        by_folder[record.folder][condition].append(record)
    for cells in by_folder.values():
        for rows in cells.values():
            rng.shuffle(rows)

    folders = sorted(by_folder)
    rng.shuffle(folders)
    cursor = dict.fromkeys(folders, 0)  # which condition cell each folder is up to
    picked: list[FrameRecord] = []
    start = 0
    while len(picked) < count:
        took_any = False
        for offset in range(len(folders)):
            if len(picked) >= count:
                break
            folder = folders[(start + offset) % len(folders)]
            taken = _take_next(by_folder[folder], cursor, folder)
            if taken is not None:
                picked.append(taken)
                took_any = True
        if not took_any:
            break  # every folder is exhausted
        start = (start + 1) % len(folders)
    return picked


def _take_next(
    cells: dict[tuple[str, str], list[FrameRecord]],
    cursor: dict[str, int],
    folder: str,
) -> FrameRecord | None:
    """One frame from this folder's next non-empty condition cell, or None."""
    keys = sorted(cells)
    for step in range(len(keys)):
        key = keys[(cursor[folder] + step) % len(keys)]
        if cells[key]:
            cursor[folder] = (cursor[folder] + step + 1) % len(keys)
            return cells[key].pop()
    return None


def _top_up_large_boxes(
    chosen: dict[str, FrameRecord],
    records: Sequence[FrameRecord],
    quota: int,
    quotas: dict[str, int],
    brightness_cuts: Sequence[float],
    blur_cuts: Sequence[float],
    rng: random.Random,
) -> None:
    """Make sure frames with a frame-filling box are represented.

    Swaps rather than adds, so the sample size does not drift: each large-box
    frame brought in displaces an ordinary frame from the same confidence
    band, taken from whichever folder is currently most over-represented.
    """
    have = sum(1 for r in chosen.values() if r.large_box)
    if have >= quota:
        return
    candidates = [r for r in records if r.large_box and r.frame_id not in chosen]
    rng.shuffle(candidates)
    for candidate in candidates:
        if have >= quota:
            return
        victim = _most_expendable(chosen, candidate.conf_band)
        if victim is None:
            return
        del chosen[victim.frame_id]
        chosen[candidate.frame_id] = candidate
        have += 1


def _most_expendable(chosen: dict[str, FrameRecord], band: str) -> FrameRecord | None:
    """The non-large frame in `band` from the most over-represented folder."""
    in_band = [r for r in chosen.values() if r.conf_band == band and not r.large_box]
    if not in_band:
        return None
    per_folder = _counts(r.folder for r in in_band)
    return max(in_band, key=lambda r: (per_folder[r.folder], r.frame_id))


def _counts(values: Iterable[str]) -> dict[str, int]:
    result: dict[str, int] = defaultdict(int)
    for value in values:
        result[value] += 1
    return dict(result)


def write_sample(result: SampleResult, path: Path) -> None:
    """Write sample.csv, one row per frame, keyed by frame_id."""
    write_sample_rows(result.frames, path, result.brightness_cuts, result.blur_cuts)


def write_sample_rows(
    frames: Sequence[FrameRecord],
    path: Path,
    brightness_cuts: Sequence[float],
    blur_cuts: Sequence[float],
    *,
    extra: dict[str, dict[str, str]] | None = None,
) -> None:
    """Write frames in the sample.csv format the labeling tool reads.

    `extra` adds columns, as column name -> {frame_id: value}; the labeling
    tool reads columns by name, so extra ones pass through it untouched.
    """
    extra = extra or {}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([*SAMPLE_HEADER, *extra])
        for record in frames:
            writer.writerow(
                [
                    record.frame_id,
                    record.source,
                    record.frame,
                    record.camera,
                    record.folder,
                    "" if record.blur is None else f"{record.blur:.1f}",
                    "" if record.brightness is None else f"{record.brightness:.1f}",
                    bin_of(record.blur, blur_cuts),
                    bin_of(record.brightness, brightness_cuts),
                    f"{record.max_conf:.3f}",
                    record.conf_band,
                    record.n_boxes,
                    f"{record.max_area_frac:.6f}",
                    "yes" if record.large_box else "no",
                    *(values.get(record.frame_id, "") for values in extra.values()),
                ]
            )


def write_sample_meta(result: SampleResult, path: Path, sources: dict[str, str]) -> None:
    """Write the cut points, quotas and per-stratum counts beside the sample."""
    payload = {
        "seed": result.seed,
        "population": result.population,
        "sample_size": len(result.frames),
        "sources": sources,
        "brightness_cuts": [round(v, 1) for v in result.brightness_cuts],
        "blur_cuts": [round(v, 1) for v in result.blur_cuts],
        "conf_bands": {name: [low, high] for name, low, high in CONF_BANDS},
        "band_quotas": result.band_quotas,
        "band_counts": result.band_counts,
        "camera_counts": result.camera_counts,
        "folder_counts": result.folder_counts,
        "large_box_frac": LARGE_BOX_FRAC,
        "large_box_count": result.large_box_count,
        "forced_include": result.forced,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
