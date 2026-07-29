"""Three-stage pipeline: base pass over a whole folder, then high-recall
re-runs on only the frames that base found real fish in.

Stage 1 (base):     default settings over every image -> <out>/base
Stage 2 (dense):     --dense settings, only frames with >= min_count fish -> <out>/dense
Stage 3 (thorough):  --dense --thorough (SAHI), same subset          -> <out>/thorough

The subset is chosen from base's counts.csv, so the slow, recall-first (and
false-positive-prone) passes never touch empty frames. A top-level summary.csv
compares the three counts per frame.
"""

from __future__ import annotations

import csv
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from fishcount.batch import BatchSummary, run_batch
from fishcount.config import AppConfig
from fishcount.count import TOTAL_ROW_LABEL
from fishcount.detector import Detector

# Builds a detector for a stage: (config, thorough) -> Detector.
DetectorFactory = Callable[[AppConfig, bool], Detector]


@dataclass(slots=True)
class PipelineSummary:
    out_dir: Path
    min_count: int
    base: BatchSummary
    dense: BatchSummary | None
    thorough: BatchSummary | None
    subset_size: int


def run_pipeline(
    input_dir: Path,
    out_dir: Path,
    *,
    base_config: AppConfig,
    dense_config: AppConfig,
    make_detector: DetectorFactory,
    min_count: int = 2,
    write_images: bool = True,
    show_progress: bool = True,
    model_path: Path | None = None,
) -> PipelineSummary:
    """Run base -> dense -> thorough and write a comparison summary.

    Frames with at least `min_count` fish in the base pass are carried into the
    dense and thorough passes. If none qualify, those passes are skipped.
    """
    base_dir = out_dir / "base"
    base_summary = run_batch(
        input_dir,
        base_dir,
        base_config,
        make_detector(base_config, False),
        write_images=write_images,
        show_progress=show_progress,
        model_path=model_path,
    )

    subset = _select_subset(base_dir / "counts.csv", input_dir, min_count)
    dense_summary: BatchSummary | None = None
    thorough_summary: BatchSummary | None = None

    if subset:
        dense_summary = run_batch(
            input_dir,
            out_dir / "dense",
            dense_config,
            make_detector(dense_config, False),
            write_images=write_images,
            show_progress=show_progress,
            model_path=model_path,
            only=subset,
        )
        thorough_summary = run_batch(
            input_dir,
            out_dir / "thorough",
            dense_config,
            make_detector(dense_config, True),
            write_images=write_images,
            show_progress=show_progress,
            model_path=model_path,
            only=subset,
        )

    _write_summary(out_dir / "summary.csv", out_dir, subset_names(subset, input_dir))
    return PipelineSummary(
        out_dir=out_dir,
        min_count=min_count,
        base=base_summary,
        dense=dense_summary,
        thorough=thorough_summary,
        subset_size=len(subset),
    )


def _select_subset(counts_csv: Path, input_dir: Path, min_count: int) -> list[Path]:
    """Absolute paths of frames whose base count is >= min_count, in file order."""
    selected: list[Path] = []
    for filename, count in _read_counts(counts_csv):
        if count >= min_count:
            selected.append((input_dir / filename).resolve())
    return selected


def subset_names(subset: list[Path], input_dir: Path) -> list[str]:
    base = input_dir.resolve()
    return [path.relative_to(base).as_posix() for path in subset]


def _read_counts(counts_csv: Path) -> list[tuple[str, int]]:
    """(filename, count) rows from a counts.csv, excluding the TOTAL row."""
    if not counts_csv.is_file():
        return []
    rows: list[tuple[str, int]] = []
    with counts_csv.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            name = row["filename"]
            if name == TOTAL_ROW_LABEL:
                continue
            rows.append((name, int(row["fish_count"])))
    return rows


def _write_summary(path: Path, out_dir: Path, subset_names: list[str]) -> None:
    """Per-frame base/dense/thorough counts for the subset, plus a TOTAL row."""
    base = dict(_read_counts(out_dir / "base" / "counts.csv"))
    dense = dict(_read_counts(out_dir / "dense" / "counts.csv"))
    thorough = dict(_read_counts(out_dir / "thorough" / "counts.csv"))
    totals = [0, 0, 0]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filename", "base_count", "dense_count", "thorough_count"])
        for name in subset_names:
            counts = [base.get(name, 0), dense.get(name, 0), thorough.get(name, 0)]
            totals = [t + c for t, c in zip(totals, counts, strict=True)]
            writer.writerow([name, *counts])
        writer.writerow([TOTAL_ROW_LABEL, *totals])
