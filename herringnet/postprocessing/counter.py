"""Fish counting with frame-residence-rate (FRR) corrections.

Implements the counting methodology from Marjadi et al. (2024).
Raw per-frame detection counts are adjusted by the mean frame
residence rate to estimate true fish passage counts, avoiding
overcounting fish that appear across multiple consecutive frames.

Reference:
    Marjadi, M.N., et al. (2024). Automated video monitoring estimates
    high abundances of juvenile anadromous fish emigrating from a coastal
    river. Limnology and Oceanography: Methods, 22, 295-310.
"""

from __future__ import annotations

import pandas as pd

from herringnet.inference.result_types import CountPair, CountResult, FrameResult


class FishCounter:
    """Count fish from frame-level detections with FRR corrections.

    The frame residence rate (FRR) is the average number of frames a
    fish appears in as it swims through the camera's field of view.
    Dividing raw counts by the FRR corrects for the overcounting that
    occurs when the same fish is detected across multiple extracted
    frames.

    Args:
        mean_frr: Mean frame residence rate. Default is 4.55, from the
            Marjadi et al. (2024) Monument River study.
    """

    def __init__(self, mean_frr: float = 4.55):
        if mean_frr <= 0:
            raise ValueError(f"mean_frr must be positive, got {mean_frr}")
        self.mean_frr = mean_frr

    def count_from_frames(
        self,
        frame_results: list[FrameResult],
    ) -> CountResult:
        """Compute FRR-corrected fish counts from frame-level detections.

        The correction divides the raw count by the mean FRR to estimate
        the number of unique fish that passed through the field of view.

        Args:
            frame_results: Detection results for each extracted frame.

        Returns:
            CountResult with raw totals, corrected totals, and per-species
            breakdowns.
        """
        raw_total = 0
        species_raw: dict[str, int] = {}
        frames_with_fish = 0

        for result in frame_results:
            count = result.fish_count
            raw_total += count
            if count > 0:
                frames_with_fish += 1

            for species, n in result.species_counts.items():
                species_raw[species] = species_raw.get(species, 0) + n

        # Apply FRR correction
        corrected_total = raw_total / self.mean_frr

        species_counts = {}
        for species, raw in species_raw.items():
            species_counts[species] = CountPair(
                raw=raw,
                corrected=raw / self.mean_frr,
            )

        return CountResult(
            raw_total=raw_total,
            corrected_total=corrected_total,
            species_counts=species_counts,
            frames_processed=len(frame_results),
            frames_with_fish=frames_with_fish,
        )

    def generate_time_series(
        self,
        frame_results: list[FrameResult],
        bin_duration_minutes: int = 60,
    ) -> pd.DataFrame:
        """Generate time-binned detection counts from frame results.

        Groups detections into time bins and computes both raw and
        FRR-corrected counts per bin. Useful for visualizing fish
        passage patterns over time.

        Args:
            frame_results: Detection results with timestamps.
            bin_duration_minutes: Duration of each time bin in minutes.

        Returns:
            DataFrame with columns: time_bin_start, time_bin_end,
            species, raw_count, corrected_count.
        """
        if not frame_results:
            return pd.DataFrame(
                columns=[
                    "time_bin_start",
                    "time_bin_end",
                    "species",
                    "raw_count",
                    "corrected_count",
                ]
            )

        bin_duration_sec = bin_duration_minutes * 60
        rows = []

        for result in frame_results:
            if result.timestamp is None:
                continue

            bin_start = (
                int(result.timestamp / bin_duration_sec) * bin_duration_sec
            )
            bin_end = bin_start + bin_duration_sec

            for species, count in result.species_counts.items():
                rows.append({
                    "time_bin_start": bin_start,
                    "time_bin_end": bin_end,
                    "species": species,
                    "raw_count": count,
                })

        if not rows:
            return pd.DataFrame(
                columns=[
                    "time_bin_start",
                    "time_bin_end",
                    "species",
                    "raw_count",
                    "corrected_count",
                ]
            )

        df = pd.DataFrame(rows)
        grouped = (
            df.groupby(["time_bin_start", "time_bin_end", "species"])
            .agg(raw_count=("raw_count", "sum"))
            .reset_index()
        )
        grouped["corrected_count"] = grouped["raw_count"] / self.mean_frr

        return grouped
