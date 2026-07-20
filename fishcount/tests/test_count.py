from pathlib import Path

import pandas as pd

from fishcount.count import ImageResult, total_fish, write_counts_csv
from helpers import fish


def test_total_counts_every_detection_and_ignores_errored_images() -> None:
    results = [
        ImageResult("a.jpg", 10, 10, [fish(), fish()]),
        ImageResult("b.jpg", 10, 10, []),
        ImageResult("bad.jpg", error="unreadable image"),
    ]
    assert total_fish(results) == 2


def test_csv_has_total_row(tmp_path: Path) -> None:
    results = [ImageResult("a.jpg", 10, 10, [fish()])]
    path = tmp_path / "counts.csv"

    write_counts_csv(results, path)

    table = pd.read_csv(path)
    assert table.iloc[-1]["filename"] == "TOTAL"
    assert int(table.iloc[-1]["fish_count"]) == 1
