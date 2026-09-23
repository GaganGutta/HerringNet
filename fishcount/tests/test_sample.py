"""The sample decides what the evaluation can see, so its coverage rules are tested."""

import csv
import json
from pathlib import Path

from fishcount.sample import (
    FrameRecord,
    bin_of,
    load_run,
    quantile_bins,
    stratified_sample,
    write_sample,
    write_sample_meta,
)


def _record(
    frame: str,
    *,
    source: str = "RUN",
    conf: float = 0.0,
    boxes: int = 0,
    blur: float = 50.0,
    brightness: float = 100.0,
    area: float = 0.001,
) -> FrameRecord:
    parts = frame.split("/")
    camera = parts[0].removesuffix(" camera") if len(parts) > 1 else source
    folder = f"{source}/{'/'.join(parts[:-1])}" if len(parts) > 1 else source
    return FrameRecord(
        frame_id=f"{source}/{frame}",
        source=source,
        frame=frame,
        camera=camera,
        folder=folder,
        blur=blur,
        brightness=brightness,
        max_conf=conf,
        n_boxes=boxes,
        max_area_frac=area,
    )


def _population() -> list[FrameRecord]:
    """Two cameras, three folders each, and every confidence band represented."""
    records = []
    for camera in ("Primary camera", "Predator camera"):
        for folder in ("100GOPRO", "101GOPRO", "102GOPRO"):
            for i in range(60):
                conf, boxes = (0.0, 0)
                if i % 4 == 1:
                    conf, boxes = (0.15, 1)
                elif i % 4 == 2:
                    conf, boxes = (0.35, 1)
                elif i % 4 == 3:
                    conf, boxes = (0.80, 2)
                records.append(
                    _record(
                        f"{camera}/DCIM/{folder}/GOPR{i:04d}.JPG",
                        conf=conf,
                        boxes=boxes,
                        blur=10.0 + i,
                        brightness=40.0 + i * 2,
                    )
                )
    return records


def test_bands_split_on_whether_the_detector_fired_at_all() -> None:
    assert _record("a.jpg", conf=0.0, boxes=0).conf_band == "none"
    assert _record("a.jpg", conf=0.11, boxes=1).conf_band == "low"
    assert _record("a.jpg", conf=0.25, boxes=1).conf_band == "mid"
    assert _record("a.jpg", conf=0.90, boxes=1).conf_band == "high"


def test_sample_includes_frames_the_detector_found_nothing_in() -> None:
    """The whole point: a sample of only confident frames cannot measure recall."""
    result = stratified_sample(_population(), size=120, seed=1)

    assert len(result.frames) == 120
    assert result.band_counts["none"] > 0
    assert result.band_counts["low"] > 0  # low-conf-only frames, the threshold's crux
    assert result.band_counts["mid"] > 0
    assert result.band_counts["high"] > 0


def test_every_camera_and_folder_is_represented() -> None:
    result = stratified_sample(_population(), size=120, seed=1)

    assert set(result.camera_counts) == {"Primary", "Predator"}
    assert len(result.folder_counts) == 6  # 2 cameras x 3 folders, none crowded out


def test_sample_spans_brightness_and_blur_bins() -> None:
    result = stratified_sample(_population(), size=120, seed=1)

    brightness = {bin_of(r.brightness, result.brightness_cuts) for r in result.frames}
    blur = {bin_of(r.blur, result.blur_cuts) for r in result.frames}
    assert brightness == {"low", "mid", "high"}
    assert blur == {"low", "mid", "high"}


def test_sample_is_exactly_the_size_asked_for() -> None:
    for size in (20, 97, 300):
        assert len(stratified_sample(_population(), size=size, seed=3).frames) == size


def test_a_thin_band_hands_its_share_to_the_others() -> None:
    """One high-confidence frame in the whole run must not shrink the sample."""
    records = [_record(f"f{i}.jpg") for i in range(200)]
    records.append(_record("rare.jpg", conf=0.9, boxes=1))

    result = stratified_sample(records, size=50, seed=2)

    assert len(result.frames) == 50
    assert result.band_counts.get("high", 0) == 1  # took all there was, no more


def test_large_box_frames_get_their_own_quota() -> None:
    """The rule that demoted GOPR7891 can only be judged on labelled large boxes."""
    records = _population()
    records += [
        _record(f"Primary camera/DCIM/100GOPRO/BIG{i:03d}.JPG", conf=0.6, boxes=1, area=0.30)
        for i in range(40)
    ]

    result = stratified_sample(records, size=120, seed=1, large_box_quota=20)

    assert result.large_box_count >= 20
    assert len(result.frames) == 120  # topped up by swapping, so the size holds


def test_a_named_frame_can_be_forced_into_the_sample() -> None:
    records = _population()
    wanted = records[7].frame_id

    result = stratified_sample(records, size=40, seed=1, force_include=[wanted])

    assert wanted in {r.frame_id for r in result.frames}
    assert result.forced == [wanted]
    assert len(result.frames) == 40


def test_the_same_seed_gives_the_same_sample() -> None:
    first = stratified_sample(_population(), size=60, seed=99)
    second = stratified_sample(_population(), size=60, seed=99)
    other = stratified_sample(_population(), size=60, seed=100)

    assert [r.frame_id for r in first.frames] == [r.frame_id for r in second.frames]
    assert [r.frame_id for r in first.frames] != [r.frame_id for r in other.frames]


def test_frames_are_keyed_by_path_so_repeated_filenames_stay_distinct() -> None:
    """GoPro reuses basenames across folders; the key must survive that."""
    records = [
        _record("Primary camera/DCIM/100GOPRO/GOPR0001.JPG"),
        _record("Predator camera/DCIM/100GOPRO/GOPR0001.JPG"),
        _record("GOPR0001.JPG", source="120GOPRO"),
    ]

    ids = {r.frame_id for r in records}

    assert len(ids) == 3
    assert ids == {
        "RUN/Primary camera/DCIM/100GOPRO/GOPR0001.JPG",
        "RUN/Predator camera/DCIM/100GOPRO/GOPR0001.JPG",
        "120GOPRO/GOPR0001.JPG",
    }


def test_camera_and_folder_come_out_of_the_path() -> None:
    nested = _record("Primary camera/DCIM/104GOPROSOURCE/GOPR4523.JPG")
    flat = _record("GOPR6994.JPG", source="120GOPRO")

    assert nested.camera == "Primary"
    assert nested.folder == "RUN/Primary camera/DCIM/104GOPROSOURCE"
    assert flat.camera == "120GOPRO"
    assert flat.folder == "120GOPRO"


def test_quantile_bins_and_bin_names() -> None:
    cuts = quantile_bins(list(range(100)), 3)
    assert len(cuts) == 2
    assert bin_of(0, cuts) == "low"
    assert bin_of(50, cuts) == "mid"
    assert bin_of(99, cuts) == "high"
    assert bin_of(None, cuts) == "unknown"


def test_load_run_counts_recorded_boxes_not_boxes_above_threshold(tmp_path: Path) -> None:
    """frames.csv counts past the threshold; the band split needs the true count."""
    out = tmp_path / "run"
    out.mkdir()
    (out / "frames.csv").write_text(
        "frame,max_conf,n_boxes_above_threshold,blur,brightness,error\n"
        "a.jpg,0.150,0,30.0,90.0,\n"  # fired weakly: 0 above threshold, 1 recorded
        "b.jpg,0.000,0,31.0,91.0,\n"
        "bad.jpg,,,,,unreadable image\n",
        encoding="utf-8",
    )
    (out / "detections.csv").write_text(
        "frame,x1,y1,x2,y2,confidence,area_frac\na.jpg,0,0,10,10,0.150,0.250000\n",
        encoding="utf-8",
    )

    records = {r.frame: r for r in load_run(out, "RUN")}

    assert set(records) == {"a.jpg", "b.jpg"}  # the unreadable frame has nothing to label
    assert records["a.jpg"].n_boxes == 1
    assert records["a.jpg"].conf_band == "low"  # not "none": the detector did fire
    assert records["a.jpg"].max_area_frac == 0.25
    assert records["b.jpg"].conf_band == "none"


def test_written_sample_is_keyed_by_frame_id_and_meta_records_the_choice(
    tmp_path: Path,
) -> None:
    result = stratified_sample(_population(), size=30, seed=5)

    write_sample(result, tmp_path / "sample.csv")
    write_sample_meta(result, tmp_path / "sample_meta.json", {"RUN": "output/RUN"})

    with (tmp_path / "sample.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 30
    assert rows[0]["frame_id"].startswith("RUN/")
    assert "/" in rows[0]["frame"]  # a path, never a bare filename
    assert rows == sorted(rows, key=lambda r: r["frame_id"])

    meta = json.loads((tmp_path / "sample_meta.json").read_text(encoding="utf-8"))
    assert meta["seed"] == 5
    assert meta["sample_size"] == 30
    assert len(meta["brightness_cuts"]) == 2
    assert meta["band_counts"]
