"""The test set is only worth anything if training provably never touched it."""

import csv
import json
from pathlib import Path

import pytest

from fishcount.sample import FrameRecord
from fishcount.training import (
    Stratum,
    TestLeakError,
    TestSet,
    TestSetError,
    enriched_sample,
    frame_number,
    load_training_frames,
)

DCIM = "RUN/Primary camera/DCIM"


def _write_labels(
    labels_dir: Path, frames: dict[str, list[tuple[str, str]]], **excluded: bool
) -> None:
    """frames: frame_id -> [(origin, label), ...]. Keyword `excluded` by frame name."""
    labels_dir.mkdir(parents=True, exist_ok=True)
    with (labels_dir / "labels.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "frame_id",
                "source",
                "frame",
                "box_index",
                "origin",
                "label",
                "x1",
                "y1",
                "x2",
                "y2",
                "confidence",
            ]
        )
        for frame_id, boxes in frames.items():
            for index, (origin, label) in enumerate(boxes):
                writer.writerow(
                    [
                        frame_id,
                        "RUN",
                        frame_id[4:],
                        index,
                        origin,
                        label,
                        10,
                        20,
                        30,
                        40,
                        "" if origin == "missed" else "0.5",
                    ]
                )
    with (labels_dir / "labeled_frames.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "frame_id",
                "source",
                "frame",
                "n_model_boxes",
                "n_fish",
                "n_not_fish",
                "n_missed",
                "has_fish",
                "excluded",
            ]
        )
        for frame_id in frames:
            name = frame_id.rsplit("/", 1)[-1].split(".")[0]
            writer.writerow(
                [
                    frame_id,
                    "RUN",
                    frame_id[4:],
                    0,
                    0,
                    0,
                    0,
                    "no",
                    "yes" if excluded.get(name) else "no",
                ]
            )


def _frozen(tmp_path: Path, test_ids: list[str], buffer: int = 5) -> TestSet:
    _write_labels(tmp_path / "labels", {fid: [("model", "fish")] for fid in test_ids})
    return TestSet.freeze(
        tmp_path / "labels", tmp_path / "splits" / "test.json", lambda f: True, buffer=buffer
    )


def _record(frame_id: str, *, conf: float = 0.6, boxes: int = 1) -> FrameRecord:
    folder = frame_id.rsplit("/", 1)[0]
    return FrameRecord(
        frame_id=frame_id,
        source="RUN",
        frame=frame_id[4:],
        camera="Primary",
        folder=folder,
        blur=50.0,
        brightness=100.0,
        max_conf=conf,
        n_boxes=boxes,
        max_area_frac=0.01,
    )


def test_frame_number_reads_the_camera_sequence() -> None:
    assert frame_number(f"{DCIM}/104GOPROSOURCE/GOPR4523.JPG") == 4523
    assert frame_number("RUN/odd name.png") is None


def test_a_test_frame_can_never_reach_training(tmp_path: Path) -> None:
    test = _frozen(tmp_path, [f"{DCIM}/104GOPRO/GOPR0100.JPG"])

    with pytest.raises(TestLeakError, match="is a test frame"):
        test.assert_disjoint([f"{DCIM}/104GOPRO/GOPR0100.JPG"])


def test_a_neighbouring_frame_is_refused_because_it_can_hold_the_same_fish(tmp_path: Path) -> None:
    """One frame a minute from a fixed camera: frame 103 can show frame 100's fish."""
    test = _frozen(tmp_path, [f"{DCIM}/104GOPRO/GOPR0100.JPG"], buffer=5)

    with pytest.raises(TestLeakError, match="3 frame"):
        test.assert_disjoint([f"{DCIM}/104GOPRO/GOPR0103.JPG"])
    test.assert_disjoint([f"{DCIM}/104GOPRO/GOPR0106.JPG"])  # outside the buffer: fine


def test_the_same_number_in_another_folder_is_not_a_neighbour(tmp_path: Path) -> None:
    """Folders are separate deployments; GOPR0100 recurs with no relation."""
    test = _frozen(tmp_path, [f"{DCIM}/104GOPRO/GOPR0100.JPG"])

    test.assert_disjoint([f"{DCIM}/105GOPRO/GOPR0100.JPG"])


def test_a_frozen_test_set_cannot_be_redefined_by_freezing_again(tmp_path: Path) -> None:
    _frozen(tmp_path, [f"{DCIM}/104GOPRO/GOPR0100.JPG"])

    with pytest.raises(TestSetError, match="already exists"):
        TestSet.freeze(tmp_path / "labels", tmp_path / "splits" / "test.json", lambda f: True)


def test_editing_the_frozen_frame_list_is_detected(tmp_path: Path) -> None:
    test = _frozen(tmp_path, [f"{DCIM}/104GOPRO/GOPR0100.JPG"])
    payload = json.loads(test.path.read_text(encoding="utf-8"))
    payload["frame_ids"].append(f"{DCIM}/104GOPRO/GOPR0999.JPG")
    test.path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(TestSetError, match="edited"):
        TestSet.load(test.path)


def test_changing_a_test_label_after_the_freeze_is_detected(tmp_path: Path) -> None:
    test = _frozen(tmp_path, [f"{DCIM}/104GOPRO/GOPR0100.JPG"])
    test.verify_labels()  # unchanged: fine

    _write_labels(tmp_path / "labels", {f"{DCIM}/104GOPRO/GOPR0100.JPG": [("model", "not_fish")]})

    with pytest.raises(TestSetError, match="changed"):
        test.verify_labels()


def test_the_test_set_can_be_claimed_for_final_evaluation_only_once(tmp_path: Path) -> None:
    test = _frozen(tmp_path, [f"{DCIM}/104GOPRO/GOPR0100.JPG"])

    test.claim_final_evaluation("base vs fine-tuned")

    with pytest.raises(TestSetError, match="already used"):
        test.claim_final_evaluation("one more look")
    test.claim_final_evaluation("re-print", allow_reuse=True)  # explicit, and recorded


def test_training_frames_are_refused_if_any_overlaps_the_test_set(tmp_path: Path) -> None:
    test = _frozen(tmp_path, [f"{DCIM}/104GOPRO/GOPR0100.JPG"])
    _write_labels(
        tmp_path / "train",
        {f"{DCIM}/104GOPRO/GOPR0500.JPG": [("model", "fish")], f"{DCIM}/104GOPRO/GOPR0101.JPG": []},
    )

    with pytest.raises(TestLeakError):
        load_training_frames(tmp_path / "train", test)


def test_training_frames_keep_fish_boxes_and_drop_the_rest(tmp_path: Path) -> None:
    """Not-fish boxes become background; drawn boxes are fish the model missed."""
    test = _frozen(tmp_path, [f"{DCIM}/104GOPRO/GOPR0100.JPG"])
    _write_labels(
        tmp_path / "train",
        {
            f"{DCIM}/104GOPRO/GOPR0500.JPG": [
                ("model", "fish"),
                ("model", "not_fish"),
                ("missed", "fish"),
            ],
            f"{DCIM}/104GOPRO/GOPR0600.JPG": [("model", "not_fish")],
        },
    )

    frames = {
        f.frame_id.rsplit("/", 1)[-1]: f for f in load_training_frames(tmp_path / "train", test)
    }

    assert len(frames["GOPR0500.JPG"].fish) == 2
    assert frames["GOPR0600.JPG"].fish == ()  # a pure hard negative, still used


def test_a_frame_left_out_is_not_trained_as_a_negative(tmp_path: Path) -> None:
    """A school too dense to box is not "no fish here"; it is skipped."""
    test = _frozen(tmp_path, [f"{DCIM}/104GOPRO/GOPR0100.JPG"])
    _write_labels(
        tmp_path / "train",
        {f"{DCIM}/104GOPRO/GOPR0500.JPG": [], f"{DCIM}/104GOPRO/GOPR0700.JPG": []},
        GOPR0700=True,
    )

    frames = load_training_frames(tmp_path / "train", test)

    assert [f.frame_id.rsplit("/", 1)[-1] for f in frames] == ["GOPR0500.JPG"]


def test_enriched_sample_fills_strata_in_order_and_never_touches_the_test_set(
    tmp_path: Path,
) -> None:
    test = _frozen(tmp_path, [f"{DCIM}/A/GOPR0100.JPG"], buffer=5)
    records = [_record(f"{DCIM}/A/GOPR{n:04d}.JPG", conf=0.8) for n in range(90, 200)]
    records += [_record(f"{DCIM}/B/GOPR{n:04d}.JPG", conf=0.15) for n in range(0, 60)]

    result = enriched_sample(
        records,
        [
            Stratum("confident", 20, lambda r: r.max_conf >= 0.5),
            Stratum("faint", 10, lambda r: 0 < r.max_conf < 0.25),
        ],
        test,
        seed=1,
    )

    assert result.counts == {"confident": 20, "faint": 10}
    ids = [r.frame_id for r in result.frames]
    assert len(ids) == len(set(ids)) == 30  # no frame drawn twice
    assert all(test.leaks(fid) is None for fid in ids)
    assert result.refused == {"is a test frame": 1, "is beside a test frame": 10}
    assert {result.stratum_of[fid] for fid in ids} == {"confident", "faint"}


def test_a_frame_goes_to_the_first_stratum_that_wants_it(tmp_path: Path) -> None:
    test = _frozen(tmp_path, [f"{DCIM}/Z/GOPR9000.JPG"])
    records = [_record(f"{DCIM}/A/GOPR{n:04d}.JPG", conf=0.8) for n in range(5)]

    result = enriched_sample(
        records,
        [Stratum("first", 3, lambda r: True), Stratum("second", 10, lambda r: True)],
        test,
    )

    assert result.counts == {"first": 3, "second": 2}  # the leftovers, not repeats
