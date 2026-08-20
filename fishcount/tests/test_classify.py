import csv
import json
from pathlib import Path

from fishcount.classify import classify

W, H = 1000, 800  # frame size used by all synthetic entries (area 800k)
SMALL = [10, 10, 110, 60]  # 5k px, 0.6% of frame
GIANT = [0, 0, 500, 400]  # 200k px, 25% of frame


def _frame(file: str, dets: list[tuple[list[int], float]], blur: float = 100.0) -> dict:
    return {
        "file": file,
        "width": W,
        "height": H,
        "blur": blur,
        "detections": [{"box": box, "confidence": conf} for box, conf in dets],
    }


def _write_results(out: Path, images: list[dict]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    payload = {"input": "x", "model": "m.pt", "conf": 0.10, "imgsz": 1536, "images": images}
    (out / "results.json").write_text(json.dumps(payload), encoding="utf-8")


def _rows(out: Path, tier: str) -> list[list[str]]:
    with (out / f"{tier}.csv").open(encoding="utf-8", newline="") as handle:
        return list(csv.reader(handle))[1:]


def test_basic_tiers(tmp_path: Path) -> None:
    out = tmp_path / "out"
    _write_results(
        out,
        [
            _frame("strong.jpg", [(SMALL, 0.9)]),
            _frame("two_real.jpg", [(SMALL, 0.3), ([200, 200, 300, 250], 0.3)]),
            _frame("one_real.jpg", [(SMALL, 0.3)]),
            _frame("weak.jpg", [(SMALL, 0.15)]),
            _frame("empty.jpg", []),
        ],
    )
    summary = classify(out, floor=0.10, static_min_frames=0, move_images=False)

    assert summary.frames_per_tier == {"confident": 2, "under_review": 1, "not_confident": 1}
    assert summary.no_detection_frames == 1
    assert [r[0] for r in _rows(out, "confident")] == ["strong.jpg", "two_real.jpg", "two_real.jpg"]
    assert [r[0] for r in _rows(out, "under_review")] == ["one_real.jpg"]
    weak_rows = _rows(out, "not_confident")
    assert weak_rows[0][0] == "weak.jpg" and weak_rows[0][6] == "weak"


def test_oversized_boxes_are_demoted_not_dropped(tmp_path: Path) -> None:
    out = tmp_path / "out"
    _write_results(
        out,
        [
            _frame("giant_strong.jpg", [(GIANT, 0.9)]),  # murk as one huge fish
            _frame("giant_weak.jpg", [(GIANT, 0.15)]),
            _frame(
                "giant_plus_fish.jpg", [(GIANT, 0.9), (SMALL, 0.3), ([300, 300, 400, 350], 0.3)]
            ),
        ],
    )
    summary = classify(out, floor=0.10, static_min_frames=0, move_images=False)

    # A frame-filling box never makes a frame confident, but it is never deleted either.
    assert summary.frames_per_tier == {"confident": 1, "under_review": 1, "not_confident": 1}
    review = _rows(out, "under_review")
    assert review[0][0] == "giant_strong.jpg" and "oversized" in review[0][6]
    weak = _rows(out, "not_confident")
    assert weak[0][0] == "giant_weak.jpg" and weak[0][6] == "oversized;weak"
    confident = _rows(out, "confident")  # the real fish carry the frame; giant row rides along
    assert [r[0] for r in confident] == ["giant_plus_fish.jpg"] * 3


def test_static_detections_demote_to_not_confident(tmp_path: Path) -> None:
    out = tmp_path / "out"
    rock = [50, 50, 120, 100]
    images = [_frame(f"f{i:02d}.jpg", [(rock, 0.6)]) for i in range(9)]
    images.append(_frame("fishy.jpg", [(rock, 0.6), ([700, 600, 800, 650], 0.55)]))
    _write_results(out, images)

    summary = classify(out, floor=0.10, static_min_frames=8, move_images=False)

    assert summary.frames_per_tier["not_confident"] == 9  # rock-only frames
    assert summary.frames_per_tier["confident"] == 1  # the frame with a moving fish
    assert summary.static_detections == 10
    rows = _rows(out, "confident")
    assert {r[0] for r in rows} == {"fishy.jpg"}
    notes = {r[5]: r[6] for r in rows}  # conf -> note
    assert notes["0.600"] == "static" and notes["0.550"] == ""


def test_blur_caps_thin_evidence_but_exempts_schools(tmp_path: Path) -> None:
    out = tmp_path / "out"
    school = [([100 * i, 100, 100 * i + 80, 160], 0.3) for i in range(1, 5)]  # 4 real dets
    _write_results(
        out,
        [
            _frame("blurry_two.jpg", [(SMALL, 0.3), ([300, 300, 400, 350], 0.3)], blur=5.0),
            _frame("blurry_school.jpg", school, blur=5.0),
            _frame("sharp_a.jpg", [(SMALL, 0.9)], blur=100.0),
            _frame("sharp_b.jpg", [], blur=100.0),
            _frame("sharp_c.jpg", [], blur=100.0),
            _frame("sharp_d.jpg", [], blur=100.0),
        ],
    )
    summary = classify(out, floor=0.10, blur_percentile=40, static_min_frames=0, move_images=False)

    # Both blurry frames are below the folder's 40th percentile; the two-box frame
    # is capped to under_review, the 4-detection school keeps confident.
    assert summary.blur_capped == 1
    assert summary.school_exempt == 1
    assert [r[0] for r in _rows(out, "under_review")] == ["blurry_two.jpg"] * 2
    confident_files = {r[0] for r in _rows(out, "confident")}
    assert confident_files == {"blurry_school.jpg", "sharp_a.jpg"}


def test_blur_percentile_zero_disables_cap(tmp_path: Path) -> None:
    out = tmp_path / "out"
    _write_results(
        out,
        [
            _frame("blurry_two.jpg", [(SMALL, 0.3), ([300, 300, 400, 350], 0.3)], blur=1.0),
            _frame("sharp.jpg", [], blur=100.0),
        ],
    )
    summary = classify(out, floor=0.10, blur_percentile=0, static_min_frames=0, move_images=False)
    assert summary.blur_threshold is None
    assert summary.frames_per_tier["confident"] == 1
    assert summary.blur_capped == 0


def test_annotated_images_move_into_tier_folders(tmp_path: Path) -> None:
    out = tmp_path / "out"
    _write_results(
        out,
        [
            _frame("a.jpg", [(SMALL, 0.9)]),
            _frame("sub/b.jpg", [(SMALL, 0.15)]),
            _frame("c.jpg", []),
        ],
    )
    (out / "annotated" / "sub").mkdir(parents=True)
    (out / "annotated" / "a.jpg").write_bytes(b"annotated-a")
    (out / "annotated" / "sub" / "b.jpg").write_bytes(b"annotated-b")

    classify(out, floor=0.10, static_min_frames=0, move_images=True)

    assert (out / "confident" / "a.jpg").read_bytes() == b"annotated-a"
    assert (out / "not_confident" / "sub" / "b.jpg").read_bytes() == b"annotated-b"
    assert not (out / "annotated").exists()
    for tier in ("confident", "under_review", "not_confident"):
        assert (out / f"{tier}.csv").is_file()  # all three files always exist
