"""Keep the test set out of training, and draw training frames where they teach most.

The held-out test set is only worth anything if nothing learned from it. That
is enforced here rather than trusted to care:

- The test set is frozen to a file listing its frame_ids, with a hash of that
  list and a hash of the labels behind it. Loading a test set whose list has
  been edited fails; evaluating against labels that changed after the freeze
  fails.
- Every path into training goes through `load_training_frames`, which raises
  if any training frame is a test frame. Not a warning, an exception.
- Frames a few minutes either side of a test frame are refused as well. The
  cameras shoot one frame a minute from a fixed mount, so a neighbouring frame
  can hold the same fish or the same school in the same place; training on it
  would let the model memorise a test answer under a different filename.
- Final evaluation claims the test set once and writes a ledger. A second
  claim raises. Looking at the test set to choose anything turns it into a
  validation set, and then there is no test set.

Training frames are drawn by `enriched_sample`: deliberately not uniform. A
uniform draw would be mostly empty water, which the model already handles; the
frames that change its behaviour are the ones where it is confidently wrong,
the ones holding fish, and the ones where it only just noticed a fish.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fishcount.sample import FrameRecord, quantile_bins, spread

_FRAME_NUMBER = re.compile(r"(\d+)(?=\.[A-Za-z]+$)")


class TestLeakError(RuntimeError):
    """A frame from the held-out test set, or one beside it, reached training."""

    __test__ = False  # a name starting "Test" is not a pytest test class


class TestSetError(RuntimeError):
    """The frozen test set is missing, edited, or has already been used."""

    __test__ = False


def frame_number(frame_id: str) -> int | None:
    """The camera's sequence number, e.g. 4523 from ".../GOPR4523.JPG"."""
    match = _FRAME_NUMBER.search(frame_id.rsplit("/", 1)[-1])
    return int(match.group(1)) if match else None


def frame_folder(frame_id: str) -> str:
    return frame_id.rsplit("/", 1)[0] if "/" in frame_id else ""


def _sha256(value: Any) -> str:
    blob = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _label_fingerprint(labels_dir: Path, frame_ids: Iterable[str]) -> str:
    """Hash of every label row belonging to these frames, order-independent."""
    wanted = set(frame_ids)
    boxes = sorted(
        tuple(sorted(row.items()))
        for row in _rows(labels_dir / "labels.csv")
        if row["frame_id"] in wanted
    )
    frames = sorted(
        tuple(sorted(row.items()))
        for row in _rows(labels_dir / "labeled_frames.csv")
        if row["frame_id"] in wanted
    )
    return _sha256({"boxes": boxes, "frames": frames})


@dataclass(frozen=True, slots=True)
class TestSet:
    """The frozen held-out frames. Build with `freeze`, read with `load`."""

    __test__ = False

    path: Path
    frame_ids: frozenset[str]
    labels_dir: Path
    labels_sha256: str
    buffer: int  # frames either side of a test frame that training may not use
    _near: dict[str, set[int]] = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self) -> None:
        for frame_id in self.frame_ids:
            number = frame_number(frame_id)
            if number is not None:
                self._near.setdefault(frame_folder(frame_id), set()).add(number)

    @property
    def ledger(self) -> Path:
        return self.path.with_suffix(".used.json")

    @classmethod
    def freeze(
        cls,
        labels_dir: Path,
        path: Path,
        predicate: Callable[[str], bool],
        *,
        buffer: int = 5,
    ) -> TestSet:
        """Freeze the labelled frames matching `predicate` as the test set.

        Refuses to overwrite an existing frozen set: redefining the test set
        after seeing results is the thing the freeze exists to prevent.
        """
        if path.exists():
            raise TestSetError(
                f"{path} already exists. A test set is frozen once; delete it by hand "
                "only if you are deliberately starting a new experiment."
            )
        frame_ids = sorted(
            row["frame_id"]
            for row in _rows(labels_dir / "labeled_frames.csv")
            if predicate(row["frame_id"]) and row.get("excluded") != "yes"
        )
        if not frame_ids:
            raise TestSetError(f"No labelled frames in {labels_dir} match the test-set rule.")
        payload = {
            "frame_ids": frame_ids,
            "frame_ids_sha256": _sha256(frame_ids),
            "labels_dir": str(labels_dir),
            "labels_sha256": _label_fingerprint(labels_dir, frame_ids),
            "buffer_frames": buffer,
            "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return cls.load(path)

    @classmethod
    def load(cls, path: Path) -> TestSet:
        if not path.is_file():
            raise TestSetError(f"No frozen test set at {path}.")
        payload = json.loads(path.read_text(encoding="utf-8"))
        frame_ids = list(payload["frame_ids"])
        if _sha256(frame_ids) != payload["frame_ids_sha256"]:
            raise TestSetError(
                f"{path} has been edited since it was frozen: its frame list no longer "
                "matches its own hash."
            )
        return cls(
            path=path,
            frame_ids=frozenset(frame_ids),
            labels_dir=Path(payload["labels_dir"]),
            labels_sha256=payload["labels_sha256"],
            buffer=int(payload["buffer_frames"]),
        )

    def leaks(self, frame_id: str) -> str | None:
        """Why `frame_id` may not be used for training, or None if it may."""
        if frame_id in self.frame_ids:
            return "is a test frame"
        number = frame_number(frame_id)
        if number is None or self.buffer <= 0:
            return None
        neighbours = self._near.get(frame_folder(frame_id), set())
        close = [n for n in neighbours if abs(n - number) <= self.buffer]
        if close:
            return f"is {min(abs(n - number) for n in close)} frame(s) from a test frame"
        return None

    def assert_disjoint(self, frame_ids: Iterable[str]) -> None:
        """Raise TestLeakError if any of these frames touches the test set."""
        problems = [(fid, why) for fid in frame_ids if (why := self.leaks(fid)) is not None]
        if problems:
            shown = "\n".join(f"  {fid} {why}" for fid, why in problems[:10])
            more = f"\n  ... and {len(problems) - 10} more" if len(problems) > 10 else ""
            raise TestLeakError(
                f"{len(problems)} training frame(s) overlap the held-out test set:\n{shown}{more}"
            )

    def verify_labels(self) -> None:
        """Raise if the test labels changed after the freeze."""
        if _label_fingerprint(self.labels_dir, self.frame_ids) != self.labels_sha256:
            raise TestSetError(
                "The test labels have changed since the test set was frozen. Numbers "
                "measured against them would not be measured against the frozen set."
            )

    def claim_final_evaluation(self, note: str, *, allow_reuse: bool = False) -> None:
        """Record that the test set is being used. The second claim raises."""
        self.verify_labels()
        if self.ledger.exists() and not allow_reuse:
            previous = json.loads(self.ledger.read_text(encoding="utf-8"))
            raise TestSetError(
                f"The test set was already used on {previous['used_at']} "
                f"({previous['note']}). Choosing anything after looking at it makes it a "
                "validation set. Pass allow_reuse only to re-print numbers already reported."
            )
        self.ledger.write_text(
            json.dumps(
                {"used_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "note": note},
                indent=2,
            ),
            encoding="utf-8",
        )


@dataclass(frozen=True, slots=True)
class Stratum:
    """One slice of the training sample: which frames qualify, and how many."""

    name: str
    quota: int
    qualifies: Callable[[FrameRecord], bool]


@dataclass(slots=True)
class EnrichedSample:
    frames: list[FrameRecord]
    stratum_of: dict[str, str]
    counts: dict[str, int]
    available: dict[str, int]  # eligible frames per stratum when it was drawn
    refused: dict[str, int]  # candidates dropped for touching the test set
    brightness_cuts: list[float]
    blur_cuts: list[float]
    seed: int


def enriched_sample(
    records: Sequence[FrameRecord],
    strata: Sequence[Stratum],
    test_set: TestSet,
    *,
    seed: int = 20260924,
) -> EnrichedSample:
    """Fill each stratum in order, never drawing a frame twice or near the test set.

    Order matters where strata overlap: a frame goes to the first stratum that
    takes it. Within a stratum frames are spread round robin over folders and
    then over brightness and blur, so no single folder or lighting condition
    fills a quota on its own.
    """
    rng = random.Random(seed)
    refused = {"is a test frame": 0, "is beside a test frame": 0}
    eligible: list[FrameRecord] = []
    for record in records:
        why = test_set.leaks(record.frame_id)
        if why is None:
            eligible.append(record)
        elif why == "is a test frame":
            refused["is a test frame"] += 1
        else:
            refused["is beside a test frame"] += 1

    brightness_cuts = quantile_bins([r.brightness for r in eligible if r.brightness is not None], 3)
    blur_cuts = quantile_bins([r.blur for r in eligible if r.blur is not None], 3)

    chosen: dict[str, FrameRecord] = {}
    stratum_of: dict[str, str] = {}
    counts: dict[str, int] = {}
    available: dict[str, int] = {}
    for stratum in strata:
        pool = [r for r in eligible if r.frame_id not in chosen and stratum.qualifies(r)]
        available[stratum.name] = len(pool)
        picked = spread(pool, stratum.quota, brightness_cuts, blur_cuts, rng)
        for record in picked:
            chosen[record.frame_id] = record
            stratum_of[record.frame_id] = stratum.name
        counts[stratum.name] = len(picked)

    test_set.assert_disjoint(chosen)  # the filter above makes this impossible; prove it
    return EnrichedSample(
        frames=sorted(chosen.values(), key=lambda r: r.frame_id),
        stratum_of=stratum_of,
        counts=counts,
        available=available,
        refused=refused,
        brightness_cuts=brightness_cuts,
        blur_cuts=blur_cuts,
        seed=seed,
    )


@dataclass(frozen=True, slots=True)
class TrainingFrame:
    """One labelled training frame, reduced to what a detector trains on."""

    frame_id: str
    source: str
    frame: str
    fish: tuple[tuple[int, int, int, int], ...]  # every fish box, model-found or drawn


def load_training_frames(labels_dir: Path, test_set: TestSet) -> list[TrainingFrame]:
    """The only way into training. Raises TestLeakError on any test overlap.

    A frame's fish are the model's boxes marked fish plus the boxes drawn
    around fish it missed. Boxes marked not fish are dropped, so that region
    is trained as background, which is the lesson those boxes exist to teach.
    Frames left out with E are skipped entirely: they carry no verdict.
    """
    reviewed = [r for r in _rows(labels_dir / "labeled_frames.csv") if r.get("excluded") != "yes"]
    test_set.assert_disjoint(r["frame_id"] for r in reviewed)

    fish: dict[str, list[tuple[int, int, int, int]]] = {}
    for row in _rows(labels_dir / "labels.csv"):
        if row["label"] == "fish":
            fish.setdefault(row["frame_id"], []).append(
                (int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"]))
            )
    return [
        TrainingFrame(
            frame_id=row["frame_id"],
            source=row["source"],
            frame=row["frame"],
            fish=tuple(fish.get(row["frame_id"], [])),
        )
        for row in reviewed
    ]
