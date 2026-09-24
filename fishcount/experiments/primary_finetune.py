"""Fine-tune the detector on the Primary camera: the experiment, step by step.

Every dataset-specific choice lives here, where it can be read and changed;
the machinery it calls lives in the fishcount package, where it is tested.

    python experiments/primary_finetune.py freeze     # freeze the held-out test set
    python experiments/primary_finetune.py sample     # draw the frames to label for training

Later steps (export, train, choose the threshold, evaluate) are added once the
training labels exist.

Scope is the Primary camera only. The Predator camera and 120GOPRO are out of
scope for every step, including labelling.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from fishcount.sample import FrameRecord, load_run, write_sample_rows  # noqa: E402
from fishcount.training import Stratum, TestSet, enriched_sample  # noqa: E402

SOURCE = "ALLFISHDATA"
RUN = REPO / "output" / "ALLFISHDATA"
PRIMARY = f"{SOURCE}/Primary camera/DCIM/"

EVAL_LABELS = REPO / "labels"  # the original 300-frame evaluation sample
TEST_SET = REPO / "splits" / "test_primary.json"
TRAIN_DIR = REPO / "labels_train"

# Frames either side of a test frame that training may not use. One frame a
# minute from a fixed camera: five frames is five minutes, long enough that a
# fish or school in a test frame has usually moved on.
BUFFER = 5

# Where the labelled evaluation found fish, and where the detector fired
# confidently on nothing. These drive the enrichment below.
FISH_FOLDERS = {"104GOPROSOURCE", "101GOPRO"}
HARD_NEGATIVE_FOLDERS = {"102GOPRO", "103GOPRO", "105GOPRO"}


def _leaf(record: FrameRecord) -> str:
    return record.folder.rsplit("/", 1)[-1]


# Filled in order; a frame goes to the first stratum that takes it.
STRATA = (
    # Confident boxes, anywhere: where both the real fish and the confident
    # false positives live, so each label here moves the threshold decision.
    Stratum("confident", 120, lambda r: r.conf_band == "high"),
    # The folders that actually hold fish, at any confidence the model fired.
    Stratum("fish_folders", 120, lambda r: _leaf(r) in FISH_FOLDERS and r.n_boxes > 0),
    # Confident firing where labelling found no fish at all: substrate, murk,
    # cobble. The false positives a threshold cannot remove.
    Stratum(
        "hard_negatives",
        90,
        lambda r: _leaf(r) in HARD_NEGATIVE_FOLDERS and r.max_conf >= 0.25,
    ),
    # Fired, but only faintly: where a fish the model barely noticed would be.
    Stratum("faint", 70, lambda r: r.conf_band == "low"),
)


def freeze() -> None:
    test = TestSet.freeze(
        EVAL_LABELS, TEST_SET, lambda frame_id: frame_id.startswith(PRIMARY), buffer=BUFFER
    )
    print(f"froze {len(test.frame_ids)} Primary frames as the test set -> {TEST_SET}")


def sample(seed: int) -> None:
    if (TRAIN_DIR / "labeled_frames.csv").exists():
        sys.exit(
            f"{TRAIN_DIR} already holds labels. Drawing a new sample would orphan them; "
            "move them aside first if that is really what you want."
        )
    test = TestSet.load(TEST_SET)
    records = [r for r in load_run(RUN, SOURCE) if r.frame_id.startswith(PRIMARY)]
    result = enriched_sample(records, STRATA, test, seed=seed)

    write_sample_rows(
        result.frames,
        TRAIN_DIR / "sample.csv",
        result.brightness_cuts,
        result.blur_cuts,
        extra={"stratum": result.stratum_of},
    )
    meta = {
        "seed": result.seed,
        "test_set": str(TEST_SET.relative_to(REPO)),
        "buffer_frames": BUFFER,
        "candidates": len(records),
        "refused": result.refused,
        "strata": {
            s.name: {
                "quota": s.quota,
                "drawn": result.counts[s.name],
                "eligible": result.available[s.name],
            }
            for s in STRATA
        },
        "per_folder": dict(sorted(Counter(_leaf(r) for r in result.frames).items())),
        "brightness_cuts": result.brightness_cuts,
        "blur_cuts": result.blur_cuts,
    }
    (TRAIN_DIR / "sample_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    steps = parser.add_subparsers(dest="step", required=True)
    steps.add_parser("freeze", help="Freeze the labelled Primary frames as the test set.")
    draw = steps.add_parser("sample", help="Draw the training frames to label.")
    draw.add_argument("--seed", type=int, default=20260924)
    args = parser.parse_args()
    if args.step == "freeze":
        freeze()
    elif args.step == "sample":
        sample(args.seed)


if __name__ == "__main__":
    main()
