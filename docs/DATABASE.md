# Image database and FiftyOne review

This replaces the one-sheet-per-week Excel workflow with a single SQLite
database that is the source of truth for camera-trap imagery and labels,
plus FiftyOne as a visual surface for reviewing and labeling images
instead of typing into a grid.

## Why

The legacy workbook keyed images by a `GOPRO####` id that repeated across
weeks (so it was not unique), never stored a path to the actual image
file, and let categorical fields drift (invalid `uncountable_reason`
codes, `OBS_002` vs `OBS002`, two different column orders). The database
fixes all of that: images are keyed by `(session, original_name)` and
linked to real files by relative path plus a content hash, and every
categorical column is a foreign key that cannot drift.

## Data model

```
sites ──< sessions ──< images ──< observations      (human labels)
                          │   └──< expert_reviews     (expert confirmation)
                          │   └──< model_predictions  (HerringNet output)
observers ─────────────────┘ (referenced by observations / expert_reviews)
count_bins, uncountable_categories  (lookups, enforced as foreign keys)
```

A `session` is one site-week (originally one spreadsheet tab). An `image`
belongs to a session and carries its file path, content hash, and
capture time. The same image can hold a human `observation`, an
`expert_review`, and a `model_prediction` at once without one
overwriting another. Reviewer edits from FiftyOne are written as new
observations with `source='fiftyone'`, so the original record is never
destroyed.

## One-time setup

```bash
pip install -e ".[label]"     # installs FiftyOne for the review UI
```

## Workflow

```bash
# 1. Migrate the existing Excel workbook into the database.
#    Cleans Y/N -> 1/0, normalizes observer ids, and quarantines invalid
#    values into the observation notes (nothing is dropped silently).
herringnet db import-excel "River_Herring_Camera_Trap_Database.xlsx"

# 2. Link the image files. Point at the archive whose folders are your
#    site/week folders; paths are stored relative to this root.
herringnet db scan /path/to/image_archive

# 3. (Optional) Load model predictions so they overlay during review.
herringnet detect /path/to/image_archive --save-json
herringnet db ingest-predictions outputs/directory_results.json --model cfd

# 4. Check what you have.
herringnet db stats

# 5. Review and label visually. Builds a FiftyOne dataset from the DB and
#    opens the app. Filter by tag (e.g. model_found_fish), confirm or
#    correct, tag samples 'confirmed' / 'corrected' / 'needs_review'.
herringnet db review /path/to/image_archive

# 6. Write your review edits back into the database.
herringnet db sync

# 7. (Optional) Export confirmed/model boxes to YOLO format for training.
herringnet db export-yolo data/processed/yolo_export --field model
```

## Reviewer conventions (read by `db sync`)

Only samples you actually touch are written back. Mark a sample by either:

- tagging it `confirmed`, `corrected`, or `needs_review`, and/or
- setting the sample fields `rh_present` (bool), `rh_count` (int), or
  `review_notes` (text) in the FiftyOne app.

## Notes

- The `.db` file is generated and git-ignored; regenerate it any time
  from the Excel workbook and a folder scan.
- Paths are stored relative to the scan root and every file has a
  content hash, so the archive can move between machines or drives
  without breaking links.
- Going forward you do not need Excel: scanning a new week's folder
  creates its session and image rows automatically, and labeling happens
  in FiftyOne.
