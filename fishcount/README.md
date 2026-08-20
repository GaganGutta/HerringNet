# fishcount

Detect fish in a folder of images, fully offline on a Windows laptop CPU. Put a
folder in, run one command (or drag the folder onto a .bat file), get every
frame that holds a fish, tiered by confidence. No database, no server, no
cloud, and **no counting**: this tool reports detections, not fish counts.

Detection uses the Community Fish Detector (`cfd-yolov12x.pt`): a YOLOv12x
object detector with a single class (`fish`). Every detection is "fish"; there
is no species classification. Input images are never modified; all output goes
to a separate folder.

## Setup (once)

1. Install Python 3.10+ from python.org (tick "Add python.exe to PATH").
2. In this folder, run:

   ```
   python -m venv .venv
   .venv\Scripts\activate
   pip install -e .
   ```

   This downloads roughly 2 GB of packages (the torch CPU wheel is large).
   It is a one-time step; everything runs offline afterwards.
3. Put the model weights at `models\cfd-yolov12x.pt`. The tool never downloads,
   retrains, or substitutes a model: if the file is missing it stops and tells
   you where to put it. Weight files are git-ignored.

## Use it

Drag any folder of images onto **Detect Folder.bat**, or run:

```
fishcount detect "C:\path\to\your images"
```

One inference pass per image (imgsz 1536, confidence floor 0.10, IoU 0.7,
max_det 3000), then every frame is tiered. Output goes to `output\<name>\`:

```
output\120GOPRO\
  confident.csv        detections in frames that almost certainly hold fish
  under_review.csv     detections worth a human glance
  not_confident.csv    weak / static / oversized detections, quarantined
  confident\           annotated images for each tier (review is visual)
  under_review\
  not_confident\
  results.json         the raw record: every detection + per-frame blur score
```

Each CSV row is one detection: `filename, x1, y1, x2, y2, confidence, note,
frame_blurry`. All of a frame's detections land in its frame's tier file.
Frames with no detections at all appear in no file.

## How tiering works (detection-first)

The rule is **demote, never delete**: nothing above the confidence floor is
ever discarded. Three signals demote a detection or cap a frame; all three are
auditable in the `note` column and the annotated images:

| Signal | What it means | Effect |
| --- | --- | --- |
| `static` | the box recurs at the same pixels in `--static-min-frames` (default 8) distinct frames: a rock, shell, or debris, not a fish | does not count as fish evidence |
| `oversized` | the box covers more than 10% of the frame: murky water misread as one giant fish (real fish here are under 5% of the frame) | routed to review, never confident |
| blur cap | the frame is blurrier than `--blur-percentile` (default 25) of its own folder | frame tier capped at under_review unless the school exemption applies |

Frame tiers, computed from the non-static, non-oversized detections:

| Tier | Rule |
| --- | --- |
| `confident` | 2+ detections at >= 0.25, or any detection at >= 0.50, in a sharp frame |
| `under_review` | exactly one moderate detection; or confident-level evidence in a blurry frame; or an oversized box at moderate confidence |
| `not_confident` | only weak (0.10-0.25), static, or weak-oversized detections |

**The blur cap and the school exemption.** Blur is normalized per folder
(percentile of the run's own distribution) because absolute sharpness tracks
turbidity and lighting as much as focus, so an absolute cutoff would not
transfer between sites. A blurry frame cannot reach `confident`, no matter how
many boxes it has, **unless** it has 4 or more real detections. The exemption
exists because dense fish schools are blurry (the fish move); note that it was
fit on six school frames from the 104GOPRO and 120GOPRO deployments and should
be re-checked on new sites.

## The recall/precision tradeoff, stated plainly

- Recall first: the detector runs at a 0.10 floor and nothing above it is
  deleted, so anything the model fires on appears in one of the three files.
  Blur tiering is triage, not a detector fix: it does not stop sediment from
  firing the model, it keeps those firings out of `confident`.
- Precision is bought only inside `confident.csv`, by demotion. The cost: a
  real fish in a blurry frame with thin evidence lands in `under_review`
  instead of `confident`. It is never lost, but `under_review` is a real part
  of the workflow, not a dumping ground.
- Validation to date: on 80 randomly sampled no-detection frames across the
  two GOPRO deployments, an independent tiled-inference oracle surfaced zero
  verified fish (every strong hit was hand-checked and was equipment, rock,
  flare, or murk). Rule-of-three 95% upper bound: under ~4% of no-detection
  frames could hold an oracle-findable fish.

## Known limitations

- **A fish holding station is demoted like a rock.** The static rule cannot
  tell a rock from a fish that sits at the same spot in 8+ distinct frames.
  It only demotes (to `not_confident`, note `static`), so such a fish stays
  visible and recoverable in review. Lower `--static-min-frames` risk-free is
  not possible; disabling it (`--static-min-frames 0`) restores those frames
  at the cost of hundreds of rock frames.
- The blur percentile always marks the blurriest quarter of any folder as
  blurry, even in a uniformly sharp deployment; this only matters for frames
  with thin evidence.
- The school exemption threshold (4 real detections) is fit on six frames from
  two deployments at one site type. Re-validate before trusting it elsewhere.

## Flags

| Flag | Default | Meaning |
| --- | --- | --- |
| `--name NAME` / `--out DIR` | input folder's name | where results go under `output\` |
| `--conf FLOAT` | `0.10` | confidence floor; everything above it is recorded |
| `--imgsz INT` | `1536` | inference size; 1024 is faster but misses dense schools of small fish |
| `--iou FLOAT` | `0.7` | NMS IoU; higher keeps tightly packed fish |
| `--max-det INT` | `3000` | max detections per image |
| `--batch-size INT` | `8` | images per model pass; lower if RAM is tight |
| `--blur-percentile FLOAT` | `25` | per-folder blur cutoff; `0` disables the blur cap |
| `--static-min-frames INT` | `8` | recurrence threshold for stationary objects; `0` disables |
| `--no-images` | off | CSVs and results.json only |
| `--open` | off | open the output folder when finished |

Defaults can also be set in [config.yaml](config.yaml). Precedence:
CLI flags > config.yaml > built-in defaults.

## Speed on CPU

One pass at imgsz 1536 costs roughly 6 s per 4000x3000 frame on a laptop CPU
(batched, model loaded once, each image read from disk exactly once; the blur
score is computed from the already-decoded image). `--imgsz 1024` is about
2.5x faster but misses dense schools of small fish entirely; use it only when
that is acceptable.

## Development

```
pip install -e ".[dev]"
pytest            # tests mock the detector; no model weights needed
ruff check .
ruff format .
mypy fishcount
```

Layout: `detector.py` (YOLO wrapper behind a small Detector protocol),
`batch.py` (folder pipeline + results.json), `classify.py` (tiering and the
three-file output), `sequence.py` (static-object detection across frames),
`draw.py` (boxes), `config.py` (pydantic + config.yaml), `cli.py`.

The last counting-capable version (base/dense/thorough stages, SAHI tiled
inference) is preserved at tag `v0.2-three-stage` / branch
`three-stage-pipeline` for reproducibility of earlier results.

## License

Ultralytics and the Community Fish Detector weights are AGPL-3.0, so this tool
is AGPL-3.0 as well. Local use is unaffected.
