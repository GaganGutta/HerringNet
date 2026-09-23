# fishcount

Detect fish in a folder of images, fully offline on a Windows laptop. Put a
folder in, run one command (or drag the folder onto a .bat file), get every
detection the model made and one row per frame. No database, no server, no
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

   This downloads roughly 2 GB of packages (the torch wheel is large).
   It is a one-time step; everything runs offline afterwards.
3. Put the model weights at `models\cfd-yolov12x.pt`. The tool never downloads,
   retrains, or substitutes a model: if the file is missing it stops and tells
   you where to put it. Weight files are git-ignored.

## Use it

Drag any folder of images onto **Detect Folder.bat**, or run:

```
fishcount detect "C:\path\to\your images"
```

One inference pass per image (imgsz 1536, recording floor 0.10, IoU 0.7,
max_det 3000). Output goes to `output\<name>\`:

```
output\120GOPRO\
  detections.csv   one row per box
  frames.csv       one row per frame, including frames with no boxes
  annotated\       annotated copies of frames at or above the threshold
  results.json     the raw record: every detection + per-frame statistics
```

`detections.csv` — every box the model produced above the recording floor:

| Column | Meaning |
| --- | --- |
| `frame` | path relative to the input folder |
| `x1 y1 x2 y2` | box in pixels on the original image |
| `confidence` | model confidence |
| `area_frac` | box area as a fraction of the frame area |

`frames.csv` — one row per frame, a complete census of the input folder:

| Column | Meaning |
| --- | --- |
| `frame` | path relative to the input folder |
| `max_conf` | highest confidence on the frame (`0.000` if no boxes) |
| `n_boxes_above_threshold` | boxes at or above the reporting threshold |
| `blur` | variance of the Laplacian at quarter resolution |
| `brightness` | mean grayscale value, 0-255 |
| `error` | set if the frame could not be read; the metrics are then blank |

## Two numbers, and no other rules

**`conf` (default 0.10) is the recording floor.** Every box above it is written
to `results.json` and `detections.csv` and is never dropped.

**`threshold` (default 0.25) is the reporting threshold.** It is applied in
exactly one place — `n_boxes_above_threshold`, which also decides which frames
get an annotated image. Because `detections.csv` keeps every box with its
confidence, any threshold at or above the floor can be evaluated later without
re-running the detector.

Nothing else filters, demotes, or caps a detection. There is no size rule, no
blur rule, no cross-frame recurrence rule, and no tier system.

`blur` and `brightness` are recorded but acted on by nothing. They exist so
that evaluation can break detector performance down by condition.

### Why the heuristics were removed

Earlier versions (tag `v0.3-heuristics`) stacked four geometric rules to demote
false positives: a static-recurrence rule, an oversized-box rule, a per-folder
blur cap, and a dense-school exemption from that cap. They were removed rather
than tuned, for one reason: **none of them had ever been measured against
labelled frames.** They also interacted badly — the static rule fired on
essentially every large box, so a real large fish (GOPR7891, confidence 0.80)
was silently demoted to `not_confident`.

The replacement policy is deliberately strict: no filtering rule goes back in
unless it is measured on labelled frames and shows a net gain, stated as fish
lost against false positives removed.

The full heuristic implementation is preserved at tag `v0.3-heuristics`, and
the last counting-capable version (base/dense/thorough stages, SAHI tiled
inference) at tag `v0.2-three-stage` / branch `three-stage-pipeline`.

## Known limitations

- **The reporting threshold is not yet validated.** 0.25 is a placeholder
  inherited from the previous version's `DETECT_CONF`, not a measured
  operating point. It will be chosen from labelled data.
- **Per-condition recall is unknown.** The detector's behaviour in murky,
  dark, or blurry frames has not been measured, which is what `blur` and
  `brightness` in `frames.csv` are recorded for.
- Earlier spot-checking found no fish in 80 randomly sampled no-detection
  frames across two GOPRO deployments (rule-of-three 95% upper bound: under
  ~4% of no-detection frames hold a findable fish). That was a check of one
  operating point on one site, not a recall measurement.

## Flags

| Flag | Default | Meaning |
| --- | --- | --- |
| `--name NAME` / `--out DIR` | input folder's name | where results go under `output\` |
| `--conf FLOAT` | `0.10` | recording floor; every box above it is written out |
| `--threshold FLOAT` | `0.25` | reporting threshold |
| `--imgsz INT` | `1536` | inference size; 1024 is faster but misses dense schools of small fish |
| `--iou FLOAT` | `0.7` | NMS IoU; higher keeps tightly packed fish |
| `--max-det INT` | `3000` | max detections per image |
| `--batch-size INT` | `8` | images per model pass; lower if RAM is tight |
| `--no-images` | off | CSVs and results.json only |
| `--open` | off | open the output folder when finished |

Defaults can also be set in [config.yaml](config.yaml). Precedence:
CLI flags > config.yaml > built-in defaults.

## Speed

One pass at imgsz 1536 costs roughly 6 s per 4000x3000 frame on a laptop CPU
(batched, model loaded once, each image read from disk exactly once; blur and
brightness are computed from the already-decoded image). `--imgsz 1024` is
about 2.5x faster but misses dense schools of small fish entirely; use it only
when that is acceptable.

## Development

```
pip install -e ".[dev]"
pytest            # tests mock the detector; no model weights needed
ruff check .
ruff format .
mypy fishcount
```

Layout: `detector.py` (YOLO wrapper behind a small Detector protocol),
`batch.py` (folder pipeline, per-frame statistics, results.json),
`report.py` (results.json to the two CSVs), `draw.py` (boxes),
`config.py` (pydantic + config.yaml), `cli.py`.

## License

Ultralytics and the Community Fish Detector weights are AGPL-3.0, so this tool
is AGPL-3.0 as well. Local use is unaffected.
