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

   That installs the CPU build of torch. For an NVIDIA GPU, install a CUDA
   build instead (pick the index matching your driver):

   ```
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
   ```

   Nothing else changes: the GPU is picked up automatically when it works.
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
  results.jsonl    the raw record: one frame per line, written as the run goes
  run.json         the detection settings this run used
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

Frames are identified by their path relative to the input folder, never by
filename alone. GoPro recycles filenames between cards and deployments, so the
same basename can name completely different photographs in different folders.

## Two numbers, and no other rules

**`conf` (default 0.10) is the recording floor.** Every box above it is written
to `results.jsonl` and `detections.csv` and is never dropped.

**`threshold` (default 0.25) is the reporting threshold.** It is applied in
exactly one place — `n_boxes_above_threshold`, which also decides which frames
get an annotated image.

Detection never sees the threshold, which is what lets you change your mind:

```
fishcount report output\120GOPRO --threshold 0.4 --folder "C:\path\to\your images"
```

rewrites both CSVs and redraws the annotated images from the existing run. No
inference, so it takes seconds rather than hours.

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

## Long runs

A pass over thousands of 4000x3000 frames takes hours, so a run is built to be
interrupted.

**It resumes by itself.** Each frame is appended to `results.jsonl` and flushed
to disk as its batch finishes. Re-running the same command skips every frame
already in there and finishes the rest. Killing the process — Ctrl-C, a closed
laptop, a power cut — costs at most the batch in flight. A half-written final
line is detected and that one frame is simply redone.

**It refuses to mix runs.** `run.json` records the settings the journal was
built under. Resuming with a different model, `conf`, `iou`, `imgsz` or
`max_det` stops with an explanation instead of blending incompatible results;
`--restart` detects the folder again from scratch. The reporting threshold is
deliberately not on that list, so changing it never costs a re-run.

**Memory is flat.** Nothing accumulates: frames are streamed a batch at a time
into the journal, and read back one at a time when the CSVs are written. Peak
memory measured at 1455 MiB over 6 frames and 1444 MiB over 50 — eight times
the work, no growth.

**Throughput and ETA are logged** every 100 frames as well as on the progress
bar, so a run redirected to a log file still says where it is.

## Speed

Measured on this project's frames (4000x3000, imgsz 1536), end to end,
including JPEG decode and the per-frame statistics:

| Device | s/frame | img/s | 15,000 frames |
| --- | --- | --- | --- |
| RTX 4070 Laptop (8 GB) | 0.43 | 2.3 | ~1h 50m |
| Laptop CPU | 5.4 | 0.19 | ~23h |

The GPU is used automatically when torch can see one. CPU and GPU results are
equivalent but not bit-identical: boxes land within a pixel and confidences
within 0.001, which is ordinary floating-point difference between CUDA and CPU
kernels, not a bug.

**Batch size is not a speed knob on a GPU.** This model at imgsz 1536 needs
about 2.4 GB of VRAM for a single image, and one image already saturates an
RTX 4070. Measured on an 8 GB card:

| Batch | s/frame | Peak VRAM | Share of an 8 GB card |
| --- | --- | --- | --- |
| 1 | 0.31 | 2.4 GB | 29% |
| 2 | 0.34 | 4.5 GB | 56% |
| 3 | 0.35 | 6.4 GB | 81% |
| 4 | 1.05 | 8.4 GB | 105% — spills to system RAM |

Three is the largest batch that fits; one is the fastest. Past the card's
capacity Windows lets CUDA spill into system RAM over PCIe and throughput
collapses, so the default is 1 on a GPU and 8 on the CPU. A batch that will not
fit is split in half and retried rather than killing the run.

`--imgsz 1024` is about 2.5x faster again but misses dense schools of small
fish entirely; use it only when that is acceptable.

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

`fishcount detect FOLDER`:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--name NAME` / `--out DIR` | input folder's name | where results go under `output\` |
| `--conf FLOAT` | `0.10` | recording floor; every box above it is written out |
| `--threshold FLOAT` | `0.25` | reporting threshold |
| `--imgsz INT` | `1536` | inference size; 1024 is faster but misses dense schools of small fish |
| `--iou FLOAT` | `0.7` | NMS IoU; higher keeps tightly packed fish |
| `--max-det INT` | `3000` | max detections per image |
| `--device` | `auto` | `auto`, `cpu`, or `cuda` / `cuda:N` |
| `--batch-size INT` | 1 on GPU, 8 on CPU | images per model pass |
| `--restart` | off | discard previous results for this output folder instead of resuming |
| `--no-images` | off | CSVs only |
| `--open` | off | open the output folder when finished |

`fishcount report OUT_DIR` re-derives the CSVs from a finished run:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--threshold FLOAT` | `0.25` | the threshold to report at |
| `--folder DIR` | none | the original image folder, needed only to redraw annotated images |
| `--no-images` | off | CSVs only |

Defaults can also be set in [config.yaml](config.yaml). Precedence:
CLI flags > config.yaml > built-in defaults.

## Development

```
pip install -e ".[dev]"
pytest            # tests mock the detector; no model weights needed
ruff check .
ruff format .
mypy fishcount
```

Layout: `detector.py` (YOLO wrapper behind a small Detector protocol, device
choice, out-of-memory backoff), `batch.py` (folder pipeline, per-frame
statistics), `journal.py` (the resumable on-disk record), `report.py` (journal
to CSVs and annotated images), `draw.py` (boxes), `config.py` (pydantic +
config.yaml), `cli.py`.

## License

Ultralytics and the Community Fish Detector weights are AGPL-3.0, so this tool
is AGPL-3.0 as well. Local use is unaffected.
