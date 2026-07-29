# fishcount

Count fish in a folder of images, fully offline on a Windows laptop CPU. Put a
folder in, run one command (or drag the folder onto a .bat file), get annotated
images and a fish-count file out. No database, no server, no cloud.

Detection uses the Community Fish Detector (`cfd-yolov12x.pt`): a YOLOv12x
object detector with a single class (`fish`), trained at image size 1024.
Every detection is "fish"; there is no species classification. Input images are
never modified; all output goes to a separate folder.

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

### Drag and drop (Windows)

Drag any folder of images onto **Detect Folder.bat**. A console window shows
progress and the summary, then the results folder opens. Results land in
`output\<your folder name>\` inside this project folder.

### Command line

```
fishcount detect "C:\path\to\your images"
```

Subfolders are included. Outputs go to `output\<folder name>\` under your
current directory (change with `--out`):

```
output\dive1\
  annotated\     every image with green boxes and a "Fish: N" overlay
  counts.csv     one row per image (filename, fish_count) + a TOTAL row
  results.json   per image: boxes [x1, y1, x2, y2] in pixels + confidence
```

### Flags

| Flag | Default | Meaning |
| --- | --- | --- |
| `--out DIR` | `output/<folder name>` | Where to write results |
| `--conf FLOAT` | `0.25` | Confidence threshold (raise to cut false positives) |
| `--imgsz INT` | `1024` | Inference size; higher finds more small fish, much slower |
| `--iou FLOAT` | `0.7` | NMS IoU; higher keeps tightly packed fish |
| `--max-det INT` | `1000` | Max detections per image (Ultralytics silently caps at 300 otherwise) |
| `--batch-size INT` | `8` | Images per model pass; lower it if RAM is tight |
| `--dense` | off | Recall-first preset for dense schools of small fish (see below) |
| `--slice-size INT` | `640` | SAHI tile size (`--thorough` only); smaller finds smaller fish |
| `--overlap FLOAT` | `0.2` | SAHI tile overlap (`--thorough` only) |
| `--no-images` | off | Counts only, skip writing annotated images |
| `--thorough` | off | SAHI tiled inference for very small fish (much slower) |
| `--open` | off | Open the output folder when finished |

Defaults can also be set in [config.yaml](config.yaml). Precedence:
CLI flags > `--dense` preset > config.yaml > built-in defaults.

## Dense schools of small fish (important)

The default settings are tuned for sparse-to-moderate scenes with clearly
separated fish. On a **dense school of small fish**, full-frame detection at
1024 px misses most of them and can report **zero** on a frame that visibly
holds hundreds, because each fish shrinks to a few pixels after downscaling.

Use `--dense` for these frames. It lowers the confidence threshold, raises the
inference size, and removes the 300-detection cap. Add `--thorough` to also
switch to SAHI tiling, which is the strongest recall setting:

```
fishcount detect "C:\path\to\images" --dense            # much better, still CPU-friendly
fishcount detect "C:\path\to\images" --dense --thorough # maximum recall, slow
```

Measured on one laptop CPU, on two 4000x3000 frames that the default run scored
**0**, plus a moderate frame the default scored 11:

| Setting | dense frame A | dense frame B | moderate frame | speed |
| --- | --- | --- | --- | --- |
| default (1024) | 0 | 0 | 11 | ~2 s/img |
| `--dense` (1536) | 16 | 17 | 33 | ~6 s/img |
| `--dense --thorough` (tile 640) | ~67 | ~94 | ~87 | ~30-45 s/img |

### The honest ceiling

None of these settings counts a dense school *accurately*. Bounding-box
detection systematically **undercounts** crowds, and the error grows with
density: overlapping fish get merged by NMS, few-pixel bodies have no clear
edges, and a packed 3-D school is under-determined from one 2-D frame. Even the
strongest setting recovers on the order of dozens-to-low-hundreds from a frame
that may hold a thousand fry. So:

- For **sparse, separated fish**, treat the count as a real count.
- For **dense schools**, treat the number as a **relative-abundance index / lower
  bound**, good for comparing frames, sites, or time points under similar
  conditions, not for stating an exact population. Validate against hand counts
  on a few frames before trusting absolute numbers, and note the density band
  where the tool stays reliable.
- A truly accurate dense count needs a different method (density-map estimation,
  or acoustics such as imaging sonar), which is out of scope for this offline
  box detector.

## Pipeline: base pass, then dense only where there are fish

`fishcount pipeline` runs the whole workflow in one command. Because `--dense`
(low confidence) produces false positives on empty/murky water, the pipeline
first runs a **presence gate** over every frame, then re-runs the high-recall
settings only on frames that actually held fish, so they never touch empty frames.

```
fishcount pipeline "C:\path\to\104GOPRO" --name 104GOPROFINAL --min-count 1
```

Three stages, written to separate folders under `output\<name>\`:

1. `base\`     presence gate over every image: is there a fish here?
2. `dense\`    `--dense`, only frames with at least `--min-count` fish.
3. `thorough\` `--dense --thorough` (SAHI tiling), same subset.

**The base pass is a gate, not a count.** It answers "does this frame contain a
fish, big or small?" so the frame can go on to be counted. Two things make it
reliable where a plain pass fails:

- It runs at `--base-imgsz` (default 1536) so dense schools of tiny fish
  register at all (at 1024 they read as 0 and get dropped before the dense passes).
- It discards any detection larger than `--base-max-box-frac` of the frame
  (default 0.10). Empty/murky water gets misread as one giant fish-shaped box;
  real fish are only a few percent of the frame (measured: <5%, while these
  false positives are >=10%, with a clean gap between), so this removes them
  without losing real detections. The same size filter is applied in the
  `--dense` passes.

Use `--min-count 1` to send every frame with any fish to the counting passes
(maximum recall of fish-bearing frames), or a higher value to be stricter.

Two false-positive types to know about: the size filter removes *oversized*
boxes (empty water read as one big fish). It cannot remove *small* boxes on
water-surface ripple/caustic texture, which look like a distant fish and are
genuinely ambiguous. Those show up as single-detection frames (`base_count == 1`
in `summary.csv`), so with `--min-count 1` you can eyeball just those to reject
ripple hits by hand.

Plus `summary.csv` at the top, comparing base vs dense vs thorough counts per
frame. Flags: `--name` / `--out`, `--min-count N`, `--base-imgsz`,
`--base-max-box-frac`, `--no-images`, `--open`.

The dense/thorough counts here are recall-first; read them as a relative
abundance index, not exact totals (see the ceiling note above).

## Speed on CPU

YOLOv12x at image size 1024 is the accurate-but-heavy setting. On a laptop CPU
expect a few seconds per image. Levers:

- `--imgsz 640` is roughly 2-3x faster, with loss of small-fish recall.
  `--imgsz 1536/2048` finds more small fish, at ~2.5x / ~4x the cost.
- `--no-images` skips encoding and writing annotated images.
- `--batch-size` mostly trades RAM, not speed, on CPU; lower it if memory is tight.

The model is loaded once per run and images are processed in batches, with each
image read from disk exactly once.

## Thorough mode (very small fish)

```
pip install -e ".[thorough]"
fishcount detect "C:\path\to\images" --thorough
```

Slices each image into overlapping tiles (SAHI, default 640 px via
`--slice-size`), detects per tile at native resolution, and merges. Catches
small distant fish that full-frame inference misses, at many times the runtime.
Smaller tiles find smaller fish. Off by default on purpose; combine with
`--dense` for the strongest recall on schools.

## Phase 2 design: sequences and video (not built yet)

For image sequences or video frames, summing per-image counts double-counts a
fish that stays in view across frames. The code is already split so this slots
in cleanly:

- `fishcount/detector.py` produces per-image detections and will not change.
- `fishcount/batch.py` already yields results in sorted filename order, which
  doubles as frame order.
- `fishcount/count.py` is the only place counts are aggregated. Phase 2 adds a
  sequence-aware aggregator there: either frame-residence-rate correction
  (scale raw detections by how long the average fish stays in view) or a
  lightweight IoU/centroid tracker that counts track births. A future
  `--sequence` flag will select it.

## Development

```
pip install -e ".[dev]"
pytest            # tests mock the detector; no model weights needed
ruff check .
ruff format .
mypy fishcount
```

Layout: `detector.py` (YOLO wrapper behind a small Detector protocol),
`batch.py` (folder pipeline), `draw.py` (boxes + overlay), `count.py`
(aggregation, Phase 2 seam), `config.py` (pydantic + config.yaml),
`cli.py` (argparse + rich summary).

## Troubleshooting

- "Fish detector weights not found": put `cfd-yolov12x.pt` in `models\`.
- "'fishcount' is not recognized": activate the venv
  (`.venv\Scripts\activate`) or use the .bat file, which finds the venv itself.
- Corrupt or unreadable images are skipped with a warning; they appear in
  `results.json` with an `"error"` field and are excluded from `counts.csv`.

## License

Ultralytics and the Community Fish Detector weights are AGPL-3.0, so this tool
is AGPL-3.0 as well. Local use is unaffected.
