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
| `--imgsz INT` | `1024` | Inference size; lower is faster, 1024 matches training |
| `--batch-size INT` | `8` | Images per model pass; lower it if RAM is tight |
| `--no-images` | off | Counts only, skip writing annotated images |
| `--thorough` | off | SAHI tiled inference for very small fish (much slower) |
| `--open` | off | Open the output folder when finished |

Defaults can also be set in [config.yaml](config.yaml). Precedence:
CLI flags > config.yaml > built-in defaults.

## Speed on CPU

YOLOv12x at image size 1024 is the accurate-but-heavy setting. On a laptop CPU
expect a few seconds per image. If you need it faster:

- `--imgsz 640` is roughly 2-3x faster, with some loss of small-fish recall.
- `--no-images` skips encoding and writing annotated images.
- `--batch-size` mostly trades RAM, not speed, on CPU; lower it if memory is tight.

The model is loaded once per run and images are processed in batches, with each
image read from disk exactly once.

## Thorough mode (very small fish)

```
pip install -e ".[thorough]"
fishcount detect "C:\path\to\images" --thorough
```

Slices each image into overlapping 1024 px tiles (SAHI), detects per tile, and
merges. Catches small distant fish that full-frame inference misses, at many
times the runtime. Off by default on purpose.

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
