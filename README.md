# HerringNet

Fish detection and species classification for underwater camera trap data.

HerringNet is a two-stage detection pipeline built for Jordaan Labs at UMass Amherst. It uses the Community Fish Detector (CFD) for locating fish in camera trap imagery, then classifies each detection by species and life stage using a fine-tuned YOLOv8 classifier.

Primary target: juvenile river herring (Alosa pseudoharengus, Alosa aestivalis) emigration monitoring.

## Features

- Two-stage pipeline: robust fish detection + species/life-stage classification
- Video processing with frame-residence-rate (FRR) corrected counting (Marjadi et al. 2024)
- Active learning: flags uncertain detections for human review
- Site-specific configuration for deployment across monitoring locations
- CLI and Gradio web demo interfaces
- CPU-optimized inference

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/herringnet.git
cd herringnet

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install in development mode
pip install -e ".[dev]"
```

## Quick Start

### Fastest: detect a whole folder

Drop a folder of images in, get annotated images plus a counts CSV out.
On Windows, drag the image folder onto **`Detect Folder.bat`**. Or:

```bash
herringnet batch path/to/image_folder
# -> outputs/<folder name>/annotated/*.jpg  and  counts.csv
```

This runs one fast batched pass per image (no SAHI tiling), which is the
right tool for processing many images. Use `herringnet detect ... --sahi`
only when you need maximum small-fish recall on a few images.

### Other commands

```bash
# Run detection on an image
herringnet detect path/to/image.jpg --save-images

# Run detection on a video with counting
herringnet detect path/to/video.mp4 --save-csv --save-images

# Extract frames from video
herringnet extract path/to/video.mp4 --output-dir frames/

# Launch the web demo
herringnet demo --share

# Download training datasets
herringnet setup-data --dataset all

# Train a species classifier
herringnet train classify --data data/processed/data.yaml --epochs 50
```

## Architecture

```
Input (image/video)
  -> FrameExtractor (video only, FRR-corrected intervals)
  -> FishDetector (CFD: locate all fish, output bounding boxes)
  -> Crop detected regions
  -> SpeciesClassifier (species + life-stage per crop)
  -> ActiveLearningManager (flag uncertain detections)
  -> FishCounter (FRR-corrected passage counts)
  -> Output: JSON results, annotated images, CSV summaries
```

## Configuration

Default settings are in `configs/default.yaml`. Site-specific overrides go in `configs/sites/`. See `configs/sites/monument_river.yaml` for an example.

## License

AGPL-3.0. This project uses the Community Fish Detector (CFD) which is AGPL-licensed.

## Citation

If you use HerringNet in your research, please cite:

```
Marjadi, M.N., et al. (2024). Automated video monitoring estimates high abundances
of juvenile anadromous fish emigrating from a coastal river. Limnology and
Oceanography: Methods, 22, 295-310.
```

## Acknowledgments

- Jordaan Labs, UMass Amherst
- Community Fish Detector (WildHackers / Dan Morris)
- Ultralytics YOLOv8
