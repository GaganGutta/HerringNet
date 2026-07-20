# Model weights go here

Place the Community Fish Detector weights in this folder:

    models/cfd-yolov12x.pt

That model is a YOLOv12x object detector with a single class (`fish`), trained at
image size 1024, in standard Ultralytics format, licensed AGPL-3.0.

Weight files (`*.pt`) are git-ignored because they are large. The tool never
downloads, retrains, or substitutes another model: if this file is missing,
`fishcount` stops with an error telling you to put it here.
