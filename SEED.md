# eye-seg-et

A research repo exploring eye segmentation as the front-end of an eye-tracking
pipeline. The goal is to understand the full path from raw eye images to gaze
estimates, with a particular focus on how segmentation quality affects downstream
tracking performance.

## Scope

Active areas of interest, roughly in order of depth:

- **Segmentation models** — architectures suited to eye region parsing
  (pupil, iris, sclera, eyelids). Lightweight encoders (e.g. EfficientNet-style
  backbones), UNet variants, and recent transformer-based approaches.
- **Annotation & active learning** — model-assisted labeling loops, sample
  selection strategies, and annotation tooling for eye imagery. Practical
  pipelines that reduce labeling cost on small subject pools.
- **Evaluation** — segmentation metrics (mIoU per class, boundary metrics),
  robustness to per-subject variation, and how segmentation error propagates
  into gaze error.
- **Seg → gaze** — how segmentation outputs (pupil ellipses, iris boundaries,
  eyelid occlusion) feed model-based or learned gaze estimators.
- **Robustness** — augmentation strategies (contrast, geometric jitter),
  cross-subject generalization, and handling of partial occlusion.

## Constraints

- Small-sample regime: a handful of subjects, not internet-scale datasets.
- Near-eye camera imagery (HMD-style), not remote webcam.
- Per-subject variation is a first-class concern.

## Status

Early-stage. This repo is currently a research scaffold; prior art is being
collected before implementation work begins.