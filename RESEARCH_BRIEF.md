# Research Brief: Eye Segmentation for Eye Tracking

## Objective

Identify all useful prior art — papers, repositories, datasets, and product documentation — relevant to using semantic segmentation of the eye region as the front-end of a gaze estimation pipeline. Prioritize work that operates under constraints similar to ours: near-eye (HMD-mounted) cameras, small subject pools, real-time inference requirements, and per-subject variation.

---

## Relevance Tiers

Based on cross-referencing with the active SegUNET project (EfficientNet-B0 UNet, 7-class eye parsing, ~1500 frames, 22+ subjects, model-assisted labeling loop, targeting 90fps on Bigscreen Beyond):

### DIRECTLY RELATED (answers problems we face right now)

These map 1:1 to current SegUNET pain points:

| Topic | Why it's urgent | Current state in SegUNET |
|-------|----------------|--------------------------|
| UpperLid segmentation failure modes | Weakest class (IoU 0.665), worst fold drops to 0.61 | Loss weighting + more labels attempted, still weak |
| Cross-subject generalization | Fold 5 mIoU=0.761 vs fold 1 mIoU=0.859 — massive subject gap | GroupKFold splits, but no explicit domain adaptation |
| Lightweight transformer architectures | MobileViT, EfficientFormer, TopFormer, SegFormer-B0 may beat EfficientNet-B0 at same latency | Not explored — still on CNN encoder |
| Synthetic eye generation for augmentation | Need to simulate subject variation (lash density, lid droop, iris texture) that geometric aug can't produce | Only using contrast/geometric jitter currently |
| Model-assisted label quality / confirmation bias | Approve rates 0-42%, risk of feedback loop corruption | Review tool exists, no correction-rate tracking yet |
| Small-class boundary precision (Pupil, BrightSpot) | BrightSpot <1% of pixels, Pupil ~2% | Dice+Focal helps but boundary quality unverified |
| Inference optimization for 90fps/11ms | End goal for Bigscreen Beyond deployment | No optimization work started |

### NEEDED IN NEAR FUTURE (next 2-3 months)

These become critical as the project matures past Phase 1 labeling:

| Topic | When it becomes relevant |
|-------|------------------------|
| Seg-to-gaze pipeline (pupil ellipse → gaze vector) | After seg model stabilizes at mIoU >0.80 |
| Test-time adaptation / per-subject fine-tuning | When deploying to new users outside training pool |
| Confidence-calibrated pseudo-labels (see note below) | Phase 2-3 labeling (model-assisted with quality gates) |
| Knowledge distillation (large→small model) | When transitioning from training model to deployment model |
| Boundary IoU metrics | After base mIoU exceeds 0.65 per class (close now) |
| Label cleaning (cleanlab, loss-based prioritization) | After reaching 603+ labels |

**Key pseudo-label techniques to investigate:**
- **ACT** (Adaptive Confidence Thresholding) — part of the ENCORE method, arXiv:2505.07691 ("Feedback-Driven Pseudo-Label Reliability Assessment...", May 2025)
- **CSL** (Confidence Separable Learning) — arXiv:2509.16704, "When Confidence Fails: Revisiting Pseudo-Label Selection in Semi-supervised Semantic Segmentation"

### PERIPHERAL (good to know, not blocking)

Useful context but not actionable in the current project phase:

| Topic | Why it's peripheral |
|-------|-------------------|
| Remote/webcam eye tracking | Different domain — we're near-eye HMD only |
| Full-face landmark models (MediaPipe, etc.) | Different input modality, not periocular crops |
| Multi-task gaze+seg joint training | Requires gaze labels we don't have |
| Eye tracking for accessibility/marketing | Application domain, not technical prior art |

---

## Key Research Questions

### 1. Segmentation Architectures for Near-Eye Parsing [DIRECTLY RELATED]

- What lightweight encoder-decoder architectures achieve competitive mIoU at sub-10ms inference on edge hardware for eye region segmentation?
- Specifically: what beats EfficientNet-B0 UNet at the same latency, or matches it at lower latency? Include lightweight transformer variants (MobileViT, EfficientFormer, TopFormer, distilled SegFormer-B0) — this is one of the most active areas of lightweight-seg research.
- Are there architectures that handle extreme class imbalance (BrightSpot <1% pixels) structurally rather than through loss tricks?

Search terms: eye segmentation, ocular segmentation, periocular parsing, pupil segmentation network, iris segmentation deep learning, near-eye semantic segmentation, lightweight eye parsing, MobileNetV3 segmentation

### 2. Datasets and Benchmarks [DIRECTLY RELATED]

- What public datasets provide dense 5-7 class pixel labels from HMD/near-eye IR cameras?
- Which include per-subject splits and multiple gaze directions per subject?
- Can any be used for pretraining or evaluation alongside our proprietary data?

Key datasets to investigate: OpenEDS (Facebook), NVGaze (NVIDIA), RITEyes (synthetic), TEyeD, LPW, S-General

Search terms: eye segmentation dataset, OpenEDS, NVGaze, RITEyes, LPW dataset, near-eye benchmark, periocular dataset HMD, IR eye dataset semantic labels

### 3. Cross-Subject Generalization [DIRECTLY RELATED]

- What domain adaptation or personalization techniques work for subject-dependent eye appearance (lid shape, iris color/pattern, lash density)?
- How do top methods handle the "new user" problem — deploying to a subject with zero labels?
- What augmentation strategies specifically target inter-subject variation rather than noise/lighting?

Search terms: cross-subject eye segmentation, domain adaptation near-eye, personalization eye model, few-shot periocular adaptation, subject-invariant features eye

### 4. Active Learning and Label Efficiency [DIRECTLY RELATED]

- What model-assisted labeling pipelines work best for iterative segmentation with <1000 labels?
- How to detect and prevent pseudo-label confirmation bias in an approve/reject workflow?
- What active learning sample selection strategies maximize mIoU gain per annotation hour in multi-class seg?

Search terms: active learning segmentation, model-assisted labeling eyes, semi-supervised eye segmentation, few-shot periocular, confirmation bias pseudo-labels, sample selection semantic segmentation

### 5. Segmentation-to-Gaze Pipeline [NEAR FUTURE]

- How do segmentation outputs (pupil ellipse fits, iris boundary, eyelid contours) feed into model-based gaze estimation?
- What is the quantified relationship between segmentation error (mIoU degradation) and gaze angular error?
- What are the best pupil/iris ellipse fitting algorithms that consume segmentation masks?

Search terms: segmentation-based gaze estimation, pupil ellipse fitting gaze, iris boundary gaze direction, eye model geometric gaze, segmentation error gaze accuracy, model-based eye tracking

### 6. Real-Time Inference and Deployment [NEAR FUTURE]

- What optimization techniques (quantization, pruning, TensorRT, ONNX) have been applied to eye segmentation at 90+ fps?
- What is the actual measured latency of EfficientNet-B0 UNet at 256x256 on typical HMD compute (Qualcomm XR2, similar ARM/mobile GPU)?
- Are there FPGA or NPU deployments for HMD eye tracking with published latency numbers?

Search terms: real-time eye segmentation, eye tracking inference optimization, quantized segmentation model edge, ONNX eye tracking, TensorRT pupil segmentation, XR2 neural processing eye

### 7. Robustness and Augmentation [DIRECTLY RELATED]

- What augmentation strategies improve generalization in near-eye IR imagery specifically?
- How to simulate subject variation synthetically (eyelash density, lid droop, iris texture)?
- What is the effect of IR illumination variation on segmentation stability?

- What GAN or diffusion-based synthetic eye generators exist (e.g. RITEyes, UnityEyes, SynthesEyes)? Which produce labeled output usable for segmentation pretraining?

Search terms: eye segmentation augmentation, IR image augmentation, synthetic eye generation, domain randomization near-eye, illumination invariant segmentation, GAN eye synthesis, UnityEyes, RITEyes synthetic, diffusion model eye generation

---

## Source Priorities

1. **arXiv / peer-reviewed papers** — CVPR, ECCV, ICCV, ETRA, ISMAR, IEEE TPAMI, Medical Image Analysis
2. **Git repositories** — reference implementations, training pipelines, pretrained weights, annotation tools
3. **Datasets** — public benchmarks with download links, license info, and class definitions
4. **Product/SDK documentation** — Tobii (eye tracking SDK), Pupil Labs (open-source ET), Meta Quest Pro, Apple Vision Pro, HTC Vive Pro Eye
5. **Theses and technical reports** — especially those with reproducible results or released code

## Output Format

For each source found, provide:

- **Citation** (authors, title, year, venue or URL)
- **Relevance tier** (Directly Related / Near Future / Peripheral)
- **Key findings** (2-3 bullet points of actionable takeaways)
- **Links** (paper URL, code repo, dataset download)
- **Applicability to SegUNET** (1 sentence: what would we change or build based on this?)

Group results by research question. Flag any survey papers that cover multiple questions.

---

## Context: What SegUNET Already Has

So the research agent doesn't re-discover what we already know:

- **Architecture:** EfficientNet-B0 encoder + UNet decoder (segmentation_models_pytorch), 7 classes, 256x256 input
- **Loss:** 0.5 Dice + 0.5 Focal (gamma=2.0), optional per-class CE weights
- **Training:** PyTorch Lightning, AdamW, cosine LR, 25-50 epoch cap, subject-level GroupKFold splits
- **Data:** ~1500 frames, 22 subjects, IR grayscale, 2-channel input (grayscale + eye-side flag)
- **Labeling:** LabelMe polygons → PNG masks, model-assisted review loop (mask_review_server.py)
- **Current mIoU:** 0.813 mean across 5 folds (weakest: UpperLid 0.665, Sclera 0.756)
- **Known references already in DESIGN_REFERENCE.md:** See Appendix C of that document (loss surveys, pseudo-label papers, cleanlab, OpenEDS, data leakage studies)

## Constraints

- **Small-sample regime** — handful of subjects, not internet-scale
- **Near-eye HMD cameras** — high magnification, IR illumination, partial face only
- **Per-subject variation** — eye shape, lid geometry, lash density are first-class concerns
- **Inference budget: under 11ms** at 90fps on consumer hardware (Bigscreen Beyond)
- **7 classes:** background, lower lid, upper lid, pupil, bright spot (specular), iris, sclera
