# Learning Guide: Prior Art for SegUNET

A guided tour of the research corpus, synthesized from `docs/research/index/`.
Designed for an engineer building a 7-class near-eye IR segmentation model
targeting 90fps on Bigscreen Beyond.

---

## 1. Field Map

Eye segmentation for VR sits at the intersection of three subfields:

```
                    SEMANTIC SEGMENTATION
                   (architectures, losses)
                          |
              +-----------+-----------+
              |                       |
    MOBILE/EDGE DEPLOYMENT    SEMI-SUPERVISED LEARNING
    (quantization, distill)   (pseudo-labels, SSL)
              |                       |
              +-----------+-----------+
                          |
                  NEAR-EYE IR PARSING
                 (eye anatomy priors,
                  off-axis geometry,
                  VR-specific degradation)
```

**Upstream inputs:** IR camera frames from off-axis sensors inside an HMD.
**Downstream consumer:** Ellipse fitting on pupil/iris masks → 3D gaze vector.
**Constraint envelope:** <11ms end-to-end, <500K params, INT8-friendly.

The corpus covers 7 clusters (see `index/master.yaml`):
- **Critical:** Architectures, Training Efficiency, Deployment
- **High:** Datasets, Robustness
- **Medium:** Gaze Pipeline, Label Quality Tools

---

## 2. Top 10 Reading Order

Ordered by relevance to SegUNET's current pain points. Each entry cites the
index file where its full metadata lives.

| # | Paper/Repo | Why read it | Pain point |
|---|-----------|-------------|------------|
| 1 | **Cross Pseudo Supervision** (CVPR 2021) | Simplest credible SSL baseline for your 1500-label situation. Two identical nets, different seeds, mutual pseudo-supervision. Validated at +13 mIoU on a 22-label IR task. | Label scarcity |
| 2 | **DistillGaze** (Meta, 2026) | The production deployment blueprint: adapt VFM on mixed data → distill to 256K-param student. Proves the whole pipeline works for eye tracking at scale. | Deployment latency |
| 3 | **TopFormer** (CVPR 2022) | 1.4ms on iPhone 12. Token pyramid architecture proves sub-11ms is achievable. Study the scale-aware injection mechanism. | Latency budget |
| 4 | **RITnet** (ICCVW 2019) | Direct architectural ancestor. 248K params, 95.3% on OpenEDS 4-class at >300Hz. Dense skip connections at 32ch are the baseline to beat for 7-class. | Architecture baseline |
| 5 | **SAM Zero-Shot Eye** (ETRA 2024) | 93.34% pupil IoU, 86.63% iris — zero-shot with bbox prompts. Use as auto-annotator to halve annotation work. | Annotation bottleneck |
| 6 | **EllSeg** (IEEE TVCG 2021) | Ellipse-aware segmentation + soft-argmax center loss. Directly applicable auxiliary loss for Pupil/Iris geometric consistency. | Pupil/Iris boundaries |
| 7 | **CSL** (ICCV 2025) | Replaces fixed confidence thresholds with convex optimization for pseudo-label selection. No manual tuning — solves BrightSpot's class imbalance problem. | Pseudo-label quality |
| 8 | **SSL Eye Segmentation** (ETRA 2021) | Radial/concentric augmentations designed for eye anatomy. Targets exactly the UpperLid/Iris boundary confusion SegUNET struggles with. | UpperLid weakness |
| 9 | **Boundary IoU** (2021) | Standard mIoU under-penalizes boundary errors. Use Boundary IoU as primary metric for UpperLid/LowerLid/BrightSpot where edges matter most. | Evaluation blindspot |
| 10 | **DNN Eye Tracking Degradation** (ETRA 2021) | Defines exact augmentation conditions: exposure, blur, reflection, occlusion. Your test set must cover these axes. | Robustness gaps |

*See: `papers-training.yaml`, `papers-architectures.yaml`, `papers-datasets-robustness.yaml`*

---

## 3. Concept Primers

### 3.1 Cross Pseudo Supervision (CPS)

When you have 1500 labeled frames and 50K+ unlabeled ones, CPS is the simplest
way to exploit the unlabeled pool. Train two networks with identical architecture
but different random initialization. Each generates pseudo-labels for the other on
unlabeled data. The disagreement between them acts as a natural quality filter —
where both agree, the pseudo-label is likely correct.

Why it works for SegUNET: your EfficientNet-B0 UNet is small enough to train two
copies simultaneously. The mutual supervision provides a free consistency
regularizer that's especially valuable for ambiguous boundaries (UpperLid touching
Sclera). Validated on IR eye data in an IOVS 2023 follow-up showing +13 mIoU with
only 22 labeled frames.

*Reference: `papers-training.yaml` → cross-pseudo-supervision*

### 3.2 Knowledge Distillation for Deployment

The DistillGaze pipeline (Meta Reality Labs, 2026) establishes a two-stage
pattern: (1) adapt a large Visual Foundation Model (ViT-B/DINOv2) via
self-supervised learning on your unlabeled real + synthetic data, then (2) distill
its knowledge into a tiny student (256K params) that runs on-device.

The key insight is that the teacher doesn't need eye-tracking labels — it learns
useful representations from unlabeled frames via SSL. The student then compresses
those representations into a deployable model. EdgeSAM shows the same pattern:
SAM's ViT-H encoder → purely CNN student for edge devices.

For SegUNET Phase 3: train a large teacher (SAM/DINOv2-adapted) on all available
data, then distill to your deployment model. The teacher captures knowledge the
small model couldn't learn directly.

*Reference: `papers-architectures.yaml` → distillgaze, edgesam*

### 3.3 Off-Axis Camera Geometry

Most academic eye-tracking datasets (OpenEDS, LPW) use fronto-parallel cameras —
the camera looks straight at the eye. Modern HMDs like Beyond use off-axis
cameras positioned at the rim of the lens. This creates perspective distortion:
the pupil appears elliptical even when circular, iris boundaries are foreshortened
on one side, and sclera visibility is asymmetric.

The VRGaze dataset (2.1M images, 68 participants, CVPR 2026) is the first
large-scale off-axis near-eye IR dataset matching Beyond's geometry. Classical
methods (Swirski 2012) handle off-axis via explicit geometric correction;
deep methods can learn it implicitly but need representative training data.

This is why VRGaze is the #1 pretraining source — it's the only dataset that
doesn't require domain adaptation to match Beyond's camera geometry.

*Reference: `papers-datasets-robustness.yaml` → gazeshift-vrgaze; `repos.yaml` → gazeshift*

### 3.4 Uncertainty-Aware Segmentation

EyeSeg (IJCAI 2025) adds a per-pixel uncertainty score alongside the
segmentation prediction. Instead of binary "is this mask good enough?" the model
outputs a calibrated confidence map. Pixels near ambiguous boundaries (UpperLid
merging with Iris under motion blur) get high uncertainty; clear interior regions
get low uncertainty.

For the labeling loop: flag frames for human review only where the model is
genuinely uncertain — rather than reviewing all predictions or using a crude
global confidence threshold. This concentrates annotation effort exactly where it
helps most.

*Reference: `papers-architectures.yaml` → eyeseg-uncertainty; `repos.yaml` → EyeSeg*

---

## 4. Reading Questions

Use these while studying each paper to extract maximum value for SegUNET.

**Architecture papers (RITnet, TopFormer, SegFormer, EfficientFormer):**
- What is the parameter count and measured latency on comparable hardware?
- Could this encoder replace EfficientNet-B0 without changing the decoder?
- How do they handle multi-scale features — is it compatible with skip connections?
- What input resolution do they benchmark at, and how does downscaling affect boundary classes?

**SSL/Training papers (CPS, CSL, ACT, SAM zero-shot):**
- What's the minimum labeled fraction tested? Does it match our ~3% labeled ratio?
- How do they handle class imbalance in pseudo-label generation? (BrightSpot is <1% of pixels)
- Is the method compatible with GroupKFold subject-based splits, or does it assume i.i.d.?
- What's the compute overhead vs supervised-only training?

**Robustness papers (degradation, slippage, signal quality):**
- Which failure modes overlap with Beyond's hardware (off-axis IR, Fresnel reflections)?
- Can their degradation model generate synthetic augmentation transforms?
- Do they provide quantitative thresholds for "acceptable" vs "failed" frames?

**Deployment papers (DistillGaze, EdgeSAM, PTQ):**
- What's the accuracy drop at INT8 vs FP32? Is it uniform across classes?
- Does quantization disproportionately affect small classes (BrightSpot, pupil)?
- What calibration dataset size is needed for PTQ?

---

## 5. Open Questions in the Field

These are unresolved problems where the corpus offers partial answers but no
definitive solution:

1. **7-class at sub-250K params** — RITnet proves 4-class at 248K params works.
   No one has published 7-class (adding UpperLid, LowerLid, BrightSpot) at
   comparable size. The boundary classes may require more capacity.

2. **Off-axis domain gap quantification** — VRGaze provides the data, but no
   paper measures exactly how much performance drops when training on
   fronto-parallel and deploying on off-axis. The domain-adaptation-eye-tracking
   repo handles sim→real but not geometry shifts between real datasets.

3. **BrightSpot as a class** — No paper in the corpus treats specular IR
   reflections (BrightSpot) as a segmentation class. It's always either removed
   in preprocessing (CLAHE, Starburst's CR removal) or ignored. SegUNET's
   decision to segment it is novel but unsupported by prior work.

4. **Cross-subject generalization at test time** — GroupKFold trains on some
   subjects, tests on others. The few-shot personalization papers (few-shot-
   scanpath, GazeShift's per-user calibration) hint at solutions, but none
   address segmentation specifically. How many frames per new user are needed?

5. **Temporal consistency without temporal models** — SegUNET processes single
   frames. OpenEDS2020 provides sequences but the corpus lacks a lightweight
   temporal smoothing method that fits within the 11ms budget.

6. **Interaction between quantization and class imbalance** — PTQ papers test on
   balanced datasets (Cityscapes, ADE20K). Whether INT8 quantization
   disproportionately harms rare classes (BrightSpot at <1% pixels) is untested.

---

## 6. What's Missing from the Corpus

Gaps where additional research or experimentation is needed:

| Gap | What we have | What we need |
|-----|-------------|-------------|
| **Lid segmentation** | No paper specifically addresses eyelid parsing as a class. EyeSeg mentions it in passing. | Lid-specific augmentation strategies, boundary loss weights, or auxiliary tasks. |
| **BrightSpot detection** | Starburst removes reflections; RITnet augments with starburst patterns. | A principled approach to segmenting (not removing) specular IR reflections. |
| **TensorRT INT8 on Snapdragon XR2** | PTQ paper covers general segmentation. DistillGaze deploys on "VR headset GPU" (unspecified). | Actual latency benchmarks on Beyond's compute platform. |
| **Multi-dataset training for 7 classes** | EllSeg trains across 5 datasets (OpenEDS, NVGaze, RITEyes, LPW, Fuhl) for pupil/iris. | Protocol for harmonizing datasets with different class ontologies (4-class → 7-class mapping). |
| **Annotation tool integration** | Cleanlab finds label errors post-hoc. SAM generates pseudo-labels. | End-to-end active learning loop: model predicts → human corrects → model improves, with uncertainty-guided frame selection. |
| **Fresnel lens artifacts** | Periocular NIR paper handles generic degradation. | Beyond-specific: Fresnel zone reflections, pancake lens ghosting, variable illumination patterns. |
| **Confirmation bias in pseudo-labels** | CSL and ACT address threshold selection. CPS uses dual-network disagreement. | Empirical measurement of confirmation bias accumulation over multiple pseudo-label rounds with SegUNET's specific class distribution. |

---

## Quick Reference: Index Navigation

```
docs/research/index/
  master.yaml              ← Start here. 7 clusters, priority-ranked.
  tags.yaml                ← "What covers topic X?" inverted index.
  papers-architectures.yaml ← Model design papers (12 entries)
  papers-training.yaml     ← SSL, active learning, label efficiency (10 entries)
  papers-datasets-robustness.yaml ← Data, robustness, classical (19 entries)
  repos.yaml               ← Cloned repos with verified file paths (14 entries)
  README.md                ← Navigation guide for coding agents
```

Each YAML entry has a `segunet_takeaway` field — the single most actionable
insight for our project. Start there before reading the full paper.
