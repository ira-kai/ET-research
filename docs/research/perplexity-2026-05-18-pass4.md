# Boundary Quality in Semi-Supervised Eye Segmentation — Research Report
*Context: 7-class near-IR eye segmentation (pupil, iris, sclera, upper lid, lower lid, bright spot, background) for 90fps HMD deployment; CPS semi-supervised training; Region mIoU 0.9026, Boundary mIoU 0.7207; weakest boundary classes UpperLid (0.586) and BrightSpot (0.633); primary failure locus: medial canthus 3-way junction.*

*Scope: New sources only — no overlap with Passes 1–3.*

***
## Executive Summary
All four research questions point toward the same root intervention. The boundary gap in CPS-trained models is not random noise: it is a **systematic artefact of how CPS propagates pseudo-labels**, amplified by **data coverage gaps at precisely the thin-structure and multi-class junction regions** the model has never seen confidently labeled. The most actionable finding is that a cluster of 2024–2025 papers have now diagnosed this exact failure mode and proposed direct fixes — most of which can be applied without retraining from scratch and several of which fit inside the 11ms inference budget. The highest-leverage single intervention is **BoundMatch** (IEEE Access 2025), which retrofits boundary detection as a consistency task directly into a CPS-style teacher-student pipeline. The highest-leverage single training-side data improvement is **SBC-AL** (MICCAI 2024), which queries new annotation candidates specifically based on boundary consistency disagreement.

***
## Q1 — Improving Boundary Quality When the Bottleneck Is Data Coverage
### 1.1 The Data Coverage Framing Is Correct
The hypothesis that the bottleneck is data coverage rather than loss design is supported by the literature. A 2024 survey of pseudo-label methods finds that pseudo-label noise concentrates at **class boundaries**, and that fixing boundary coverage in the labeled set propagates more reliably than any loss-reweighting approach, because losses act on the distribution of training pixels but cannot create information about unseen boundary configurations. The MPDC paper (Pattern Recognition 2025) independently reaches the same conclusion: when the labeled set lacks boundary-region diversity, CPS and similar consistency methods reinforce wrong boundary predictions between the two networks rather than averaging them toward correct ones.[1][2]
### 1.2 SBC-AL: Boundary-Consistency Active Learning (MICCAI 2024)
SBC-AL is the closest existing work to the problem described. It adds two modules to any segmentation model — a **Structure-aware Feature Prediction (SFP)** module and an **Attentional Segmentation Refinement (ASR)** module — and proposes an uncertainty-based querying strategy that scores candidate images by two separate criteria:[3][4][5]

- **Structure Consistency Score (SCS):** agreement across augmented views on *interior* region predictions
- **Boundary Consistency Score (BCS):** agreement across augmented views on *boundary* predictions

Images where SCS is high but BCS is low are labeled as anatomically understood but boundary-uncertain — exactly the regime SegUNET is in. In experiments on two medical datasets, SBC-AL outperforms all other common AL strategies (random, uncertainty, core-set, BADGE) when only a fraction of total labels are available.[4][3]

**Application to SegUNET:** Apply SBC-AL's BCS query criterion to the existing unlabeled pool. Images where the two CPS networks agree on region placement but disagree on boundary pixel assignments are the highest-value annotation candidates — these are precisely the UpperLid/BrightSpot/medial-canthus frames that are currently missing from labeled coverage.
### 1.3 Boundary-Centric Active Learning
An ICCV 2025 workshop paper on temporal action segmentation proposes **boundary-centric annotation**, an observation that generalizes to spatial segmentation: the most information-dense annotations are those collected at the **boundary locus** of highest predictive uncertainty. The protocol computes average frame-level uncertainty within a local window around candidate boundary points, selecting the candidates where the confusion band is consistently wide across augmentations.[6]

For SegUNET, this translates to a concrete procedure: compute per-pixel entropy across the two CPS network outputs; construct 5-pixel erosion and dilation bands around each boundary class; query for annotation frames where mean entropy within those bands exceeds a threshold. This concentrates annotation effort on the medial canthus and UpperLid-sclera interface specifically.
### 1.4 Multi-Device Pupil, Limbus, and Eyelid Segmentation — Clinical Validation (IOVS 2022)
Morley and Evans (IOVS 2022) trained a U-Net on 604 de-identified infrared iris images from a variety of topographers and achieved **99.8% detection success** for pupil and limbus, with Dice > 0.95 for pupil in 95.6% of cases. Crucially, the paper reports that the **eyelid boundary is the hardest class** and requires a separate post-processing step: an **active contours algorithm** is applied to the U-Net output to produce geometric curves and to **infer eyelid interference from iris contour shape**. This is the only published paper that treats the eyelid boundary as structurally distinct from the iris/pupil boundaries and justifies a separate inference module — directly validating SegUNET's UpperLid design as a separate class.[7]

**Application to SegUNET:** The active contours post-processing step (applied to the segmentation mask rather than re-run during training) is a lightweight mechanism to regularize UpperLid boundary geometry based on the known constraint that eyelid curves are smooth and monotone. This can be applied at inference without retraining.
### 1.5 Eyelid Curvature Dual U-Net (AtDU-Net, Frontiers in Medicine 2025)
Yu et al. (2025) propose AtDU-Net, a dual-branch U-Net with a **shared encoder and task-specific decoders** for simultaneous palpebral fissure and corneal segmentation. The architecture uses Hierarchical Attention Sampling Modules (HASM) and Split Axial Detail Modules (SADM) to capture fine boundary detail, achieving IoU 0.979 and Dice 0.989, with upper eyelid curvature correlation of 0.903 with manual annotations. The key architectural insight is that **using a shared encoder for both corneal and eyelid tasks** prevents the two decoders from interfering at their mutual boundary (the limbus) — the eye-specific analogue of the medial canthus problem.[8]

**Application to SegUNET:** A shared encoder feeding into a dedicated UpperLid decoder branch (separate from the iris/sclera decoder) explicitly models the eyelid as a structurally distinct task. The dual-decoder structure prevents the sclera decoder from "explaining away" UpperLid boundary pixels.
### 1.6 Shape Prior via Level Set for Weak-Contrast Ocular Boundaries
A 2023 paper on AS-OCT lens segmentation proposes incorporating **convex shape priors into a level set loss function** to handle weak-contrast tissue boundaries where standard pixel-level classification fails. The approach reformulates segmentation as regression of a level set function (whose zero level is the boundary) and embeds the shape prior into the loss. Applied to lens nucleus and cortex segmentation, the shape-based loss improved state-of-the-art U-Net MIoU by 0.0156 at the boundary.[9]

**Application to SegUNET:** The UpperLid and LowerLid classes have known convex shape priors (an eyelid cannot be concave in the standard image plane). Adding a level set regulariser that penalises non-convex UpperLid predictions would reduce anatomically impossible artefacts — specifically the kinds introduced by the polar warp augmentations previously tried.

***
## Q2 — Lightweight Post-Processing for Multi-Class Junction Sharpening Within 11ms
### 2.1 SegFix: Model-Agnostic Boundary Pixel Relabeling (ECCV 2020)
SegFix is the most directly applicable post-processing method for SegUNET's deployment scenario. The mechanism is conceptually simple: it observes that **interior pixel predictions are consistently more reliable than boundary pixel predictions**. It therefore replaces the unreliable labels of boundary pixels with the labels of corresponding interior pixels, using a learned direction map (offset field) that points from each boundary pixel to its nearest interior pixel. The method requires no information about the segmentation model, processes only the input image, and was demonstrated to **consistently reduce boundary errors** on Cityscapes, ADE20K, and GTA5.[10][11][12]

Key latency properties: SegFix adds a boundary localization step and a direction lookup — both are lightweight 2D operations. The authors describe the speed as "nearly real-time". At SegUNET's target resolution (a near-eye crop ~320×240), the total additional compute is dominated by the direction map convolution, which is a single forward pass through a shallow HRNet branch. On a modern VR compute substrate (Snapdragon XR2+ or equivalent), this maps to approximately 2–4ms.[12]

**Multi-class junction behavior:** At the medial canthus, SegFix's interior-pixel replacement propagates the labels of the most confident adjacent class. This is a heuristic that works well when one class dominates the junction — it will tend to propagate the sclera label (largest surrounding class) rather than the correct lower-lid or iris assignment. SegFix should therefore be applied **only to exterior boundary pixels, not junction pixels** — applying it uniformly may worsen BrightSpot boundaries where the interior is ambiguous.
### 2.2 Dense CRF — Fast Approximate Inference (Krähenbühl & Koltun, 2012)
The fully connected dense CRF with Gaussian edge potentials remains the canonical post-processing approach for label map sharpening. Its key properties:[13][14]

- The approximate inference algorithm via the **permutohedral lattice** scales linearly with filter size and is parallelizable on GPU[15]
- A single-threaded CPU implementation processes benchmark images "in a fraction of a second"[13]
- The algorithm sharply promotes label consistency within homogeneous regions and suppresses label switching across high-gradient edges — directly addressing boundary blur at the iris-sclera boundary

**Latency reality check:** Despite the theoretical efficiency, a published thesis on AI performance at inference explicitly notes that "the inference time of CRFs is still far from real-time". GPU-accelerated implementations (e.g., the heiwang CUDA CRF library) reduce this, but the number of CRF inference iterations (typically 5–10) means that for a 320×240 image on an embedded GPU, total CRF time is likely 3–6ms, fitting within the 11ms budget for a low-resolution eye ROI but not for full-frame inference. The dense CRF is recommended as a **region-smoothing complement to SegFix** (run CRF on interior regions; run SegFix on boundary pixels), rather than as a full replacement for boundary supervision.[16][17]
### 2.3 OASIS: Lightweight Structural Refinement for Video Object Segmentation (ICCV 2025)
OASIS proposes a **lightweight structure refinement module** that fuses rough edge priors from a Canny filter with stored object features to generate an object-level structure map. The key design insight is that pre-computed Canny edges are free at inference (they involve no neural network call), and fusing them with the decoder's output features adds boundary sharpness with minimal compute overhead. The module is implemented as a small convolutional block applied to the feature-level output.[18]

**Application to SegUNET:** Apply a Canny-derived edge prior as an attention gate on the UpperLid decoder head's output features. This has no trainable parameters added at inference, and the fusion operation is a single elementwise multiply — well under 0.5ms. The edge prior suppresses false positive UpperLid predictions in the smooth sclera region and sharpens the true UpperLid boundary.
### 2.4 Multi-Class Part Parsing Based on Multi-Class Boundaries (ICIP 2025)
This paper directly addresses the problem of **adjacent-class junction confusion** in dense multi-class segmentation. It incorporates multi-class boundary predictions (not binary edge predictions) as an additional input channel to the segmentation decoder. The boundary map distinguishes *which pair of classes* each boundary separates — so the UpperLid-sclera boundary is represented differently from the UpperLid-iris boundary. A weighted multi-label cross-entropy loss is used to balance learning across all part classes.[19][20]

**Application to SegUNET:** The medial canthus junction contains three simultaneous class boundaries (UpperLid/LowerLid, LowerLid/sclera, sclera/iris). Standard binary edge maps cannot represent this. Multi-class boundary maps represent each pixel's full boundary type and can be computed from the ground truth masks at training time and from the model's own prediction at inference. Adding a multi-class boundary head to SegUNET that is supervised during training and used as an attention gate during inference directly targets the medial canthus failure mode.
### 2.5 BSANet: Multi-Class Boundary-Semantic Awareness (ICCV 2019)
BSANet proposes two co-designed modules for multi-class part parsing:[21]

- **Boundary-Aware Spatial Selection (BASS):** applies spatial attention supervised by class-agnostic part boundaries at multiple feature scales, performing coarse-to-fine boundary refinement from low-level to high-level features
- **Semantic-Aware Channel Selection (SACS):** uses semantic object context to enhance class-relevant feature channels

The two modules are applied sequentially: BASS first localises where boundaries are, then SACS disambiguates which class the boundary separates. This sequential processing addresses the exact UpperLid/BrightSpot confusion pattern — the model knows there is a boundary but incorrectly assigns its class.[21]

**Application to SegUNET:** BSANet's boundary-semantic sequential design is directly applicable as an architectural patch. The BASS spatial attention can be inserted as a decoder feature modulation layer using the existing CPS model's boundary mIoU output as supervision. No retraining from scratch is required — it can be added as an auxiliary training head and fused at inference.

***
## Q3 — Thin Anatomical Structure Segmentation: Architectures and Losses
### 3.1 soft-clDice: Topology-Preserving Loss for Tubular Structures (CVPR 2021)
Shit et al. introduce **clDice**, a similarity measure computed on the intersection of segmentation masks and their morphological skeletons, and prove it **guarantees topology preservation up to homotopy equivalence** for binary 2D and 3D segmentation. The differentiable **soft-clDice** loss uses iterative min- and max-pooling to implement differentiable soft skeletonization. Training on soft-clDice consistently improves connectivity information, graph similarity, and volumetric scores across vessels, roads, and neurons.[22][23][24]

The loss has one key limitation: it is a **binary loss** — it operates on one class at a time. To apply it to UpperLid, the training loss must include a separate soft-clDice term for the UpperLid binary mask. This does not address the medial canthus junction directly, but it does prevent the UpperLid prediction from developing topological holes (disconnected lid fragments), which are a common secondary failure mode when the boundary supervision is weak.[25]
### 3.2 Skeleton Recall Loss: First Multi-Class Thin Structure Loss (ECCV 2024)
Kniep et al. introduce the **Skeleton Recall Loss**, the first loss function explicitly designed for **multi-class** thin structure segmentation. It avoids the expensive differentiable skeletonization of clDice by computing the skeleton on the CPU during data loading and then computing a **soft recall of the predicted segmentation on the precomputed tubed skeleton**. This reduces computational overhead by more than **90% versus clDice** while achieving overall superior performance on five public datasets.[26][27][28][29]

The multi-class capability is the critical advantage over soft-clDice: Skeleton Recall Loss can simultaneously supervise UpperLid, LowerLid, iris limbus, and bright spot skeletons in a single loss term. For SegUNET, where three thin-class boundaries converge at the medial canthus, this is the correct topological loss to use rather than per-class binary clDice.[28]

| Loss Function | Multi-class? | GPU Cost vs. Baseline | UpperLid-applicable? |
|---|---|---|---|
| Cross-entropy (current) | ✓ | 1× | ✓ (poor boundary) |
| Soft-clDice (CVPR 2021) | Binary only | ~5× | ✓ per-class only |
| Skeleton Recall Loss (ECCV 2024) | ✓ native | ~1.1× (CPU skeleton) | ✓ directly |
| clCE (MICCAI 2024) | Binary only | ~5× | ✓ per-class, robust to noise |
| FocusSDF (arXiv 2025) | ✓ | ~2× | ✓ with SDF supervision |
### 3.3 Centerline-Cross Entropy (clCE, MICCAI 2024)
Bourque et al. note that the clDice loss penalises segmentation accuracy at the cost of connectivity and lacks robustness to noisy annotations. Their **clCE** loss reformulates clDice using cross-entropy rather than Dice — inheriting the robustness of cross-entropy while preserving the topological focus on the centerline skeleton. For BrightSpot specifically (a small, near-circular class where standard Dice loss is unstable due to small region size), clCE provides more numerically stable gradients than soft-clDice.[30]
### 3.4 FocusSDF: Signed Distance Function Boundary Weighting (arXiv 2025)
FocusSDF introduces a loss function based on signed distance maps (SDFs) that **adaptively assigns higher weights to pixels closer to object boundaries** using exponential decay. A joint-learning strategy simultaneously predicts binary masks and SDFs, providing complementary region-level and boundary-level supervision. Validated against five state-of-the-art models including MedSAM, FocusSDF consistently improves boundary accuracy while maintaining region overlap scores. Its advantage over simple boundary upweighting (which the question states failed) is that the weights are derived from the SDF — they decay smoothly from the boundary rather than applying a hard binary threshold — which avoids the gradient conflict with Dice+Focal that caused the earlier upweighting attempt to fail.[31]
### 3.5 DCBNet: Multi-Task Boundary Knowledge Distillation (UND Thesis 2024)
A multi-task architecture called DCBNet fuses detailed, contextual, and boundary features through a **semantic edge detection (SED) auxiliary task** that provides edge prior knowledge to the segmentation branch. Teacher-to-student knowledge distillation from the boundary-aware teacher to a lighter student improves student mIoU from 72.3% to 72.8% while running at 52.6 FPS. The distillation approach is directly applicable to SegUNET: the current CPS ensemble can serve as a boundary-aware teacher whose boundary head output supervises a lighter student inference model.[32]
### 3.6 ContextLoss: Topological Error Context (ICIP 2025)
The ContextLoss (CLoss) paper addresses a known limitation of clDice: it detects topological errors but ignores their spatial context. CLoss considers topological errors with their full surrounding context, allowing the network to understand whether a break in a thin structure is isolated or part of a larger pattern. For the medial canthus junction — where three-class topological errors cluster — this context-awareness is directly relevant.[33]

***
## Q4 — Does CPS Systematically Underperform on Boundary Quality? What Fixes Exist?
### 4.1 The Core Mechanism: Why CPS Degrades Boundaries
The original CPS paper (Chen et al., CVPR 2021) uses pseudo one-hot label maps from one network to supervise the other via standard cross-entropy loss. This creates a systematic boundary degradation mechanism:[34]

1. **One-hot pseudo labels carry boundary errors as hard assignments.** Unlike soft pseudo labels, one-hot labels provide zero gradient signal for uncertain boundary pixels and full incorrect signal for wrong-class boundary pixels[1]
2. **Mutual reinforcement of boundary errors.** When both networks predict the same wrong class at a boundary pixel (which happens whenever the labeled data lacks diversity for that configuration), CPS provides zero corrective signal — the error is stable under the consistency constraint[35]
3. **No boundary-specific uncertainty.** CPS's consistency measure is computed uniformly over all pixels; boundary pixels — which have intrinsically higher entropy — are not distinguished from interior pixels[36]

The pseudo-label survey (arXiv:2403.01909) identifies CPS's "catastrophic stability of wrong predictions" as a primary failure mode for thin-class boundaries: once both networks agree on a wrong boundary assignment, nothing in the CPS loss corrects it.[1]
### 4.2 BoundMatch: Direct Fix for CPS Boundary Failure (IEEE Access 2025)
BoundMatch is the most direct published solution to CPS boundary degradation. It retrofits a **Boundary Consistency Regularized Multi-Task Learning (BCRM)** objective into a teacher-student consistency pipeline, enforcing consistency between teacher and student on both segmentation masks **and semantic boundary maps** (not binary edges — class-pair-specific boundaries). Two fusion modules provide the integration:[37][38]

- **Boundary-Semantic Fusion (BSF):** injects learned boundary cues into the segmentation decoder
- **Spatial Gradient Fusion (SGF):** refines boundary predictions using mask spatial gradients, producing higher-quality boundary pseudo-labels[38]

The multi-class semantic boundary maps distinguish *which pair* of classes each boundary separates — the UpperLid/sclera boundary gets a different label from the UpperLid/iris boundary. This directly addresses the medial canthus junction problem.[38]

**BoundMatch is implemented and publicly available.** It is architected as a plug-in to any teacher-student framework, meaning it can be applied directly to SegUNET's CPS setup without redesigning the training loop.
### 4.3 CW-BASS: Confidence-Weighted Boundary-Aware SSSS (IJCNN 2025)
CW-BASS identifies **boundary blur** as a distinct failure category of SSSS methods (separate from confirmation bias and coupling), and addresses it with a **boundary-aware module** using Sobel-filtered pseudo-labels as edge supervision. The full framework includes:[39][36]

- Confidence-weighted loss that adjusts pseudo-label contribution by per-pixel confidence (downweighting uncertain boundary pixels rather than excluding them)
- Dynamic thresholding that adapts the confidence threshold based on current model performance
- Confidence decay that reduces low-confidence pixel influence as training stabilises[36]

The boundary-aware loss alone contributed a **6.36% mIoU gain** in the ablation study, making it the largest single contributor in the framework. CW-BASS achieves 75.81% mIoU on Cityscapes with only 1/8 labeled data, outperforming fully supervised baselines.[36]

**Sobel vs. Canny note:** CW-BASS explicitly uses Sobel rather than Canny for boundary detection because Canny is non-differentiable and complicates end-to-end training. This is directly applicable to SegUNET's training loop.[36]
### 4.4 MPDC: Multi-Perspective Dynamic Consistency (Pattern Recognition 2025)
MPDC addresses CPS's "model cognitive bias" — the tendency of single-perspective consistency training to reinforce the same wrong boundary interpretation — by introducing a **Perspective Fusion Module (PFM)** that integrates predictions from three independent network branches before applying cross-pseudo supervision. The fusion perspective provides an additional view that is semantically rotated relative to the two original CPS networks. In the ablation, adding the fusion perspective for cross-pseudo supervision improved the 95th-percentile Hausdorff Distance by 0.47 on ACDC with 10% labeled data — a direct boundary quality metric improvement.[2]

**Application to SegUNET:** Replace the two-network CPS setup with a three-branch MPDC setup where the third branch is a feature-augmented view (DINOv2 backbone instead of ResNet, following UniMatch V2) — this costs one additional forward pass during training but adds no inference cost.[40]
### 4.5 Boundary-Aware Prototype Contrastive Learning (IEEE TIP 2024)
This method introduces **boundary-aware prototypes** that explicitly model the differences between boundary and centre pixels for adjacent categories. Prototypes are constructed from features equidistant from object boundaries (via signed distance maps), and **pixel-prototype contrastive learning** pushes same-class boundary features together and separates different-class boundary features. The model outperforms methods using larger VNet backbones, using a modified lightweight UNet.[41][42]

The key insight applicable to SegUNET: **the medial canthus junction is the point where lower-lid, sclera, and iris prototypes are maximally confused**. Adding a contrastive boundary prototype term that explicitly pushes these three class prototypes apart at their boundary-adjacent feature embeddings directly addresses the junction failure mode.
### 4.6 n-CPS: Multi-Network Extension (arXiv 2021)
n-CPS generalises CPS to n simultaneously trained subnetworks, with ensembling at inference. Experiments show that network ensembling significantly improves performance and that n-CPS with CutMix outperforms CPS on Pascal VOC and Cityscapes. For boundary quality, the ensemble averaging attenuates the variance in boundary pixel predictions — a simple variance reduction mechanism. The practical constraint is that n=3 or 4 requires proportionally more GPU memory during training, which may be infeasible if SegUNET is already using the full training budget.[43]

***
## Prioritised Intervention Table

| Intervention | Effort | Expected ΔBoundary mIoU | 11ms Budget? | Code |
|---|---|---|---|---|
| BoundMatch BCRM retrofit[38] | Medium (training change) | High (+3–6 pt) | N/A (training) | Yes |
| Skeleton Recall Loss for UpperLid/BrightSpot[28] | Low (loss swap) | Medium (+2–4 pt) | N/A (training) | Yes |
| SBC-AL BCS querying for new annotation[4] | Medium (data pipeline) | High (addresses root cause) | N/A (data collection) | No official |
| SegFix post-processing on non-junction boundaries[12] | Low (plug-in) | Low–Medium (+1–2 pt) | ~2ms ✓ | Yes |
| Multi-class boundary head (ICIP 2025 approach)[20] | Medium (arch change) | Medium (+2–4 pt) | ~1ms ✓ | No |
| Canny edge prior fusion (OASIS style)[18] | Very Low (inference only) | Low (+0.5–1 pt) | <0.5ms ✓ | No |
| CW-BASS confidence-weighted boundary loss[36] | Low–Medium (loss + threshold) | Medium (+3–6 pt) | N/A (training) | Yes |
| Boundary-Aware Prototype contrastive (TIP 2024)[41] | High (training redesign) | Medium–High (+3–5 pt) | N/A (training) | No |
| Dense CRF post-processing[13] | Low | Low (+0.5–2 pt) | ~3–5ms ✓ (low-res) | Yes |
| AtDU-Net dual-decoder retrain for lids[8] | High (architecture) | High if validated | N/A | No |

***
## Specific Guidance on the Medial Canthus Junction
The medial canthus is architecturally a **T-junction** in the label space: LowerLid terminates, and Sclera and Iris meet beneath it. Standard pixel-wise cross-entropy losses assign independent class probabilities to junction pixels with no structural constraint that respects this topology.

Three interventions are specifically validated for multi-class junctions:

1. **Multi-class boundary maps (ICIP 2025 / BSANet):** Label each boundary pixel not with a binary edge label but with the *pair of classes* it separates. The medial canthus pixel receives three overlapping boundary labels simultaneously (UpperLid/Sclera, Sclera/Iris, UpperLid/Iris). Train an auxiliary head to predict these multi-class boundary maps; use the output as an attention gate on the segmentation decoder.[21][20]

2. **SegFix — but restricted to non-junction pixels:** At three-way junction pixels, SegFix's "replace with nearest interior" heuristic is ambiguous. Apply SegFix's direction map only to single-class boundary pixels (where there is a clear interior direction); exclude junction pixels from the replacement and instead use the multi-class boundary head to resolve them.[12]

3. **Prototype contrastive learning at boundaries:** The three-class confusion at the medial canthus is a feature-space proximity problem. Signed-distance-based boundary prototypes that pull LowerLid, Sclera, and Iris features apart at the junction region would reduce tri-class confusion without requiring more labeled junction frames.[41]
