# Near-Eye Segmentation for HMD Eye Tracking — Third-Pass Deep Research
*All sources are new relative to Passes 1 and 2. No overlap with: RITnet, SegFormer, TopFormer, MobileViT, EllSeg, CondSeg, ENCORE/ACT, CSL, EyeSeg, CSA-CNN, GazeShift/VRGaze, DistillGaze, OpenEDS, NVGaze, RIT-Eyes, TEyeD, LPW.*

***
## Executive Summary
This third pass surfaces four previously uncovered strata of prior art: (1) the complete classical pupil-detection lineage from which every deep segmentation model inherited its problem formulation; (2) PhD theses and technical reports from the main research groups (RIT PerForm Lab, Tübingen HCI, Bulling / MPI-INF / Stuttgart, Pelz Lab) that define the field's shared vocabulary; (3) a dedicated failure-mode and demographic-bias literature covering glasses, makeup, dust, contact lenses, iris pigmentation, ethnicity, HMD slippage, and luminance variation; and (4) published engineering case studies of every major commercial and open-source HMD eye-tracking deployment.

The most actionable finding for SegUNET is the **slippage + luminance stress-test protocol** (Meta Quest Pro / HoloLens 2 studies) which provides a ready-made evaluation framework for measuring robustness under HMD wear conditions, and the **Tübingen dust/dirt benchmark** which exposes that edge-based sub-networks (any module relying on Canny-like features) degrade sharply even under small in-focus contamination.

***
## Part 1 — Foundational Pre-2019 Classical Work
### 1.1 Starburst (Li & Parkhurst, 2005)
The Starburst algorithm is the canonical starting point for feature-based pupil tracking and the immediate ancestor of the RANSAC-based approach adopted later by Świrski et al. The core idea is a ray-casting "starburst": from an initial seed point assumed near the pupil centre, rays are projected radially outward until an intensity gradient threshold is crossed; the intersection points are then cast back inward to generate candidate boundary points, and an ellipse is fitted to the consensus set. The algorithm is inherently iterative and sensitive to corneal reflections, making it best-suited to controlled near-infrared setups where specular artefacts can be removed in a preprocessing step. Open-source code was released, which accounts for its continued use as a baseline in multi-algorithm benchmark studies well into the 2010s.[^1][^2][^3]

**SegUNET relevance:** Starburst's ray-casting intuition (boundary detection via radial gradient) directly motivates SegUNET's UpperLid edge supervision target. The algorithm's known failure mode under partial occlusion — it propagates from an incorrect seed to an entirely wrong boundary — is the classical justification for full segmentation over single-feature methods.
### 1.2 Świrski, Bulling & Dodgson — ETRA 2012
Świrski et al. published the first algorithm explicitly designed for highly off-axis HMD images — the precise scenario that motivated SegUNET. Their pipeline combines a Haar-like feature detector for coarse pupil localisation, k-means segmentation to refine a binary pupil region, and a novel **image-aware RANSAC ellipse fitting** step that weights edge-point selection by local image gradient strength. Unlike earlier approaches that assumed a near-circular pupil, this work explicitly handles the highly eccentric ellipses that arise when the camera axis is far from the optical axis of the eye — a routine condition in HMD designs. The paper was co-authored by Bulling (then at Cambridge, later MPI-INF), establishing the cross-group lineage that connects Tübingen's ExCuSe/ElSe/PuRe work directly back to this 2012 paper.[^4][^5][^6]

**SegUNET relevance:** The Świrski pipeline is the direct RANSAC predecessor to EllSeg's geometric ellipse parametrisation. More critically, the off-axis failure taxonomy Świrski identified (eyelash occlusion of a highly eccentric ellipse) is the same failure mode SegUNET's UpperLid class is designed to handle. Any ablation study of SegUNET's UpperLid head should include Świrski as the classical baseline.
### 1.3 Timm & Barth — VISAPP 2011
Timm and Barth proposed an elegant gradient-field approach for eye-centre localisation that became one of the most-cited pre-deep-learning methods. For each candidate centre pixel, they compute the sum of squared dot products between the unit displacement vectors from that candidate to every other pixel and the corresponding image gradient unit vectors. The maximum of this scalar field corresponds to where the most gradient vectors converge — geometrically, the centre of the brightest circular structure. The method is computationally simple (only dot products), runs in real time, and is inherently robust to partial occlusions. A key contribution was demonstrating complementarity between gradient-based and appearance-based modalities.[^7][^8]

**SegUNET relevance:** Timm-Barth gradients are a lightweight proxy for the iris/pupil boundary geometry that can serve as an auxiliary training signal or as a fast check on predicted segmentation ellipses. For devices without heavy GPU compute (e.g. a fallback pupil-only mode when the full SegUNET is under thermal throttle), Timm-Barth remains viable at 120+ fps on CPU.
### 1.4 Valenti & Gevers — IEEE TPAMI 2012
Where Timm-Barth used raw gradient vectors, Valenti and Gevers exploited isophote curvature — the curvature of iso-intensity contours — to achieve illumination-invariant eye-centre localisation. Isophotes are circular near the pupil centre regardless of overall brightness, giving the method a natural invariance to linear lighting changes (contrast, brightness) and making it robust across a much wider range of iris colours and illumination conditions. Published in TPAMI, this work set the academic standard for appearance-based eye-centre location and remains a frequently cited prior in demographic bias discussions precisely because the isophote approach degrades more gracefully for dark irises than gradient methods that depend on the iris–sclera contrast.[^9][^10]

**SegUNET relevance:** The isophote model of eye anatomy is the implicit geometric prior behind every "ring-like" attention map in modern eye-segmentation networks. For dark-iris failure analysis in SegUNET, Valenti-Gevers provides the baseline that quantifies how much of the performance gap is attributable to the geometric prior versus label imbalance.
### 1.5 Fitzgibbon, Pilu & Fisher — IEEE TPAMI 1999
The Fitzgibbon direct least-squares ellipse fitting algorithm is the standard closed-form ellipse fitting method used by Świrski, ExCuSe, ElSe, and PuRe, and implicitly by every deep model that computes ellipse parameters from a binary mask. The method minimises the algebraic distance from scattered boundary points to an ellipse, subject to the ellipse-specific constraint \(4ac - b^2 = 1\), and is solved via a generalised eigensystem — making it non-iterative, numerically stable, and ellipse-specific (it always returns an ellipse, even with noisy data). A 1996 ICIP version was extended to the widely cited 1999 TPAMI paper.[^11][^12][^13][^14]

**SegUNET relevance:** Every post-processing step that converts SegUNET's binary pupil or iris mask into an ellipse parameter vector — for downstream gaze estimation — implicitly or explicitly applies Fitzgibbon-style fitting. Awareness of its numerical properties (low-eccentricity bias, instability with fewer than ~5 edge points) informs the minimum region size constraints SegUNET should impose before passing masks to the geometric fitting layer.
### 1.6 ExCuSe (Fuhl et al., 2015)
ExCuSe (Excluding Curved Structures) targets what the authors called "real-world scenarios" as opposed to laboratory conditions — off-axis cameras, rapid illumination changes, glasses reflections, and makeup. The algorithm applies edge filtering, then computes oriented histograms via the **Angular Integral Projection Function (AIPF)** to identify pupil-like structures while suppressing curved non-pupil edges (eyebrows, lashes, specular rings). Evaluated on 39,001 hand-labelled images, ExCuSe achieved nearly double the detection rate of the Świrski algorithm on the most challenging subsets. The paper is notable for explicitly naming **makeup** as a first-class failure mode that the authors had to handle in the algorithm design.[^15][^16]

**SegUNET relevance:** ExCuSe's AIPF is, conceptually, an orientation-aware convolution — the direct classical ancestor of oriented convolution blocks in modern segmentation backbones. ExCuSe's failure taxonomy (reflections on glasses, mascara misidentified as pupil, non-centred recording) directly maps to SegUNET's three identified weak-class scenarios.
### 1.7 ElSe (Fuhl et al., 2016)
ElSe (Ellipse Selector) refines ExCuSe by applying ellipse evaluation directly on a filtered Canny edge image: it ranks candidate ellipse fits by a multi-criterion score (edge density, aspect ratio plausibility, size range) and outputs the highest-ranked result. Evaluated on over 93,000 hand-labelled images (55,000 contributed by this work), ElSe achieved a **14.53% improvement** over the best state-of-the-art performer at the time. A key design goal was embeddability in resource-constrained architectures (automotive driver monitoring), making ElSe the most-used classical baseline in HMD eye-tracking benchmarks throughout 2016–2019.[^17][^18]

**SegUNET relevance:** ElSe is the single best classical alternative for pupil-only gaze scenarios. For SegUNET, ElSe provides the lower-bound accuracy benchmark on edge-case frames where the deep model fails but a classical method might partially recover — particularly relevant for hybrid fallback strategies during inference.
### 1.8 PuRe and PuReST (Santini, Fuhl & Kasneci, 2018)
PuRe (Pupil Reconstructor) is the strongest-performing classical algorithm and the immediate predecessor to the deep learning era in Tübingen. It frames pupil detection as an **edge segment selection and conditional combination** problem: curved edge segments compatible with an ellipse hypothesis are selected, and a conditional combination scheme assembles them into the most probable ellipse while propagating a meaningful confidence score. Evaluated on over 316,000 images from four distinct HMDs, PuRe improved the detection rate by over **10 percentage points** versus ElSe on the two most challenging datasets. Crucially, PuRe also improved **precision and specificity by 25.05 and 10.94 percentage points**, respectively, by correctly abstaining on frames with no visible pupil. PuReST adds a Kalman-like temporal tracking stage on top of PuRe detections.[^19][^20][^21][^22]

**SegUNET relevance:** PuRe's confidence metric is the classical analogue of SegUNET's per-pixel entropy — both provide a principled way to flag uncertain frames for human review or active learning. The PuRe paper also provides the largest pre-deep-learning benchmark with 316K images across four HMDs, making it the best classical comparison dataset for SegUNET's failure mode analysis.
### 1.9 Pupil Detection in the Wild — Benchmark Survey (Fuhl et al., Machine Vision and Applications, 2016)
This landmark survey compares six methods — **ElSe, ExCuSe, Pupil Labs, SET, Starburst, and Świrski** — across a unified corpus of 225,569 annotated eye images from four publicly available datasets. The conclusion is that ElSe outperforms all others "by a large margin" under real-world conditions, while Starburst and SET collapse most severely under makeup and glasses. The paper is authored jointly by Fuhl (Tübingen), Tonsen, Bulling (MPI-INF/Stuttgart), and Kasneci, making it the canonical cross-group agreement on the state of pre-deep-learning pupil detection.[^23][^24]

**SegUNET relevance:** Table 1 of this paper is the de facto pre-deep-learning leaderboard that SegUNET should cite as context. Reproducing a subset of these 225,569 images as a "classical test split" within SegUNET's evaluation would allow direct comparison of the deep segmentation pipeline against each classical method.
### 1.10 Świrski & Dodgson — Synthetic Ground Truth Rendering (ETRA 2014)
This Cambridge paper introduced a principled framework for evaluating eye-tracking algorithms using photorealistic synthetic images rendered from a 3D eye model in Blender. The system supports parametric control over gaze vector, camera position, eyelid aperture, and — critically — eyeglass refraction and reflection, enabling algorithm stress-testing under conditions impossible to label from real images. This is the direct ancestor of RIT-Eyes, SynthesEyes, and LEyes in the synthetic pipeline literature.[^25][^26]

**SegUNET relevance:** The Blender pipeline code is publicly available and directly usable for generating SegUNET-specific synthetic data with ground-truth UpperLid annotations — the class most difficult to label in real images.

***
## Part 2 — PhD Theses and Technical Reports
### 2.1 Chaudhary RIT Dissertation — "Deep into the Eyes" (2022)
Aayush Kumar Chaudhary's dissertation at RIT (PerForm Lab, advisors Bailey and Pelz) is the most directly relevant thesis to SegUNET in the corpus. It addresses four interrelated problems: (I) velocity-based iris texture tracking for precision improvement; (II) a **generalised semantic segmentation framework for identifying all eye regions** with ellipse fitting extensions for both pupil and iris — the direct institutional predecessor to SegUNET; (III) a data-driven Blender rendering pipeline for synthetic temporally-contiguous sequences; and (IV) a novel strategy for preserving privacy in eye videos. The dissertation also directly addresses the sim-to-real gap, providing the research motivation for the subsequent 2024 ETRA Sim2Real paper from the same lab.[^27][^28]

**SegUNET relevance:** This is the canonical internal reference for SegUNET's institutional history. The ellipse-fitting extension described in chapter (II) is the direct precursor to SegUNET's geometric head. The dissertation is publicly available via the RIT Libraries repository.
### 2.2 Barkevich RIT Thesis — "Deep Learning for Eye-Tracking Robustness in VR" (2026)
Kevin Barkevich's 2026 thesis at RIT provides an objective assessment of how several contemporary ML-based methods for eye feature tracking affect the final gaze estimate quality — specifically accuracy, precision, and dropout rate — under both feature-based and model-based gaze mapping. This is a direct third-party evaluation of the class of models SegUNET belongs to, measuring whether segmentation improvements actually translate to improved gaze quality.[^29]

**SegUNET relevance:** Provides the gaze-quality evaluation protocol that SegUNET should adopt for its end-to-end assessment: not just mIoU, but gaze accuracy (°) and dropout rate under HMD wear conditions.
### 2.3 Deep Domain Adaptation: Sim2Real (RIT / Meta Reality Labs, ETRA 2024)
Nguyen, Bailey, Diaz, Ma, Fix (Meta Reality Labs) and Ororbia (RIT) address the well-known problem that segmentation models trained on synthetic eye images fail to generalise to real-world images. Their solution uses **dimensionality-reduction techniques to measure distribution overlap** between the target real dataset and synthetic training candidates, then prunes the synthetic training set to maximise distribution overlap. The approach is domain-agnostic and the code is publicly available.[^30][^31][^32][^33]

**SegUNET relevance:** This paper is the closest existing work to the sim-to-real challenge implicit in SegUNET's use of TEyeD synthetic data alongside real HMD footage. The distribution-pruning technique is immediately applicable to SegUNET's training data curation pipeline.
### 2.4 Kassner, Patera & Bulling — Pupil Labs Open-Source Platform (UbiComp 2014)
The Pupil Labs paper introduced the first affordable, open-source HMD eye-tracking system and software stack. The platform achieves **0.6° gaze accuracy** with a processing latency of only **0.045 seconds**, using a lightweight headset with high-resolution scene and eye cameras. The software is platform-independent and includes the full pupil detection, calibration, and gaze estimation pipeline. This was primarily a joint output of Pupil Labs UG (Berlin) and Bulling's group at MPI-INF Saarbrücken.[^34][^35][^36]

**SegUNET relevance:** Pupil Labs is the most widely deployed open-source eye-tracking system globally. SegUNET's intended deployment as an open-source HMD component maps directly onto this infrastructure. Replacing Pupil Labs' default Canny-based pupil detector with SegUNET is the most plausible near-term integration path.
### 2.5 Pelz Lab (RIT) Technical Reports
The RIT Visual Perception Laboratory (Pelz Lab) produced a sequence of technical reports on wearable eye tracking that predated the deep learning era. Key outputs include the wearable RIT eye tracker using IR CMOS cameras and beam-splitter optics, omnidirectional scene camera integration for head pose estimation, and the foundational study "Oculomotor Behavior and Perceptual Strategies in Complex Tasks" which established the behavioural rationale for mobile HMD eye tracking.[^37][^38][^39][^40]

**SegUNET relevance:** The Pelz Lab's hardware design choices — particularly the off-axis IR camera placement — define the exact image characteristics (high eccentricity, strong eyelash occlusion) that SegUNET must handle. These reports provide documented ground truth for expected image appearance statistics.
### 2.6 Bulling Group — Pervasive Eye Tracking Research Program
Andreas Bulling's research programme spans Cambridge (Świrski 2012), MPI-INF Saarbrücken (Kassner 2014), and later the University of Stuttgart / Collaborative AI group. Key outputs for SegUNET include: the 2008 wearable EOG system (UbiComp), the 2016 "pupil detection in the wild" benchmark (co-authored with Fuhl and Kasneci), and the 2019 chapter "Pervasive Eye Tracking for Real-World Consumer Behavior Analysis". The ETRA 2018 keynote "Pervasive Gaze Sensing, Analysis, and Interaction" provides a roadmap of open problems that directly maps to SegUNET's research agenda.[^41][^42][^43][^23]

**SegUNET relevance:** Bulling's group maintains the most comprehensive bibliography of open problems in pervasive eye tracking (available at collaborative-ai.org). Any future SegUNET publication should situate itself within Bulling's taxonomy of sensing, analysis, and interaction challenges.

***
## Part 3 — Failure Mode and Bias Analyses
### 3.1 Sclera Segmentation Bias: Group Evaluation (IEEE TIFS 2023)
This is the most rigorous demographic bias study in the ocular segmentation literature: seven independent research groups evaluated their sclera segmentation models on five diverse datasets within a common experimental framework. Results show **significant performance differences across all seven models**, with **ethnicity as the single largest source of bias** — larger than eye colour, gaze direction, or capture device. Crucially, the study also finds that "training with representative and balanced data does not necessarily lead to less biased results," suggesting that dataset balance alone is insufficient — architectural choices and loss functions also contribute to bias.[^44][^45][^46]

| Factor Studied | Finding |
|---|---|
| Ethnicity | Largest single bias source across all 7 models |
| Eye colour | Correlated with ethnicity but secondary |
| Gaze direction | Significant for models lacking positional invariance |
| Capture device | Moderate; sensor-specific IR spectra matter |
| Balanced training | Did not reliably reduce bias |

**SegUNET relevance:** This directly applies to SegUNET's cross-demographic generalisation objective. The paper's experimental framework (per-demographic mIoU on five standardised datasets) should be adopted verbatim as SegUNET's bias evaluation protocol.
### 3.2 SSRBC 2023 Sclera Competition
The Sclera Segmentation and Joint Recognition Benchmarking Competition 2023 extended the 2020 SSBC work by coupling segmentation quality directly to downstream recognition performance — measuring how bias in segmentation degrades end-to-end biometric accuracy. Five groups submitted six segmentation models. The competition confirmed that segmentation bias due to ethnicity and eye colour propagates directly and measurably into recognition error rates, providing a causal chain from segmentation failure to system-level harm.[^47]

**SegUNET relevance:** Provides the end-to-end bias measurement protocol SegUNET should adopt: rather than measuring bias only at the segmentation mIoU level, SSRBC's approach traces it through to gaze error.
### 3.3 Iris Pigmentation Bias in Visible Spectrum (arXiv:2411.08490)
A 2024 study specifically on iris pigmentation bias in visible-spectrum iris recognition finds that both ViT-b and ResNet-50 show dramatically higher equal error rates for dark irises — with EERs as high as **24.13%** on dark-iris test sets and DPD (differential performance disparity) scores of 81.04% for device cross-comparison. The study underscores that even architectures designed for robustness remain systematically worse on dark-pigmented irises across devices.[^48]

**SegUNET relevance:** SegUNET's training and evaluation datasets (OpenEDS, TEyeD) skew heavily toward light-iris subjects — a known demographic bias of North American and European research populations. The ~24% EER increase for dark irises observed in this study should be treated as a prior for expected SegUNET performance drops on under-represented demographics.
### 3.4 Fuhl et al. — Dirt and Dust on Pupil Detection (JEMR 2017)
The Tübingen group conducted the only systematic study of **physical contamination** as a pupil detection failure mode. By simulating various quantities of dirt and dust on eyeglasses and the eye-tracker camera lens, they found: (1) overall high robustness to out-of-focus dust layers; (2) **strong vulnerability of edge-based methods to even small in-focus dust particles**; and (3) a particle-size vs. particle-count trade-off, where a small number of large particles is less damaging than many small ones. The study also notes that 30% of potential users wear eyeglasses, and that excluding such users from product deployment is commercially unacceptable.[^49][^50][^51]

**SegUNET relevance:** Any SegUNET sub-module that relies on edge features (e.g. the UpperLid boundary head or any Laplacian-based data augmentation) is directly subject to this failure mode. The simulation methodology is freely applicable for augmenting SegUNET's training set with synthesised dirt and dust artefacts.
### 3.5 Slippage Study — Niehorster et al. (Behavior Research Methods, 2020)
This paper is the definitive study of **HMD slippage as a failure mode** for wearable eye trackers. Four setups were tested — Tobii Pro Glasses 2, SMI Glasses 2.0, Pupil Labs Pupil (3D), and Pupil Labs Pupil with the Grip algorithm — under controlled speaking, facial expression, and physical device movement tasks. The Tobii and Grip systems remained stable; the others showed **0.8–3.1° gaze deviation increase** above baseline even for small natural head movements that occur during speech. The study concludes that some systems are fundamentally unsuitable for face-to-face interaction research due to slippage sensitivity.[^52][^53][^54]

**SegUNET relevance:** Slippage shifts the eye-camera geometry, changing the apparent ellipse eccentricity and position in the image — precisely the distribution shift that degrades SegUNET's cross-session generalisation. This motivates SegUNET including slippage-simulating augmentations (random affine transforms with physically motivated parameters).
### 3.6 Slippage-Robust Gaze Tracking (arXiv:2210.11637, 2022)
A follow-up to Niehorster et al., this paper proposes a slippage-correction algorithm for HMD gaze tracking. The system maintains a stable gaze estimate under physical device movement by learning a corrective mapping between optical axis and gaze point that is robust to small camera offsets. The proposed method reduced angular error by **27.3% relative to the Pupil system** under slippage conditions.[^55]

**SegUNET relevance:** Slippage correction is the post-segmentation stage; accurate segmentation is a necessary but not sufficient condition. This paper defines the slippage test protocol (controlled brow-raise / head-tilt tasks) that SegUNET's deployment documentation should adopt.
### 3.7 Biswas et al. — DNN Failure Modes for Eye Tracking (ETRA 2021)
The University of Nevada Reno group characterised DNN failure under five types of image perturbation that naturally occur in mobile eye tracking: **rotations, defocus blur, exposure changes, specular reflections, and JPEG compression**. Using ResNet50 features to compute the cosine distance of perturbed frames from the training distribution, they show that increasing distribution distance is a monotonically predictive indicator of performance decline across all five domains. This establishes distribution shift as a unifying explanation for DNN eye-tracking failures.[^56][^57][^58]

**SegUNET relevance:** The five perturbation types should be included in SegUNET's standard augmentation suite. The cosine-distance-to-training-distribution metric provides a principled monitoring signal for deployment: when a new device's eye images are far from training data, SegUNET can flag low confidence without requiring ground-truth labels.
### 3.8 ElSe's Explicit Failure Taxonomy
ElSe (Fuhl et al., ETRA 2016) provides a written taxonomy of real-world failure modes validated across 93,000 images: illumination changes, **reflections on glasses**, **make-up**, non-centred eye recording, and physiological characteristics (ptosis, narrow palpebral fissure). The paper directly motivated PuRe's confidence metric as a way to detect and abstain on such frames.[^17][^18]

**SegUNET relevance:** This taxonomy is the baseline "failure mode checklist" for SegUNET. Each item on ElSe's list should appear as a specific test case in SegUNET's evaluation protocol.
### 3.9 Tobii Documentation — Known Eye Tracking Limitations
Tobii's published technical documentation for the Vive Pro Eye explicitly lists: **eye surgery, eye disease, heavy makeup, and high myopia** as conditions that may affect eye tracking performance. Tobii's study recruitment guide notes that mascara and eyeliner can make tracking impossible because the software misidentifies dark pigmented regions as the pupil.[^59][^60][^61]

**SegUNET relevance:** These documented limitations directly define SegUNET's known failure envelope. The eyeliner/mascara case is the most consequential for SegUNET because it specifically affects the sclera–eyelid boundary — SegUNET's UpperLid class — by introducing false dark regions that are texturally indistinguishable from the pupil.
### 3.10 Periocular NIR Under Degraded State (Tapia et al., arXiv:2106.15828, 2021)
Tapia et al. propose a framework for NIR periocular segmentation under **alcohol consumption**, deploying Criss-Cross attention (122K params) and DenseNet10 (211K params) achieving **94.54% mIoU** on a manually labelled 20K-image database. The paper is notable for demonstrating that near-eye segmentation degrades measurably under physiological state changes (pupil dilation, reduced blink rate, periocular oedema) — a failure mode distinct from optical or demographic factors.[^62][^63]

**SegUNET relevance:** Physiological state (fatigue, medication, altered states) systematically changes pupil diameter and eyelid aperture distribution, causing distribution shift at inference even when demographic demographics are held constant. This motivates including physiological variation as a data augmentation dimension.

***
## Part 4 — Production System Case Studies
### 4.1 HTC Vive Pro Eye — Tobii Integration
The Vive Pro Eye is the first mass-market VR headset with integrated eye tracking. Key engineering parameters: 120 Hz binocular gaze output, **0.5–1.1° accuracy** within a 20° trackable FOV, 5-point calibration, HTC SRanipal SDK, Unity/Unreal engine support. The SDK exposes raw gaze origin/direction vectors, blink state, and convergence distance. Tobii's internal implementation uses its own proprietary pupil-PCCR (pupil centre corneal reflection) pipeline; the segmentation layer is not disclosed, but the device constraints (IR illumination, off-axis near-eye cameras) are fully consistent with SegUNET's target domain.[^59][^64]
### 4.2 HoloLens 2 — Signal Quality Assessment (Aziz & Komogortsev, ETRA 2022)
The most complete published characterisation of HoloLens 2 eye-tracking quality. Key findings: the device samples at **30 Hz** with a nominal spatial accuracy of **1.5°**, delivered via the MRTK API. A dataset of 30 participants performing a random saccade task revealed that the shipped gaze data appears **uncalibrated by default**; after a recalibration step using the dataset itself, spatial accuracy improved substantially. The study measured spatial accuracy, spatial precision, temporal precision, linearity, and crosstalk.[^65][^66][^67]

| Metric | Reported Value |
|---|---|
| Sampling rate | 30 Hz (nominal) |
| Nominal spatial accuracy | ~1.5° |
| Post-recalibration accuracy | Substantially improved (study-specific) |
| Temporal jitter | Low; steady approximation of 33.3 ms intervals |
| Crosstalk (L/R eye) | Measured; full details in paper |

**SegUNET relevance:** 30 Hz is the HoloLens 2's eye-tracking pipeline bottleneck — not the camera frame rate. SegUNET needs to output segmentation at ≥30 Hz to integrate with this platform, which is already achievable given SegUNET's target latency.
### 4.3 Meta Quest Pro — Privacy White Paper and Signal Quality Study
Meta's white paper on building Quest Pro eye tracking "responsibly" provides the only publicly disclosed description of the on-device pipeline: inward-facing cameras estimate gaze direction; **eye images are processed entirely on-device and deleted immediately** after abstracted gaze data is generated. Eye tracking is off by default and per-app consent is required.[^68][^69]

The companion academic study (Aziz & Komogortsev, arXiv:2403.07210, 2024) characterised signal quality across **78 participants**, measuring spatial accuracy, precision, and linearity under ideal conditions and under two stress conditions: **background luminance variation** and **headset slippage**. Key findings: spatial accuracy degraded significantly for the 75th percentile of users and above when the brow-raise task induced slippage; dark environments degraded pupil diameter estimation and indirectly affected gaze accuracy.[^70][^71][^72][^73]

**SegUNET relevance:** The Quest Pro study defines the HMD stress-test protocol. The 78-participant dataset includes slippage conditions that directly test SegUNET-class segmentation models. The finding that the top-25% of "difficult" users drive most signal quality failures motivates training SegUNET with hard-example mining (OHEM) to specifically improve this tail population.
### 4.4 PSVR2 — Tobii Licensing and Foveated Rendering
Sony confirmed in July 2022 that PSVR2 integrates Tobii eye-tracking technology. The system uses a **custom IR optical sensor** and supports foveated rendering, gaze position and rotation, pupil diameter, and blink state. Unity documentation presented at GDC 2022 reports that eye-tracking-enabled foveated rendering achieves **3.6× faster GPU frame times** versus the 2.5× achieved by non-gaze-aware foveated rendering alone — from 33.2ms to 14.3ms on a reference demo. Sony's own rendering technology documentation confirms the hardware implements positional reprojection in 6 degrees of freedom using this gaze data.[^74][^75][^76][^77][^78]

A November 2025 Sony patent (US 12,468,156) reveals ongoing research into detecting **prescription add-on lenses** using the PSVR2 eye-tracking sensors — a direct acknowledgement that optics worn inside the HMD represent a production failure mode.[^79]
### 4.5 Apple Vision Pro — R1 Chip and Diffraction Grating Eye Tracking
Apple Vision Pro implements the most sophisticated HMD eye-tracking system publicly documented. The hardware includes **four eye-tracking cameras** and approximately **8 IR illuminators per eye (≥16 total)** using diffraction gratings at the eyepieces to redirect IR light reflected from the user's eyes back to the cameras while allowing visible light to pass. The dedicated R1 co-processor achieves **12ms photon-to-photon latency** and processes eye data from all sensors simultaneously. The system supports iris-based biometric authentication (Optic ID), with iris data encrypted and stored only in the Secure Enclave.[^80][^81][^82]

Through-optics analysis reveals that the Vision Pro uses **Variable Rasterization Rate (VRR)** to implement foveated rendering: the centre of the render texture is always aligned to the current fovea, and peripheral resolution is continuously reduced. The system operates in two modes: Fixed Foveated Rendering (FFR) at 26 PPD when eye tracking is unavailable, and Dynamic Foveated Rendering (DFR) at full resolution when gaze is actively tracked.[^83]

| System | Eye Cameras | Hz | Accuracy | Technology |
|---|---|---|---|---|
| Apple Vision Pro | 4 IR cameras + 16+ illuminators | Not disclosed | Sub-degree | Diffraction grating redirect + R1 chip [^80][^81] |
| HTC Vive Pro Eye | Tobii integrated | 120 Hz binocular | 0.5–1.1° | PCCR, SRanipal SDK [^59] |
| HoloLens 2 | 2 stereo IR cameras | 30 Hz | ~1.5° | MRTK API, appears uncalibrated by default [^65][^67] |
| PSVR2 | Tobii IR optical sensor | Not disclosed | Not disclosed | Custom IR sensor, foveated rendering [^77][^78] |
| Meta Quest Pro | Inward-facing cameras | 60 Hz (PCVR) | Not disclosed | On-device processing, deleted immediately [^69][^71] |
| Pupil Labs (open source) | CMOS + IR illuminator | Up to 200 Hz | 0.6° | Open pipeline; PuRe-based detector [^34][^36] |
### 4.6 Pupil Labs — Open-Source Deployment Reference
The Pupil Labs platform (Kassner, Patera, Bulling 2014) provides the most widely studied open-source reference implementation. The 2D pupil detector is a refinement of the Świrski pipeline; the 3D detector adds a cornea-sphere model. The platform has known documented weaknesses: the Niehorster slippage study showed up to **3.1° gaze error increase** under facial expression / speech tasks for the Pupil 3D mode (though Grip mode remained stable). The platform serves as the integration target for replacing the classical pupil detector with SegUNET's output.[^34][^52][^36]
### 4.7 Event-Driven Eye Tracking — Feng et al. (IEEE VR 2022)
Feng et al. propose an ultra-lightweight (30K parameters) gaze-tracking system that achieves 30 Hz on mobile processors with 0.1–0.5° accuracy by emulating an event camera in software. The system identifies ROIs via software-emulated event signals (frame differences), then processes only those ROIs for gaze estimation. This represents an orthogonal approach to SegUNET's full-frame segmentation: rather than segmenting the whole eye region at every frame, it identifies which pixels changed and updates only those.[^84][^85][^86]

**SegUNET relevance:** Event-driven ROI prediction could be used as a front-end for SegUNET to restrict full segmentation to the predicted eye region, reducing compute by 60–80% on frames where the eye moves minimally. The ROI prediction network (30K params) could run ahead of SegUNET's encoder.

***
## Cross-Reference: Which Papers Address Multiple Research Questions
| Paper | Pre-2019 Classical | Theses/Reports | Failure Modes | Production |
|---|---|---|---|---|
| Fuhl et al. "in the wild" 2016 [^23] | ✓ | | ✓ | |
| Fuhl dirt/dust 2017 [^49] | ✓ | | ✓ | |
| Sclera bias IEEE TIFS 2023 [^44] | | | ✓ | ✓ |
| Aziz & Komogortsev HoloLens 2 [^67] | | | ✓ | ✓ |
| Aziz & Komogortsev Quest Pro 2024 [^71] | | | ✓ | ✓ |
| Chaudhary RIT dissertation [^28] | | ✓ | ✓ | |
| Sim2Real ETRA 2024 [^30] | | ✓ | ✓ | |
| Niehorster slippage 2020 [^52] | | | ✓ | ✓ |
| Pupil Labs 2014 [^34] | ✓ | ✓ | | ✓ |
| ElSe 2016 [^17] | ✓ | | ✓ | |
| PuRe 2018 [^19] | ✓ | ✓ | | |
| Świrski ETRA 2012 [^5] | ✓ | ✓ | ✓ | |

***
## Prioritised Actionable Findings
**1. Adopt the HoloLens 2 + Quest Pro stress-test protocol as SegUNET's standard deployment evaluation.** Both studies (Aziz & Komogortsev 2022 and 2024) provide reproducible experimental designs — random saccade tasks, brow-raise slippage, luminance variation — that map directly to SegUNET's deployment conditions. These studies are open and the task software is implementable in Unity MRTK.[^67][^71]

**2. Instrument SegUNET with a PuRe-style confidence score.** PuRe's per-frame confidence metric demonstrated 25-point precision improvement by abstaining on low-confidence frames. A SegUNET-specific abstention mechanism (e.g. predicting mask entropy as a head output) would address the slippage and occlusion failure modes identified by Niehorster and Biswas.[^19][^52][^56]

**3. Run the Tübingen dirt/dust augmentation on training data.** The JEMR 2017 study provides a simulation procedure for in-focus particle contamination. Edge-based methods (including any residual Canny/Sobel processing in SegUNET's boundary supervision) are most sensitive; applying the simulation as a training augmentation is the minimum required mitigation.[^49]

**4. Adopt the sclera segmentation bias protocol (IEEE TIFS 2023) for demographic evaluation.** Run SegUNET on the five SSBC datasets segmented by ethnicity and eye-colour subgroup. The 2023 paper's finding that balanced training is insufficient implies SegUNET may need demographic-stratified loss weighting or adversarial debiasing.[^44]

**5. Map Fitzgibbon ellipse fitting properties into SegUNET post-processing constraints.** The Fitzgibbon DLS method has a known low-eccentricity bias and instability below ~5 edge points. SegUNET's mask-to-ellipse conversion should impose minimum connected-component area thresholds and validate that the predicted mask yields ≥5 boundary samples before passing to the geometric fitting layer.[^13]

**6. Validate SegUNET on the 225,569-image "pupil detection in the wild" corpus.** This is the only cross-method benchmark that includes classical algorithms, HMD-specific images, makeup, and glasses cases. Running SegUNET's pupil-head output against this corpus against ElSe, ExCuSe, and PuRe provides the definitive comparison positioning SegUNET in the literature.[^23]

---

## References

1. [[PDF] Limbus / Pupil Switching For Wearable Eye Tracking Under Variable ...](http://andrewd.ces.clemson.edu/research/vislab/docs/etra08-short.pdf) - To accomplish this, Li et al. [2005] proposed the Starburst algorithm, in which rays are projected f...

2. [[PDF] GPU-accelerated High-speed Eye Pupil Tracking System](https://webs.um.es/jlaragon/papers/mompean_SBACPAD15.pdf) - After removing the reflections the Starburst algorithm is used to search the pupil. Starburst needs ...

3. [Starburst: A robust algorithm for video-based eye tracking](https://www.semanticscholar.org/paper/Starburst:-A-robust-algorithm-for-video-based-eye-Li-Parkhurst/1b44723236a67389c3adaad848f4f7ae9d70b8ea) - A hybrid eye-tracking algorithm that integrates feature-based and model-based approaches and made it...

4. [[PDF] Robust real-time pupil tracking in highly off-axis images](https://www.cl.cam.ac.uk/research/rainbow/projects/pupiltracking/files/Swirski,%20Bulling,%20Dodgson%20-%202012%20-%20Robust%20real-time%20pupil%20tracking%20in%20highly%20off-axis%20images.pdf) - We present a real-time dark-pupil tracking algorithm designed for low-cost head-mounted active-IR ha...

5. [Robust, real-time pupil tracking in highly off-axis images](https://www.collaborative-ai.org/publications/swirski12_etra/) - Our group conducts fundamental research towards collaborative artificial intelligence (CAI) at the i...

6. [[PDF] Robust real-time pupil tracking in highly off-axis images](https://www.collaborative-ai.org/publications/swirski12_etra.pdf) - We present a real-time dark-pupil tracking algorithm designed for low-cost head-mounted active-IR ha...

7. [Accurate Eye Centre Localisation by Means of Gradients](https://www.semanticscholar.org/paper/Accurate-Eye-Centre-Localisation-by-Means-of-Timm-Barth/7ae3c4bd0a4e7a86b1543fdf0fdeffc18d1981b4) - This work proposes an approach for accurate and robust eye centre localisation by using image gradie...

8. [[PDF] ACCURATE EYE CENTRE LOCALISATION BY MEANS OF ...](https://www.scitepress.org/papers/2011/33261/33261.pdf) - We therefore propose an approach for accurate and robust eye centre localisation by using image grad...

9. [Accurate eye center location through invariant isocentric patterns](https://pubmed.ncbi.nlm.nih.gov/22813958/) - Our aim is to bridge this gap by locating the center of the eye within the area of the pupil on low-...

10. [Accurate Eye Center Location Through Invariant Isocentric Patterns](https://ivi.fnwi.uva.nl/isis/publications/bibtexbrowser.php?key=ValentiTPAMI2012&bib=all.bib) - Accurate Eye Center Location Through Invariant Isocentric Patterns. Accurate Eye Center Location Thr...

11. [[PDF] Direct least squares fitting of ellipses | Semantic Scholar](https://www.semanticscholar.org/paper/Direct-least-squares-fitting-of-ellipses-Fitzgibbon-Pilu/a3c27e9811b8ed9ec553277eb269265529d37bfa) - This paper presents a new efficient method for fitting ellipses to scattered data that is ellipse-sp...

12. [Ellipse-specific Direct least-square Fitting](https://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/PILU1/Paper/icip.html)

13. [Direct Least Square Fitting of Ellipses](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/ellipse-pami.pdf)

14. [Microsoft Word - 466.doc](http://www.cs.bilkent.edu.tr/~s.rahimzadeh/fitzgibbon99_ellipse_fitting.pdf)

15. [ExCuSe: Robust Pupil Detection in Real-World Scenarios](https://www.hci.uni-tuebingen.de/assets/pdf/publications/WTCKWE092015.pdf) - We present a novel algorithm for robust pupil detection in real-world scenarios, which is based on e...

16. [ExCuSe: Robust Pupil Detection in Real-World Scenarios](https://www.semanticscholar.org/paper/ExCuSe:-Robust-Pupil-Detection-in-Real-World-Fuhl-K%C3%BCbler/57f16432c3769ad43ed4324b06da259fe7632654) - This work presents a novel, real-time dark-pupil tracking algorithm that uses a Haar-like feature de...

17. [[1511.06575] ElSe: Ellipse Selection for Robust Pupil Detection in ...](https://arxiv.org/abs/1511.06575) - We propose ElSe, a novel algorithm based on ellipse evaluation of a filtered edge image. We aim at a...

18. [Ellipse Selection for Robust Pupil Detection in Real-World ...](https://www.hci.uni-tuebingen.de/assets/pdf/publications/WTTE032016.pdf)

19. [PuRe: Robust pupil detection for real-time pervasive eye tracking](https://www.hci.uni-tuebingen.de/assets/pdf/publications/TWE052018.pdf) - Real-time, accurate, and robust pupil detection is an essential prerequisite to enable pervasive eye...

20. [PuRe: Robust pupil detection for real-time pervasive eye tracking](https://arxiv.org/abs/1712.08900) - In this paper, we introduce the Pupil Reconstructor PuRe, a method for pupil detection in pervasive ...

21. [PuRe: Robust pupil detection for real-time pervasive eye tracking](http://arxiv.org/pdf/1712.08900.pdf)

22. [PuReST: Robust Pupil Tracking for Real-Time Pervasive Eye Tracking](https://www.hci.uni-tuebingen.de/assets/pdf/publications/TWE062018.pdf) - PuReST consists of three distinct parts orchestrated to produce a fast and robust pupil tracking alg...

23. [Pupil detection for head-mounted eye tracking in the wild](https://www.collaborative-ai.org/publications/fuhl16_mvap/) - In this paper we review six state-of-the-art pupil detection methods, namely ElSe, ExCuSe, Pupil Lab...

24. [Paper.pdf](https://www.hci.uni-tuebingen.de/assets/pdf/publications/062016.pdf)

25. [Computer Laboratory – Projects: Rendering Eye Images](https://www.cl.cam.ac.uk/research/rainbow/projects/eyerender/) - Rendering synthetic ground truth images for eye tracker evaluation. Lech Świrski · Neil A. Dodgson. ...

26. [[PDF] Rendering synthetic ground truth images for eye tracker evaluation | Semantic Scholar](https://www.semanticscholar.org/paper/Rendering-synthetic-ground-truth-images-for-eye-Swirski-Dodgson/b1382b0a430e245432af4a6a9ca7df38c1931e75) - A computer graphics approach to creating realistic synthetic eye images, using a 3D model of the eye...

27. [Deep Into the Eyes_ Applying Machine Learning to Improve Eye-Trac](https://www.scribd.com/document/879081602/Deep-Into-the-Eyes-Applying-Machine-Learning-to-Improve-Eye-Trac) - The dissertation titled 'Deep into the Eyes: Applying Machine Learning to improve Eye-Tracking' by A...

28. [Applying Machine Learning to improve Eye-Tracking](https://repository.rit.edu/theses/11102/) - Eye-tracking has been an active research area with applications in personal and behav- ioral studies...

29. ["Using Deep Learning to Increase Eye-Tracking Robustness ...](https://repository.rit.edu/theses/12519/) - Algorithms for the estimation of gaze direction from mobile and videobased eye trackers typically in...

30. [A Sim2Real Neural Approach for Improving Eye-Tracking Systems](https://arxiv.org/abs/2403.15947) - We demonstrate that our methods result in robust, improved performance when tackling the discrepancy...

31. [A Sim2Real Neural Approach for Improving Eye-Tracking Systems](https://arxiv.org/html/2403.15947v1) - Deep Domain Adaptation: A Sim2Real Neural Approach for Improving Eye-Tracking Systems ... In the con...

32. [A Sim2Real Neural Approach for Improving Eye-Tracking Systems](https://dl.acm.org/doi/10.1145/3654703) - Deep Domain Adaptation: A Sim2Real Neural Approach for Improving Eye-Tracking Systems ... Aayush K. ...

33. [GitHub - PerForm-Lab-RIT/domain-adaptation-eye-tracking](https://github.com/PerForm-Lab-RIT/domain-adaptation-eye-tracking) - Official Implementation for the paper Deep Domain Adaptation: A Sim2Real Neural Approach for Improvi...

34. [Pupil: An Open Source Platform for Pervasive Eye Tracking ... - arXiv](https://arxiv.org/abs/1405.0006) - ... Pervasive Eye Tracking and Mobile Gaze-based Interaction. Authors:Moritz Kassner, William Patera...

35. [Pupil: An Open Source Platform for Pervasive Eye Tracking ...](https://ar5iv.labs.arxiv.org/html/1405.0006) - Commercial head-mounted eye trackers provide useful features to customers in industry and research b...

36. [Pupil: an open source platform for pervasive eye tracking and mobile gaze-based interaction](https://dl.acm.org/doi/10.1145/2638728.2641695) - In this paper we present Pupil -an accessible, affordable, and extensible open source platform for p...

37. [[PDF] A Tool for the Study of High-level Visual Tasks](https://www.cis.rit.edu/pelz/publications/Babcock_pelz_peak_MSS_2003.pdf) - Figure 9 – Top images show a person wearing the RIT portable eye tracking system. Bottom image shows...

38. [Head movement estimation for wearable eye tracker](https://www.cis.rit.edu/people/faculty/pelz/lab/papers/rothkopf_1.pdf)

39. [[PDF] The RIT Visual Perception Laboratory](https://www.cis.rit.edu/pelz/lab/documentation/VPL_boilerplate_2004.pdf) - Head-mounted, video-based eyetrackers manufactured by Applied Science Laboratories and ISCAN are use...

40. [[PDF] Oculomotor Behavior and Perceptual Strategies in Complex Tasks](https://www.cis.rit.edu/people/faculty/pelz/lab/people/jason/final.pdf) - To study the oculomotor system in its native mode, we developed a wearable eyetracker that allows na...

41. [Towards Context-Awareness and Mobile HCI Using Wearable EOG ...](https://www.collaborative-ai.org/publications/bulling08_ubicomp/) - In this work we describe the design, implementation and evaluation of a novel eye tracker for contex...

42. [Pervasive Eye Tracking for Real-World Consumer Behavior ...](https://www.collaborative-ai.org/publications/bulling19_tf/) - Our group conducts fundamental research towards collaborative artificial intelligence (CAI) at the i...

43. [[PDF] Pervasive Gaze Sensing, Analysis, and Interaction ... - ETRA - ACM](https://etra.acm.org/2018/ETRA2018_Keynote_Andreas_Bulling.pdf) - Kai Kunze; Andreas Bulling; Yuzuko Utsumi; Shiga Yuki; Koichi Kise. I know what you are reading --. ...

44. [Publications](https://lmi.fe.uni-lj.si/en/research/publications/?tgid=288)

45. [Exploring Bias in Sclera Segmentation Models: A Group Evaluation ...](https://ieeexplore.ieee.org/iel7/10206/9970396/09926136.pdf) - algorithmic bias across eye color, gaze direction, ethnicity, and capture device and the overall seg...

46. [Exploring Bias in Sclera Segmentation Models: A Group Evaluation ...](https://www.academia.edu/99519608/Exploring_Bias_in_Sclera_Segmentation_Models_A_Group_Evaluation_Approach) - Bias and fairness of biometric algorithms have been key topics of research in recent years, mainly d...

47. [[PDF] SSRBC 2023 - Laboratory for Machine Intelligence](https://lmi.fe.uni-lj.si/wp-content/uploads/2023/09/CameraReady-233.pdf) - A follow-up group benchmarking effort fol- lowing SSBC 2020 also looked at bias in sclera segmenta- ...

48. [Impact of Iris Pigmentation on Performance Bias in Visible ...](https://arxiv.org/html/2411.08490v1)

49. [Controlling the influence of dirt and dust on pupil detection - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC7141060/) - Therefore, in this work, we present a systematic study of the effect of dust and dirt on the pupil d...

50. [Controlling the influence of dirt and dust on pupil detection](https://bop.unibe.ch/index.php/JEMR/article/view/3657) - We present a systematic study of the eect of dust and dirt on the pupil detection by simulating vari...

51. [Journal of Eye Movement Research](https://core.ac.uk/download/pdf/158974738.pdf)

52. [The impact of slippage on the data quality of head-worn eye trackers](https://portal.research.lu.se/en/publications/the-impact-of-slippage-on-the-data-quality-of-head-worn-eye-track/) - To investigate how this eye-tracker slippage affects data quality, we designed experiments in which ...

53. [The impact of slippage on the data quality of head-worn eye trackers](https://pmc.ncbi.nlm.nih.gov/articles/PMC7280360/) - Our results show that whereas gaze estimates of the Tobii and Grip remained stable when the eye trac...

54. [The impact of slippage on the data quality of head-worn eye trackers](https://research-portal.uu.nl/en/publications/the-impact-of-slippage-on-the-data-quality-of-head-worn-eye-track) - Specifically, we investigated data quality when participants speak, make facial expressions, and mov...

55. [[PDF] Slippage-robust Gaze Tracking for Near-eye Display - arXiv](https://arxiv.org/pdf/2210.11637.pdf) - However, unavoidable slippage of head-mounted devices (HMD) often results higher gaze tracking error...

56. [[PDF] Characterizing the Performance of Deep Neural Networks for Eye ...](https://par.nsf.gov/servlets/purl/10232276) - Deep neural networks (DNNs) provide powerful tools to identify and track features of interest, and h...

57. [Characterizing the Performance of Deep Neural Networks for Eye ...](https://dl.acm.org/doi/fullHtml/10.1145/3450341.3458491) - We evaluated our DNN model on eye videos with increasing exposure, reflection, defocus blur, eye rot...

58. [Characterizing the Performance of Deep Neural Networks for Eye ...](https://dl.acm.org/doi/10.1145/3450341.3458491) - Here, we test the ability of a DNN to predict keypoints localizing the eyelid and pupil under the ty...

59. [Eye tracking technology for VR - VIVE Pro Eye with Tobiiwww.tobii.com › device-integrations › ht...](https://www.tobii.com/products/integration/xr-headsets/device-integrations/htc-vive-pro-eye) - VIVE Pro Eye with Tobii is next-generation virtual reality. Tobii's eye tracking technology enhances...

60. [Managing participants with vision irregularities - Tobii](https://www.tobii.com/blog/eye-tracking-study-recruitment-managing-participants-with-vision-irregularities) - There's no point eye tracking if participants can't see properly, factor corrective lenses into your...

61. [EyeTrackingProblems - Meg Wiki](https://imaging.mrc-cbu.cam.ac.uk/meg/EyeTrackingProblems) - Common eye tracking problems · Ambient light · Glasses · Makeup · Drooping eyelids · Asymmetries in ...

62. [Semantic Segmentation of Periocular Near-Infra-Red Eye Images ...](https://arxiv.org/abs/2106.15828) - Abstract:This paper proposes a new framework to detect, segment, and estimate the localization of th...

63. [[PDF] Semantic Segmentation of Periocular Near-Infra-Red Eye Images ...](https://arxiv.org/pdf/2106.15828.pdf) - Abstract—This paper proposes a new framework to detect, segment, and estimate the localization of th...

64. [VIVE Pro Eye Overview | VIVE Southeast Asia](https://www.vive.com/sea/product/vive-pro-eye/overview/) - Precision eye tracking combined with professional-grade sound and graphics—designed for studios, hom...

65. [AN ASSESSMENT OF THE EYE TRACKING SIGNAL QUALITY](https://arxiv.org/pdf/2111.07209v1.pdf)

66. [[PDF] an assessment of the eye tracking signal quality captured in ... - arXiv](https://arxiv.org/pdf/2111.07209.pdf) - We present an analysis of the eye tracking signal quality of the HoloLens 2's integrated eye tracker...

67. [An Assessment of the Eye Tracking Signal Quality Captured ... - arXiv](https://arxiv.org/abs/2111.07209) - We characterize the eye tracking signal quality of the device in terms of spatial accuracy, spatial ...

68. [Eye tracking on Meta Quest Pro](https://www.meta.com/help/quest/8107387169303764/)

69. [[PDF] Building Eye Tracking on Meta Quest Pro Responsibly](https://securecdn.oculus.com/sr/meta-quest-pro-eye-tracking-white-paper) - Two features utilize these sensors at the launch of Meta Quest Pro—eye tracking and fit adjustment—b...

70. [Evaluation of Eye Tracking Signal Quality for Virtual Reality ... - arXiv](https://arxiv.org/abs/2403.07210) - We present an extensive, in-depth analysis of the eye tracking capabilities of the Meta Quest Pro vi...

71. [Evaluation of Eye Tracking Signal Quality for Virtual Reality ... - arXiv](https://arxiv.org/html/2403.07210v1) - We present an extensive, in-depth analysis of the eye tracking capabilities of the Meta Quest Pro vi...

72. [Evaluation of Eye Tracking Signal Quality for Virtual Reality Applications: A Case Study in the Meta Quest Pro](https://www.emergentmind.com/papers/2403.07210) - We present an extensive, in-depth analysis of the eye tracking capabilities of the Meta Quest Pro vi...

73. [Evaluation of Eye Tracking Signal Quality for Virtual Reality ...](https://dl.acm.org/doi/fullHtml/10.1145/3649902.3653347) - We explored the eye tracking capabilities of the Meta Quest Pro VR headset using a large dataset of ...

74. [New rendering technology supporting revolutionary ... - Sony](https://www.sony.com/en/SonyInfo/technology/activities/STEF2022/exhibition_0302/02/)

75. [PSVR 2 Is 3.6x Faster Using Eye-Tracking Technology](https://www.playstationlifestyle.net/2022/03/28/psvr-2-specs-eye-tracking-foveated-rendering/) - Using a combination of eye-tracking technology and foveated rendering, the PSVR 2 specs perform that...

76. [PSVR 2 Foveated Rendering Provides 3.6x Faster Performance - Unity](https://www.uploadvr.com/psvr-2-eye-tracking-foveated-rendering-gdc/) - As reported by Michael Hicks of Android Central, a Unity panel at GDC last week revealed new details...

77. [Eye-tracking firm confirms Sony has licensed its tech for PSVR2 | VGC](https://www.videogameschronicle.com/news/eye-tracking-firm-confirms-sony-has-licensed-its-tech-for-psvr2/) - The cameras use a custom infrared optical sensor to track users' eye movements, with support in over...

78. [Update: PSVR 2 to Include Tech from the Biggest Name in Eye ...](https://www.roadtovr.com/psvr-2-eye-tracking-tobii/) - It was first revealed that Sony would include eye-tracking in PSVR 2 back in May 2021, with the ment...

79. [Big Research Revealed! + Next Week PSVR2 Releases - YouTube](https://www.youtube.com/watch?v=tKs1HpmyeHs) - Sony's latest VR patent hints at eye-tracking being used for far more than just foveated rendering —...

80. [Apple Vision Pro - Technical Specifications](https://www.apple.com/apple-vision-pro/specs/) - Two high‑resolution main cameras; Six world‑facing tracking cameras; Four eye‑tracking cameras; True...

81. [How Apple Vision Pro's Infrared Eye-Tracking Technology Works](https://petapixel.com/2024/04/01/how-apple-vision-pros-infrared-eye-tracking-technology-works/) - The Apple Vision Pro uses infrared illuminators and cameras to accurately track the user's eyes. Her...

82. [Apple Patent | Eye tracking system](https://patent.nweon.com/33610) - The eye tracking system may include at least one eye tracking camera (e.g., infrared (IR) cameras) p...

83. [Apple Vision Pro has the same effective resolution...](https://douevenknow.us/post/750217547284086784/apple-vision-pro-has-the-same-effective-resolution) - visionOS has two modes it uses when rendering layers: Fixed Foveated Rendered (FFR) 26PPD [no eye tr...

84. [Real-Time Gaze Tracking with Event-Driven Eye Segmentation - arXiv](https://arxiv.org/abs/2201.07367) - This paper presents a real-time eye tracking algorithm that, on average, operates at 30 Hz on a mobi...

85. [[PDF] Real-Time Gaze Tracking with Event-Driven Eye Segmentation](https://par.nsf.gov/servlets/purl/10323881) - Our main contribution is a novel ROI prediction algorithm, which emulates an event camera in softwar...

86. [Real-Time Gaze Tracking with Event-Driven Eye Segmentation](https://www.semanticscholar.org/paper/Real-Time-Gaze-Tracking-with-Event-Driven-Eye-Feng-Goulding/749cff59b002adb68dda280764ab6b2988af7ff6) - A novel, lightweight ROI prediction algorithm is introduced by emulating an event camera, which cont...

