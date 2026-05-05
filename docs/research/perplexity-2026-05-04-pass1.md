# SegUNET Prior Art Report: Eye Segmentation for Eye Tracking

## Executive Summary

This report catalogs prior art relevant to the SegUNET project — an EfficientNet-B0 UNet performing 7-class eye parsing at ~256×256 input, targeting 90fps on the Bigscreen Beyond HMD. Sources are grouped by the seven research questions in the brief and tagged by relevance tier (**Directly Related**, **Near Future**, **Peripheral**). Each entry includes citation, key takeaways, links, and a one-sentence applicability note for SegUNET.

***

## Q1 — Segmentation Architectures for Near-Eye Parsing

### RITnet: Real-time Semantic Segmentation of the Eye for Gaze Tracking

**Citation:** Chaudhary, Kothari, Acharya et al. *RITnet: Real-time Semantic Segmentation of the Eye for Gaze Tracking.* ICCVW 2019.[^1][^2]

**Relevance Tier:** ⚡ Directly Related

**Key Findings:**
- Combines U-Net and DenseNet into a sub-1 MB model that achieves 95.3% accuracy on the 2019 OpenEDS Semantic Segmentation challenge[^2]
- Runs at >300 Hz on a GeForce GTX 1080 Ti, far exceeding the 90fps target[^3]
- Handles the background, sclera, iris, and pupil classes with dense skip connections; the DenseNet blocks mitigate gradient vanishing in shallow networks without heavy parameter cost[^1]

**Links:** arXiv:1910.00694 · Code: https://bitbucket.org/eye-ush/ritnet/

**Applicability to SegUNET:** RITnet is the most direct architectural competitor — its dense skip connections on a UNet scaffold at sub-1 MB size set a latency target for SegUNET's deployment model, and its 4-class structure (vs. SegUNET's 7) is the baseline to beat on class resolution.

***

### SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers

**Citation:** Xie, Wang, Yu et al. *SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers.* NeurIPS 2021.[^4][^5]

**Relevance Tier:** ⚡ Directly Related

**Key Findings:**
- SegFormer-B0 (3.8M params, 8.4G FLOPs) achieves 37.4% mIoU on ADE20K, outperforming all real-time CNN counterparts on latency and is 7.4 FPS faster than DeepLabV3+ (MobileNetV2) while being 3.4% better in mIoU[^6]
- Without TensorRT acceleration, SegFormer-B0 already runs at 48 FPS on a V100 at Cityscapes resolution — with optimization this translates to plausible sub-11ms at 256×256 on edge hardware[^4]
- The Mix Transformer (MiT) encoder eliminates positional encodings, preventing the resolution-mismatch degradation that plagues ViT-based encoders when inference size differs from training[^7]

**Links:** arXiv:2105.15203 · Code: https://github.com/NVlabs/SegFormer

**Applicability to SegUNET:** Drop-in encoder swap from EfficientNet-B0 to SegFormer-B0; the hierarchical multi-scale features likely improve BrightSpot and UpperLid boundary quality due to local+global attention.

***

### TopFormer: Token Pyramid Transformer for Mobile Semantic Segmentation

**Citation:** Zhang, Huang, Luo et al. *TopFormer: Token Pyramid Transformer for Mobile Semantic Segmentation.* CVPR 2022.[^8][^9]

**Relevance Tier:** ⚡ Directly Related

**Key Findings:**
- TopFormer-T achieves 33.4% mIoU on ADE20K with only 1.4M params and 0.5G FLOPs, designed explicitly for ARM CPU inference (benchmarked on Qualcomm Snapdragon 865)[^8]
- On ARM-based mobile devices with 448×448 input, TopFormer-T achieves real-time inference; at 512×512 TopFormer-S beats MobileNetV3 by 5% higher mIoU at lower latency and outperforms SegFormer with 4× less computation (1.8 vs 8.4 GFLOPs)[^9]
- Scale-aware token pooling addresses the extreme size imbalance between small classes (BrightSpot ~1% of pixels) and large ones without requiring special loss tricks[^9]

**Links:** arXiv:2204.05525 · Code: https://github.com/hustvl/TopFormer

**Applicability to SegUNET:** TopFormer-T/S is the strongest candidate for SegUNET's deployment encoder — 1.4M params vs. EfficientNet-B0's 5.3M, with ARM-benchmarked latency that better predicts Bigscreen Beyond compute than GPU benchmarks.

***

### EfficientFormer: Vision Transformers at MobileNet Speed

**Citation:** Li, Yuan, Wen et al. *EfficientFormer: Vision Transformers at MobileNet Speed.* NeurIPS 2022.[^10][^11]

**Relevance Tier:** ⚡ Directly Related

**Key Findings:**
- EfficientFormer-L1 achieves 79.2% top-1 on ImageNet with only 1.6ms inference on iPhone 12 (CoreML), matching MobileNetV2×1.4 (1.6ms, 74.7%) while being 4.5% more accurate[^10]
- The design uses a dimension-consistent 4D MetaBlock for early stages and 3D MHSA only in the final stage, avoiding the throughput bottlenecks of full attention at large spatial resolutions[^12]
- Demonstrates strong results on detection and segmentation downstream tasks, confirming that the backbone transfers well beyond classification[^11]

**Links:** arXiv:2206.01191 · Code: https://github.com/snap-research/EfficientFormer

**Applicability to SegUNET:** EfficientFormer-L1 is the direct replacement for EfficientNet-B0 in segmentation_models_pytorch — same parameter count regime, mobile-optimized, with transformer global context expected to help the cross-subject generalization gap.

***

### MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer

**Citation:** Mehta, Rastegari. *MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer.* ICLR 2022 (Apple Research).[^13][^14]

**Relevance Tier:** ⚡ Directly Related

**Key Findings:**
- MobileViT achieves 78.4% top-1 on ImageNet-1K with ~6M parameters, outperforming MobileNetV3 (CNN) by 3.2% and DeiT (ViT) by 6.2% at similar parameter count[^13]
- Treats transformer blocks as convolutions with global receptive fields, enabling direct integration into any CNN without positional encoding overhead[^14]
- A DeepLabV3 + MobileViT segmentation model is available on Hugging Face (pretrained on PASCAL VOC), providing a ready fine-tuning starting point[^15]

**Links:** arXiv:2110.02178 · Code: https://github.com/apple/ml-cvnets · HuggingFace: Matthijs/deeplabv3-mobilevit-small

**Applicability to SegUNET:** MobileViT-XXS/XS as the SegUNET encoder with a UNet decoder head is the most directly reproducible transformer experiment, with pretrained segmentation weights already publicly available.

***

### MobileViTv3: Improved Lightweight Vision Transformer for Segmentation

**Citation:** Wadekar, Chaurasia. *MobileViTv3: Mobile-Friendly Vision Transformer with Simple and Effective Fusion of Local, Global and Input Features.* 2022.[^16]

**Relevance Tier:** ⚡ Directly Related

**Key Findings:**
- MobileViTv3-0.5 and 0.75 outperform MobileViTv2-0.5 and 0.75 by 2.1% and 1.0% respectively on ImageNet-1K[^16]
- MobileViTv3-1.0 achieves 2.07% better mIoU than MobileViTv2-1.0 on ADE20K segmentation, and 1.1% better on PascalVOC[^16]
- The improved fusion block combines local, global, and input features at each MobileViT block, which is likely beneficial for small-class boundary detail

**Links:** OpenReview — MobileViTv3 paper

**Applicability to SegUNET:** MobileViTv3's improved fusion is worth comparing directly against MobileViTv1 under identical UNet decoder setup to measure the segmentation delta on SegUNET's 7-class task.

***

### Edge-Guided Near-Eye Image Analysis for HMDs

**Citation:** Wang et al. *Edge-Guided Near-Eye Image Analysis for Head Mounted Displays.* ISMAR 2021.[^17]

**Relevance Tier:** ⚡ Directly Related

**Key Findings:**
- Introduces E2-Net, an adversarially-trained edge extraction network that guides pupil/iris ellipse fitting for AR/VR HMD images[^17]
- Compares directly against EllSeg and ESF-Net, showing that edge guidance reduces ellipse fitting failure under eyelid occlusion — a SegUNET UpperLid failure mode[^17]
- Demonstrates that combining a segmentation mask with an explicit edge map is superior to masks alone for ellipse parameter accuracy near eyelid boundaries

**Links:** PDF from ISMAR 2021 proceedings (linked above)

**Applicability to SegUNET:** Complementary post-processing: rather than fixing UpperLid IoU alone, an edge-extraction step can recover the eyelid boundary even when mask quality is imperfect.

***

## Q2 — Datasets and Benchmarks

### OpenEDS (2019): Open Eye Dataset

**Citation:** Garbin, Shen, Koberle et al. *OpenEDS: Open Eye Dataset.* arXiv:1905.03702, Facebook Research, ICCVW 2019.[^18]

**Relevance Tier:** ⚡ Directly Related

**Key Findings:**
- 152 participants, VR HMD-mounted synchronized cameras at 200 Hz, IR illumination — the closest published hardware profile to Bigscreen Beyond[^18]
- Provides 12,759 pixel-annotated images for iris, pupil, and sclera (3 classes); 252,644 additional images for synthesis; data captured at various gaze directions[^18]
- Winning model RITnet achieved 95.3% accuracy; challenge used mIoU as the primary metric, providing a direct comparison baseline[^2]

**Links:** Request at http://research.fb.com/programs/openeds/ · Paper: arXiv:1905.03702

**Applicability to SegUNET:** Primary pretraining corpus — OpenEDS 2019's 3-class masks need re-annotation for UpperLid, LowerLid, and BrightSpot, but the raw IR images are valid pretraining data to overcome the small-sample limitation.

***

### OpenEDS2020: Open Eyes Dataset (Second Edition)

**Citation:** Palmero, Sharma, Behrendt et al. *OpenEDS2020: Open Eyes Dataset.* arXiv:2005.03876, Facebook Research, 2020.[^19][^20]

**Relevance Tier:** ⚡ Directly Related

**Key Findings:**
- 80 participants, 100 Hz HMD-mounted cameras; Eye Segmentation Dataset contains up to 29,500 images at 5 Hz with semantic segmentation labels for ~5% of frames[^19]
- Baseline mIoU for semantic segmentation: 84.1%, establishing the floor that SegUNET's 81.3% mean is approaching[^19]
- The Gaze Prediction dataset (550,400 images with gaze vectors) is separately available and can provide the gaze ground truth needed for future seg-to-gaze pipeline evaluation[^19]

**Links:** arXiv:2005.03876 · Dataset request: http://research.fb.com/programs/openeds-2020-challenge/

**Applicability to SegUNET:** The 29,500 unlabeled sequences can serve as unlabeled data in a semi-supervised ENCORE/CSL training loop, while the 5% labeled portion supplements SegUNET's 1,500-frame annotation set.

***

### TEyeD: Over 20 Million Real-World Eye Images

**Citation:** Fuhl et al. *TEyeD: Over 20 million real-world eye images with Pupil, Eyelid, and Iris 2D and 3D Segmentation, 3D Gaze, and 360° Head-Pose.* CVPR 2021.[^21][^22]

**Relevance Tier:** ⚡ Directly Related

**Key Findings:**
- World's largest unified HMD eye dataset: 20+ million carefully annotated images captured with 7 different head-mounted eye trackers, including two VR/AR devices[^21]
- Provides 2D and 3D landmarks, semantic segmentation (pupil, iris, eyelids), 3D eyeball annotation, gaze vector, and eye movement types for all images[^21]
- Dataset integrates NNGaze, LPW, GIW, ElSe, ExCuSe, and PNET sub-datasets under unified annotation[^22]

**Links:** FTP download: ftp://nephrit.cs.uni-tuebingen.de (user TEyeDUser, no password)

**Applicability to SegUNET:** Highest-value pretraining source for UpperLid and LowerLid classes specifically — TEyeD provides eyelid annotations missing from OpenEDS, at HMD camera geometry that matches SegUNET's domain.

***

### NVGaze: Anatomically-Informed Dataset for Near-Eye Gaze Estimation

**Citation:** Kim, Stengel, Majercik et al. *NVGaze: An Anatomically-Informed Dataset for Low-Latency, Near-Eye Gaze Estimation.* CHI 2019, NVIDIA.[^23][^24]

**Relevance Tier:** ⚡ Directly Related

**Key Findings:**
- Two datasets: 2M synthetic images (1280×960, anatomically-informed 3D eye/face models with gaze, pupil, iris, skin tone variation) and 2.5M real images from 35 subjects (640×480)[^23]
- Synthetic images labeled with exact 2D gaze vector, 3D eye location, 2D pupil center, and eyelid segmentation mask, making them directly usable for segmentation pretraining[^24]
- Network achieves sub-millisecond inference latency, demonstrating that the dataset design supports real-time target hardware[^23]

**Links:** NVIDIA Research: https://research.nvidia.com/publication/2019-05_nvgaze-anatomically-informed-dataset

**Applicability to SegUNET:** NVGaze's 2M synthetic images provide massive subject variation (lash density, lid droop, iris texture) that SegUNET's geometric-only augmentation cannot simulate; direct pretraining candidate.

***

### RIT-Eyes: Rendering of Near-Eye Images for Eye Tracking

**Citation:** Nair, Kothari, Chaudhary et al. *RIT-Eyes: Rendering of Near-Eye Images for Eye-Tracking Applications.* ETRA 2020.[^25][^26]

**Relevance Tier:** ⚡ Directly Related

**Key Findings:**
- Blender-based synthetic pipeline with active deformable iris, aspherical cornea, retinal retro-reflection, gaze-coordinated eyelid deformations, and blink simulation — all features SegUNET's training data lacks[^25]
- Benchmarks SegNet and RITnet trained on RIT-Eyes synthetic data and tested on NVGaze/OpenEDS — the sim-to-real performance is the critical metric for SegUNET's augmentation strategy[^25]
- *Temporal RIT-Eyes* (2022) extends the pipeline to temporally continuous sequences with natural gaze dynamics, enabling future temporal label propagation experiments[^26]

**Links:** arXiv:2006.03642 · ACM DL: https://doi.org/10.1145/3385955.3407935

**Applicability to SegUNET:** Enables generation of thousands of synthetic frames with labeled UpperLid/LowerLid/BrightSpot/Pupil class variations tuned to SegUNET's class distribution — the most actionable short-term augmentation path.

***

### SynthesEyes: Rendering Eyes for Eye-Shape Registration and Gaze Estimation

**Citation:** Wood, Baltrusaitis, Zhang et al. *Rendering of Eyes for Eye-Shape Registration and Gaze Estimation.* ICCV 2015, University of Cambridge.[^27]

**Relevance Tier:** **Near Future**

**Key Findings:**
- 11,382 synthesized close-up eye images from 10 dynamic eye-region models (5 female, 5 male) with wide range of head pose, gaze, and illumination[^27]
- Each image comes with a pickle file containing per-image metadata (shape, gaze, pose); the dataset structure is well-documented for immediate integration[^27]
- Demonstrates out-of-distribution generalization for cross-dataset gaze estimation — the same principle applies to cross-subject segmentation generalization

**Links:** Project + download: https://www.cl.cam.ac.uk/research/rainbow/projects/syntheseyes/

**Applicability to SegUNET:** SynthesEyes is small (11K images) but provides labeled eye-shape data usable for pretraining UpperLid/LowerLid — worth combining with RIT-Eyes for a richer synthetic corpus.

***

### LPW: Labelled Pupils in the Wild

**Citation:** Tonsen, Zhang, Sugano, Bulling. *Labelled Pupils in the Wild: A Dataset for Studying Pupil Detection in Unconstrained Environments.* ETRA 2016, MPI-INF.[^28][^29]

**Relevance Tier:** **Peripheral** (pupil-only labels)

**Key Findings:**
- 66 high-speed (95 FPS) videos from 22 participants recorded in everyday locations with a dark-pupil head-mounted eye tracker[^28]
- Covers glasses, contact lenses, makeup, strong shade, eyelid occlusion, and outdoor illumination variation — exactly the hard cases SegUNET faces[^30]
- Pixel-level labels are pupil only, not multi-class; benchmarks five pupil detection algorithms[^29]

**Links:** MPI-INF: https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/gaze-based-human-computer-interaction/labelled-pupils-in-the-wild-lpw · Download (2.4 GB)

**Applicability to SegUNET:** LPW's hard cases (strong shade, eyelid occlusion, glasses) are useful augmentation inspiration and failure case analysis, though labels are pupil-only and require re-annotation for SegUNET's 7-class setup.

***

## Q3 — Cross-Subject Generalization

### Domain Adaptation for Eye Segmentation (OpenEyes 2020 Workshop)

**Citation:** Shen et al. *Domain Adaptation for Eye Segmentation.* OpenEyes Workshop @ ECCV 2020.[^31]

**Relevance Tier:** ⚡ Directly Related

**Key Findings:**
- Compares three domain adaptation frameworks on eye segmentation: unsupervised (UDA), supervised (SDA), and semi-supervised (SSDA) DA
- Using only 200 samples of labeled target data, SDA shows +5.4% mIoU and SSDA shows +6.6% mIoU over the unadapted baseline[^31]
- Uses adversarial training to align probability map distributions between source and target domains, directly addressing the cross-subject feature distribution gap SegUNET faces in Fold 5 vs Fold 1[^31]

**Links:** PDF: https://openeyes-workshop.github.io/downloads/openeyes2020_yiru_shen_domain_adaptation_for_eye_segmentation.pdf

**Applicability to SegUNET:** The +6.6% mIoU from SSDA with 200 target labels maps directly to SegUNET's per-subject generalization problem — adding a discriminator head to align feature maps across the subject GroupKFold splits could close the 0.761→0.859 fold gap.

***

### Cross-Domain Adaptation and Geometric Synthesis for Near-Eye to Remote Gaze Tracking

**Citation:** (University of Texas thesis). *Cross-domain adaptation and geometric data synthesis for near-eye to remote gaze tracking.* 2023.[^32]

**Relevance Tier:** **Near Future**

**Key Findings:**
- Uses TEyeD as source domain and UnityEyes as intermediate to test generalization to MPIIGaze, validating that head-mounted gaze data can adapt to remote gaze estimators[^32]
- Demonstrates the value of 3D computer vision as a data synthesis technique for bridging domain gaps when labeled data is scarce[^32]
- Results show complementarity with other state-of-the-art gaze adaptation approaches

**Links:** Repository: http://dx.doi.org/10.26153/tsw/46018

**Applicability to SegUNET:** The TEyeD → UnityEyes intermediate domain strategy can be adapted to generate SegUNET-targeted synthetic frames with subject-specific eye geometry, bridging the gap for subjects underrepresented in the training set.

***

### Few-Shot Personalized Scanpath Prediction (CVPR 2025)

**Citation:** Xue, Xu, Mondal et al. *Few-shot Personalized Scanpath Prediction.* CVPR 2025.[^33][^34]

**Relevance Tier:** **Near Future**

**Key Findings:**
- Proposes Subject-Embedding Network (SE-Net) that generates per-subject embeddings from minimal support data, conditioning a downstream model without requiring test-time fine-tuning[^33]
- Experiments on three eye-tracking datasets demonstrate strong FS-PSP performance with only a few examples per subject[^34]
- The subject embedding approach is modality-agnostic — the same principle applies to eye appearance (lid geometry, iris texture) rather than gaze patterns

**Links:** arXiv:2504.05499 · Code: https://github.com/cvlab-stonybrook/few-shot-scanpath

**Applicability to SegUNET:** When deploying to new users on Bigscreen Beyond, SE-Net-style subject conditioning (trained on 2–5 frames from the new user) could replace expensive per-user fine-tuning.

***

## Q4 — Active Learning and Label Efficiency

### ENCORE: Feedback-Driven Pseudo-Label Reliability Assessment (May 2025)

**Citation:** Ghamsarian, Nasirihaghighi, Schoeffmann, Sznitman. *Feedback-Driven Pseudo-Label Reliability Assessment: Redefining Thresholding for Semi-Supervised Semantic Segmentation.* arXiv:2505.07691, May 2025.[^35][^36][^37]

**Relevance Tier:** ⚡ Directly Related

**Key Findings:**
- ENCORE (Ensemble-of-Confidence Reinforcement) integrates Class-Aware Confidence Calibration (CAC) and Adaptive Confidence Thresholding (ACT) to select pseudo-labels without any manual threshold tuning[^35]
- CAC estimates class-wise pseudo-label confidence using only the labeled dataset, making it suitable for SegUNET's small-pool regime (~1,500 frames)[^36]
- ACT dynamically adjusts per-class thresholds based on real-time student model feedback — directly addressing SegUNET's risk of confirmation bias in the approve/reject loop[^37]

**Links:** arXiv:2505.07691 · PDF: https://arxiv.org/pdf/2505.07691.pdf

**Applicability to SegUNET:** ENCORE should be the first pseudo-label quality gate to implement — replace the binary approve/reject decision with ACT-calibrated per-class confidence scores, and track ACT threshold curves over labeling rounds to detect feedback loop drift.

***

### CSL: Confidence Separable Learning (ICCV 2025)

**Citation:** Liu, Liu. *When Confidence Fails: Revisiting Pseudo-Label Selection in Semi-supervised Semantic Segmentation.* ICCV 2025. arXiv:2509.16704.[^38][^39][^40]

**Relevance Tier:** ⚡ Directly Related

**Key Findings:**
- Diagnoses the core failure mode of confidence-threshold pseudo-label selection: network overconfidence causes correct and incorrect predictions to overlap in the high-confidence region, making simple thresholding ineffective[^38]
- CSL formulates selection as a convex optimization problem in confidence distribution feature space, establishing sample-specific (rather than global) decision boundaries[^40]
- Introduces random masking of reliable pixels to force the network to learn from low-reliability spatial context — directly counteracting the spatial-semantic discontinuity that harms UpperLid boundary learning[^39]

**Links:** arXiv:2509.16704 · Code: https://github.com/PanLiuCSU/CSL

**Applicability to SegUNET:** CSL's sample-specific selection boundaries are the right tool for SegUNET's BrightSpot and UpperLid classes, where the model is likely overconfident about background mis-classified as small classes.

***

### Integrating Semi-Supervised and Active Learning for Semantic Segmentation (2025)

**Citation:** Ma, Karakus, Rosin. *Integrating Semi-Supervised and Active Learning for Semantic Segmentation.* arXiv:2501.19227, 2025.[^41]

**Relevance Tier:** ⚡ Directly Related

**Key Findings:**
- Proposes Pseudo-Label Auto-Refinement (PLAR) that corrects erroneous pseudo-label pixels by comparing feature representations with labeled regions, without increasing the labeling budget[^41]
- Pinpoints pixels where pseudo-labels are likely inaccurate and applies manual labeling only to those regions — an efficient strategy for SegUNET's annotation hours
- Outperforms state-of-the-art methods on both natural image and remote sensing benchmarks[^41]

**Links:** arXiv:2501.19227

**Applicability to SegUNET:** PLAR's automatic correction loop can be integrated into mask_review_server.py as a pre-filtering step, reducing the fraction of frames that require full human review.

***

### Overcoming Confirmation Bias in Semi-Supervised Learning

**Citation:** Grendel, Hüllermeier, Fischer. *How to Overcome Confirmation Bias in Semi-Supervised Image Classification via Active Learning.* MCML / Pattern Recognition, 2023.[^42]

**Relevance Tier:** ⚡ Directly Related

**Key Findings:**
- Demonstrates that random sampling does not mitigate SSL confirmation bias in realistic data scenarios (class imbalance, within-class imbalance, class similarity) and can perform worse than fully supervised learning[^42]
- Active learning with uncertainty-based selection can overcome confirmation bias even when SSL with random sampling cannot[^42]
- Identifies SegUNET's exact challenge: between-class imbalance (BrightSpot <1% vs. Background ~60%) combined with class similarity (UpperLid ↔ Sclera boundary confusion) creates a confirmation bias environment

**Links:** https://mcml.ai/publications/ghf+23/

**Applicability to SegUNET:** This paper provides theoretical justification for prioritizing uncertainty-based sample selection (rather than random round-robin) in SegUNET's annotation loop, especially as BrightSpot and UpperLid are the weakest classes.

***

### cleanlab: Label Error Detection for Semantic Segmentation

**Citation:** Northcutt et al. *cleanlab: Confident Learning — Estimating Uncertainty in Dataset Labels.* JAIR 2021, ongoing framework.[^43][^44]

**Relevance Tier:** **Near Future**

**Key Findings:**
- cleanlab now directly supports semantic segmentation label error detection (`find_label_issues` for N×H×W label arrays) using out-of-sample predicted probabilities[^43]
- Masks with mislabeled pixels are ranked by quality score — the tool requires a trained model's predicted probability maps, making it usable after SegUNET's first training run[^43]
- Supports PyTorch natively and has been used to detect label errors in image segmentation datasets comparable to SegUNET's scale[^44]

**Links:** cleanlab docs: https://docs.cleanlab.ai/stable/tutorials/segmentation.html · GitHub: https://github.com/cleanlab/cleanlab

**Applicability to SegUNET:** After reaching 603+ labels (Phase 2 trigger), run cleanlab's segmentation filter over all existing LabelMe annotations using the current model's pred_probs — prioritize re-labeling the highest-error frames before the next training cycle.

***

## Q5 — Segmentation-to-Gaze Pipeline

### EllSeg: An Ellipse Segmentation Framework for Robust Gaze Tracking

**Citation:** Kothari, Chaudhary, Bailey, Pelz, Diaz. *EllSeg: An Ellipse Segmentation Framework for Robust Gaze Tracking.* IEEE TVCG 2021.[^45][^46]

**Relevance Tier:** **Near Future**

**Key Findings:**
- Trains a CNN to directly segment complete (full) elliptical pupil and iris regions rather than visible-only masks, making ellipse fitting robust to eyelid occlusion — the most frequent cause of SegUNET UpperLid/Pupil interaction failures[^45]
- Achieves at least 10% and 24% increase in pupil and iris center detection rate respectively within a two-pixel error margin vs. standard part segmentation[^45]
- Code is available; the framework processes standard segmentation masks and outputs ellipse parameters compatible with downstream gaze models[^47]

**Links:** arXiv:2007.09600 · IEEE: https://doi.org/10.1109/TVCG.2021.3067765 · Code: https://github.com/RSKothari/EllSeg

**Applicability to SegUNET:** EllSeg is the primary pupil→gaze bridge when SegUNET's mIoU exceeds 0.80 — it converts SegUNET's Pupil and Iris masks to ellipse parameters, feeding the geometric gaze model without needing direct gaze labels.

***

### CondSeg: Ellipse Estimation of Pupil and Iris via Conditioned Segmentation

**Citation:** Jia, Deng, Chi, Long, Du. *CondSeg: Ellipse Estimation of Pupil and Iris via Conditioned Segmentation.* arXiv:2408.17231, 2024.[^48][^49][^50]

**Relevance Tier:** **Near Future**

**Key Findings:**
- Jointly trains segmentation and ellipse parameter regression using two priors: (1) pupil/iris project as ellipses, and (2) visibility is controlled by eye-region openness (eyelid mask)[^48]
- Tests on OpenEDS-2019 and OpenEDS-2020, achieving competitive segmentation mIoU while simultaneously outputting accurate elliptical parameters — eliminating a post-processing step[^49]
- Does not require explicit ellipse annotation in the training set — only standard segmentation masks needed[^50]

**Links:** arXiv:2408.17231

**Applicability to SegUNET:** CondSeg's joint loss can be added to SegUNET's training objective once eyelid segmentation stabilizes — conditioning the ellipse estimator on SegUNET's eyelid masks ensures the ellipse is physically consistent with the visible eye region.

***

### DeepVOG: Open-Source Pupil Segmentation and Gaze Estimation

**Citation:** Yiu, Aboulatta, Raiser et al. *DeepVOG: Open-source Pupil Segmentation and Gaze Estimation in Neuroscience Using Deep Learning.* Journal of Neuroscience Methods 2019.[^51][^52][^53]

**Relevance Tier:** **Near Future**

**Key Findings:**
- Integrates an FCNN pupil segmenter with a two-sphere anatomical eyeball model for gaze estimation; provides confidence measure from segmentation output to weight the ellipse fit[^51]
- Open source with complete pipeline from raw frame to gaze vector — deployable as a reference implementation for SegUNET Phase 2[^53]
- 3DeepVOG (2025) extends to 3D gaze (horizontal, vertical, torsional) with corneal refraction correction, building on DeepVOG's segmentation pipeline[^54]

**Links:** GitHub: https://github.com/pydsgz/DeepVOG · PubMed: https://pubmed.ncbi.nlm.nih.gov/31176683/

**Applicability to SegUNET:** DeepVOG's two-sphere model provides a ready geometric gaze calculator — feed SegUNET's Pupil mask through EllSeg → pupil ellipse → DeepVOG gaze vector, with confidence gating from the segmentation Dice score.

***

### Two-Stage Gaze Estimation Using Eye Image Segmentation (Thesis, TU Wien)

**Citation:** Fuerst. *Evaluating Two-Stage Gaze Estimation Using Eye Image Segmentation.* Master's Thesis, TU Wien, 2023.[^55]

**Relevance Tier:** **Near Future**

**Key Findings:**
- Directly quantifies the relative influence of the segmentation stage on end-to-end gaze estimation error, using Pupil Labs internal datasets with paired segmentation/gaze annotations[^55]
- Concludes that no prior published work has explicitly used a segmentation feature extraction step combined with a deep-learning gaze estimator, making this the first ablation of seg error vs. gaze error[^55]
- Establishes a two-stage evaluation protocol that can be adapted to SegUNET: measure gaze angular error as a function of UpperLid and Pupil IoU degradation

**Links:** TU Wien repository: https://repositum.tuwien.at/bitstream/20.500.12708/188367/

**Applicability to SegUNET:** This thesis provides the evaluation methodology for Phase 2 — use its protocol to quantify how each mIoU point gained on SegUNET translates to angular gaze error reduction.

***

## Q6 — Real-Time Inference and Deployment

### RITnet at >300Hz: Real-Time Benchmark Reference

**Citation:** Chaudhary et al. (same as Q1 above) — deployment benchmark.[^3][^2]

**Relevance Tier:** **Near Future**

**Key Findings:**
- RITnet runs at >300Hz on GTX 1080 Ti (sub-1 MB model, 4-class, ~256×256 input) — this is the most directly comparable latency baseline to SegUNET's target[^2]
- SegUNET adds 3 more classes and uses a heavier encoder (EfficientNet-B0 vs. DenseNet-like), so direct benchmarking should expect ~3–5× higher latency before optimization[^3]
- RITnet's performance sets a floor: a deployment model smaller than 1 MB can run at sufficient frequency, suggesting aggressive quantization/distillation is tractable

**Links:** Same as Q1.

**Applicability to SegUNET:** Benchmark SegUNET against RITnet at identical 256×256 resolution on the same GPU before starting TensorRT work to establish the baseline latency gap.

***

### TensorRT Quantization for Segmentation Models

**Citation:** Multiple: (a) Mahdi et al. *Accelerated Real-Time Face Recognition and Segmentation with YOLOv8 Optimized through TensorRT.* 2025. (b) TensorRT PTQ framework, arXiv:2501.17343.[^56][^57]

**Relevance Tier:** **Near Future**

**Key Findings:**
- TensorRT API Quantization (INT8 PTQ) reduces inference latency by up to 58% (4.28 ms/image) on segmentation models with minimal mAP loss[^56]
- TensorRT engines can achieve up to 14.87× faster inference than full-precision PyTorch on MobileNet; for smaller architectures, speedups are more consistent and predictable[^58]
- FP8 TensorRT engines for vision models can run as small as 0.32× FP32 size with up to 5.97× speedup at minimal accuracy cost[^59]

**Links:** YOLOv8-TRT paper: JISEM 2025 · PTQ framework: arXiv:2501.17343

**Applicability to SegUNET:** The deployment path is: PyTorch Lightning checkpoint → ONNX export → TensorRT INT8 PTQ engine — calibrate on ~200 representative frames; expect 4–15× latency reduction making the 11ms target achievable even from EfficientNet-B0.

***

### EdgeSAM: Knowledge Distillation for On-Device Segmentation

**Citation:** Zhou, Li, Loy, Dai. *EdgeSAM: Prompt-In-the-Loop Distillation for On-Device Deployment of SAM.* CVPR 2024.[^60]

**Relevance Tier:** **Near Future**

**Key Findings:**
- Distills the ViT-based SAM encoder into a CNN architecture better suited for edge devices, achieving a 37-fold speed increase vs. original SAM with minimal performance drop[^60]
- Shows that task-agnostic encoder distillation is insufficient — the decoder and task-specific prompt must be included in the distillation loop for full knowledge transfer[^60]
- Provides a blueprint for SegUNET Phase 3: distill a large SegFormer-B2 or B4 teacher into TopFormer-T student for deployment

**Links:** arXiv:2312.06660 · Project: https://www.mmlab-ntu.com/project/edgesam

**Applicability to SegUNET:** When transitioning from training model to deployment model, use EdgeSAM's task-in-loop distillation approach — include SegUNET's 7-class segmentation head in the distillation objective, not just the encoder.

***

## Q7 — Robustness and Augmentation

### EyeGAN: Gaze-Preserving, Mask-Mediated Eye Image Synthesis

**Citation:** Kaur, Manduchi. *EyeGAN: Gaze-Preserving, Mask-Mediated Eye Image Synthesis.* WACV 2020.[^61][^62][^63]

**Relevance Tier:** ⚡ Directly Related

**Key Findings:**
- EyeGAN generates eye images in the style of a target domain while inheriting annotations from a source domain (UnityEyes synthetic masks), using ternary masks as domain-independent gaze proxies[^62]
- Using EyeGAN-generated images for training leads to superior performance vs. SimGAN and CycleGAN on eye region segmentation, pupil localization, and gaze estimation[^63]
- The approach directly enables subject-specific synthesis: given 3–5 real frames of a new subject, EyeGAN generates an arbitrarily large set of annotated training frames with that subject's skin/iris style[^64]

**Links:** WACV 2020 proceedings · PDF: escholarship.org/content/qt3vk9w8k9

**Applicability to SegUNET:** EyeGAN provides the most practical path to synthetic subject variation — feed SegUNET's existing LabelMe masks as ternary proxies, generate subject-specific images for under-represented subjects, and use them to close the Fold 5 generalization gap.

***

### Seg2Eye: Content-Consistent Generation of Realistic Eyes with Style

**Citation:** Bühler, Park, De Mello, Zhang, Hilliges. *Content-Consistent Generation of Realistic Eyes with Style.* ICCVW 2019 (winner of OpenEDS Synthetic Eye Generation Challenge).[^65][^66]

**Relevance Tier:** ⚡ Directly Related

**Key Findings:**
- Synthesizes person-specific eye images from segmentation masks with the style of a target individual using only a few reference images — exactly the per-subject augmentation SegUNET needs[^65]
- Won the OpenEDS Synthetic Eye Generation Challenge at ICCV 2019; code and model weights are publicly available[^66]
- Multi-scale injection of style and content information produces higher semantic consistency (mIoU) between source masks and generated images compared to single-scale approaches[^66]

**Links:** arXiv via NVIDIA Research · Code: https://github.com/mcbuehler/Seg2Eye

**Applicability to SegUNET:** Seg2Eye is immediately applicable for augmenting under-represented subjects — input the existing per-subject LabelMe masks, generate 200–500 synthetic IR images per subject, and use for training without additional annotation.

***

### UnityEyes 2: Open Source Synthetic Eye Generation for Camera-Based Eye Tracking

**Citation:** (University of Illinois). *UnityEyes 2: Open Source Robust Synthetic Eye Generation for Camera-Based Eye Tracking with Machine Learning.* 2025.[^67]

**Relevance Tier:** ⚡ Directly Related

**Key Findings:**
- UnityEyes 2 supports customization of eye pose, camera intrinsics/extrinsics, multi-camera setups, and varied lighting conditions — with a GUI for rapid prototyping of data distributions[^67]
- Shows that camera-specific synthetic datasets (matching the real camera FOV and mounting) transfer much better to real-world deployment than generic-viewpoint synthetic data[^67]
- Directly addresses SegUNET's sim-to-real gap: customizing the synthetic dataset to Bigscreen Beyond's specific IR camera geometry is a first-class feature

**Links:** Publication page: experts.illinois.edu/en/publications/unityeyes-2

**Applicability to SegUNET:** Build a Bigscreen Beyond-specific UnityEyes 2 data distribution matching camera FOV, lens, and IR wavelength — use it to augment BrightSpot (specular highlight) and lash density variation that are not well-covered in the current 1,500-frame pool.

***

### IR Image Augmentation and Domain Randomization for Near-Eye Imagery

**Citation:** Odinokikh. *Iris Segmentation in Challenging Conditions.* IDP 2018.[^68]

**Relevance Tier:** ⚡ Directly Related

**Key Findings:**
- Catalogs the specific degradation modes in near-eye IR imagery: under/over-exposure, poor contrast, glass reflection, eyelid occlusion, and gaze-away positions[^68]
- Each mode maps directly to a SegUNET class failure: overexposure bleaches the BrightSpot; under-illumination washes out the Pupil/Iris boundary; occlusion is the primary UpperLid failure trigger[^68]
- Recommends explicit augmentation for each degradation type (simulated overexposure, synthetic specular highlights, random occlusion masks over the lid region)

**Links:** PDF at machinelearning.ru

**Applicability to SegUNET:** Use this paper's degradation taxonomy to design an IR-specific augmentation pipeline: add simulated specular highlights (BrightSpot augmentation), random eyelid occlusion strips (UpperLid augmentation), and exposure jitter — beyond the current contrast/geometric-only approach.

***

## Boundary Quality Metric

### Boundary IoU: Improving Object-Centric Image Segmentation Evaluation

**Citation:** Cheng, Girshick, Dollár, Berg, Kirillov. *Boundary IoU: Improving Object-Centric Image Segmentation Evaluation.* CVPR 2021.[^69][^70]

**Relevance Tier:** **Near Future**

**Key Findings:**
- Standard Mask IoU is insensitive to boundary quality — a mask with blobby boundaries can score 89% Mask IoU while Boundary IoU correctly measures it at 69%[^70]
- Boundary IoU computes IoU over the set of pixels within distance *d* from the predicted and ground truth contours — more sensitive to large-object boundary errors and not over-penalizing small-object errors[^69]
- Proposes Boundary AP and Boundary PQ as drop-in replacements for instance and panoptic segmentation evaluation[^69]

**Links:** arXiv:2103.16562 · Code integrated in panopticapi

**Applicability to SegUNET:** Add Boundary IoU per class to SegUNET's evaluation suite — it will quantify the BrightSpot and Pupil boundary quality that standard mIoU masks, and flag UpperLid contour degradation that the current IoU 0.665 does not fully capture.

***

## Architecture Comparison Table

| Model | Params | FLOPs | mIoU (ADE20K) | Mobile Latency | Key Advantage for SegUNET |
|---|---|---|---|---|---|
| EfficientNet-B0 + UNet (current) | ~5.3M | ~0.39G | — | ~9ms (est. GPU) | Strong baseline, well-integrated in smp |
| RITnet (DenseNet-UNet) | <1M | <0.1G | — | <3ms GPU (300Hz) [^2] | Sub-1 MB deployment target; 4-class only |
| SegFormer-B0 | 3.8M | 8.4G | 37.4% | 48 FPS (V100) [^4] | Best-in-class mIoU for real-time tier |
| TopFormer-T | 1.4M | 0.5G | 33.4% | ARM real-time [^8] | Lowest params, ARM-benchmarked |
| TopFormer-S | 3.1M | 1.2G | 36% | <81ms ARM [^9] | Best accuracy/compute at ARM latency |
| EfficientFormer-L1 | 12.3M | 1.3G | — | 1.6ms iPhone12 [^10] | MobileNet speed, transformer accuracy |
| MobileViT-XXS | 1.3M | 0.5G | — | — | Apple pretrained seg weights available [^15] |
| MobileViTv3-0.5 | ~2M | ~0.5G | ↑2.1% vs MViTv2 | — | Best MobileViT seg mIoU [^16] |

***

## Cross-Reference: Papers Addressing Multiple Questions

| Paper | Q1 Arch | Q2 Data | Q3 Domain | Q4 Labels | Q5 Gaze | Q6 Inference | Q7 Aug |
|---|---|---|---|---|---|---|---|
| RITnet (Chaudhary 2019) | ✓ | — | — | — | — | ✓ | — |
| OpenEDS2020 (Palmero 2020) | — | ✓ | — | — | ✓ | — | — |
| TEyeD (Fuhl 2021) | — | ✓ | ✓ | — | ✓ | — | — |
| NVGaze (Kim 2019) | — | ✓ | — | — | ✓ | ✓ | ✓ |
| RIT-Eyes (Nair 2020) | — | ✓ | — | — | — | — | ✓ |
| CondSeg (Jia 2024) | ✓ | — | — | — | ✓ | — | — |
| EllSeg (Kothari 2020) | ✓ | — | — | — | ✓ | — | — |
| ENCORE (Ghamsarian 2025) | — | — | — | ✓ | — | — | — |
| CSL (Liu 2025) | — | — | — | ✓ | — | — | — |
| EyeGAN (Kaur 2020) | — | — | ✓ | — | — | — | ✓ |
| Seg2Eye (Bühler 2019) | — | — | ✓ | — | — | — | ✓ |
| Domain Adaptation (Shen 2020) | — | — | ✓ | ✓ | — | — | — |

***

## Recommended Immediate Actions for SegUNET

Based on the directly-related tier, the following actions are ranked by expected mIoU gain per implementation hour:

1. **Synthetic augmentation with RIT-Eyes or Seg2Eye** — address UpperLid class imbalance and subject variation without requiring new real annotations. Highest expected gain for weakest class (IoU 0.665).
2. **ENCORE integration in the label loop** — replace static approve/reject with ACT-calibrated per-class confidence thresholds; prevents feedback loop corruption especially for BrightSpot and UpperLid.
3. **Encoder swap experiment: TopFormer-S vs. EfficientNet-B0** — same UNet decoder, identical training config; expected improvement in cross-subject mIoU from transformer global context.
4. **Domain adversarial head for subject generalization** — add a DANN-style discriminator on subject identity following the OpenEyes 2020 DA paper's +6.6% mIoU result with only 200 target labels.
5. **Add Boundary IoU to evaluation** — no training cost; immediately reveals whether UpperLid boundary quality is improving independently of bulk mIoU.
6. **TEyeD pretraining** — use eyelid-labeled HMD frames from TEyeD as pretrain source before fine-tuning on SegUNET's proprietary data; directly addresses UpperLid's persistent weakness.

---

## References

1. [[PDF] RITnet: Real-time Semantic Segmentation of the Eye for Gaze ...](https://www.semanticscholar.org/paper/RITnet:-Real-time-Semantic-Segmentation-of-the-Eye-Chaudhary-Kothari/330dc33a7f5b3ede14a01b9bb03d2e88f9d7a926) - The RITnet model, which is a deep neural network that combines U-Net and DenseNet, achieves 95.3% ac...

2. [Real-time Semantic Segmentation of the Eye for Gaze Tracking - arXiv](https://arxiv.org/abs/1910.00694) - RITnet is under 1 MB and achieves 95.3\% accuracy on the 2019 OpenEDS Semantic Segmentation challeng...

3. [RITnet: Real-time Semantic Segmentation of the Eye for Gaze Tracking](https://ar5iv.labs.arxiv.org/html/1910.00694) - Accurate eye segmentation can improve eye-gaze estimation and support interactive computing based on...

4. [[PDF] SegFormer: Simple and Efficient Design for Semantic Segmentation ...](https://proceedings.neurips.cc/paper_files/paper/2021/file/64f1f27bf1b4ec22924fd0acb550c235-Paper.pdf) - We present SegFormer, a simple, efficient yet powerful semantic segmentation framework which unifies...

5. [SegFormer: Simple and Efficient Design for Semantic Segmentation ...](https://arxiv.org/abs/2105.15203) - We present SegFormer, a simple, efficient yet powerful semantic segmentation framework which unifies...

6. [[PDF] SegFormer: Simple and Efficient Design for Semantic Segmentation ...](https://arxiv.org/pdf/2105.15203.pdf)

7. [SegFormer: Vision Transformer for Segmentation - Emergent Mind](https://www.emergentmind.com/topics/vision-transformer-segformer) - SegFormer is a robust semantic segmentation architecture that fuses a hierarchical Transformer encod...

8. [TopFormer: Token Pyramid Transformer for Mobile Semantic ...](https://github.com/hustvl/TopFormer) - The proposed TopFormer takes Tokens from various scales as input to produce scale-aware semantic fea...

9. [[PDF] Token Pyramid Transformer for Mobile Semantic Segmentation - arXiv](https://arxiv.org/pdf/2204.05525.pdf) - The proposed TopFormer takes tokens from different scales as input, and pools the tokens to the very...

10. [[2206.01191] EfficientFormer: Vision Transformers at MobileNet Speed](https://arxiv.org/abs/2206.01191) - Our work proves that properly designed transformers can reach extremely low latency on mobile device...

11. [[PDF] EfficientFormer: Vision Transformers at MobileNet Speed](https://proceedings.neurips.cc/paper_files/paper/2022/file/5452ad8ee6ea6e7dc41db1cbd31ba0b8-Paper-Conference.pdf) - Extensive experiments on image classification, object detection, and segmentation tasks show that Ef...

12. [EfficientFormer: Vision Transformers at MobileNet](https://arxiv.org/pdf/2206.01191.pdf)

13. [Light-weight, General-purpose, and Mobile-friendly Vision Transformer](https://arxiv.org/abs/2110.02178) - We introduce MobileViT, a light-weight and general-purpose vision transformer for mobile devices. Mo...

14. [Light-weight, General-purpose, and Mobile-friendly Vision Transformer](https://openreview.net/forum?id=vh-0sUt8HlG) - This paper introduces a lightweight and general-purpose vision transformer, termed MobileViT, for mo...

15. [README.md · Matthijs/deeplabv3-mobilevit-small at main](https://huggingface.co/Matthijs/deeplabv3-mobilevit-small/blob/main/README.md) - We’re on a journey to advance and democratize artificial intelligence through open source and open s...

16. [[PDF] MOBILEVITV3: MOBILE-FRIENDLY VISION TRANS - OpenReview](https://openreview.net/pdf?id=wtr-9AKxCI5)

17. [[PDF] Edge-Guided Near-Eye Image Analysis for Head Mounted Displays](https://zhimin-wang.github.io/publication/ismar_2021/pages/pdf/wang21_ISMAR.pdf) - In this paper, we propose a novel near-eye image analysis method that estimates pupil and iris ellip...

18. [OpenEDS: Open Eye Dataset](https://arxiv.org/pdf/1905.03702v2.pdf)

19. [[2005.03876] OpenEDS2020: Open Eyes Dataset](https://arxiv.org/abs/2005.03876) - We present the second edition of OpenEDS dataset, OpenEDS2020, a novel dataset of eye-image sequence...

20. [OpenEDS2020: Open Eyes Dataset](https://www.academia.edu/119894818/OpenEDS2020_Open_Eyes_Dataset) - We present the second edition of OpenEDS dataset, OpenEDS2020, a novel dataset of eye-image sequence...

21. [TEyeD: Over 20 million real-world eye im...](https://axi.lims.ac.uk/paper/2102.02115)

22. [[PDF] TEyeD: Over 20 million real-world eye images with Pupil, Eyelid ...](https://www.hci.uni-tuebingen.de/assets/pdf/publications/fuhl2021teyed.pdf)

23. [NVGaze: An Anatomically-Informed Dataset for Low-Latency, Near ...](https://research.nvidia.com/publication/2019-05_nvgaze-anatomically-informed-dataset-low-latency-near-eye-gaze-estimation) - We create two datasets satisfying these criteria for near-eye gaze estimation under infrared illumin...

24. [[PDF] An Anatomically-Informed Dataset for Low-Latency, Near-Eye Gaze ...](https://openreview.net/pdf?id=VCvUq-f60r) - NVGaze: An Anatomically-Informed Dataset, for. Low-Latency, Near-Eye Gaze ... Each image is labeled ...

25. [RIT-Eyes: Rendering of near-eye images for eye-tracking applications](https://arxiv.org/abs/2006.03642) - We introduce a synthetic eye image generation platform that improves upon previous work by adding fe...

26. [Temporal RIT-Eyes: From Real Infrared Eye-Images to Synthetic ...](https://discovery.researcher.life/article/temporal-rit-eyes-from-real-infrared-eye-images-to-synthetic-sequences-of-gaze-behavior/d61348c9d57a3b14b25aad4a954e99e9) - We present Temporal RIT-Eyes, a Blender pipeline that draws data from real eye videos for the render...

27. [SynthesEyes - University of Cambridge](https://www.cl.cam.ac.uk/research/rainbow/projects/syntheseyes/) - We render photorealistic images of eyes for use as training data. We prepare our dynamic eye region ...

28. [Labelled Pupils in the Wild (LPW) - MPI-INF](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/gaze-based-human-computer-interaction/labelled-pupils-in-the-wild-lpw) - The videos in our dataset were recorded from 22 participants in everyday locations at about 95 FPS u...

29. [A dataset for studying pupil detection in unconstrained environments](https://arxiv.org/abs/1511.05768) - We present labelled pupils in the wild (LPW), a novel dataset of 66 high-quality, high-speed eye reg...

30. [LPW | Collaborative Artificial Intelligence](https://www.collaborative-ai.org/research/datasets/LPW/) - We present labelled pupils in the wild (LPW), a novel dataset of 66 high-quality, high-speed eye reg...

31. [[PDF] Domain Adaptation for Eye Segmentation - OpenEyes 2020](https://openeyes-workshop.github.io/downloads/openeyes2020_yiru_shen_domain_adaptation_for_eye_segmentation.pdf) - Abstract. Domain adaptation (DA) has been widely investigated as a framework to alleviate the labori...

32. [Cross-domain adaptation and geometric data synthesis for near-eye ...](https://repositories.lib.utexas.edu/items/f472f814-63b9-4b44-b1b9-be9197df8c17) - This work contributes a new data adaption approach of combining the comparably economical annotated ...

33. [cvlab-stonybrook/few-shot-scanpath - GitHub](https://github.com/cvlab-stonybrook/few-shot-scanpath) - Experiments on multiple eye-tracking datasets demonstrate that our method excels in FS-PSP settings ...

34. [[PDF] Few-shot Personalized Scanpath Prediction - CVF Open Access](https://openaccess.thecvf.com/content/CVPR2025/papers/Xue_Few-shot_Personalized_Scanpath_Prediction_CVPR_2025_paper.pdf) - A personalized model for scanpath prediction provides insights into the visual preferences and atten...

35. [Redefining Thresholding for Semi-Supervised Semantic Segmentation](https://arxiv.org/html/2505.07691v1) - We propose Adaptive Confidence Thresholding (ACT), a dynamic threshold adjustment mechanism that con...

36. [[PDF] Feedback-Driven Pseudo-Label Reliability Assessment - arXiv](https://arxiv.org/pdf/2505.07691.pdf) - We propose Adaptive Confidence Thresholding (ACT), a dynamic threshold adjustment mechanism that con...

37. [Feedback-Driven Pseudo-Label Reliability Assessment: Redefining Thresholding for Semi-Supervised Semantic Segmentation](https://www.arxiv.org/abs/2505.07691) - Semi-supervised learning leverages unlabeled data to enhance model performance, addressing the limit...

38. [When Confidence Fails: Revisiting Pseudo-Label Selection in Semi ...](https://arxiv.org/abs/2509.16704) - We propose Confidence Separable Learning (CSL) to address these limitations. CSL formulates pseudo-l...

39. [CSL - Semi supervised semantic segmentation in ICCV 2025 - GitHub](https://github.com/PanLiuCSU/CSL) - CSL proposes a novel approach that formulates pseudo-label selection as a convex optimization proble...

40. [Revisiting Pseudo-Label Selection in Semi-supervised Semantic ...](https://arxiv.org/html/2509.16704v1) - We propose the Confidence Separable Learning (CSL) framework to address the challenge of insufficien...

41. [Integrating Semi-Supervised and Active Learning for Semantic ...](https://arxiv.org/abs/2501.19227) - In this paper, we propose a novel active learning approach integrated with an improved semi-supervis...

42. [How to Overcome Confirmation Bias in Semi-Supervised ... - MCML](https://mcml.ai/publications/ghf+23/) - The rise of strong deep semi-supervised methods raises doubt about the usability of active learning ...

43. [Find Label Errors in Semantic Segmentation Datasets - cleanlab](https://docs.cleanlab.ai/v2.7.1/tutorials/segmentation.html) - This 5-minute quickstart tutorial shows how you can use cleanlab to find potentially mislabeled imag...

44. [cleanlab now supports all major ML tasks — including Regression ...](https://cleanlab.ai/blog/learn/cleanlab-2.5) - Errors found in regression dataset. Learn how to apply it to your own regression data within 5 minut...

45. [EllSeg: An Ellipse Segmentation Framework for Robust Gaze Tracking](https://arxiv.org/abs/2007.09600) - We propose training a convolutional neural network to directly segment entire elliptical structures ...

46. [EllSeg: An Ellipse Segmentation Framework for Robust Gaze Tracking](https://www.semanticscholar.org/paper/EllSeg:-An-Ellipse-Segmentation-Framework-for-Gaze-Kothari-Chaudhary/73cf09982fafb59eb07da7d9900ee2c0aac56d38) - This work proposes training a convolutional neural network to directly segment entire elliptical str...

47. [RSKothari/EllSeg - GitHub](https://github.com/RSKothari/EllSeg) - Ellipse fitting, an essential component in pupil or iris tracking based video oculography, is perfor...

48. [Ellipse Estimation of Pupil and Iris via Conditioned Segmentation](https://arxiv.org/html/2408.17231v1) - A novel method CondSeg to estimate elliptical parameters of pupil/iris directly from segmentation la...

49. [Ellipse Estimation of Pupil and Iris via Conditioned Segmentation](https://arxiv.org/abs/2408.17231) - A novel method CondSeg to estimate elliptical parameters of pupil/iris directly from segmentation la...

50. [CondSeg: Ellipse Estimation of Pupil and Iris via Conditioned Segmentation](https://arxiv.org/html/2408.17231)

51. [DeepVOG: Open-source pupil segmentation and gaze estimation in ...](https://pubmed.ncbi.nlm.nih.gov/31176683/) - Our proposed FCNN-based pupil segmentation framework is accurate, robust and generalizes well to new...

52. [DeepVOG: Open-source pupil segmentation and gaze estimation in ...](https://www.sciencedirect.com/science/article/pii/S0165027019301578) - We propose novel tools for video-oculography powered by deep-learning. Robust pupil segmentation usi...

53. [GitHub - pydsgz/DeepVOG: Pupil segmentation and gaze estimation ...](https://github.com/pydsgz/DeepVOG) - DeepVOG is a framework for pupil segmentation and gaze estimation based on a fully convolutional neu...

54. [3DeepVOG: An Open-Source Framework for Real-Time, Accurate 3D Gaze Tracking with Deep Learning - PubMed](https://pubmed.ncbi.nlm.nih.gov/41658975/) - 3DeepVOG enables accurate, quantitative eye movement tracking across three dimensions under diverse ...

55. [[PDF] Evaluating two-stage gaze estimation using eye image segmentation](https://repositum.tuwien.at/bitstream/20.500.12708/188367/1/Fuerst%20Patrick%20-%202023%20-%20Evaluating%20two-stage%20gaze%20estimation%20using%20eye%20image...pdf) - the pupil ellipse should be used to build the eye model or quantify the gaze estimation quality. 3.4...

56. [[PDF] Accelerated Real-Time Face Recognition and Segmentation with ...](https://jisem-journal.com/index.php/journal/article/download/5987/3236/11657) - TensorRT is an optimization tool for inference that performs six types of optimizations to reduce th...

57. [[PDF] arXiv:2501.17343v1 [cs.CV] 28 Jan 2025](https://arxiv.org/pdf/2501.17343.pdf) - In this study, we introduce a real post-training quantization (PTQ) framework that success- fully im...

58. [[PDF] TensorRT Implementations of Model Quantization on Edge SoC](https://userweb.cs.txstate.edu/~k_y47/webpage/pubs/mcsoc23.pdf) - Quantization in TensorRT involves mapping the high-precision floating-point values in a model to low...

59. [Float8 (FP8) Quantized LightGlue in TensorRT with NVIDIA Model ...](https://fabio-sim.github.io/blog/fp8-quantized-lightglue-tensorrt-nvidia-model-optimizer/) - FP8 quantization via NVIDIA Model Optimizer shrinks TensorRT engines for SuperPoint + LightGlue and ...

60. [Prompt-In-the-Loop Distillation for On-Device Deployment of SAM](https://arxiv.org/html/2312.06660v2) - Our approach involves distilling the original ViT-based SAM image encoder into a purely CNN-based ar...

61. [EyeGAN: Gaze-Preserving, Mask-Mediated Eye Image Synthesis](https://openaccess.thecvf.com/content_WACV_2020/papers/Kaur_EyeGAN_Gaze-Preserving_Mask-Mediated_Eye_Image_Synthesis_WACV_2020_paper.pdf)

62. [EyeGAN: Gaze–Preserving, Mask–Mediated Eye Image Synthesis](https://escholarship.org/content/qt3vk9w8k9/qt3vk9w8k9_noSplash_92708aa4704e0f9af5f11b68242d98e5.pdf?t=q3yo42)

63. [UC Santa Cruz](https://escholarship.org/content/qt3vk9w8k9/qt3vk9w8k9.pdf?t=q3yo42)

64. [Subject Guided Eye Image Synthesis with Application to Gaze Redirection](https://pmc.ncbi.nlm.nih.gov/articles/PMC8040934/pdf/nihms-1648872.pdf)

65. [Content-Consistent Generation of Realistic Eyes with Style | Research](https://research.nvidia.com/publication/2019-11_content-consistent-generation-realistic-eyes-style) - In this work, we synthesize person-specific eye images that satisfy a given semantic segmentation ma...

66. [mcbuehler/Seg2Eye: Official implementation of "Content-Consistent ...](https://github.com/mcbuehler/Seg2Eye) - In this work, we synthesize person-specific eye images that satisfy a given semantic segmentation ma...

67. [UnityEyes 2: Open source robust synthetic eye generation ...](https://experts.illinois.edu/en/publications/unityeyes-2-open-source-robust-synthetic-eye-generation-for-camer/)

68. [[PDF] Gleb Odinokikh - Iris Segmentation in Challenging Conditions - IDP18](http://www.machinelearning.ru/wiki/images/a/a2/OdinokikhIDP18.pdf)

69. [Boundary IoU: Improving Object-Centric Image Segmentation ... - arXiv](https://arxiv.org/abs/2103.16562) - We present Boundary IoU (Intersection-over-Union), a new segmentation evaluation measure focused on ...

70. [Boundary IoU for Image Segmentation Evaluation | PDF - Scribd](https://www.scribd.com/document/595876239/Boundary-IOU) - This document proposes a new segmentation evaluation metric called Boundary IoU that is more sensiti...

