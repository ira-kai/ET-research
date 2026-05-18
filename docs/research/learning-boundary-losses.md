# Learning File: Boundary-Aware Losses for SegUNET

A hands-on guide to the thin-structure and boundary-aware loss functions
in the corpus, with implementation notes for SegUNET's 7-class eye segmentation.

---

## The Problem

SegUNET's current Dice + Focal + CE loss treats all pixels equally.
Boundary pixels (especially UpperLid at 0.586 boundary mIoU and BrightSpot
at 0.633) are underweighted because they represent a tiny fraction of total
pixels. Previous attempts at hard boundary upweighting caused gradient
conflicts with Dice loss.

---

## Loss Landscape: From Simplest to Most Powerful

### Level 1: soft-clDice (CVPR 2021)
**What it does:** Computes Dice overlap between predicted and GT *skeletons*
rather than full masks. Uses differentiable soft skeletonization via
iterative min/max pooling.

**Limitation:** Binary only. Must add one clDice term per class:
```
L = L_base + λ₁·clDice(UpperLid) + λ₂·clDice(LowerLid) + λ₃·clDice(BrightSpot)
```

**GPU cost:** ~5x baseline (the iterative pooling is expensive).

**When to use:** Quick experiment to test if topology preservation helps
UpperLid connectivity. If it helps, switch to Skeleton Recall.

**Code:** `repos/clDice/cldice_loss/pytorch/cldice.py`

---

### Level 2: clCE (MICCAI 2024)
**What it does:** Replaces Dice in clDice with cross-entropy. More
numerically stable for small regions.

**Why it matters for SegUNET:** BrightSpot is often <50 pixels. Dice-based
losses are unstable when the denominator is tiny. clCE gives stable
gradients on the BrightSpot skeleton.

**Code:** `repos/centerline_CE/nnUNet/losses/cldice_loss.py`

---

### Level 3: Skeleton Recall Loss (ECCV 2024) — RECOMMENDED
**What it does:** Precomputes skeletons on CPU during data loading.
At training time, computes soft recall of the model's prediction
restricted to the skeleton tube (dilated skeleton).

**Why it's the best choice:**
1. **Multi-class native** — one loss term handles all 7 classes simultaneously
2. **90% cheaper than clDice** — skeleton computed offline, no iterative pooling
3. **Recall-focused** — penalizes missing skeleton pixels (disconnections)
   without penalizing extra predictions (false positives handled by Dice)

**Implementation pattern for SegUNET:**
```
L_total = Dice + Focal + λ_skel · SkeletonRecall(pred, skeleton_tubes)
```
Where `skeleton_tubes` are precomputed per-class skeleton masks dilated by
2-3 pixels, generated during data loading on CPU.

**Code:** `repos/Skeleton-Recall/nnunetv2/training/loss/compound_losses.py`

---

### Level 4: FocusSDF (arXiv 2025)
**What it does:** Computes signed distance field (SDF) from each class
boundary and uses exponential decay to weight boundary-proximal pixels higher.

**Why it fixes the gradient conflict:** Hard boundary upweighting creates
a discontinuity — pixels at distance 0 get weight W, pixels at distance 1
get weight 1. This conflicts with Dice's smooth gradient landscape.
FocusSDF's exponential decay `w(d) = e^{-α|d|}` is smooth, compatible
with Dice + Focal.

**When to use:** If Skeleton Recall improves topology but boundary mIoU
is still low, add FocusSDF for smooth boundary emphasis.

---

## Decision Tree

```
Start: Current Dice + Focal + CE
  │
  ├─ Want quick topology test?
  │   → Add soft-clDice on UpperLid only (Level 1)
  │
  ├─ Want production boundary loss?
  │   → Add Skeleton Recall Loss (Level 3) — multi-class, cheap
  │
  ├─ BrightSpot still unstable?
  │   → Replace clDice with clCE for BrightSpot (Level 2)
  │
  └─ Boundary mIoU still plateaued?
      → Add FocusSDF smooth boundary weighting (Level 4)
```

---

## Reading Order for Implementation

1. `repos/Skeleton-Recall/readme.md` → understand the API
2. `repos/Skeleton-Recall/nnunetv2/training/loss/compound_losses.py` → loss implementation
3. `repos/clDice/cldice_loss/pytorch/soft_skeleton.py` → understand soft skeletonization (for comparison)
4. `papers/focussdf-2025.pdf` → SDF weighting math (pages 3-5)

---

## Key Numbers to Remember

| Loss | Multi-class? | GPU overhead | Best for |
|------|-------------|-------------|----------|
| Dice + Focal (current) | Yes | 1x | Region overlap |
| soft-clDice | Binary per-class | ~5x | Quick topology test |
| clCE | Binary per-class | ~5x | Small classes (BrightSpot) |
| Skeleton Recall | Yes native | ~1.1x | **Production use** |
| FocusSDF | Yes | ~2x | Smooth boundary emphasis |
