# Analysis of the Initial RGB-D Mass Prediction Architecture

## 1. Initial Architecture

The model extends Mask R-CNN (Detectron2) for simultaneous detection,
segmentation, and mass prediction of tomato bunches from RGB-D images.

Depth enters the model through **two paths**:

**Path 1 — Backbone (shared with detection/segmentation):**
```
4-channel input [B, G, R, Depth] → ResNet-50 conv1 (modified: 4 input channels)
→ ResNet stages → FPN → shared features for all heads
```

**Path 2 — Mass head (dedicated depth channel):**
```
Raw depth → cached separately → ROI-pooled to 14x14
→ concatenated as channel 257 to FPN features → mass head convolutions
```

**Mass head architecture:**
```
Input: (N, 257, 14, 14) — 256 FPN channels + 1 depth channel
→ 4x Conv3x3 (257→256→256→256→256)
→ ConvTranspose2d (2x upsample)
→ AdaptiveAvgPool2d (to 4x4)
→ FC(4096, 128) → ReLU → FC(128, 1) → log(mass)
Loss: MSE in log-space
```

---

## 2. Baseline Performance (Before Any Fixes)

| Metric      | Value   |
|-------------|---------|
| Mass MAE    | 0.112 kg |
| Mass MAPE   | 14.8%   |
| Mass R²     | 0.870   |
| Segm AP     | 98.6%   |

These numbers look good on paper. But testing revealed the model was not
using depth at all.

---

## 3. How I Tested Depth Usage

### Test 1: Depth Sensitivity Test (test_depth_sensitivity.py)

I ran inference on the same images under different depth conditions:

| Condition       | Description                              |
|-----------------|------------------------------------------|
| **Real depth**  | Original depth maps from the sensor      |
| **Zero depth**  | All depth values set to zero             |
| **Const 1m**    | All depth values set to 1000mm           |
| **Const 3m**    | All depth values set to 3000mm           |
| **Random depth**| Random depth values                      |

**Result:** Predictions changed by only **3-4%** between real depth and
zero/constant depth. The model produces nearly identical mass predictions
regardless of what depth information it receives.

### Test 2: Correlation Analysis (inference_analysis.py)

I computed correlations between predicted mass and input features:

| Correlation              | Value  |
|--------------------------|--------|
| pred_mass vs mask_area   | r=0.49 |
| pred_mass vs mean_depth  | r=0.02 |

The model's predictions correlate with mask area (visual size) but have
essentially **zero correlation with depth**.

### Test 3: Gradient Analysis

Examined gradient magnitudes for the first conv layer weights:

| Weight group               | Gradient magnitude |
|----------------------------|--------------------|
| FPN channels (0-255)       | ~0.0012            |
| Depth channel (256)        | ~0.00001           |

The depth channel receives **100x smaller gradients** than FPN channels,
meaning the network barely updates its depth-processing weights during
training.

---

## 4. Bugs Discovered

### Bug 1: Depth Boundary Artifacts (validity threshold)
- **Location:** `rcnn.py`, `_cache_depth()`
- **Problem:** Depth values near object edges from the RealSense sensor
  are unreliable (noisy, zero, or very small values). These invalid pixels
  were being included in the depth cache, and during ROI pooling's bilinear
  interpolation, they contaminated neighboring valid pixels.
- **Fix:** Set validity threshold to >200mm. Invalid pixels are set to 0
  in normalized space (which maps to the dataset mean depth of 915mm after
  un-normalization), acting as a neutral fill value.

### Bug 2: FPN/Depth Scale Mismatch in Mass Head
- **Location:** `roi_heads.py`, `_concat_cached_depth_to_features()`
- **Problem:** When the depth channel is concatenated to FPN features before
  the mass head, the two have very different magnitudes:
  - FPN features: mean≈0, **std≈4.0**, range [-25, +25]
  - Depth channel: mean≈0, **std≈0.5**, range [-1.6, +2.2]

  The first conv layer (mass_fcn1) processes all 257 channels with the same
  kernel. With depth being 8x smaller, the conv layer naturally learns to
  ignore it — the FPN channels dominate the activations.
- **Fix:** Applied 8x scaling to the depth channel after concatenation,
  matching its std to the FPN features' std (~4.0).

### Bug 3: RGB/Depth Normalization Mismatch in Backbone
- **Location:** `train.py`, PIXEL_MEAN/PIXEL_STD configuration
- **Problem:** The backbone normalizes all 4 input channels using:
  ```
  PIXEL_MEAN = [103.530, 116.280, 123.675, 915.043]
  PIXEL_STD  = [1.0,     1.0,     1.0,     411.683]
  ```
  After normalization:
  - RGB channels: std ≈ 60 (because STD=1.0 doesn't truly normalize)
  - Depth channel: std ≈ 1.0 (properly normalized by 411.683)

  This creates a **60x activation imbalance** in the backbone. Conv1 weights
  for the depth channel produce 60x smaller activations than RGB weights,
  so depth features propagating through ResNet and into FPN are negligible.
- **Status:** Identified but NOT fixed. Changing backbone normalization would
  require re-initializing the entire backbone, which is risky. Instead,
  focused on fixing the mass head path (Bug 2).

### Bug 4: Mass Evaluator Matching Error
- **Location:** `mass_evaluator_mean.py`
- **Problem:** The evaluator matched predictions to ground truth by index
  order, which broke when detections came in a different order than
  annotations.
- **Fix:** Switched to IoU-based matching (match each prediction to the
  ground truth box with highest IoU overlap).

---

## 5. Root Cause: Why the Model Ignores Depth

Even after fixing bugs 1, 2, and 4, the model still does not use depth
effectively. The fundamental reasons are:

### 5.1 Capacity Imbalance (256 vs 1)
The mass head receives 256 FPN channels and only 1 depth channel. The 4
conv layers (256 filters each) have 256x more signal from FPN than from
depth. Even with matched scales (after the 8x fix), a single channel
cannot compete with 256 channels of rich visual features.

### 5.2 Memorization Shortcut (36 clusters)
The training set contains only **36 unique tomato bunches**, each
photographed ~23 times. The 256 FPN channels encode enough visual detail
(leaf patterns, fruit arrangement, stem shape) to uniquely identify each
cluster. The model learns a **lookup table**:

```
"this visual pattern" → 0.765 kg
"that visual pattern" → 0.608 kg
...
```

This memorization achieves low training loss without requiring any
understanding of depth, area, or physics. Since memorization is a
lower-loss strategy than learning the noisy depth-to-mass relationship,
gradient descent naturally converges to it.

### 5.3 ROI Pooling Destroys Scale Information
All detected objects are ROI-pooled to the same 14x14 spatial resolution.
A small nearby tomato bunch and a large far-away bunch produce identical
spatial feature maps. The absolute depth value (channel 256) could help
recover scale, but it's 1 channel among 257 — easily ignored.

### 5.4 Backbone Depth Path is Dead
Due to the 60x normalization mismatch (Bug 3), depth features in the FPN
are negligible. Path 1 (backbone) contributes essentially nothing from
depth. Only Path 2 (direct concatenation to mass head) has any chance of
providing depth information, and it suffers from the 256:1 imbalance.

---

## 6. What I Figured Out: The Physical Area Proxy

To understand whether depth information is actually useful for mass
prediction (independent of the model's architecture), I computed a
physics-based proxy using ground-truth masks and raw depth maps.

### Pinhole Camera Model

A single pixel at depth Z meters covers a physical area of Z²/(FX×FY)
square meters. Summing over all N mask pixels:

```
area_m² = Σ(Z_i²) / (FX × FY)  ≈  N_pixels × Z_mean² / (FX × FY)
```

This is **distance-invariant**: pixel count scales as 1/Z² (farther =
fewer pixels), and multiplying by Z² cancels this out. The result is the
true physical cross-sectional area of the object in square meters.

Since mass ∝ volume ∝ R³ and area_m² ∝ R², the relationship is:
```
log(mass) ≈ (3/2) × log(area_m²) + constant
```

### Proxy Results (mean_volume.py)

| Metric                                    | area_m² × depth | area_m² alone |
|-------------------------------------------|-----------------|---------------|
| Per-instance correlation (linear)         | 0.788           | **0.907**     |
| Per-instance correlation (log-log)        | 0.818           | **0.882**     |
| **Per-cluster mean correlation (linear)** | 0.944           | **0.947**     |
| Per-cluster mean correlation (log-log)    | 0.940           | **0.943**     |
| Median within-cluster CV                  | 0.180           | **0.097**     |
| 90th percentile CV                        | 0.310           | **0.228**     |

Key findings:
- **area_m² alone is the better proxy** — adding depth multiplication
  re-introduces distance dependence and increases noise (CV: 18% → 9.7%)
- **r = 0.947 at cluster level** proves depth IS highly informative for mass
- **The problem is entirely architectural** — the information exists in the
  data but the model cannot extract it

### Why area_m² × depth Was Wrong

I initially multiplied area_m² by mean_depth, thinking it would approximate
volume. But since area_m² is already distance-invariant (Z² cancels 1/Z²),
multiplying by Z again makes the proxy proportional to Z — reintroducing
the very distance dependence that the Z² conversion was meant to remove.
The same tomato bunch at 0.45m and 0.65m would get different proxy values,
not because of its mass, but because of camera distance.

---

## 7. Summary

| Finding | Evidence |
|---------|----------|
| Detection and segmentation work perfectly | AP ≈ 98% |
| Mass prediction ignores depth | Depth sensitivity: 3-4%, pred vs depth correlation: r=0.02 |
| Depth IS informative for mass | Physical area proxy: r=0.947 with GT mass |
| Model memorizes clusters via FPN features | 36 clusters, 256 FPN channels, lookup-table behavior |
| Scale mismatch suppressed depth signal | 60x in backbone, 8x in mass head (fixed) |
| ROI pooling loses object size | All objects normalized to 14x14 |
| Physical area (area_m²) is distance-invariant | CV drops from 18% to 9.7% when removing depth multiplication |
