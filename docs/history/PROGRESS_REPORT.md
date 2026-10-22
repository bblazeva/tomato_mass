# RGB-D Mass Prediction: Progress Report

## 1. Problem Statement

I'm building a multitask RGB-D model that simultaneously:
1. **Detects** tomato bunches (bounding boxes)
2. **Segments** them (instance masks)
3. **Predicts their mass** (in kg) from RGB-D images

The architecture is based on Mask R-CNN (Detectron2) extended to 4-channel
input (BGR + Depth). Detection and segmentation work perfectly (AP ~98%).
**The core problem is mass prediction: the model does not use depth
information**, despite depth being the most informative signal for mass.

---

## 2. Dataset

| Property            | Value                           |
|---------------------|---------------------------------|
| Total images        | 1190 RGB-D                      |
| Unique clusters     | 52 (each cluster = one physical tomato bunch) |
| Training clusters   | ~36                             |
| Validation clusters | ~16                             |
| Images per cluster  | ~23 (same bunch photographed from different angles/distances) |
| Annotated instances | 875 (training set)              |
| Depth source        | Intel RealSense (mm scale)      |
| Mass range          | ~0.3 kg to ~1.5 kg             |
| Camera intrinsics   | FX=921.975, FY=922.056          |

**Key dataset limitation:** Only 36 unique objects in training. The model
sees ~23 images of each object, all from different viewpoints. This means
the model can achieve low training loss by simply memorizing what each
cluster *looks like* via RGB features, without ever learning the
depth-to-mass relationship.

---

## 3. Architecture Overview

```
Input: 4-channel image (BGR + Depth)
  |
  v
ResNet-50 Backbone (modified: conv1 has 4 input channels)
  |                                     Depth Cache
  v                                        |
FPN (Feature Pyramid Network)              | raw depth in mm
  |                                        | normalized: (d - 915.043) / 411.683
  |                                        | invalid pixels (< 200mm) = 0
  v                                        v
RPN -> Proposals -----> ROI Pooling (14x14)
                            |
                    +-------+-------+
                    |       |       |
                    v       v       v
                  Box     Mask    Mass Head
                  Head    Head    (our focus)
                                    |
                         FPN features (256 ch) + Depth (1 ch, scaled x8)
                         = 257 channels, 14x14
                                    |
                                    v
                         4x Conv3x3 -> Deconv -> Pool(4x4) -> FC -> log(mass)
```

### Two Depth Paths

1. **Backbone path:** Depth enters as the 4th channel of the input image,
   processed by conv1 alongside BGR. Feeds into FPN.
2. **Mass head path:** Raw depth is cached separately, ROI-pooled to 14x14,
   and concatenated as channel 257 to the FPN features before the mass head.

### Normalization

| Channel | PIXEL_MEAN | PIXEL_STD | After normalization |
|---------|------------|-----------|---------------------|
| B       | 103.530    | 1.0       | std ~ 60            |
| G       | 116.280    | 1.0       | std ~ 60            |
| R       | 123.675    | 1.0       | std ~ 60            |
| Depth   | 915.043 mm | 411.683 mm| std ~ 1.0           |

**Scale mismatch in backbone:** After normalization, RGB values have std~60
while depth has std~1.0 -- a 60x imbalance. The backbone's conv1 naturally
learns to weight RGB 60x more than depth. This was NOT fixed in the backbone
(too risky to change backbone normalization mid-project).

**In the mass head:** The depth channel is explicitly scaled by 8x after
concatenation to match FPN feature magnitudes (FPN std~4.0, depth after
8x scaling: std~4.0). This fix was applied.

---

## 4. Bugs Found and Fixed (Previous Session)

### Bug 1: Depth Boundary Artifacts
- **Problem:** Depth values near object edges were invalid/noisy (< 200mm).
  ROI pooling interpolated these with valid pixels, corrupting the depth signal.
- **Fix:** Set validity threshold to >200mm in `_cache_depth()`. Invalid pixels
  are zeroed in normalized space (which maps to the mean depth of 915mm after
  un-normalization).

### Bug 2: Double Normalization
- **Problem:** Depth was being normalized in the cache AND again by the backbone
  PIXEL_MEAN/STD subtraction.
- **Fix:** The cache stores raw depth in mm. The backbone normalizes it once
  with PIXEL_MEAN=915.043 / PIXEL_STD=411.683.

### Bug 3: FPN/Depth Scale Mismatch in Mass Head
- **Problem:** FPN features at the mass head input had std~4.0, but the depth
  channel had std~0.5. The mass head's first conv layer (mass_fcn1) naturally
  ignored the depth channel.
- **Fix:** Applied 8x scaling to the depth channel in
  `_concat_cached_depth_to_features()`, matching std~4.0.
- **Verification:** Gradient ratio (depth/FPN) improved from 0.01 to a
  reasonable range after scaling.

### Bug 4: MassEvaluator Crash
- **Problem:** The evaluator used index-based matching between predictions
  and ground truth, which broke when detection order didn't match annotation
  order.
- **Fix:** Switched to IoU-based matching in the evaluator.

---

## 5. Experiments and Results

### Experiment 1: Baseline (mass_stage4) -- Before Depth Fixes

Trained with original architecture, depth bugs still present.

| Metric      | Value      |
|-------------|------------|
| Mass MAE    | 0.112 kg   |
| Mass MAPE   | 14.8%      |
| Mass R2     | 0.870      |
| Mass RMSE   | 0.135 kg   |
| Segm AP     | 98.6%      |

**Looks good, but misleading.** The model achieved this by memorizing the 36
training clusters via FPN features, not by using depth. Depth sensitivity
test showed only **3-4% difference** when replacing real depth with zeros.

**Key evidence:** `pred_mass vs mean_depth` correlation = 0.02 (nearly zero).
The model completely ignores depth.

---

### Experiment 2: Depth Fixes + Color Jitter + Modality Dropout (mass_drop)

Applied all depth bug fixes, then added:
- **Color jitter** (brightness 0.6-1.4, contrast 0.6-1.4, 80% probability)
  on BGR channels only. Goal: make FPN features less distinctive so the model
  can't memorize clusters as easily.
- **Modality dropout** (30% probability): zero ALL 256 FPN channels, forcing
  the mass head to predict from depth alone during those forward passes.

| Metric      | Value      | vs Baseline |
|-------------|------------|-------------|
| Mass MAE    | 0.185 kg   | +65% worse  |
| Mass MAPE   | 53.0%      | +258% worse |
| Mass R2     | 0.701      | -19% worse  |
| Mass RMSE   | 0.204 kg   | +51% worse  |
| Segm AP     | 98.0%      | unchanged   |

**Depth sensitivity improved** from 3-4% to ~12.7%, but:
- Overall mass prediction got significantly worse
- Depth was used in the **wrong direction** for some clusters
  (e.g., cluster 1010: real depth made predictions worse, not better)
- The FPN pathway was severely damaged by 30% modality dropout --
  `const_mass` predictions collapsed to ~1.0 kg for ALL clusters,
  meaning the model lost its ability to distinguish clusters via FPN
- Detection/segmentation was unaffected (those heads don't use the mass path)

**Conclusion:** Modality dropout at 30% was too aggressive. It destroyed the
FPN signal without the depth signal being strong enough to compensate. The
model couldn't learn a useful depth-to-mass mapping from the 1-channel depth
in the 14x14 ROI-pooled features.

---

### Experiment 3: Modality Dropout + Box Area Input (mass_drop_box)

Same as Experiment 2, plus:
- **Bounding box area** passed as an additional scalar to the mass head MLP.
  Motivation: ROI pooling normalizes all objects to 14x14, losing pixel area
  information. Box area tells the mass head how big the object is in the image.
- `fc1` changed from `Linear(4096, 128)` to `Linear(4097, 128)` to accept
  the extra scalar.

| Metric      | Value      | vs Exp 2    |
|-------------|------------|-------------|
| Mass MAE    | 0.182 kg   | -1.5% better|
| Mass MAPE   | 53.1%      | unchanged   |
| Mass R2     | 0.680      | -3% worse   |
| Mass RMSE   | 0.211 kg   | +3% worse   |

**No meaningful improvement.** The box area scalar (1 number out of 4097
inputs to fc1) was drowned out by the 4096 conv features. The model
effectively ignored it -- confirmed by identical `const_mass` predictions
with and without box area variation.

**Conclusion:** Appending a single scalar to a 4096-dimensional feature
vector doesn't work. The information is there but the architecture can't
use it.

---

### Experiment 4: Depth Fixes Only, No Dropout (mass_mm, mass_mm2)

Trained with depth bug fixes (8x scaling, boundary fix) but WITHOUT
modality dropout or color jitter.

| Metric      | mass_mm    | mass_mm2   |
|-------------|------------|------------|
| Mass MAE    | 0.211 kg   | 0.224 kg   |
| Mass MAPE   | 51.7%      | 53.3%      |
| Mass R2     | 0.639      | 0.586      |
| Segm AP     | 97.3%      | 97.3%      |

**Worse than baseline.** Fixing the depth bugs alone didn't help; the model
still memorizes via FPN, and something about the fixes disrupted the
previous (memorization-based) equilibrium.

---

### Summary Table

| Experiment          | MAE (kg) | MAPE   | R2    | Depth Sensitivity |
|---------------------|----------|--------|-------|-------------------|
| Baseline (stage4)   | 0.112    | 14.8%  | 0.870 | 3-4% (not using)  |
| Depth fixes only    | 0.211    | 51.7%  | 0.639 | ~3-4%             |
| + Color jitter + Dropout | 0.185 | 53.0% | 0.701 | ~12.7% (wrong dir)|
| + Box area          | 0.182    | 53.1%  | 0.680 | ~9%               |

---

## 6. Root Cause Analysis

### Why the baseline works "well" (MAE=0.112)

The baseline doesn't use depth at all. It works because:
- 36 training clusters, ~23 images each
- 256 FPN channels encode enough visual features to **uniquely identify**
  each cluster (appearance, leaf patterns, fruit arrangement)
- The mass head memorizes: "this visual pattern = 0.765 kg"
- On the validation set (different clusters), this memorization still
  partially works because tomato bunches of similar mass look somewhat similar

This is essentially a **lookup table**, not a learned physics relationship.

### Why depth gets ignored

1. **Capacity imbalance:** 256 FPN channels vs 1 depth channel. The conv
   layers have 256x more signal from FPN than from depth.
2. **Easier shortcut:** Memorizing 36 clusters from FPN features is a lower-
   loss solution than learning the noisy depth-to-mass relationship.
3. **ROI pooling destroys scale:** After pooling to 14x14, a small nearby
   object looks the same as a large far-away object. The absolute depth value
   (channel 256) is there, but it's 1 channel among 257.
4. **Backbone normalization mismatch:** In the backbone path, depth has
   60x smaller activations than RGB, so depth features in FPN are weak.

### Why modality dropout doesn't fix it

- Zeroing 256 FPN channels 30% of the time means 70% of the time the model
  still has full FPN features to memorize with
- During the 30% dropout steps, the single depth channel (14x14 spatial grid)
  doesn't contain enough information to predict mass accurately -- it's the
  *average* depth over the bounding box, not mask-specific depth
- The model can't learn a useful mapping from depth alone through 4 conv
  layers designed for 257-channel input
- Result: the FPN pathway degrades (can't train reliably when randomly zeroed)
  without the depth pathway compensating

---

## 7. Volume Proxy Analysis

Independently of the model, I computed a physics-based volume proxy using the
ground-truth masks and depth maps (script: `mean_volume_detail.py`):

```
area_m2     = sum(Z^2) / (FX * FY)     # physical area in m^2 (per pixel)
volume_proxy = area_m2 * median_depth   # proxy for volume in m^3
```

### Results

| Metric                                   | Value     |
|------------------------------------------|-----------|
| Per-instance correlation (linear)        | moderate  |
| Per-instance correlation (log-log)       | moderate  |
| **Per-cluster mean correlation (linear)**| **r = 0.94** |
| Per-cluster median CV (within-cluster)   | varies    |
| Number of instances                      | 875       |
| Number of clusters                       | 36        |
| Mean(log proxy)                          | -5.24     |
| Std(log proxy)                           | 0.54      |

**Key finding:** The analytical volume proxy (area x depth) correlates at
r=0.94 with ground-truth mass at the cluster level. This proves that
**depth IS highly informative for mass prediction** -- the problem is that
the model architecture can't extract this signal.

The proxy has noise at the instance level (same cluster, different images give
different proxy values due to pose, distance, occlusion), but the cluster-mean
proxy is an excellent mass predictor.

---

## 8. Current State of the Code

### What's currently in mass_head.py (Option A: Volume Proxy Test)

The mass head currently bypasses all conv/deconv/fc layers and instead:
1. Extracts the depth channel from ROI-pooled features (channel 256)
2. Un-normalizes it back to meters
3. Computes `volume_proxy = box_area_pixels * mean_depth^3 / (FX * FY)`
4. Feeds `log(volume_proxy)` through a small MLP: `Linear(1, 64) -> ReLU -> Linear(64, 1)`
5. Outputs `log(mass)`

**This has NOT been trained yet.** It's a pure test to see if the proxy
formula (which works analytically at r=0.94) can work through the model's
depth path (which goes through normalization, caching, ROI pooling, etc.).

### Known limitation of current Option A

The MLP receives only 1 scalar input (the combined proxy). A better version
would feed 2 separate inputs: `[log(box_area), log(mean_depth)]`, letting
the MLP learn optimal exponents for area and depth independently.

### Other active modifications in train.py

- Color jitter augmentation on BGR channels (still active)
- Stratified sampling by mass bins (still active)
- Depth cache with >200mm validity threshold (still active)

---

## 9. Ideas for Continuing

### Option A (current): Pure Volume Proxy Test
- **Goal:** Validate that the proxy formula works end-to-end through the model
- **Change:** Feed `[log(box_area), log(mean_depth)]` as 2 features to MLP
- **Expectation:** Should achieve decent R2 if depth path is working correctly
- **Limitation:** Ignores all visual features (FPN). A ceiling test.
- **If it works:** Proves depth path is functional; move to Option B
- **If it fails:** Something is wrong with the depth path (normalization,
  caching, ROI pooling corrupting depth values)

### Option B: Hybrid -- Proxy + Conv Features
- Use the volume proxy as additional features alongside the conv path
- Feed `[conv_features (4096), log(box_area), log(mean_depth)]` to the MLP
- The conv features provide cluster identity; the proxy provides physical scale
- This gives the model both signals without requiring it to discover the
  depth-to-mass relationship on its own

### Option C: Larger Dataset / More Clusters
- The fundamental issue is 36 training clusters -- too few to learn
  generalizable depth-to-mass mapping
- With 200+ clusters, the model couldn't memorize all of them and would be
  forced to learn the underlying relationship
- This is the "correct" solution but requires data collection

### Option D: Depth-Only Mass Branch (Separate Encoder)
- Separate encoder for depth (not sharing with RGB backbone)
- Avoids the 256:1 channel imbalance
- More complex architecture, harder to train

### Option E: Reduce Modality Dropout Rate
- Try 5-10% instead of 30%
- Less disruptive to FPN pathway
- Combined with color jitter, might nudge the model toward depth usage
- Risk: might still not be enough to overcome memorization

---

## 10. Key Takeaways

1. **Detection and segmentation are solved** -- AP ~98%, near-perfect.
   This is not the bottleneck.

2. **Mass prediction via FPN memorization achieves MAE=0.112 but doesn't
   generalize** -- it's a lookup table, not a learned relationship.

3. **The volume proxy (area x depth) correlates at r=0.94 with mass** --
   depth IS informative, the model just can't access it.

4. **Architectural bottleneck:** 256 FPN channels overpower 1 depth channel.
   The model always takes the memorization shortcut.

5. **Modality dropout disrupts FPN without adequate depth compensation** --
   making things worse, not better.

6. **The 36-cluster dataset is fundamentally limiting** -- with so few unique
   objects, memorization is always the lowest-loss strategy.

7. **Next critical experiment:** Test if the analytical volume proxy works
   through the model's depth path (Option A). This isolates whether the
   depth signal survives the normalization/caching/pooling pipeline.
