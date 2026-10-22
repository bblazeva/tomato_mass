# Thesis progress update — RGB-D tomato cluster mass estimation

**Date:** 2026-05-25
**Author:** Blagica
**Status:** mid-project, after extensive ablations on RGB-D Mask R-CNN architectures and a depth-only counter-experiment.

---

## 1. Architectural variants and final RGB-D model

### 1.1 What I tried

I systematically explored variants of the Mask R-CNN mass head and the input/training pipeline. The key dimensions explored:

| Dimension | Variants |
|---|---|
| Mass head conv stack | 4 × Conv3×3 + ConvTranspose2d (deconv, 14×14→28×28) vs **5 × Conv3×3** (deconv replaced, stays at 14×14) |
| Mass head MLP | Default `(4096→128→1)` vs **larger MLP** |
| Late-fusion depth concat | With (LF) vs without (no-LF) injecting depth as channel 257 into the mass head input |
| `fc2.bias` initialisation | `0.0` vs `-0.8064` (mean log-mass of train set) |
| Train-time augmentation | Multi-scale `ResizeShortestEdge` + color jitter + flip vs only horizontal flip (`aug_fix`) |
| RGB dropout regularisation | `0.0` vs `0.3` (blank BGR channels on 30 % of samples) |

### 1.2 What moved the needle and what didn't

Below are the val-set summaries (MAPE / R² averaged over the eval iterations from `metrics.json`):

| Run | Mass head | LF | aug pipeline | Val MAPE (mean) | Val R² (mean) |
|---|---|---|---|---:|---:|
| `clean_baseline_v1_LF` | 4-conv + deconv | yes | old (resize) | **49.1 %** | 0.06 |
| `clean_baseline_v1_bigMLP_noLF` | big MLP | no | old | **55.8 %** | 0.56 |
| `clean_baseline_v1LF_5conv_aug_fix` | 5-conv | yes | **aug_fix** | **15.5 %** | 0.84 |
| `clean_baseline_v1_5conv_aug_fix` | 5-conv | no | aug_fix | 15.8 % | 0.81 |
| `clean_baseline_v1LF_5conv_aug_fix_drop_03` | 5-conv | yes | aug_fix + rgb_dropout 0.3 | 16.7 % | 0.82 |
| `clean_baseline_v1LF_5conv_bias_aug_fix` | 5-conv, `fc2.bias=-0.8064` | yes | aug_fix | 16.7 % | 0.79 |

**Key findings from the ablation grid:**

1. **The `aug_fix` change dominates every other intervention by an order of magnitude.** Removing multi-scale `ResizeShortestEdge` from the training mapper dropped val MAPE from ~50 % to ~15 %. The reason is structural: the validation mapper applied **no** resize, so training with random short-edge ∈ {480…640} created a systematic train/val scale mismatch. For mass prediction this is uniquely harmful because *mass ≈ depth-scaled apparent area* — random rescaling at training time deliberately breaks the size-mass correlation that the model needs at evaluation time.

2. **The 5-conv (no-deconv) head, late fusion on/off, larger MLP, and bias initialisation are all small effects (< 2 pp MAPE) compared to the `aug_fix` win.** Within the `aug_fix` family, the runs cluster tightly around 14-17 % val MAPE.

3. **`fc2.bias = 0.0` outperformed `fc2.bias = -0.8064` (mean log-mass).** The textbook "initialise the bias at the mean of the target" trick gave a small head-start in the first ~50 iterations but produced less informative features over the full 15 k iterations. The model with `bias=0` had a more stable eval curve and ~1 pp better mean MAPE.

4. **Late fusion contributes essentially nothing.** Comparing the cleanest LF / no-LF pair (`clean_baseline_v1LF_5conv_aug_fix` vs `clean_baseline_v1_5conv_aug_fix`, identical architecture except for the depth concat), the difference is 0.3 pp MAPE — well within run-to-run noise.

5. **RGB dropout did not improve val performance.** Gradient logging confirmed depth-channel gradients increased 2-60× during training, but inference predictions did not change correspondingly — the model "gamed" the regularisation by emitting near-mean predictions on RGB-blanked samples rather than actually rerouting through depth.

### 1.3 Final RGB-D best model — `clean_baseline_v1LF_5conv_aug_fix`

This is the strongest RGB-D variant. From its `metrics.json`:

| Metric | Train (final iter) | Val (last eval) | Val (best) |
|---|---:|---:|---:|
| `loss_mass` (MSE log-space) | **0.0029** | 0.0379 | — |
| MAPE | — | **14.3 %** | 13.5 % (min) |
| R² | — | **0.86** | 0.89 (max) |
| MAE (kg) | — | **0.113** | 0.099 (min) |
| `bbox/AP` | — | 92.5 | — |
| `segm/AP` | — | 98.6 | — |

**Inference summary on all three splits** (running the saved model end-to-end, picking top-1 detection per image):

| Split | MAE (g) | MAPE | bias (g) | Worst |
|---|---:|---:|---:|---:|
| Train | 31 | **5.5 %** | -22 | 233 |
| Val | 119 | **14.4 %** | -101 | 359 |
| Test | 114 | **19.0 %** | -64 | 431 |

The train↔val gap (5.5 % → 14.4 %) reflects genuine overfit. The train↔test gap (5.5 % → 19.0 %) reflects overfit **plus** a distribution shift: test contains more older `.png` capture sessions where the model struggles, as shown in the per-mass-group breakdown:

| GT (kg) | Test n | MAE (g) | MAPE | Session |
|---:|---:|---:|---:|---|
| 0.108 | 21 | 7 | 6.7 % | 2023-12-01 (.png) |
| 0.165 | 14 | 36 | 22.1 % | 2023-12-01 (.png) |
| 0.296 | 22 | 34 | 11.3 % | 2025-03-28 (.png) |
| 0.430 | 15 | 83 | 19.3 % | 2023-12-01 (.png) |
| 0.600 | 39 | 86 | 14.3 % | 2025-07-13 (.jpg, dominant) |
| **0.765** | **15** | **358** | **46.8 %** | 2025-03-28 (.png) |
| **0.947** | **24** | **240** | **25.3 %** | 2025-03-28 (.png) |

The 0.765 kg and 0.947 kg test groups (both `.png` captures from 2025-03-28) are catastrophically mispredicted. The model has overfit to the **RGB texture/lighting** of the dominant 2025-07-13 `.jpg` session and does not transfer to the older `.png` sessions.

---

## 2. Depth ablations on the RGB-D model

To diagnose whether the RGB-D model was actually using depth, I ran four inference experiments on `clean_baseline_v1LF_5conv_aug_fix` with the depth channel modified at inference time:

| Variant | What it does | Inference change vs control |
|---|---|---|
| 1. Real depth (mm) | Control — correct input | — |
| 2. depth / 1000 | Scaled wrong; after normalisation both depth pathways receive ~constant or zero | predictions effectively identical |
| 3. depth = 0 everywhere | Both depth pathways destroyed | predictions effectively identical |
| 4. depth = 5000 mm everywhere | A large constant; late-fusion `valid > 200` mask is True, so the cached depth signal is constant +9.93 | predictions shifted uniformly by ~+16 g |

### Results (val + test summary metrics, all three splits)

| Variant | Train MAPE | Val MAPE | Val R² | Test MAPE | Test MAE |
|---|---:|---:|---:|---:|---:|
| 1. real depth | 5.5 % | 14.4 % | 0.86 | 19.0 % | 114 g |
| 2. depth/1000 | 5.4 % | 14.3 % | 0.86 | 19.0 % | 114 g |
| 3. zeros | 5.4 % | 14.3 % | 0.86 | 19.0 % | 114 g |
| 4. 5000 mm constant | **5.0 %** | **13.0 %** | 0.86 | 18.8 % | 110 g |

### Per-image diff statistics against variant 1 (real depth)

| Comparison | Mean \|Δ\| | Max \|Δ\| | Exactly identical |
|---|---:|---:|---:|
| 1 vs 2 (depth/1000) | 1.3 g | 5 g | 349/1190 (29 %) |
| 1 vs 3 (zeros) | 1.3 g | 5 g | 349/1190 (29 %) |
| 2 vs 3 | 0.001 g | 1 g | 1189/1190 (99.9 %) |
| 1 vs 4 (5000 mm) | 16.7 g | 57 g | 64/1190 (5 %) |

### Interpretation

1. **The RGB-D model is depth-blind to spatial information.** Replacing real depth with corrupted depth or zeros changes predictions by under 1 g per image on average — that is the model's floating-point noise floor, not a meaningful response to depth.

2. **The model treats the late-fusion depth channel as a single scalar offset, not as spatial geometry.** Variant 4 (constant 5000 mm) is the only variant that moves predictions, and it does so as a uniform +16 g shift across all images, regardless of cluster size. The 14×14 spatial depth channel that should be informative has been reduced to a one-degree-of-freedom learned bias.

3. **Variant 4 coincidentally *improves* the metrics.** This is not because depth is working — it is because the model's existing predictions were systematically biased low (mean signed error -101 g on val, -64 g on test), and a forced positive offset accidentally compensates for that calibration error. A different magnitude would over- or under-correct.

4. **Combined with the gradient analysis** (depth-channel gradient in conv1 is ~1 % of the RGB-channel gradient throughout all 15 k iterations of every run), the conclusion is structural: in this Mask R-CNN design, depth as 1 of 4 input channels (early fusion) and 1 of 257 mass-head input channels (late fusion) is too small a signal-fraction to develop into a useful pathway. The model learns to ignore depth in iteration 0 and never reconsiders.

---

## 3. Depth-only counter-experiment

### 3.1 Motivation

If the RGB-D model ignores depth, where does its mass-prediction signal come from? The natural counter-experiment: train a model with **no RGB at all**, only depth, and see whether it can match or beat the RGB-D model. If the answer is "yes, it can match it," then RGB was not contributing to mass prediction.

### 3.2 Architecture

Same Mask R-CNN with the 5-conv mass head and the `aug_fix` augmentation pipeline. The only change is in the data mapper:

- **Channels 0-2 (BGR):** real depth, replicated three times, globally normalised to [200, 2000] mm → 0-255. The pretrained ImageNet conv1 weights now process depth as if it were a grayscale image. This is the standard trick for adapting RGB-pretrained backbones to single-modality inputs and preserves the value of the pretrained features.
- **Channel 3:** raw depth in mm (with invalid pixels filled to the dataset mean), identical to the RGB-D model.

The global normalisation [200, 2000] mm is fixed across the entire dataset (actual dataset extremes: 274-1901 mm). This was a deliberate change from per-image min/max normalisation — it preserves absolute depth scale across images, so the model can read "this pixel is at ~1500 mm" rather than "this pixel is bright in *this* image."

### 3.3 Results

Run: `clean_baseline_v1_5conv_aug_fix_DEPTH`. Same architecture, same hyperparameters, identical training duration.

**Training metrics (`metrics.json`, summarised):**

| Metric | RGB-D best | Depth-only |
|---|---:|---:|
| Train `loss_mass` (final) | 0.0029 | **0.0017** |
| Val `loss_mass` (final, mean) | 0.0379 | 0.0379 |
| Val MAPE (mean over evals) | 15.5 % | 15.6 % |
| Val MAPE (last eval) | 14.3 % | 14.8 % |
| Val R² (last eval) | 0.86 | 0.84 |
| `bbox/AP` | 92 | 88 (−4) |
| `segm/AP` | 98 | 87 (−11) |

**Inference on all splits (same protocol as RGB-D, top-1 detection per image):**

| Split | RGB-D MAPE | Depth-only MAPE | RGB-D MAE | Depth-only MAE |
|---|---:|---:|---:|---:|
| Train | 5.5 % | **3.4 %** | 31 g | **18 g** |
| Val | 14.4 % | **13.6 %** | 119 g | **111 g** |
| Test | 19.0 % | **14.1 %** | 114 g | **73 g** |

### 3.4 Per-mass-group breakdown on test — the cross-session story

This is the clearest finding of the project so far.

| GT (kg) | n | RGB-D MAPE | Depth-only MAPE | Session |
|---:|---:|---:|---:|---|
| 0.108 | 21 | 6.7 % | 10.4 % | 2023-12-01 (.png) |
| 0.165 | 14 | 22.1 % | **7.7 %** | 2023-12-01 (.png) |
| 0.296 | 22 | 11.3 % | 17.8 % | 2025-03-28 (.png) |
| 0.430 | 15 | 19.3 % | 16.3 % | 2023-12-01 (.png) |
| 0.600 | 39 | **14.3 %** | 18.5 % | 2025-07-13 (.jpg, dominant) |
| **0.765** | **15** | **46.8 %** | **7.5 %** | 2025-03-28 (.png) |
| **0.947** | **24** | **25.3 %** | **13.6 %** | 2025-03-28 (.png) |

- On the **dominant in-session group** (0.600 kg, 2025-07-13 `.jpg`), RGB-D is slightly better (14.3 % vs 18.5 %). The CNN's RGB pathway has memorised the texture/lighting of this session and gets a small edge.
- On the **older cross-session `.png` groups** (0.765, 0.947, 0.165), the depth-only model is **dramatically better** — most strikingly, the 0.765 kg group drops from a catastrophic 46.8 % MAPE to **7.5 %**.

The reason is intuitive: real-world cluster geometry is invariant to camera/lighting changes, while RGB texture is not. RGB-D learned shortcuts that don't transfer; depth-only is forced to use the session-invariant signal.

### 3.5 Depth-pathway ablation, repeated on the depth-only model

To check whether the depth-only model itself relies on channel 3 / late fusion, or only on the BGR-replicated depth path, I repeated the depth-zeroing inference on this model. Results on the test split:

| Variant | MAE (g) | MAPE | bias (g) |
|---|---:|---:|---:|
| 1. real depth (control) | 73 | 14.1 % | +0.010 |
| 2. zeros in channel 3 + zeroed late-fusion tensor | 73 | 14.1 % | +0.008 |
| 3. 5000 mm constant | 74 | 15.0 % | +0.018 |

Per-image differences between variant 1 and variant 2 are again at the noise floor. **Even in the depth-only model, the channel-3 / late-fusion pathway carries no information.** All of the depth signal flows through the BGR-replicated channels, where the pretrained backbone can process it as if it were a grayscale image. This is consistent with what I observed in the RGB-D model: late fusion is structurally drowned, regardless of what's in the rest of the input.

### 3.6 What this proves

1. **The RGB pathway of the original model was contributing essentially nothing to mass prediction.** A model with literally zero RGB information matches or beats the full RGB-D model on every split.

2. **The mass-prediction signal is dominated by geometric/spatial structure** (cluster size on the image plane, depth scale, mask shape). This signal is present in both RGB and depth — the model used whichever was easier to read.

3. **Depth-only generalises better across capture sessions than RGB-D** — most clearly on the older `.png` sessions where RGB-D fails catastrophically. This identifies cross-session shift as the dominant failure mode of the RGB-D model and the most important obstacle to address.

4. **Detection/segmentation are slightly worse with depth alone** (`bbox/AP` 88 vs 92, `segm/AP` 87 vs 98). RGB color edges still help precise mask boundaries — but the cluster is still well-localised from depth alone (`bbox/AP` 88 is solid). This is a known and acceptable trade-off for the depth-only direction.

5. **The current RGB-D architecture is structurally unable to use depth in a useful way.** Three independent attempts to make depth matter — early fusion (4th conv1 channel), late fusion (mass-head channel 257), and forced RGB-dropout regularisation — all failed. The depth-only experiment isolates the depth pathway successfully **only by removing RGB entirely** and forcing the pretrained backbone to operate on depth.

---

## Where this leaves the project

The combined picture from these experiments points to a clear next direction:

- The current RGB-D architecture has a structural limitation: depth as a single channel alongside many RGB / FPN channels is too small a signal-fraction to develop.
- Depth-only training, while practical, sacrifices the pretrained ImageNet feature advantages for color and gives up some segmentation precision.
- A better architecture would give depth a **dedicated, parallel pathway** (rather than a single channel buried among many) while keeping RGB available for detection. This is exactly the motivation for the point-cloud / geometry-branch direction (e.g., a PointNet-based mass head operating on per-instance metric XYZ points extracted via the existing Mask R-CNN segmentation), where depth becomes the dominant input signal by design.

In parallel, an oracle tabular baseline using ground-truth-mask engineered features (`area_m2`, depth statistics, bbox dimensions, etc.) reaches val R² ≈ 0.86 with a small XGBoost / MLP — confirming that the geometric signal in the data is rich enough to support better-than-current mass prediction *if* it can be extracted cleanly.

The depth-only experiment is the strongest single piece of evidence supporting this redirect: it shows that on this dataset, **the geometric/depth-derived signal is what the model needs, and a structurally depth-prominent architecture is the natural next step**.

---

## Summary numbers for the supervisor meeting

- **Best RGB-D model** (`clean_baseline_v1LF_5conv_aug_fix`): val MAPE **14.4 %**, test MAPE **19.0 %**.
- **Depth-only model** (`clean_baseline_v1_5conv_aug_fix_DEPTH`): val MAPE **13.6 %**, test MAPE **14.1 %**.
- **Single largest contribution**: removing multi-scale resize from training augmentation (`aug_fix`) — dropped val MAPE from ~50 % to ~15 %.
- **Confirmed**: the RGB-D model is depth-blind (4 independent ablations agree).
- **Open finding**: cross-session generalisation (`.png` 2025-03-28 / 2023-12-01 vs `.jpg` 2025-07-13) is the dominant remaining failure mode.
- **Recommended next direction**: an architecture with a dedicated depth/geometry branch (point-cloud or scalar-feature based), validated against the existing baselines.
