# Investigation record: how the model estimates mass (Oct 22 to Nov 7, 2026)

This records what was tested, what was found, and what is still open, so the context is not lost.
Numbers are copied from the result files listed in section 11. Where earlier statements turned out to be
wrong, section 9 says so.

**Verdicts used here**
- **Measured:** directly observed.
- **Not proven:** the data fits, but other explanations remain.
- **Wrong:** a test contradicted it.

**Metric notes.** MAPE is mean absolute percentage error, per image unless stated. "Per cluster" averages over
clusters, so clusters with many near-duplicate images do not dominate. Misses (no detection) are counted, not
scored. R² below 0 means worse than always guessing the average mass. Clusters, not frames, are the unit of evidence.

---

## 1. Summary

**Where things stand**
1. The failure is on new sessions: RGB-D has about 19% error with the session in training, and 31%, 38% and 249%
   without it.
2. Detection is not the problem: mask overlap with the annotations is 0.94–0.95 on unseen sessions.
3. The mass estimate is the problem: on unseen Lab 2 it barely ranks clusters (rank +0.14), though it follows mass
   when Lab 2 is in training (slope +0.70).
4. The model does not compute physical size: changing depth or apparent size barely moves the prediction.
5. Physical size does transfer: real area from the model's own masks gets 19.5%, 21.0% and 22.4% on unseen sessions
   (with tomato type taken from the hand labels), and ranks Lab 2 at +0.96.
6. RGB-D is not measurably better than RGB, in any test.

**What is not established**
- Why the model ignores geometry.
- Whether tomatoes or conditions (lighting, background) cause more of the session shift. They all change together
  in this dataset, so it cannot separate them.
- What exactly RGB and depth-only read from the cluster (count, shape, type).

The tests that would settle these are in section 8.

---

## 2. Data and names

### Sessions

| Name here | Date | Notes |
|---|---|---|
| Home | 2025-07-13 | Cart, room background |
| Lab 1 | 2023-12-01 | Blue sheet (called "2023-12-01" or "2023" in older output and in run names) |
| Lab 2 | 2025-03-28 | Blue sheet (called "lab" in run names, `test_0328` in `data_s0713`) |
| 538 g test | 2025-06-13 | One cluster, plain wall, in no training split |

### Datasets (not in git)

| Folder | What it is |
|---|---|
| `data/` | Links to the old repo: the original mixed split (875 / 165 / 150 images), used by runs 1 and 2 |
| `data_s0713/` | Home-only train (383) and val (190), plus `test_0328` (416) and `test_1201` (201). Mostly links to the old repo |
| `data_min120/` | Real copies. All 3 sessions, clusters below 120 g removed: 1,122 images, 45 mass values, 46 clusters |
| `data_min120/folds/fold_0..4/` | 5-fold splits by cluster (relative links) |
| `data_min120/loso/hold_{lab,home,2023}/` | Leave-one-session-out splits (relative links) |
| `data_home_color/` | The 16 images of the 538 g cluster, copied |

Removed from `data_min120` (below 120 g): 46, 63, 70, 77, 90, 93, 108 g; 68 images, all Lab 1. They remain in the old repo.

**136 g is two different clusters** in Lab 1: a vine (7 images at 16:45) and a rounder cluster (15 images at 17:23).
They share a mass value, so the splits keep them together (both in fold 1). There is no leakage from this. Scripts using
hand labels separate them as `136a` and `136b`.

**Other captures without annotations in `data/gt_json`:** `test_lust`, `test_lust2023`, `pics_lust`, `*_tune`. Not used.

**The old repo must not be moved or deleted:** `data/` and `data_s0713` link into it, as do some old runs.
The raw data is not on GitHub. A possibly incomplete copy exists under
`/mnt/md0/blagicab/output/multitask_RGB-D_FruitDetectionAndSizing/data/`.

### What each session contains (>= 120 g; hand labels from clusters_analysis.pdf)

| Session | Type | Clusters | Images | Mass | Tomatoes | g per tomato | Distance |
|---|---|---|---|---|---|---|---|
| Home | Cherry | 3 | 97 | 164–239 g | 8–12 | 20–21 | 350–927 mm |
| Home | Normal | 11 | 476 | 375–1160 g | 2–5 | 188–283 | 337–1371 mm |
| Lab 1 | Cherry | 7 | 77 | 122–165 g | 6–10 | 15–21 | 441–519 mm |
| Lab 1 | Normal | 4 | 56 | 183–430 g | 2–5 | 86–93 | 439–509 mm |
| Lab 2 | Cherry | 0 | | | | | |
| Lab 2 | Normal | 21 | 416 | 186–1029 g | 2–9 | 92–140 | 472–685 mm |

### Models

All recent models: normalized input, RGB dropout 0.5, seed 1 for the runs below marked "cv5" or "loso".
Zoom 0.5 (0.7–1.3). Learning rate drops at 80% of the iterations. Checkpoint used: `model_final.pth`.

| Run | Trained on | Tested on | Notes |
|---|---|---|---|
| `s0713_zoom_drop05` (model 5) | Home | Home val, Lab 2 | 15k |
| `clean_baseline_v1_5conv_aug_fix_noLF_norm_stip_drop05` (4.2) | Home | Home val, Lab 2 | No zoom |
| `s0713_zoom_drop_05_00` (7c) | Home | Home val, Lab 2 | 10k; same setup as 5 |
| `s0713_zoom_drop065` (7a) | Home | Home val, Lab 2 | RGB dropout 0.65 |
| `cv5_fold{0..4}_20k` | 4 of 5 folds, all sessions | The remaining fold | 20k; about 2 h 10 min each |
| `cv5_fold0` | Fold 0 | Fold 0 val | 10k, used only to choose 10k vs 20k |
| `loso_lab_20k` | Home + Lab 1 (706 images) | Lab 2 (416) | 20k |
| `loso_home_20k` | Lab 1 + Lab 2 (549) | Home (573) | 20k |
| `loso_2023_20k` | Home + Lab 2 (989) | Lab 1 (133) | 20k |

Mass ranges in the leave-one-session-out runs: holding out Lab 2, all test masses are inside the training range;
holding out Home, 51 images (1160 g) are above it; holding out Lab 1, 63 images (122–150 g) are below it.

---

## 3. Home-only models on the lab (before cross-validation)

RGB-D / RGB only / depth only, MAPE %, Lab 2 (`test_0328`), each input on the frames it detects:

| Model | RGB-D | RGB only | Depth only | Depth-only misses |
|---|---|---|---|---|
| 5 | 66.8 | 65.5 | 35.8 | 29 |
| 6d (mass-weight schedule) | 63.4 | 62.3 | 44.2 | 26 |
| 7a (RGB dropout 0.65) | 44.8 | 43.8 | 30.8 | 63 |
| 7b (0.4 + depth dropout 0.1) | 67.1 | 67.5 | 40.3 | 53 |
| 7c | 62.4 | 61.8 | 42.5 | 15 |

- Model 5 in-domain (Home val): 10.3 / 10.9 / 11.0%. On the same 188 frames: 10.4 / 10.8 / 11.0%.
- Model 5 on Lab 2, the 387 frames where all inputs predict: 55.7 / 55.3 / 35.8%.
- On the 29 frames depth-only missed: RGB-D 215%, RGB only 203%. Misses were in 186, 220, 296 and 368 g clusters, at
  normal distances.
- On the 344 frames every model and input predicts: model 5: 47.2 / 47.0 / 35.0; 7a: 37.6 / 36.7 / 31.2.

**Averaging RGB-D and depth-only predictions** (geometric mean, no tuning, falling back to RGB-D on a miss), on Lab 2:
model 5 66.8 → 53.2%; 6d 63.4 → 53.3%; 7a 44.8 → 40.0%; 7b 67.1 → 57.9%; 7c 62.4 → 50.9%. In-domain stayed similar.
On model 5's 387 same frames: depth-only 35.8%, average 41.0%. This is not in `docs/FINDINGS.md` yet.

**Depth ablation (Oct 23), mean change of the prediction vs real RGB-D:**

| Depth channel | Model 5 val | Model 5 Lab 2 | 7a val | 7a Lab 2 |
|---|---|---|---|---|
| Mean value (no depth) | 2.5% | 3.3% | 8.4% | 3.9% |
| 0 mm everywhere | 2.8% | 4.7% | 14.6% | 11.9% |
| Flat at the cluster's own distance | 2.5% | 2.8% | 8.8% | 4.7% |
| Shuffled from another cluster | 7.4% | 3.6% | 39.8% | 6.5% |

7a depends on depth in-domain, but real depth did not make it more accurate (13.6% real, 13.4% none, 11.9% flat).

**Does shuffled depth pull the prediction towards the other cluster's mass?** Regression of log(shuffled / real prediction) on
log(other cluster's mass / own mass) and log(other distance / own distance), RGB-D:

| Model, images | Mass ratio (95%) | Distance ratio (95%) | Changes above 50% | Of those, towards the other cluster's mass |
|---|---|---|---|---|
| 7a, Home val | +0.33 (+0.25 to +0.41) | +0.07 (−0.08 to +0.24) | 38 of 190 | 97% |
| Model 5, Home val | +0.04 (+0.01 to +0.08) | −0.04 (−0.12 to +0.03) | 4 of 190 | 100% |
| 7a, Lab 2 | +0.02 (−0.00 to +0.04) | −0.05 (−0.20 to +0.09) | 3 of 411 | 67% |
| Model 5, Lab 2 | +0.00 (−0.01 to +0.01) | −0.05 (−0.14 to +0.05) | 2 of 416 | 0% |

**Later correction:** the cleaner pasted-depth test (section "Structure tests") shows no pull when only the cluster's depth is swapped (7a RGB-D +0.01). So this effect does not come from the cluster in the depth map. It must come from elsewhere in the swapped depth map, such as the background or position, or from a moved detection box (not checked).

So for 7a on Home, the other cluster's whole depth map moves the prediction towards that cluster's mass, and not because of distance. Mean change by overlap
between the two clusters' boxes, 7a Home val: under 0.1: 63.1%; 0.1–0.5: 26.2%; 0.5 and up: 10.6%. **Not proven why:** the whole depth map is swapped, including
background and position, and whether the detection box moves was not checked. These are home-only models, not the CV models.

---

## 4. Cross-validation: new clusters, known sessions

**10k vs 20k on fold 0** (same images): RGB-D 15.9 → 13.8%; per cluster 19.1 → 17.2%; RGB only 17.0 → 14.0%; depth only
22.3 → 17.0%; depth-only misses 3 → 0. The agreed rule was 3 points or more on RGB-D, which this narrowly missed
(2.1 and 1.9). 20k was chosen anyway, because the other 4 folds only added about 2.3 hours, and depth-only gained most.
Choosing it on fold 0 makes fold 0 slightly optimistic. Same-seed runs already differed by about 9 points at iteration 8,000
(22.7% vs 14.1%), so this comparison is weak.

**5 folds, 20k, 1,122 images, each tested once by a model that did not see its cluster:**

| Input | MAPE per image | Per cluster | Misses |
|---|---|---|---|
| RGB-D | 18.9% (MAE 108 g, R² 0.80) | 20.5% | 0 |
| RGB only | 19.2% | 20.9% | 0 |
| Depth only | 20.9% | 22.6% | 2 |

| Fold | RGB-D | RGB only | Depth only |
|---|---|---|---|
| 0 | 13.8 | 14.0 | 17.0 |
| 1 | 22.2 | 21.1 | 24.0 |
| 2 | 23.3 | 25.0 | 25.1 |
| 3 | 13.0 | 12.9 | 13.5 |
| 4 | 22.4 | 23.5 | 25.3 |

Mean ± std over folds, RGB-D: 19.0 ± 5.1. On the same frames, per cluster: RGB-D − RGB only = −0.4 points (95% over clusters
−1.3 to +0.5); RGB-D − depth only = −2.1 (−3.8 to −0.4).

By session, RGB-D: Home 19.6%, Lab 2 19.0%, Lab 1 15.7%.
By mass, RGB-D: under 250 g 21.2%; 250–500 g 24.7%; 500–800 g 16.3%; 800 g and up 15.7%.

Same 70 Lab 2 images in fold 0: model 5 RGB-D 76.8%, 7a 46.5%, fold 0 (10k) 26.1%. All 416 Lab 2 images: model 5 66.8%,
7a 44.8%, cross-validation 19.0%. Depth only: model 5 35.8% (29 misses), cross-validation 20.1% (0 misses).
On Lab 2 clusters under 300 g, RGB-D: model 5 221%, cross-validation 27%.

**Fold notes.** Fold 1 tests 1160 g with nothing that heavy in training (it still got 18.3%). A possible cost: on model 5's
190 val images, cross-validation gets 14.9% where model 5 got 10.3%, which may be run-to-run variation.

**Possible leakage question.** Lab 1's capture order steps down in mass (430, 364, 280, 183; 165, 145, 136, 123, 108, 90, 70;
150, 136, 122, 108, 93, 77, 63, 46), which looked like tomatoes being removed one at a time. The user stated that each mass
is a different cluster, apart from the two at 136 g, so **there is no leakage** on that basis. If they ever turn out to
share tomatoes, Lab 1's 15.7% would be optimistic; without Lab 1, RGB-D is about 19.3%.

---

## 5. Unseen sessions

### Leave one session out

| Held out | RGB-D | RGB only | Depth only | RGB-D R² | RGB-D, session in training |
|---|---|---|---|---|---|
| Lab 2 | 37.8% (273 g) | 36.4% | 53.3% | −1.29 | 19.0% (110 g, R² +0.68) |
| Home | 30.9% (237 g), 9 misses | 32.8% | 39.2%, 312 of 573 missed | −0.06 | 19.6% (123 g, R² +0.76) |
| Lab 1 | 249% (392 g) | 243% | 155% | −20.3 | 15.7% (34 g, R² +0.82) |

- Lab 1 by mass, RGB-D: under 164 g (outside the training range) 435%; 164 g and up 82%.
- Same frames, per cluster: RGB-D − RGB only: Lab 2 +1.2 (−0.3 to +2.4), Home −0.3 (−2.8 to +3.0), Lab 1 +5.8 (+3.6 to +8.1).
  RGB-D − depth only: Lab 2 −12.5 (−20.6 to −2.3), Home −10.6 (−16.7 to −5.3), Lab 1 +93.5 (+13.3 to +189.9).
- Against model 5 on Lab 2: MAPE 66.8 → 37.8%, but MAE barely moves (297 → 273 g) and R² is slightly worse (−1.14 → −1.29).
  The bias flips: model 5 +157 g, this model −257 g. 7a remains closer in grams (216 g, R² −0.23).

**Depth-only is not reliably more robust.** It beat RGB-D on Lab 2 for home-only models, but is worse on Lab 2 and Home here,
and better only on Lab 1. `docs/FINDINGS.md` B2 still says otherwise and needs updating.

**Examples.** Lab 2 unseen: 857 g, 9 tomatoes → 248 g; 186 g, 2 tomatoes → 309 g. Lab 1 unseen: 136 g vine → 416 g;
136 g round → 1000 g; 122 g → 961 g; 150 g → 938 g.

### Model behaviour by cluster group, RGB-D

Slope of log prediction on log mass: 1 = follows mass, 0 = flat.

| Group | Pred ÷ true | Predicted range | True range | Rank | Slope |
|---|---|---|---|---|---|
| Lab 2, session in training | ×0.99 | 251–934 g | 186–1029 g | +0.89 | +0.70 |
| Lab 2, unseen | ×0.65 | 226–544 g | 186–1029 g | +0.14 | +0.08 |
| Lab 2, model 5 (home only) | ×1.39 | 426–1095 g | 186–1029 g | −0.02 | −0.03 |
| Home normal, in training | ×1.07 | 512–1068 g | 375–1160 g | +0.80 | +0.48 |
| Home normal, unseen | ×0.71 | 326–718 g | 375–1160 g | +0.45 | +0.34 |
| Lab 1 cherry, in training | ×1.03 | 131–151 g | 122–165 g | −0.18 | −0.12 |
| Lab 1 cherry, unseen | ×4.12 | 354–993 g | 122–165 g | −0.11 | −0.74 |
| Lab 1 normal, unseen | ×1.54 | 291–656 g | 183–430 g | +1.00 | +0.98 |

Predicted g per tomato on Lab 1 cherry, unseen: about 65, where the true value is about 18.

Lab 2 unseen, RGB-D average by tomato count (true → predicted, using the corrected labels):
2: 203 → 311 g; 3: 349 → 320 g; 4: 447 → 388 g; 5: 558 → 351 g; 6: 705 → 551 g (one cluster); 7: 693 → 397 g;
8: 901 → 352 g; 9: 857 → 248 g (one cluster).

### The 538 g cluster (fourth session, 16 images, 332–526 mm, plain wall)

| Model | RGB-D | RGB only | Depth only |
|---|---|---|---|
| 5 CV models averaged | 28.7% (688 g) | 20.7% (627 g) | 18.2% (619 g) |
| CV models individually | 26–34% | 21–27% | 14–26% |
| Model 5 | 38.3% (734 g) | 30.4% (687 g) | 57.0% (823 g) |
| 7c | 34.8% (717 g) | 34.5% (715 g) | 60.9% (862 g) |
| 7a | 44.8% (760 g) | 26.2% (655 g) | 54.5% (813 g) |

No misses. Masks were checked by eye on a sample only, as there are no annotations. Predictions did not shrink with distance.
Two capture bursts at the same distance differ a lot (618 g at 446 mm, 835 g at 451 mm). Everything here is one cluster.

---

## 6. Weaknesses in the dataset

**Same count, different mass (normal tomatoes):**

| Tomatoes | Home | Lab 1 | Lab 2 |
|---|---|---|---|
| 2 | 375–566 g | 183 g | 186–220 g |
| 3 | 600–790 g | 280 g | 296–421 g |
| 4 | 820–1010 g | 364 g | 368–560 g |
| 5 | 1160 g | 430 g | 507–608 g |
| 6–9 | none | none | 655–1029 g |

**Within a tomato type, more tomatoes means more mass in every session** (rank correlation: Home normal +1.00, Home cherry +1.00,
Lab 1 normal +1.00, Lab 1 cherry +0.71, Lab 2 normal +0.97). Mixed together: +0.03 over all clusters, −0.02 at Home, −0.47 in
Lab 1. The negative numbers come from mixing cherry and normal tomatoes.

**Mass per cm² of visible area is similar across sessions, within a type:** normal 6.4 / 5.3 / 5.4 g/cm² (Home / Lab 1 / Lab 2),
cherry 2.4 / 2.3 g/cm² (Home / Lab 1).

**Weaknesses**
1. Tomato size is tied to session: normal tomatoes at Home (188–283 g) do not overlap with the labs (86–140 g).
2. Only Lab 2 has 6–9 tomatoes. Holding it out, 10 of its 21 clusters have more tomatoes than anything in training.
3. Few cherry clusters, unevenly spread: 10 in total, none in Lab 2, narrow mass range.
4. Only Home teaches distance: far/near ratio within a cluster ×2.41, against ×1.09 (Lab 1) and ×1.21 (Lab 2).
5. Distance and mass are linked differently: rank +0.44 at Home, +0.01 in Lab 1, −0.60 in Lab 2.
6. Appearance identifies the session. Tomato saturation: Lab 1 116–139, others 222–237. Tomato brightness: Home 146–168, labs 227–238.
7. Imbalance: 5–64 images per cluster; Home is 51% of all images.
8. Occlusion in heavy Lab 2 clusters, so visible area underestimates them.
9. Only 3 sessions, so conclusions about new sessions rest on very few examples.

### Labels (analysis/cluster_inventory.csv), confirmed 2026-11-08
- **1160 g:** 5 tomatoes (the PDF's "4" was a mistake). **368 g:** 4 tomatoes, 92 g each (the PDF's "122 g" was a mistake).
- **Typos corrected against the file names:** 183 g in Lab 2 is 186 g; 867 g is 857 g.
- `analysis/tomato_counts.csv` holds the earlier counts (different for 165, 421, 655, 697, 836, 926, 947 g). Since 2026-11-08,
  `analysis/tomato_count_analysis.py` and `analysis/summarize_loso.py` read the corrected labels, and their reports were regenerated.
  Nothing changed a conclusion. Details that moved: Lab 2 count-to-mass correlation +0.93 → +0.96; Lab 2 tomato sizes are now 92–140 g
  with all 21 clusters inside Home's size gap (was 20); model 5 on Lab 2 by count: 2 tomatoes ×3.89, 9 tomatoes ×0.57.

---

## 7. Hypotheses and verdicts

### H1. The model learns mass visually, and the data contradicts itself

| Claim | Test | Result | Verdict |
|---|---|---|---|
| It does not compute physical size | Change depth or apparent size; a size calculation gives about ×1.79 | RGB-D: ×1.00 and ×1.05 | **Measured** |
| It relies mostly on the cluster, not the scene | Blank the background, RGB only | 19.2% → 24.6% | **Measured** |
| It counts tomatoes or reads their shape | Compare predictions with hand counts | +0.66 for count, +0.87 for size | **Not proven** |
| The same count means different masses across sessions | Hand labels | See section 6 | **Measured** |
| Those contradictions stop the model learning | Train with all sessions | 19%, slope +0.70 on Lab 2 | **Wrong** |
| Those contradictions cause the failure on new sessions | Hold out Lab 2 | Reaction to count +0.05 | **Not proven** |

Why counting is not proven: mass = count × size, so any accurate model scores well on that test, whatever it looks at.

Reaction to count and size (1 = perfect, corrected labels): CV RGB-D +0.66 / +0.87, RGB only +0.69 / +0.87, depth only +0.61 / +0.83.
Lab 2 unseen, normal tomatoes: RGB-D +0.05 / +0.47, depth only +0.18 / +0.81. Home unseen, normal tomatoes: RGB-D +0.42 / −0.28,
depth only +0.23 / −0.32. Intervals for size in the single-session fits are very wide.

### H2. Depth is used as a picture of shape ("structural use", not "metric use")

Definitions: **metric use** means the prediction changes with real-world depth values the way physical size would.
**Structural use** means it relies only on the spatial pattern of the depth map (outlines, surfaces).

| Claim | Test | Result | Verdict |
|---|---|---|---|
| With RGB present, depth values barely matter | Depth ×1.25 | Prediction ×1.00; head input changes ×0.05–0.20 as much as a different view | **Measured** |
| That is why RGB-D ≈ RGB | Same images | 18.9% vs 19.2%, difference −0.4 (−1.3 to +0.5) | **Measured** |
| Depth-only does not use depth metrically | Depth ×1.25 | ×1.08, where metric use gives about ×1.8 | **Measured** |
| Depth-only can work from the outline alone | Flat silhouette vs real depth (section "Structure tests") | 21.4% → 29.5%; flattening the surface adds +0.5 points (−0.9 to +2.0) | **Measured** |
| With RGB present, the cluster's own depth does not matter | Paste another cluster's depth in its place | Pull −0.01, change 1.3% vs control 1.0% | **Measured** |
| Depth-only follows the cluster's depth fully | Same paste, depth-only | Pull only +0.13 (+0.07 to +0.22), though change 17.1% vs control 8.0% | **Wrong as stated: weak** |

### H3. The mass head gets no size information

| Claim | Test | Result | Verdict |
|---|---|---|---|
| No size information reaches the head | Probe for pixel size at the head input | R² +0.97 in all 3 models | **Wrong** |
| The head is not given box size explicitly | Read the code | Only 14×14 features | **Measured** |
| The prediction ignores apparent size | Cluster looks 25% larger | Input changes ×0.77–0.90 of a different view; prediction ×1.04–1.11 | **Measured** |
| The final layer always discards size | Probe at the final layer | Lost in 2 models, kept in the third | **Not proven** |

### H4. What the gradient audit tells us

The audit measures how strongly a loss pushes the weights during training, not how predictions use an input.

| Audit result | Status | What it cannot tell you |
|---|---|---|
| Normalization made depth trainable (0.01 → 0.42–0.93) | **Measured** | Whether depth ends up being used |
| RGB dropout pushes depth much harder (0.6 → 2–10) | **Measured** | Whether depth is used when RGB is present |
| The mass loss is not drowned out by detection (0.66–1.27) | **Measured** | Why the model ignores geometry |

So the audit holds, but "depth is used" cannot be concluded from it. It measures the size of the pushes, not their direction,
so they could cancel out (not tested). The depth filters did end up almost where they started (similarity 0.99).

### Evidence behind the verdicts

**Input-change test (`analysis/depth_geometry_test.py`).** A size calculation predicts: depth ×1.25 → ×1.79; depth ×0.8 → ×0.56;
image zoom 1.25 with depth unchanged → ×1.79; consistent zoom (depth ÷ 1.25) → ×1.00. The ×1.79 uses mass ∝ area^1.3, a round number
chosen by hand. Fits on the data gave powers of 1.23, 1.34 and 1.51 (about ×1.73 to ×1.96); pure geometry, power 1.5, gives ×1.95.

| Change (CV models, 275 images, 46 clusters) | RGB-D | Depth only | RGB only |
|---|---|---|---|
| Depth ×1.25 | ×1.00 (1.00–1.01) | ×1.08 (1.05–1.15) | |
| Depth ×0.8 | ×1.00 (0.99–1.00) | ×0.94 (0.88–0.96) | |
| Image zoom 1.25 | ×1.05 (1.00–1.12) | ×1.15 (1.09–1.28) | ×1.06 (0.99–1.11) |
| Consistent zoom 1.25 | ×1.04 (0.99–1.12) | ×1.04 (0.99–1.16) | |

**Zoom vs no zoom** (24 Home val images, 126 Lab 2 images). Zoom augmentation is not the cause:

| Change | RGB-D, no zoom (4.2) | RGB-D, zoom (5) | Depth only, 4.2 | Depth only, 5 |
|---|---|---|---|---|
| Depth ×1.25, Home val | ×1.01 | ×1.00 | ×1.06 | ×1.03 |
| Depth ×1.25, Lab 2 | ×1.00 | ×1.00 | ×1.06 | ×1.15 |
| Image zoom, Home val | ×1.01 | ×1.02 | ×1.11 | ×1.11 |
| Image zoom, Lab 2 | ×1.05 | ×1.08 | ×1.21 | ×1.32 |

**Background test (`analysis/background_test.py`).** CV models, depth replaced by its mean, 1,121 images valid in all modes.
The baseline matched `evaluate.py` on all 1,122 images. Detection was not affected.

| RGB MAPE | Normal | Blurred background | Blank background |
|---|---|---|---|
| All | 19.2% | 24.1% | 24.6% |
| Home | 19.7% | 26.4% | 26.0% |
| Lab 2 | 19.7% | 22.3% | 21.7% |
| Lab 1 | 15.9% | 20.1% | 27.8% |
| Under 250 g | 21.0% | 41.7% | 45.0% |
| 250–500 g | 26.0% | 28.3% | 27.4% |
| 500–800 g | 16.4% | 17.5% | 18.1% |
| 800 g and up | 15.8% | 16.8% | 16.7% |

Per cluster, overall: blurred +3.3 points (+0.1 to +7.4), blank +4.6 (+0.9 to +8.7). The Home vs Lab 2 difference is not established
(Home blank +4.1, −4.6 to +15.7). One cluster drives much of the light-cluster rise: 208 g (45 images) goes 15% → 81%; the typical
cluster under 250 g shifts only ×1.06. 16 of 45 got more than 5 points worse, 5 got more than 5 points better. The mask outline stays visible.

Within a cluster at Home, twice the distance changes the RGB prediction by ×0.87 normally and ×0.78 with a blank background.
Reading only pixel size would give about ×0.125.

**Small-tomato guess (blanking pushes small tomatoes up): not supported.** Coefficient −0.07 (−0.14 to −0.01), and the range includes
zero without 208 g. With the corrected labels: −0.07 (−0.14 to −0.00), without 208 g −0.05 (−0.12 to +0.01).

**Looking inside (`analysis/look_inside.py`).**

First-layer filters: DC share of depth filters 0.18 at the start, about 0.31 after training (RGB about 0.25); similarity to the start 0.99
in every model; depth/RGB norm 0.85. Neither pure edge detectors nor absolute-value detectors: inconclusive for H2.

Probes, RGB-D, R² on clusters within trained sessions (input → final layer):

| Model | Pixel area | Distance | Mass | Which session (accuracy) |
|---|---|---|---|---|
| Held out Lab 2 | +0.97 → +0.23 | +0.91 → +0.25 | +0.94 → +0.98 | 99% → 90% (majority 58%) |
| Held out Lab 1 | +0.97 → +0.39 | +0.84 → +0.01 | +0.90 → +0.96 | 100% → 83% (majority 60%) |
| Held out Home | +0.97 → +0.85 | +0.79 → +0.75 | +0.93 → +0.98 | 97% → 91% (majority 68%) |

On unseen sessions, Lab 2, RGB-D, head input: pixel area +0.90, distance −11.9, tomato size −9.8, tomato count +0.45, real area −0.09.
Cherry/normal accuracy unseen, final layer: Lab 1 44%, Home 98%, Lab 2 93% (Lab 2 is all normal).

Head-input and final-layer change, relative to another view of the same cluster, unseen session, RGB-D:

| Change | Lab 1 input / final | Home input / final | Lab 2 input / final | Predicted mass |
|---|---|---|---|---|
| Cluster looks larger | 0.90 / 0.88 | 0.77 / 0.77 | 0.79 / 0.88 | ×1.08, ×1.04, ×1.11 |
| Depth ×1.25 | 0.05 / 0.06 | 0.20 / 0.23 | 0.07 / 0.12 | ×1.00, ×0.98, ×0.99 |

Depth-only, depth ×1.25: head input changes ×0.34–0.52 relative to another view of the same cluster; prediction ×1.05–1.11.

Probes show information is present, not that the head uses it. They are linear, and each has about 240–340 images from 25–35 clusters.
Adding real area to the final-layer probe did not help on Lab 2 (37.8%, where real area alone gets 17.0%), so this needs real training to test.

A possible reason the third model keeps size (not tested): the two models that lose it train on Home, where pixel size says little about
mass (+0.25). The third trains on the labs, where pixel size tracks mass better (+0.89 in Lab 2).

### Structure tests (`analysis/depth_structure_test.py`)

Baseline matched `evaluate.py` on all 1,041 comparisons. Annotated masks define the cluster. Up to 12 images per cluster, 521 images, 46 clusters.

**Silhouette test, depth-only input, CV models, 503 images valid in all conditions (per image):**

| Depth input | All | Home | Lab 1 | Lab 2 |
|---|---|---|---|---|
| Real depth | 21.4% | 23.0% | 21.0% | 20.7% |
| Real cluster, background neutral | 28.8% | 28.6% | 33.9% | 26.6% |
| Flat silhouette, background neutral | 29.5% | 28.9% | 35.7% | 27.1% |

For reference, always guessing the training average mass gives 78.7% on the same images. Rank with true mass over clusters: real +0.94, cluster only +0.86, silhouette +0.88.

Per cluster: removing the background +7.3 points (+3.3 to +11.4); flattening the cluster +0.5 (−0.9 to +2.0). No detection: 1, 13 and 13 of 521.
So surface detail adds nothing measurable once the background is gone. The silhouette keeps outline, pixel size and the cluster's median distance, so this
does not separate those. Edited images are unlike training, so the background cost may be partly an artefact.

**Pasted-depth test.** Only the cluster's depth is replaced, in its own place, resized to its box, at its own median distance. Control: another image of the same cluster.
Pull = slope of log(pasted / real prediction) on log(other mass / own mass), 95% over clusters.

| Model, input | Images | Pull | Median change: other / control | Over 25%, towards the other's mass | Box moved: other / control |
|---|---|---|---|---|---|
| CV, RGB-D | 521 | −0.01 (−0.02 to −0.00) | 1.3% / 1.0% | 5, 20% | 0 / 0 |
| CV, depth only | 504 | +0.13 (+0.07 to +0.22) | 17.1% / 8.0% | 173, 80% | 11 / 2 |
| Model 5, RGB-D | 48 | −0.00 (−0.02 to +0.01) | 1.1% / 0.8% | 0 | 0 / 0 |
| Model 5, depth only | 39 | +0.11 (+0.02 to +0.19) | 16.4% / 7.2% | 11, 100% | 6 / 3 |
| 7a, RGB-D | 48 | +0.01 (−0.01 to +0.03) | 2.8% / 2.2% | 0 | 0 / 0 |
| 7a, depth only | 45 | +0.07 (−0.07 to +0.24) | 18.0% / 5.6% | 15, 80% | 2 / 1 |

Model 5 and 7a use only 4 clusters here. The original cluster's depth shadow (invalid band) stays in the pasted image, and the pasted cluster is resized to the same box, which may
both limit the pull. With RGB present, swapping the cluster's depth changes nothing beyond the control.

### The real-area result (`analysis/real_area_predicted_masks.py`)

Real area = sum of Z² over the mask's valid depth pixels. The focal length is a constant the fit absorbs. `log(mass) = c + s·log(area)` is fitted
on annotated masks of the two training sessions, then applied to the masks from the model that never saw the test session. It assumes the same
camera settings in all sessions. `camera_intrinsics.json` is no longer present, but is not needed here. The CNN numbers matched `evaluate.py` on all 1,122 images.

| RGB-D input, unseen session | Home | Lab 1 | Lab 2 |
|---|---|---|---|
| CNN mass head | 30.9% (237 g) | 249% (392 g) | 37.8% (273 g) |
| Real area, type known | 19.5% (136 g) | 21.0% (46 g) | 22.4% (122 g) |
| Real area, type unknown | 47.0% (203 g) | 57.9% (101 g) | 18.2% (111 g) |
| Same with annotated masks, type known | 17.8% | 20.1% | 22.3% |

Rank over clusters, Lab 2: real area +0.96, CNN +0.14. Normal tomatoes only: real area 19.2 / 22.6 / 22.4%, CNN 33.7 / 55.8 / 37.8% (Home / Lab 1 / Lab 2).
Home cherries are the exception: CNN 17.5%, real area 21.1% (3 clusters). The type came from hand labels, so a real system would have to work it out.
Occlusion still hides mass.

Other one-measurement baselines, normal tomatoes only, annotated masks: tomato count 54.3 / 100.2 / 82.6%; pixel area 39.0 / 116.5 / 38.8%.

---

## 8. Tests that would settle what is open

| Question | Test | How to read it | Cost | Status |
|---|---|---|---|---|
| Does depth-only read shape? | Replace depth with a flat silhouette of the cluster | Done: outline works about as well as real cluster depth; see "Structure tests" | Minutes | **Done** |
| Does the rest of the depth map drive depth-only? | Keep the real cluster, swap only the depth background | Prediction moves with the other background: yes | Minutes | Not run |
| Does it count tomatoes? | Digitally erase one tomato | Prediction falls about 1/count: it counts. No change: it does not | Some hand masking | Not run |
| Is missing geometry the cause? | Train `mass = real area × predicted g per cm²`, everything else the same | Lab 2 rank rises towards +0.96 and error falls below the CNN's: yes. No gain: no | About 5 hours plus coding | Not run |
| Tomatoes or conditions? | Capture the same clusters at home and in the lab | Error changes on identical tomatoes: conditions matter | New captures | Not run |
| How much is the background? | Blank backgrounds on existing images | See below | Minutes to about 5 hours | Not run |

Blank-background versions: (1) only at test time, 3 existing models, minutes, only a clear drop is informative; (2) retrain RGB with blank
backgrounds, about 4.6 hours plus coding; (3) retrain depth-only with blank backgrounds, about 4.6 hours plus coding, which removes colour and
most lighting too. This can only isolate the background. Lighting and tomatoes stay different, so it cannot replace capturing the same clusters.
If version 2 falls close to 19%, the background was the main cause. If 2 stays high but 3 falls, lighting or colour matters. If both stay high,
tomatoes or differences like distance are the likely cause.

For new captures, break the links: large tomatoes in the lab, 90–140 g tomatoes at home, 6+ tomatoes at home, cherries in Lab 2, more distance
variation in the labs. Even 5–10 clusters would help.

**Recommended next:** decide with the mentor whether to run the real-area training, the main remaining test of cause.

---

## 9. Corrections to earlier statements

| Earlier statement | Status |
|---|---|
| "Depth is far more robust cross-session" (`docs/FINDINGS.md` B2) | Holds only for home-only models on Lab 2. Not general |
| "More tomatoes, lighter" at Home | Wrong. It came from mixing cherry and normal tomatoes |
| "Reference objects in the scene give RGB its scale" | Weakened: RGB still works with a blank background, and on the 538 g wall images |
| "Small tomatoes look bigger without surroundings" | Not supported |
| "Error doubles below 250 g without background" | Overstated: mostly the 208 g cluster |
| "Zoom augmentation causes the model to ignore size" | Wrong: the no-zoom model behaves the same |
| "7a's shuffled result shows it reads the other cluster from depth" | Wrong: pasting only the cluster's depth gives no pull with RGB present |
| "The mass head never receives size" | Wrong: pixel size is readable at the head input |
| "Depth is not used geometrically, used as a picture of shape" | First part measured; second part not proven |
| "The folds have leakage in Lab 1" | Withdrawn, based on the user's statement that the masses are different clusters |
| "Run-time about 40 minutes per 10k" | About 70 minutes per 10k, about 2 h 10 min per 20k |
| Progress update text, Oct 23: "depth-only misses almost all from the three lightest clusters" | True for 5, 6d, 7c. Not for 7b (38 of 53) or 7a (40 of 63) |
| Progress update discussion: "depth-only 35% vs average 53%" | Different frames. On the same 387: 35.8% vs 41.0% |
| `docs/FINDINGS.md`: "7a and 7b also a few heavier frames" | Actually 23 and 15. Not fixed yet |

---

## 10. Still to do

- [ ] Update `docs/FINDINGS.md`: B2 correction, the depth test, averaging, cross-validation, unseen sessions, real area, and the hypotheses above.
- [ ] Commit. Nothing from this investigation is committed yet (last commit: `d9303e1`).
- [ ] Decide the next test (section 8), and show the real-area table to the mentor.
- [ ] Update `README.md`, which still says "two capture sessions" (there are three), and `TODO.md`.
- [ ] Back up the raw data outside this machine, if not already done.

---

## 11. Where everything is

**Results (`results/eval/`)**

| Path | Contents |
|---|---|
| `cv5_20k_summary/report.txt`, `clusters.csv`, `per_image.csv` | Cross-validation combined |
| `cv5_20k_summary/tomato_counts.txt` | Count and size analysis (corrected labels since 2026-11-08) |
| `cv5_fold{0..4}_20k/val/`, `cv5_fold0/val/` | Each fold |
| `loso_20k_summary/report.txt` | Leave one session out, combined |
| `loso_{lab,home,2023}_20k/val/` | Each held-out model |
| `<run>/test_home_color/` | 538 g cluster, for 5 CV models and models 5, 7a, 7c |
| `<run>__depth_ablation/` | Depth ablation, models 5 and 7a |
| `dataset_audit.txt` | Dataset audit, baselines, behaviour by group |
| `real_area_predicted_masks/report.txt`, `per_image.csv` | Real area with predicted masks |
| `cv5_20k_background/report.txt`, `examples.jpg` | Background test |
| `depth_geometry_test/report.txt`, `<run>__<split>_report.txt` | Input-change tests |
| `look_inside/report.txt` | Filters and probes. The `.npz` feature files (about 250 MB) are ignored by git |
| `../audit_2026-10/` | Gradient audit |

**Scripts**

| Script | Purpose |
|---|---|
| `data_prep/make_min_mass_dataset.py` | Builds `data_min120/` |
| `data_prep/make_kfold_splits.py` | 5 folds, grouped by cluster |
| `data_prep/make_session_splits.py` | Leave-one-session-out splits |
| `data_prep/make_home_color_test.py` | Builds `data_home_color/` |
| `analysis/summarize_cv.py`, `analysis/summarize_loso.py` | Combine results |
| `analysis/dataset_by_mass.py` | Chart of images per cluster, ranked by mass |
| `analysis/make_tomato_count_sheet.py` | Hand-counting pages and CSV |
| `analysis/tomato_count_analysis.py` | Count and size analysis |
| `analysis/dataset_audit.py` | Dataset audit |
| `analysis/real_area_predicted_masks.py` | Real area on unseen sessions |
| `analysis/background_test.py` | Blurred and blank backgrounds |
| `analysis/depth_geometry_test.py` | Input-change test |
| `analysis/look_inside.py` | Filters and probes |

**Labels:** `analysis/cluster_inventory.csv` (current, from clusters_analysis.pdf), `analysis/tomato_counts.csv` (earlier hand counts).
`runs.csv` has a row for every run above.

**Running things:** start from the repo root with `./run.sh <script>`. Scripts in `analysis/` that import each other need `PYTHONPATH=analysis`, for example:

```bash
PYTHONPATH=analysis ./run.sh analysis/dataset_audit.py
./run.sh analysis/summarize_cv.py --name cv5_20k --run_template 'cv5_fold{}_20k' --k 5
```

Training example (leave-one-session-out, detached runs were used so they survive a closed session):

```bash
./run.sh train.py --dataset_path ./data_min120/loso/hold_lab/ --experiment_name <new_name> \
    --num_iterations 20000 --rgb_dropout 0.5 --zoom_prob 0.5 --zoom_min 0.7 --zoom_max 1.3 --seed 1
```

---

## 12. Limits on all of the above

- One seed per model. The same command can differ by several points from run to run.
- Only 3 sessions, and tomato type, tomato size, lighting and background all change together between them.
- Input-change tests use images that never occur in real life.
- Probes show information is present, not that it is used.
- Most of this describes behaviour. Only a test that changes one thing, such as the real-area training, can establish a cause.
