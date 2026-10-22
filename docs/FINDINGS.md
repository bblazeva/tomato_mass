# Findings

What we know, with the evidence for each claim. Run IDs refer to `runs.csv`.
Confidence: **solid** = holds across runs/clusters with error bars; **likely** = consistent but
one run or few clusters; **open** = hypothesis.

Unless stated otherwise: in-domain = `data_s0713` val (session 2025-07-13, 4 held-out clusters:
208 / 490 / 670 / 940 g, 190 frames). Cross-session = `test_0328` (2025-03-28 lab session, 416 frames).
MAPE values are for the final checkpoint.

---

## A. Setup and data

**A1. Removing multi-scale resize from the train mapper ("aug_fix") was the biggest win:**
val MAPE ~50% → ~15%. Random rescaling destroys the size ↔ mass signal. *Solid.*
Evidence: `docs/history/AGENTS.md` finding 1.

**A2. Late fusion and mass-head architecture changes matter little (<2 pp).** 5-conv vs deconv, MLP
size, 4×4 vs 7×7 pooling, late fusion on/off. On `data_s0713`, late fusion (6a) ended worse
(15.9% vs 10.3%). *Solid* (old split), *likely* (s0713, one run).
Evidence: AGENTS finding 2; `runs.csv` 1-LF vs 1, 6a vs 5; `docs/progress/tomato_progress.pdf` p.1.

**A3. The val set is effectively 4 clusters.** 190 frames, but 29 bursts of near-duplicates from 4
tomato clusters. Val MAPE moves 3–6 pp between checkpoints of one run (`runs.csv`: final vs best).
*Solid.*

**A4. Run-to-run noise is large.** The same command gave 10.3% (5), 18.5% (5r, stopped at 14k) and
10.6% (7c, 10k it). No seed is set. Differences of a few pp between single runs are not interpretable.
*Solid.* Evidence: `runs.csv` 5 / 5r / 7c; `docs/history/train_commands.txt` lines 1329, 1487, 1501.

---

## B. Which modality carries the mass signal

**B1. In-domain, RGB and depth are each sufficient and redundant.** With RGB dropout during training,
RGB-D ≈ RGB-only ≈ depth-only. Model 5: **10.3 / 10.9 / 11.0%**; model 4.2: 14.4 / 14.5 / 14.8%. *Solid.*
Evidence: `results/inference_legacy/results_stip_data_s0713_drop05_zoom.txt`, `..._drop05.txt`.

**B2. Cross-session, depth is far more robust, but RGB-D behaves like RGB-only.**

| Model | RGB-D | RGB only | depth only |
|---|---|---|---|
| 4.2 | 70.7 | 70.1 | 37.5 |
| 5 | 66.8 | 65.5 | 35.8 |
| 5c (colour aug) | 58.7 | 55.5 | **33.9** |
| 6b | 65.7 | 62.3 | 35.5 |

When both inputs are present the model follows RGB and inherits its failure. Colour augmentation
helps cross-session (66.8 → 58.7) but hurts in-domain (10.3 → 14.6). *Solid* (4 models).
Evidence: `results/inference_legacy/results_cross_0328*.txt`.

**B3. With RGB present, depth is only a small correction.** Replacing depth with a constant changes
model 5's predictions by **2.5% on average** (median 1.7%); its error is ~10%. Same for 6b (2.6%),
4.2 (4.1%), 5c (5.9%). Before the normalization fix (4.1): 0.2%. Cross-session similar (3–3.5%).
*Solid.* Evidence: per-frame RGB-D vs RGB-only predictions in the in-domain/cross results files.

**B4. The depth-only disadvantage in-domain comes mostly from very far frames.** Paired, 190 images,
model 7c, checkpoints 6k–10k: below 1.1 m the depth-only vs RGB-D loss gap is +0.002 to +0.009; the
3 frames beyond 1.1 m (all 490 g) have +0.26. *Open* (3 frames) — plausible: depth noise grows with
distance. Evidence: `results/audit_2026-10/val190/7c_drop_05_00/gradient_audit_per_image.csv`.

---

## C. Why depth was ignored, and what fixed it (gradient audit, Oct 2026)

Audit = per-loss gradients at frozen checkpoints (`gradient_audit_sweep2.py`). Cross-model numbers use
60 fixed val images; models 5 and 7c were also audited on all 190 with per-image values.
Summary: `results/audit_2026-10/summary.md`, figures `fig_audit_models.png`, `fig_audit_training.png`.

**C1. Normalization made depth trainable.** Mass-loss gradient on conv1 depth ÷ RGB filters:
**0.01** without the fix (4.1) vs **0.42–0.93** with it (all normalized models).
Model 5 on 190 images: 0.53, 95% interval 0.43–0.64, leave-one-cluster-out 0.47–0.57. *Solid.*

**C2. RGB dropout is where depth learns.** Logged real training gradients (depth ÷ RGB, conv1):
no dropout (run 3) ≈ **0.6** throughout; RGB-drop 0.5 → **3–5**; 0.65 → 4.4–5.9; 0.4 + depth-drop 0.1
→ 2.3–2.4 (dose–response). The mass loss pushes depth filters 2–15× harder on depth-only inputs than on
RGB-D inputs. Without dropout, depth-only inference gives no detections (run 3). *Solid.*
Evidence: `grad/depth`, `grad/rgb` in each run's `metrics.json`; `summary.csv`.

**C3. The mass loss is not drowned out by detection.** Mass ÷ detection gradient in the shared ResNet:
0.66–1.27 for all models (model 5: 0.65, interval 0.55–0.78). So reweighting the mass loss had no
bottleneck to fix — consistent with 6b/6d not improving. *Solid.*

**C4. Over training (7c, 190 images), depth ÷ RGB stays ~0.3–0.45 (one bump at 3k); RGB-D and
RGB-only mass losses are indistinguishable at every checkpoint.** Depth-only catches up around 2k–3k
but is never clearly better; from 4k RGB-D keeps a small lead. The earlier 60-image impression
"depth is easier early" does **not** hold. *Solid.*

**C5. 6b's high depth ÷ RGB (0.93) does not show in behaviour.** Its RGB gradient is normal, depth
gradient ~1.5× model 5's, but depth reliance is the same as 5 (2.6% vs 2.5%) and its depth filters
did not move more. 6d (also weight 2 for 2/3 of training) shows 0.45. Most likely a property of that
one run. *Likely.*

**C6. 60 images are enough for big contrasts, not small ones.** For model 5, 60- and 190-image values
agree (0.54 vs 0.53). Bootstrap half-width ≈ ±0.1 on depth ÷ RGB, so model differences < ~0.2 are
noise. The 60-image subset is uneven (670 g: 5 images) — see `audit_subset_60_images.md`. *Solid.*

---

## D. Loss weighting and training dynamics (June 2026)

**D1. The mass loss must not dominate early training.** Detection "locks on" at ~2k iterations
(mask loss 0.57 → 0.33). Constant mass weight 2 (6b) delayed that by ~1k (bbox AP 5.6 at 3k vs 83.7);
weight 5 (6c) prevented it entirely (AP NaN, no detections). A gentle start (6d: 0.5 → 1 → 2) gave
AP 87.8 at 2k. *Likely* (one run each). Evidence: `metrics.json` of 5, 6b, 6c, 6d.

**D2. No loss-weighting or dropout variant beat model 5 in-domain** (6a–7c: 12.0–15.9% vs 10.3%),
but all are within run-to-run noise (A4). Cross-session was only evaluated for 6b (no gain).

---

## Corrections to earlier reports

- **1.6 update, in-domain row 4.2 ("12% / 11.9% / 13.6%")** is from run 3 (no dropout), and its
  13.6% is depth = constant 50,000 mm, not depth-only. Correct row 4.2 (dropout model): 14.4 / 14.5 / 14.8%.
- **AGENTS.md "no-dropout sibling hits val 8.3%"**: that is run 3's best checkpoint (4k); final is 11.8%.
- **Gradient audits in `results/audit_legacy/gradient_audit_results2_5.txt` and `..._schechule.txt`**
  were run at mass weight 1 for models trained at 2 / 5 / schedule (unscaled gradients). The 6c audit
  is of a model that never learned detection — discard it.

---

## Open questions

1. Can a fusion use depth's cross-session robustness instead of following RGB? (B2)
2. Is the far-distance depth failure real (B4)? Check cross-session per distance.
3. Do the June variants (6a, 6d, 7a, 7b, 7c) differ cross-session? Not evaluated yet.
4. Does a detection warm-up improve mass accuracy, or only stability? Needs ≥3 runs per setting (A4).
