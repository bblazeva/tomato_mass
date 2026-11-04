# TODO

## Phase 1 — Set up the clean repo ✅ (2026-10-22)
- [x] New repo `~/tomato_thesis`, active scripts copied unchanged, old repo untouched
- [x] Conda env `tomato_thesis` cloned; detectron2 loaded from this repo
- [x] `output/` on `/mnt/md0` with links to every old run
- [x] Acceptance test: model 5 val reproduced frame-for-frame (MAE 0.066 kg, MAPE 10.3%)
- [x] First git commit

## Phase 2 — Records ✅ (2026-10-22)
- [x] `runs.csv`: one row per run since May — command (from bash history), date, val MAPE,
      in-domain ablation (RGB-D / RGB / depth), cross-session (test_0328), checkpoint location, notes
- [x] `docs/FINDINGS.md`: findings 1–10 from AGENTS.md + June experiments + gradient audit (Oct 21–22)
- [x] Record the fix for row 4.2 of the 1.6 update (12% / 11.9% were the no-dropout model;
      the dropout model is 14.4 / 14.5 / 14.8%; 13.6% was depth = 50,000 mm) — see FINDINGS "Corrections"
- [ ] **You:** read `runs.csv` + `docs/FINDINGS.md` once and fix anything that doesn't match your memory

## Phase 3 — Fix the pain points (one at a time; re-run the acceptance test after each)
- [x] Training saves `run_info.json` (command, args, seed, GPU, start/finish) + `config.yaml` into its run folder
- [x] Training refuses to start if `output/<name>` already exists (protects old runs)
- [x] `--seed` argument (random if omitted, always recorded). Same seed → losses within ~0.3% after
      20 it; different seeds differ 30–45%. Not bit-identical (GPU nondeterminism).
- [x] Cleaned `train.py`, `utils_detection.py`, `mass_evaluator.py`; `iter_4_depth_matched` → `depth`
- [x] `evaluate.py` replaces `inference_rgbd_viz.py`: `--input_modes rgbd rgb_only depth_only` flag,
      writes `results/eval/<run>/<split>/{per_image.csv, summary.json, report.txt}`, `--save_viz N`.
      Verified frame-for-frame on model 5 (1,818 frames)
- [x] Repo organized (2026-10-23): core pipeline at the root, `data_prep/`, `audit/`, `analysis/`, `legacy/`;
      duplicate `data_s0713/test` split deleted; `inspect_zoom_aug.py` uses the training code itself
- [ ] One shared input function (4-channel construction + PIXEL_MEAN/STD) for train, inference and audits
- [x] Cross-session eval (test_0328) for 6d, 7a, 7b, 7c (2026-10-23) → `runs.csv`, FINDINGS B2.
      6a (late fusion) skipped: abandoned variant, needs the old detectron2

## Phase 4 — Thesis
- [ ] Supervisor meeting: model table, audit figures (`results/audit_2026-10/fig_audit_*.png`),
      depth-reliance result (removing depth changes predictions by ~2.5%), RGB-dropout result
      (depth gets 3–6× more gradient with dropout vs 0.6× without)
- [ ] Decide the next experiment after the meeting (detection warm-up / two-stage / fusion that
      can fall back to depth)
- [ ] For any claim about small differences: ≥3 runs per setting (same command gives 10–18% MAPE)

## Habits
1. Add the `runs.csv` row **before** launching a run.
2. Never toggle an experiment by editing code — use a flag.
3. `git commit` before every training run.
4. After every result: two lines in `docs/FINDINGS.md`, with the evidence file.
