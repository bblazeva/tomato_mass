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
- [ ] Training saves `config.yaml`, the exact command and the git commit into its run folder
- [ ] Training refuses to start if `output/<name>` already exists (protects old runs)
- [ ] `--seed` argument, recorded per run
- [ ] Inference: `--input_mode rgbd|rgb_only|depth_only` flag instead of editing `build_rgbd_input`
- [ ] Inference writes a per-image CSV + summary JSON into the run folder (not just stdout)
- [ ] One shared input function (4-channel construction + PIXEL_MEAN/STD) for train, inference and audits
- [ ] Cross-session eval (test_0328) for runs that never got one: 6a LF, 6d schedule, 7a drop065,
      7b drop_04_01, 7c drop_05_00
- [ ] Later: reorganize the flat scripts into `tomato/` (shared code), `audit/`, `tools/`

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
