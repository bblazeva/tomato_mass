# The 60 images used in the unified gradient audit

Selected by `audit/gradient_audit_sweep2.py` (`select_stratified`, `--split val --n_batches 60 --seed 0`).
Identical for every audited model (checked: same file list in all `gradient_audit_subset.csv`).
Full list with per-image capture time and distance: `audit_subset_60_images.csv`.

All images are from the `data_s0713` **val** split (capture session 2025-07-13, home), i.e. the
4 held-out clusters. Distance = mean in-mask depth, from `results_stip_data_s0713_drop05_zoom.txt`.

| Cluster (GT mass) | Images in audit | Images in val | Bursts in audit / val | Distance in audit (mm) | Distance in val (mm) |
|---|---|---|---|---|---|
| 208 g | 20 | 45 | 10 / 13 | 352–721 | 351–813 |
| 490 g | 20 | 45 | 4 / 5 | 432–1024 | 384–1363 |
| 670 g | **5** | 46 | 3 / 5 | 446–705 | 349–934 |
| 940 g | 15 | 54 | 5 / 6 | 398–868 | 398–957 |
| **total** | **60** | 190 | 22 / 29 | | |

A burst = frames captured within the same minute (near-duplicate views).

## Caveats

- **Uneven by cluster.** Selection is stratified by the trainer's 3 mass bins (<250 g, <500 g, ≥500 g),
  not by cluster. 670 g and 940 g share the ≥500 g bin, so 670 g got only 5 images.
- **Only 4 physical clusters.** The 60 images are 22 bursts of 4 clusters, so they are far from 60
  independent samples.
- **Farthest views missing.** The audit stops at ~1.0 m; val goes up to ~1.36 m (490 g).
- The sweep CSV stores only the mean over the 60 images, not per-image values, so error bars
  cannot be computed from the existing audits.
