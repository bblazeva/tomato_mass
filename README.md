# Tomato cluster mass estimation from RGB-D

A Mask R-CNN (ResNet-50 + FPN, Detectron2) takes a 4-channel input (BGR + depth in mm), detects and
segments the cluster, and an added mass head predicts `log(mass_kg)`. Training uses a depth-consistent
zoom augmentation (apparent size × s, depth × 1/s) and input modality dropout, which replaces RGB with
its mean for a share of the training images.

## Repository layout

```
train.py              train a model
evaluate.py           evaluate a model with different inputs (RGB-D, RGB only, depth only, modified depth)
utils_detection.py    dataset loading
mass_evaluator.py     mass metrics during training
run.sh                runs any script in the project environment

data_prep/            build the dataset splits, preview the zoom augmentation
audit/                gradient audit: which loss trains which part of the model
analysis/             feature analysis and tabular baselines
detectron2/           Detectron2 with the project's changes (RGB-D model, mass head, mass annotations)
legacy/               scripts from earlier pipelines, kept for reference

runs.csv              every training run: command, settings, results
docs/FINDINGS.md      what has been learned so far
docs/progress/        progress updates and their figures
docs/history/         earlier notes and plans
results/eval/         evaluation results per run and test split
results/audit_2026-10/  gradient audit tables and figures
```

Commands in `runs.csv` and `docs/history/` use the script names from when each run was launched:
`train.py` was then called `train_new_clean.py`.

## Data

The images, depth maps and trained checkpoints are not included in this repository. Two capture
sessions are used, split so that no cluster appears in more than one of train and validation:

| Split | Session | Frames | Purpose |
|---|---|---|---|
| `train` | 2025-07-13 (home) | 383 (10 clusters) | training |
| `val` | 2025-07-13 (home) | 190 (4 clusters: 208, 490, 670, 940 g) | same-session evaluation |
| `test_0328` | 2025-03-28 (lab) | 416 (21 clusters) | cross-session evaluation |
| `test_1201` | 2023-12-01 | 201 | additional cross-session split |

Expected layout, with the mass in grams at the start of each file name
(e.g. `208_2025-07-13-18-11-23.jpg` is 0.208 kg):

```
data_s0713/
  gt_json/<split>/<split>.json   COCO annotations
  images/<split>/<name>.jpg|png  RGB images
  depth/<name>.npy               depth in mm, aligned to the RGB image
```

## Setup

Requires Python 3.6, PyTorch 1.7 and a CUDA GPU.

```bash
conda env create -f environment.yml
conda activate tomato_thesis
cd detectron2 && python setup.py develop && cd ..
```

`run.sh` expects the environment at `~/miniconda3/envs/tomato_thesis`; change `ENV` in that file if yours
is elsewhere. Start scripts from the repository root through `run.sh`, so they can find the shared modules.

Training also needs initial weights, Mask R-CNN weights with a 4-channel first layer, passed with `--weights`.

## Usage

```bash
# Train (model 5 setup). Writes to output/<name>/ and refuses to overwrite an existing run.
./run.sh train.py --dataset_path ./data_s0713/ --experiment_name <name> \
    --num_iterations 15000 --rgb_dropout 0.5 --zoom_prob 0.5 --zoom_min 0.7 --zoom_max 1.3 --seed 1

# Evaluate with RGB-D, RGB only and depth only. Add --save_viz 20 to save images of the 20 worst frames.
./run.sh evaluate.py --run <name> --splits val test_0328

# Test the influence of depth: real RGB with modified depth
./run.sh evaluate.py --run <name> --splits val test_0328 --tag depth_ablation \
    --input_modes rgbd rgb_only rgbd_zero rgbd_flat rgbd_shuffled

# Gradient audit over all checkpoints of a run
./run.sh audit/gradient_audit_sweep2.py --run_dir output/<name> --dataset_path ./data_s0713/ \
    --split val --n_batches 190 --out output/<name>/audit/gradient_audit_sweep.csv \
    --subset_out output/<name>/audit/subset.csv --meta_out output/<name>/audit/meta.json
```

Evaluation results go to `results/eval/<name>/<split>/`: `report.txt` is the readable summary, and
`per_image.csv` and `summary.json` hold the full data.
