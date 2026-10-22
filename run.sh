#!/bin/bash
# Run any script with the thesis environment (handles the libstdc++ workaround).
#   ./run.sh train_new_clean.py --dataset_path ./data_s0713/ ...
#   CUDA_VISIBLE_DEVICES=1 ./run.sh inference_rgbd_viz.py ...
ENV=$HOME/miniconda3/envs/tomato_thesis
cd "$(dirname "$0")" || exit 1
LD_PRELOAD=$ENV/lib/libstdc++.so.6 exec "$ENV/bin/python3" "$@"
