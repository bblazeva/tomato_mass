#!/bin/bash
# Run any script with the thesis environment (handles the libstdc++ workaround).
#   ./run.sh train.py --dataset_path ./data_s0713/ ...
#   CUDA_VISIBLE_DEVICES=1 ./run.sh evaluate.py --run <run> ...
#   ./run.sh legacy/<script>.py ...      (repo root is on PYTHONPATH, so shared modules resolve)
ENV=$HOME/miniconda3/envs/tomato_thesis
cd "$(dirname "$0")" || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
LD_PRELOAD=$ENV/lib/libstdc++.so.6 exec "$ENV/bin/python3" "$@"
