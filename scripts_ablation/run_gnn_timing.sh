#!/bin/bash
cd /TeRed+RATE/code || exit 1
PY=${HOME}/miniconda3/envs/pids/bin/python
export PYTHONUNBUFFERED=1
rm -f results/e6e_gsmoke.csv
echo "=== GNN timing: tered:rate_ratio, 3 epochs, 1 small test file ==="
time $PY scripts_ablation/e6e_gnn_baseline.py \
  --detectors gnn \
  --configs tered:rate_ratio \
  --test-files bin.6@400000 \
  --gnn-epochs 3 \
  --results results/e6e_gsmoke.csv 2>&1 | tail -30
echo "=== nvidia-smi ==="
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
