#!/bin/bash
cd /TeRed+RATE/code || exit 1
PY=${HOME}/miniconda3/envs/pids/bin/python
PYTHONPATH=/TeRed+RATE/code
export PYTHONUNBUFFERED=1
echo "=== SMOKE: cosine only, 1 config, 2 test files ==="
$PY scripts_ablation/e6e_gnn_baseline.py \
  --detectors cosine \
  --configs tered:rate_ratio,tered:dual_naive \
  --test-files bin.119@500000,bin.6@400000 \
  --results results/e6e_smoke.csv 2>&1 | tail -40
echo "=== SMOKE CSV ==="
cat results/e6e_smoke.csv
