#!/bin/bash
cd /TeRed+RATE/code || exit 1
PY=${HOME}/miniconda3/envs/pids/bin/python
export PYTHONUNBUFFERED=1
while pgrep -f "e6e_gnn_baselin[e]" > /dev/null; do sleep 20; done
echo "[followup] main finished $(date)" >> logs/followup.log

SPECS="bin.116,bin.117,bin.118,bin.119@500000,bin.120,bin.6@400000,bin.7@400000"

$PY scripts_ablation/e6e_gnn_baseline.py \
  --detectors cosine,gnn --configs identity:rate_ratio \
  --test-files $SPECS --gnn-epochs 20 \
  --results results/e6e_identity_ratestar.csv \
  > logs/e6e_identity_ratestar.log 2>&1
echo "[followup] identity:rate_ratio done $(date)" >> logs/followup.log

$PY scripts_ablation/e6e_gnn_baseline.py \
  --detectors cosine --clean-train \
  --configs identity:rate,tered:rate_ratio,tered:dual_naive \
  --test-files $SPECS \
  --results results/e6e_cleantrain.csv \
  > logs/e6e_cleantrain.log 2>&1
echo "[followup] clean-train done $(date)" >> logs/followup.log

$PY scripts_ablation/e6g_cost_profile.py \
  --specs bin.116,bin.117,bin.6@400000 \
  --results results/e6g_cost.csv \
  > logs/e6g_cost.log 2>&1
echo "[followup] cost done $(date)" >> logs/followup.log
echo "[followup] ALL DONE $(date)" >> logs/followup.log
