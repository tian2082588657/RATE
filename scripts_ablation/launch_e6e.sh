#!/bin/bash
cd /TeRed+RATE/code || exit 1
PY=${HOME}/miniconda3/envs/pids/bin/python
export PYTHONUNBUFFERED=1
mkdir -p logs results
# 主运行：3 检测器配置 × 6 编码全量（cosine + gnn，7 分区）
nohup $PY scripts_ablation/e6e_gnn_baseline.py \
  --detectors cosine,gnn \
  --configs identity:rate,tered:dual_naive,tered:rate,tered:rate_ratio,tered:rate_single,tered:none \
  --test-files bin.116,bin.117,bin.118,bin.119@500000,bin.120,bin.6@400000,bin.7@400000 \
  --gnn-epochs 20 \
  --results results/e6e_gnn_baseline.csv \
  > logs/e6e_gnn_baseline.log 2>&1 &
echo "PID=$!"
sleep 5
head -5 logs/e6e_gnn_baseline.log
