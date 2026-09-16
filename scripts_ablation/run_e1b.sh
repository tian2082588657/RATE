#!/bin/bash
# scripts_ablation/run_e1b.sh — E1b 语义-only 对照（问题-0916-2 P0 必做）
# encoding=semantic_only：X 仅语义 one-hot，无任何度/结构列（跳过 v4）。
# 每分区一个进程（identity+tered 两臂），结果按分区分文件，避免并发写同一 CSV。
set -u
ROOT=/TeRed+RATE/code2
CACHE=/TeRed+RATE/code/cache
cd "$ROOT" || exit 1
source "$HOME/tered_venv/bin/activate"

OUT=$ROOT/results/e1; LOGS=$ROOT/logs/e1
mkdir -p "$OUT" "$LOGS"

for spec in bin.116 bin.117 bin.118 bin.119@500000 bin.120 bin.6@400000 bin.7@400000; do
  tag=$(echo "$spec" | sed 's/[@.]/_/g')
  ( nohup python -u scripts_ablation/e1_enc_matrix.py \
      --test-files "$spec" --figs identity,tered --encs semantic_only \
      --results "$OUT/e1b_${tag}.csv" \
      --cache "$CACHE" --graph-cache "$CACHE/darpa/graphcache" \
      < /dev/null > "$LOGS/e1b_${tag}.log" 2>&1 & )
  echo "launched $spec"
done
echo "E1B LAUNCHED $(date)"
