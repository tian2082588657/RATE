#!/bin/bash
# scripts_ablation/chain_e2e3.sh — 等 E1 跑完，再串行跑 E2、E3。
# 之所以串行：E1 在 287k 节点 / 9.3M 边的大分区上单进程 RSS ≈ 12 GB，
# 4 路并发已占 ~50 GB；E2/E3 同量级，并发叠加会 OOM。
set -u
ROOT=/TeRed+RATE/code2
cd "$ROOT" || exit 1

echo "[chain] wait for e1 to finish $(date)"
while pgrep -f "e1_enc_matrix.py" > /dev/null; do sleep 20; done
echo "[chain] e1 done $(date)"

bash scripts_ablation/run_experiments.sh e2 3
echo "[chain] e2 done $(date)"

bash scripts_ablation/run_experiments.sh e3 4
echo "[chain] e3 done $(date)"
echo "[chain] ALL DONE $(date)"
