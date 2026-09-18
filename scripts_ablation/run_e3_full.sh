#!/usr/bin/env bash
# run_e3_full.sh — 在 4 个全尺寸分区上做归约强度扫描（与 chain_e3.sh 同口径）。
# 每个强度：先串行跑一个分区预热训练图归约缓存（缓存键不含 max_total），再并行跑其余。
set -u
cd "$(dirname "$0")/.." || exit 1
ROOT=$(pwd)
source "$HOME/tered_venv/bin/activate"
CACHE=/TeRed+RATE/code/cache
TPL=$CACHE/darpa/e5_templates/b36c705fde93a797.jsonl
GCACHE=$CACHE/darpa/graphcache
OUT=$ROOT/results/e3full
LOGS=$ROOT/logs/e3full
mkdir -p "$OUT" "$LOGS"

STRENGTHS=${1:-16000,4000,1000,250}
PARTS=${2:-bin.120,bin.116,bin.117,bin.118}

run_one () {
  local st=$1 spec=$2 tag
  tag=$(echo "$spec" | tr ./@ ___)
  PYTHONHASHSEED=0 python -u scripts_ablation/e1_enc_matrix.py     --test-files "$spec" --figs tered --encs none,dual_naive,rate     --max-total "$st" --templates "$TPL"     --cache "$ROOT/cache_sweep/mt$st" --graph-cache "$GCACHE"     --results "$OUT/mt${st}_${tag}.csv" > "$LOGS/mt${st}_${tag}.log" 2>&1
}

for st in ${STRENGTHS//,/ }; do
  first=1
  for spec in ${PARTS//,/ }; do
    if [ "$first" = "1" ]; then
      echo "[e3f] strength $st warm-up $spec $(date)"
      run_one "$st" "$spec"
      first=0
    else
      echo "[e3f] strength $st parallel $spec $(date)"
      run_one "$st" "$spec" &
    fi
  done
  wait
  echo "[e3f] strength $st complete $(date)"
done
echo "E3FULL DONE $(date)"
