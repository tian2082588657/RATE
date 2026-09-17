#!/usr/bin/env bash
# chain_e3.sh — 归约强度扫描（E3）。
# 关键：归约缓存的键不含 max_total，同一强度的多个分区会并发写同一训练图缓存 -> 竞态。
# 因此每个强度先串行跑一个分区（预热训练缓存），再把其余分区并行跑（只读缓存）。
set -u
cd "$(dirname "$0")/.." || exit 1
ROOT=$(pwd)
source "$HOME/tered_venv/bin/activate"
CACHE=/TeRed+RATE/code/cache
TPL=$CACHE/darpa/e5_templates/b36c705fde93a797.jsonl
GCACHE=$CACHE/darpa/graphcache
OUT=$ROOT/results/e3
LOGS=$ROOT/logs/e3
mkdir -p "$OUT" "$LOGS"

STRENGTHS=${1:-250,1000,4000,16000}
WARM=${2:-bin.7@400000}
REST=${3:-bin.6@400000,bin.119@500000}

run_one () {
  local st=$1 spec=$2
  local tag
  tag=$(echo "$spec" | tr './@' '___')
  # PYTHONHASHSEED=0：reduction/tered.py 的实例匹配已改为 sorted(cand)，
  # 这里再钉住哈希种子，双保险，保证跨进程/跨机结果完全一致。
  PYTHONHASHSEED=0 python -u scripts_ablation/e1_enc_matrix.py \
    --test-files "$spec" --figs tered --encs none,dual_naive,rate \
    --max-total "$st" --templates "$TPL" \
    --cache "$ROOT/cache_sweep/mt$st" --graph-cache "$GCACHE" \
    --results "$OUT/mt${st}_${tag}.csv" > "$LOGS/mt${st}_${tag}.log" 2>&1
}

for st in ${STRENGTHS//,/ }; do
  echo "[e3] strength max_total=$st warm-up on $WARM $(date)"
  run_one "$st" "$WARM"
  echo "[e3] strength $st warm-up done $(date)"
  pids=()
  for spec in ${REST//,/ }; do
    run_one "$st" "$spec" &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done
  echo "[e3] strength $st complete $(date)"
done
echo "E3 DONE $(date)"
