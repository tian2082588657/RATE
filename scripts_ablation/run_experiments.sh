#!/bin/bash
# scripts_ablation/run_experiments.sh — 补实验的并行驱动
#
# 用法（服务器）:
#   bash scripts_ablation/run_experiments.sh e1 6        # E1 编码矩阵，并发 6
#   bash scripts_ablation/run_experiments.sh e2 6        # E2 槽位级等价性
#   bash scripts_ablation/run_experiments.sh e3 6        # E3 归约强度扫描
#
# 设计：以 (分区) 或 (分区 x 强度) 为任务单元投递，内存上限 ≈ 并发数 x 6 GB。
set -u

STAGE=${1:-e1}
MAXJ=${2:-6}
ROOT=/TeRed+RATE/code2
CACHE=/TeRed+RATE/code/cache
cd "$ROOT" || exit 1
# shellcheck disable=SC1090
source "$HOME/tered_venv/bin/activate"

SPECS=(bin.116 bin.117 bin.118 bin.119@500000 bin.120 bin.6@400000 bin.7@400000)

wait_slot() {
  while [ "$(jobs -r | wc -l)" -ge "$MAXJ" ]; do sleep 3; done
}

tag_of() { echo "$1" | sed 's/[@.]/_/g'; }

case "$STAGE" in
  e1)
    OUT=$ROOT/results/e1; LOGS=$ROOT/logs/e1; mkdir -p "$OUT" "$LOGS"
    for spec in "${SPECS[@]}"; do
      tag=$(tag_of "$spec")
      for fig in identity tered; do
        wait_slot
        python -u scripts_ablation/e1_enc_matrix.py \
          --test-files "$spec" --figs "$fig" \
          --encs none,rate_single,dual_naive,rate \
          --results "$OUT/${fig}_${tag}.csv" --cache "$CACHE" \
          --graph-cache "$CACHE/darpa/graphcache" \
          > "$LOGS/${fig}_${tag}.log" 2>&1 &
      done
    done
    wait
    echo "E1 DONE $(date)"
    ;;

  e2)
    # E2：槽位级等价性。跑两组编码：
    #   m2 = identity 用 2 通道 rate（与主表 identity 同宽的前身）
    #   m4 = 两侧都用 4 通道 rate_ratio —— 这才是"匹配宽度"的干净对照
    OUT=$ROOT/results/e2; LOGS=$ROOT/logs/e2; mkdir -p "$OUT" "$LOGS"
    for spec in "${SPECS[@]}"; do
      tag=$(tag_of "$spec")
      for pair in m2 m4; do
        if [ "$pair" = m2 ]; then ENCS="rate,rate_ratio"; else ENCS="rate_ratio"; fi
        wait_slot
        python -u scripts_ablation/e2_slot_equiv.py \
          --test-files "$spec" --figs identity,tered --encs "$ENCS" \
          --results "$OUT/sum_${pair}_${tag}.csv" --slots "$OUT/slots_${pair}_${tag}.csv" \
          --cache "$CACHE" --graph-cache "$CACHE/darpa/graphcache" \
          > "$LOGS/${pair}_${tag}.log" 2>&1 &
      done
    done
    wait
    echo "E2 DONE $(date)"
    ;;

  e3)
    # 归约强度扫描：强度 = max_total 预算；每个 (强度) 一个独立缓存目录，
    # 因为归约缓存的键不含 max_total。
    OUT=$ROOT/results/e3; LOGS=$ROOT/logs/e3; mkdir -p "$OUT" "$LOGS"
    STRENGTHS=${3:-250,1000,4000,16000,100000}
    for st in ${STRENGTHS//,/ }; do
      for spec in "${SPECS[@]}"; do
        tag=$(tag_of "$spec")
        wait_slot
        python -u scripts_ablation/e1_enc_matrix.py \
          --test-files "$spec" --figs tered \
          --encs none,dual_naive,rate \
          --max-total "$st" \
          --templates "$CACHE/darpa/e5_templates/b36c705fde93a797.jsonl" \
          --cache "$ROOT/cache_sweep/mt$st" \
          --graph-cache "$CACHE/darpa/graphcache" \
          --results "$OUT/mt${st}_${tag}.csv" \
          > "$LOGS/mt${st}_${tag}.log" 2>&1 &
      done
    done
    wait
    echo "E3 DONE $(date)"
    ;;

  e4s)
    # E4b：StreamSpot 第二数据集（图级检测，scenes split）
    OUT=$ROOT/results/e4s; LOGS=$ROOT/logs/e4s; mkdir -p "$OUT" "$LOGS"
    ARCHIVE=/TeRed+RATE/dataset/streamspot/all.tar.gz
    GIDS=${3:-0-9,500-509}
    for op in identity tered; do
      for enc in none dual_naive rate_single rate; do
        wait_slot
        python -u run_pipeline.py --archive "$ARCHIVE" --gids "$GIDS" \
          --split scenes --operator "$op" --encoding "$enc" \
          --semantic onehot --tape-base 10000 --cache "$CACHE" \
          --results "$OUT/${op}_${enc}.csv" \
          > "$LOGS/${op}_${enc}.log" 2>&1 &
      done
    done
    wait
    echo "E4S DONE $(date)"
    ;;

  *)
    echo "unknown stage: $STAGE" >&2; exit 2 ;;
esac
