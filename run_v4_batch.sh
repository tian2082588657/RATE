#!/bin/bash
# v4 补跑：每批 2 个并行（实测单进程解析攻击图峰值 RSS 12-18GB，5 并行会 OOM）
# 用法: bash /tmp/run_v4_batch.sh bin.116 bin.117 bin.118 bin.120
# 内部自动两两一批，批内并行、批间串行
cd /TeRed+RATE/code
source ${HOME}/tered_venv/bin/activate

ALL=("$@")
i=0
while [ $i -lt ${#ALL[@]} ]; do
  A=${ALL[$i]}
  B=${ALL[$((i+1))]}
  echo "===== 批次 [$i]: ${A} (+ ${B:-无}) $(date '+%H:%M:%S') ====="
  # 启动 A
  python -u scripts/e5_f1_eval.py \
      --attack-file ${A} \
      --train-files bin.1,bin.2,bin.3,bin.4,bin.5 \
      --max-records-train 400000 \
      --share-k -1 \
      --detector benign \
      --v4-extras --v4-scale raw \
      --results results/e5_f1_v4_${A}_sk-1.csv > logs/e5_f1_v4_${A}_sk-1.log 2>&1 &
  PA=$!
  # 启动 B（若存在）
  if [ -n "$B" ]; then
    python -u scripts/e5_f1_eval.py \
        --attack-file ${B} \
        --train-files bin.1,bin.2,bin.3,bin.4,bin.5 \
        --max-records-train 400000 \
        --share-k -1 \
        --detector benign \
        --v4-extras --v4-scale raw \
        --results results/e5_f1_v4_${B}_sk-1.csv > logs/e5_f1_v4_${B}_sk-1.log 2>&1 &
    PB=$!
    echo "[launched] ${A} PID=$PA, ${B} PID=$PB"
  else
    echo "[launched] ${A} PID=$PA"
  fi
  wait
  echo "===== 批次结束 $(date '+%H:%M:%S') ====="
  i=$((i+2))
done
echo "=== 全部补跑完成 $(date) ==="
ls -la results/e5_f1_v4_*_sk-1.csv
