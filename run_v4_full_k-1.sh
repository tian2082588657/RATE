#!/bin/bash
# v4 全量修正版：5 文件 × share_k=-1（缓存命中，单次输出 3 operator）
# 5 进程并行，每进程峰值 ~4.3G，56 核机器无压力
cd /TeRed+RATE/code
source ${HOME}/tered_venv/bin/activate

for atk in bin.116 bin.117 bin.118 bin.119 bin.120; do
  out=results/e5_f1_v4_${atk}_sk-1.csv
  log=logs/e5_f1_v4_${atk}_sk-1.log
  echo "[start] atk=${atk} share_k=-1 -> ${out}"
  python -u scripts/e5_f1_eval.py \
      --attack-file ${atk} \
      --train-files bin.1,bin.2,bin.3,bin.4,bin.5 \
      --max-records-train 400000 \
      --share-k -1 \
      --detector benign \
      --v4-extras --v4-scale raw \
      --results ${out} > ${log} 2>&1 &
  echo "[launched] atk=${atk} PID=$!"
done

echo "=== 5 文件已并行启动，等待完成 ==="
wait
echo "=== v4 全量(share_k=-1) 完成 $(date) ==="
ls -la results/e5_f1_v4_*_sk-1.csv
