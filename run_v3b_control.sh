#!/bin/bash
# v3 benign 对照补跑：--no-v4-extras，share_k=-1，与 v4 全量同口径唯一差异=无 v4 特征
# 每批 1 个（bin.119 v4 在后台跑，避免内存竞争），串行 4 文件
cd /TeRed+RATE/code
source ${HOME}/tered_venv/bin/activate

for atk in bin.116 bin.117 bin.118 bin.120; do
  out=results/e5_f1_v3b_${atk}_sk-1.csv
  log=logs/e5_f1_v3b_${atk}_sk-1.log
  echo "[start] v3对照 ${atk} $(date '+%H:%M:%S')"
  python -u scripts/e5_f1_eval.py \
      --attack-file ${atk} \
      --train-files bin.1,bin.2,bin.3,bin.4,bin.5 \
      --max-records-train 400000 \
      --share-k -1 \
      --detector benign \
      --no-v4-extras \
      --results ${out} > ${log} 2>&1
  echo "[done] ${atk} $(date '+%H:%M:%S')"
done
echo "=== v3 benign 对照完成 $(date) ==="
ls -la results/e5_f1_v3b_*_sk-1.csv
