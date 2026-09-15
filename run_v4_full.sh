#!/bin/bash
# v4 全量 5 文件 × 3 share_k × naive+RATE 检测器=benign
cd /TeRed+RATE/code
source ${HOME}/tered_venv/bin/activate

for atk in bin.116 bin.117 bin.118 bin.119 bin.120; do
  for sk in 0 1 -1; do
    out=results/e5_f1_v4_${atk}_sk${sk}.csv
    log=logs/e5_f1_v4_${atk}_sk${sk}.log
    echo "[start] atk=${atk} share_k=${sk} -> ${out}"
    python -u scripts/e5_f1_eval.py \
        --attack-file ${atk} \
        --train-files bin.1,bin.2,bin.3,bin.4,bin.5 \
        --max-records-train 400000 \
        --share-k ${sk} \
        --detector benign \
        --v4-extras --v4-scale raw \
        --results ${out} > ${log} 2>&1
    echo "[done ] atk=${atk} share_k=${sk} ($(date))"
  done
done
echo "=== v4 全量完成 ==="