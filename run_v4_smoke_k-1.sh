#!/bin/bash
cd /TeRed+RATE/code
source ${HOME}/tered_venv/bin/activate
exec python -u scripts/e5_f1_eval.py \
    --attack-file bin.118 \
    --train-files bin.1,bin.2,bin.3,bin.4,bin.5 \
    --max-records-train 400000 \
    --share-k -1 \
    --detector benign \
    --v4-extras --v4-scale raw \
    --results results/e5_f1_v4_smoke_bin118_k-1.csv