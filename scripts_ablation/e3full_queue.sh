#!/usr/bin/env bash
# e3full_queue.sh — 大分区强度扫描波次队列：等当前 mt16000 波结束后接力 4000/250/1000。
set -u
cd /TeRed+RATE/code2 || exit 1
source ~/tered_venv/bin/activate
TPL=/TeRed+RATE/code/cache/darpa/e5_templates/b36c705fde93a797.jsonl
GCACHE=/TeRed+RATE/code/cache/darpa/graphcache
OUT=/TeRed+RATE/code2/results/e3full
LOGS=/TeRed+RATE/code2/logs/e3full
PARTS="bin.120 bin.116 bin.117 bin.118"

wait_wave () {
  local st=$1
  while [ "$(pgrep -fc "max-total $st")" -gt 0 ]; do sleep 60; done
}

launch_wave () {
  local st=$1 spec tag
  for spec in $PARTS; do
    tag=$(echo "$spec" | tr ./@ ___)
    if [ -f "$OUT/mt${st}_${tag}.csv" ]; then continue; fi
    PYTHONHASHSEED=0 nohup python -u scripts_ablation/e1_enc_matrix.py       --test-files "$spec" --figs tered --encs none,dual_naive,rate       --max-total "$st" --templates "$TPL"       --cache "/TeRed+RATE/code2/cache_sweep/mt$st" --graph-cache "$GCACHE"       --results "$OUT/mt${st}_${tag}.csv" > "$LOGS/mt${st}_${tag}.log" 2>&1 < /dev/null &
  done
}

wait_wave 16000
echo "[queue] mt16000 wave done $(date)"
for st in 4000 250 1000; do
  launch_wave "$st"
  echo "[queue] wave $st launched $(date)"
  wait_wave "$st"
  echo "[queue] wave $st done $(date)"
done
echo "[queue] ALL DONE $(date)"
