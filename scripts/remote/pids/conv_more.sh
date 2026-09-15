#!/usr/bin/env bash
# conv_more.sh — 并行转换 05-16/17 缺失的 23 个 cadets bin → JSON
set -u
DATA=/TeRed+RATE/dataset/darpae5/cadets
OUT=/TeRed+RATE/dataset/darpae5/cadets_json3
PY=${HOME}/pids-lite/bin/python
AV=${HOME}/pidsmaker/scripts/avro2json.py
mkdir -p "$OUT"
BINS="94 95 96 97 98 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 118 119 120"
echo "$BINS" | tr ' ' '\n' > /tmp/conv3_list.txt
echo "[conv3] 开始 $(date '+%H:%M:%S')，共 $(wc -l < /tmp/conv3_list.txt) bin，23 路并行"
cat /tmp/conv3_list.txt | xargs -P 23 -I{} bash -c '
  B="$1"
  F=$(ls /TeRed+RATE/dataset/darpae5/cadets/ta1-cadets-1-e5-official-2.bin.${B}.gz 2>/dev/null)
  [ -z "$F" ] && { echo "[conv3] MISS bin.$B" >> /tmp/conv3_progress; exit 1; }
  N=$(basename "$F" .gz)
  if [ -s "/TeRed+RATE/dataset/darpae5/cadets_json3/$N.json" ]; then
    echo "[conv3] SKIP bin.$B (json 已存在)" >> /tmp/conv3_progress
    exit 0
  fi
  ${HOME}/pids-lite/bin/python ${HOME}/pidsmaker/scripts/avro2json.py \
      --in "$F" --out "/TeRed+RATE/dataset/darpae5/cadets_json3/$N.json" \
      > "${HOME}/conv3_$B.log" 2>&1
  rc=$?
  echo "[conv3] bin.$B rc=$rc $(date "+%H:%M:%S")" >> /tmp/conv3_progress
' _ {}
echo "[conv3] 全部完成 $(date '+%H:%M:%S')"
ls -la "$OUT" | tail -3
