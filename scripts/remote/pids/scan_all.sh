#!/usr/bin/env bash
# scan_all.sh — 28 路并行全量扫描 122 bin 的 GT uuid 落点
set -u
cd ~
DATA=/TeRed+RATE/dataset/darpae5/cadets
GT=~/gt_missing.txt
OUT=/tmp/scan_out
rm -rf "$OUT" && mkdir -p "$OUT"
ls "$DATA"/ta1-cadets-1-e5-official-2.bin.*.gz | sort > /tmp/bin_list.txt
echo "[scan_all] 共 $(wc -l < /tmp/bin_list.txt) 个 bin，28 路并行开始 $(date '+%H:%M:%S')"
cat /tmp/bin_list.txt | xargs -P 28 -I{} bash -c '
  F="$1"; B=$(basename "$F")
  ~/pids-lite/bin/python ~/scan_gt_bins.py --gt ~/gt_missing.txt --file "$F" --out "/tmp/scan_out/$B.tsv" > "/tmp/scan_out/$B.err" 2>&1
  echo "[done] $B rc=$?" >> /tmp/scan_out/.progress
' _ {}
echo "[scan_all] 全部完成 $(date '+%H:%M:%S')"
# 合并（过滤 0 命中行由分析端做）
head -q /tmp/scan_out/*.tsv | sort > ~/gt_bin_hits.tsv
echo "[scan_all] 合并 -> ~/gt_bin_hits.tsv ($(wc -l < ~/gt_bin_hits.tsv) 行)"
