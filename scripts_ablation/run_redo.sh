#!/usr/bin/env bash
# run_redo.sh — 用修复后的确定性归约代码，重算全部 7 个评估单元的 identity/tered 归约图。
#
# 背景：09-17 之前的 reduction/tered.py 在 set 上迭代未排序，Python 字符串哈希随机化使
# 归约结果随进程变化；论文主结果（tab:mechan / tab:main / 各消融表）都建立在那些
# 不可复现的归约图之上。09-21 补齐了剩余两处顺序修复后，本脚本在干净缓存上重算。
#
# 要点：
#   - 归约缓存的键不含 max_total，因此每个强度必须一个独立 --cache；本脚本只用 PRODCACHE。
#   - 训练图（bin.1..bin.5, tr400000）的归约结果由 warm-up 串行写入，避免并发同写。
#   - PYTHONHASHSEED=0 + 代码内 sorted => 结果可复现（detC/detD 双跑已验证）。
set -u
cd /TeRed+RATE/code2 || exit 1
source ~/tered_venv/bin/activate

TPL=/TeRed+RATE/code/cache/darpa/e5_templates/b36c705fde93a797.jsonl
GCACHE=/TeRed+RATE/code/cache/darpa/graphcache
PRODCACHE=/TeRed+RATE/code2/cache_redo
OUT=/TeRed+RATE/code2/results/redo
LOGS=/TeRed+RATE/code2/logs/redo
MI=300
MT=5000

mkdir -p "$OUT" "$LOGS"

# 1) 预热：最小单元串行跑完，写入 5 个训练图 + 自身测试图的 identity/tered 归约产物
if [ ! -f "$OUT/redo_bin_7_400000.csv" ]; then
  PYTHONHASHSEED=0 python -u scripts_ablation/e1_enc_matrix.py \
      --test-files bin.7@400000 --figs identity,tered --encs rate \
      --max-instances $MI --max-total $MT \
      --templates "$TPL" --cache "$PRODCACHE" --graph-cache "$GCACHE" \
      --results "$OUT/redo_bin_7_400000.csv" > "$LOGS/warmup.log" 2>&1
  echo "[redo] warm-up rc=$? $(date)" >> "$LOGS/progress"
fi

# 2) 其余单元并行（每单元 ~2GB RSS，4 路安全）
for spec in bin.116 bin.117 bin.118 bin.120; do
  tag=$(echo "$spec" | tr ./@ ___)
  [ -f "$OUT/redo_${tag}.csv" ] && continue
  PYTHONHASHSEED=0 nohup python -u scripts_ablation/e1_enc_matrix.py \
      --test-files "$spec" --figs identity,tered --encs rate \
      --max-instances $MI --max-total $MT \
      --templates "$TPL" --cache "$PRODCACHE" --graph-cache "$GCACHE" \
      --results "$OUT/redo_${tag}.csv" > "$LOGS/redo_${tag}.log" 2>&1 < /dev/null &
done
wait
echo "[redo] large wave 1 done $(date)" >> "$LOGS/progress"

for spec in bin.119@500000 bin.6@400000; do
  tag=$(echo "$spec" | tr ./@ ___)
  [ -f "$OUT/redo_${tag}.csv" ] && continue
  PYTHONHASHSEED=0 nohup python -u scripts_ablation/e1_enc_matrix.py \
      --test-files "$spec" --figs identity,tered --encs rate \
      --max-instances $MI --max-total $MT \
      --templates "$TPL" --cache "$PRODCACHE" --graph-cache "$GCACHE" \
      --results "$OUT/redo_${tag}.csv" > "$LOGS/redo_${tag}.log" 2>&1 < /dev/null &
done
wait
echo "[redo] ALL DONE $(date)" >> "$LOGS/progress"
