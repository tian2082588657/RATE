#!/usr/bin/env bash
# run_e3_redo.sh — 用修复后的确定性归约代码重做 fig2(b) 的归约强度扫描。
#
# 只扫论文 fig2(b) 实际用到的三个截断窗口：bin.119@500000 / bin.6@400000 / bin.7@400000。
# 每个 (强度, 单元) 用独立 --cache —— 归约缓存键不含 max_total，必须物理隔离防竞态。
# 产出 CSV 供 figures/make_fig_png.py 画横坐标=实际边归约率 的曲线。
set -u
cd /TeRed+RATE/code2 || exit 1
source ~/tered_venv/bin/activate

TPL=/TeRed+RATE/code/cache/darpa/e5_templates/b36c705fde93a797.jsonl
GCACHE=/TeRed+RATE/code/cache/darpa/graphcache
ROOTCACHE=/TeRed+RATE/code2/cache_e3
OUT=/TeRed+RATE/code2/results/e3redo
LOGS=/TeRed+RATE/code2/logs/e3redo
MI=300

mkdir -p "$OUT" "$LOGS"

joblist=$(mktemp)
for st in 250 1000 4000 16000; do
  for spec in bin.119@500000 bin.6@400000 bin.7@400000; do
    tag=$(echo "$spec" | tr ./@ ___)
    echo "$st $spec $tag" >> "$joblist"
  done
done

run_one () {
  st=$1; spec=$2; tag=$3
  out="$OUT/st${st}_${tag}.csv"
  [ -f "$out" ] && { echo "[e3redo] skip $st $spec"; return 0; }
  PYTHONHASHSEED=0 python -u scripts_ablation/e1_enc_matrix.py \
      --test-files "$spec" --figs tered --encs none,rate \
      --max-instances $MI --max-total "$st" \
      --templates "$TPL" --cache "$ROOTCACHE/st${st}_${tag}" --graph-cache "$GCACHE" \
      --results "$out" > "$LOGS/st${st}_${tag}.log" 2>&1
  echo "[e3redo] rc=$? st=$st $spec $(date)"
}
export -f run_one
export TPL GCACHE ROOTCACHE OUT LOGS MI

cat "$joblist" | xargs -P 6 -n 3 bash -c 'run_one "$0" "$1" "$2"'
rm -f "$joblist"
echo "[e3redo] ALL DONE $(date)"
