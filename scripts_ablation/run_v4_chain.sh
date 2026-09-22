#!/usr/bin/env bash
# run_v4_chain.sh — 用确定性归约（cache_redo）重跑论文的全部下游评测链路。
#
# 前提：run_redo.sh 已把 7 个单元的 identity/tered 归约图写进 CACHE，
#       下面每一步都只是「读缓存归约图 + 特征 + 评分 + 聚合」，不再重新归约。
#
# 产出（results/v4/）：
#   e1v4_<unit>.csv            编码谱系（semantic_only / none / rate_single / dual_naive / rate
#                              × identity / tered）  -> fig2(a) 与 tab:encgraph 的 semantic-only 行
#   e6_alert_full_v3.csv       四配置全流程告警指标           -> tab:main, tab:abl-budget, tab:runtime
#   e6d_ablation_v3.csv        A1/A3/A4/A5 消融             -> tab:abl-a1, abl-a3, abl-v4, abl-enc
#   e6d_ratestar_v3.csv        TeRed+RATE* 的 A1/A3/A4      -> 同上（RATE* 口径）
#   e7_origunit.csv            统一评估单元                  -> tab:equivalence / 正文等价性
#
# 注：文件名沿用 v3 后缀，是为了让 scripts_ablation/gen_paper_tex.py 不加改动直接吃下；
#     目录 results/v4/ 本身即代表「确定性重算版」。
set -u
cd /TeRed+RATE/code2 || exit 1
source ~/tered_venv/bin/activate

TPL=/TeRed+RATE/code/cache/darpa/e5_templates/b36c705fde93a797.jsonl
CACHE=/TeRed+RATE/code2/cache_redo
GCACHE=/TeRed+RATE/code/cache/darpa/graphcache
OUT=/TeRed+RATE/code2/results/v4
LOGS=/TeRed+RATE/code2/logs/v4
TESTS=bin.116,bin.117,bin.118,bin.119@500000,bin.120,bin.6@400000,bin.7@400000
ENCS=semantic_only,none,rate_single,dual_naive,rate

mkdir -p "$OUT" "$LOGS"
export PYTHONHASHSEED=0

# ---------- 阶段 1：编码谱系（按单元并行，3 路控内存） ----------
run_unit () {
  spec=$1
  tag=$(echo "$spec" | tr ./@ ___)
  out="$OUT/e1v4_${tag}.csv"
  [ -f "$out" ] && { echo "[v4] skip e1 $spec"; return 0; }
  PYTHONHASHSEED=0 python -u scripts_ablation/e1_enc_matrix.py \
      --test-files "$spec" --figs identity,tered --encs "$ENCS" \
      --max-instances 300 --max-total 5000 \
      --templates "$TPL" --cache "$CACHE" --graph-cache "$GCACHE" \
      --results "$out" > "$LOGS/e1v4_${tag}.log" 2>&1
  echo "[v4] rc=$? e1 $spec $(date)"
}
export -f run_unit
export TPL CACHE GCACHE OUT LOGS ENCS

joblist=$(mktemp)
printf '%s\n' bin.116 bin.117 bin.118 bin.119@500000 bin.120 bin.6@400000 bin.7@400000 > "$joblist"
cat "$joblist" | xargs -P 3 -I{} bash -c 'run_unit "{}"'
rm -f "$joblist"
echo "[v4] stage1 done $(date)" >> "$LOGS/progress"

# ---------- 阶段 2：四配置全流程 ----------
if [ ! -f "$OUT/e6_alert_full_v3.csv" ]; then
  PYTHONHASHSEED=0 python -u scripts/e6_alert_eval.py \
      --test-files "$TESTS" --templates "$TPL" --cache "$CACHE" \
      --results "$OUT/e6_alert_full_v3.csv" > "$LOGS/e6_alert.log" 2>&1
  echo "[v4] e6 rc=$? $(date)" >> "$LOGS/progress"
fi

# ---------- 阶段 3：消融（含 A5 编码消融；RATE* 单列一份） ----------
if [ ! -f "$OUT/e6d_ablation_v3.csv" ]; then
  PYTHONHASHSEED=0 python -u scripts_ablation/e6d_ablation_replay.py \
      --test-files "$TESTS" --templates "$TPL" --cache "$CACHE" \
      --groups A1,A3,A4,A5 --configs identity+rate,TeRed+naive,TeRed+RATE \
      --a5-enc dual_naive,rate_single,none,rate_ratio,rate_ratio_only,rate_rank \
      --results "$OUT/e6d_ablation_v3.csv" > "$LOGS/e6d.log" 2>&1
  echo "[v4] e6d rc=$? $(date)" >> "$LOGS/progress"
fi

if [ ! -f "$OUT/e6d_ratestar_v3.csv" ]; then
  PYTHONHASHSEED=0 python -u scripts_ablation/e6d_ablation_replay.py \
      --test-files "$TESTS" --templates "$TPL" --cache "$CACHE" \
      --groups A1,A3,A4 --configs 'TeRed+RATE*' \
      --results "$OUT/e6d_ratestar_v3.csv" > "$LOGS/e6d_rs.log" 2>&1
  echo "[v4] e6d* rc=$? $(date)" >> "$LOGS/progress"
fi

# ---------- 阶段 4：统一评估单元 ----------
if [ ! -f "$OUT/e7_origunit.csv" ]; then
  PYTHONHASHSEED=0 python -u scripts_ablation/e7_origunit_eval.py \
      --test-files "$TESTS" --templates "$TPL" --cache "$CACHE" \
      --results "$OUT/e7_origunit.csv" > "$LOGS/e7.log" 2>&1
  echo "[v4] e7 rc=$? $(date)" >> "$LOGS/progress"
fi

echo "[v4] ALL DONE $(date)" >> "$LOGS/progress"
