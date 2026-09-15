# -*- coding: utf-8 -*-
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), *[".."] * 2))

"""聚合 e6d_ablation_v2.csv（新口径：wmax=1.0 + cmax 排序 + v4 三列）-> 论文表。

输出:
  aggregate_tables_v2.txt   全部表
  base 基准核对 vs e6_alert_full_v2.csv
  A1 管线解构（含 legacy_agg / legacy_cmax / legacy_w1 的 2x2 分解）
  A3 预算×排序
  A4 v4 逐列（3 列）
"""
import csv
from collections import defaultdict

HERE = _os.path.join(_R, "results", "ablation_full")
SRC = HERE + r"\e6d_ablation_v2.csv"
REF = HERE + r"\e6_alert_full_v2.csv"

MCOLS = ("best_F1_alert", "node_cov", "node_best_f1", "F1_alert@1",
         "F1_alert@2", "F1_alert@3", "F1_alert@5", "F1_alert@10",
         "F1_alert@20", "alert_P", "first_hit", "n_alerts")

rows = list(csv.DictReader(open(SRC, encoding="utf-8")))
for r in rows:
    for k in MCOLS:
        try:
            r[k] = float(r[k])
        except (KeyError, ValueError, TypeError):
            r[k] = 0.0

FILES = ["bin.116", "bin.117", "bin.118", "bin.119@500000", "bin.120",
         "bin.6@400000", "bin.7@400000"]
CFGS = ["identity+rate", "TeRed+naive", "TeRed+RATE"]

variants_seen = sorted({r["variant"] for r in rows})
print("变体清单:", variants_seen)
print("文件数:", len({r["file"] for r in rows}), "配置数:", len({r["config"] for r in rows}))


def mean(v):
    return sum(v) / len(v) if v else 0.0


def ms(v):
    m = mean(v)
    s = (sum((x - m) ** 2 for x in v) / len(v)) ** 0.5 if v else 0.0
    return f"{m:.3f}±{s:.3f}"


def sel(c, v):
    return [r for r in rows if r["config"] == c and r["variant"] == v]


out = []


def emit(s=""):
    print(s)
    out.append(s)


# ---------- base 基准核对 ----------
emit("=== base 基准核对（e6d base  vs  e6_alert_full_v2.csv, best_F1_alert）===")
try:
    ref = list(csv.DictReader(open(REF, encoding="utf-8")))
    idx = {(r["config"], r["file"]): float(r["best_F1_alert"]) for r in ref}
    nmatch = ntot = 0
    for r in rows:
        if r["variant"] != "base":
            continue
        ntot += 1
        key = (r["config"], r["file"])
        if key in idx and abs(r["best_F1_alert"] - idx[key]) < 0.005:
            nmatch += 1
        else:
            emit(f"  MISMATCH {key}: abl={r['best_F1_alert']:.4f} "
                 f"ref={idx.get(key)}")
    emit(f"  {nmatch}/{ntot} 行一致 (容差 0.005)")
except FileNotFoundError:
    emit("  (e6_alert_full_v2.csv 不在，跳过核对)")

# ---------- A1 表：管线解构 ----------
emit("\n=== 表 A1 · 告警管线解构（7 文件 mean±std）===")
a1_vars = ["base", "legacy_agg", "legacy_cmax", "legacy_w1",
           "no_bfs", "no_rank", "bfs_q50", "bfs_q90", "w100", "w050", "w000"]
hdr = (f"{'config':<15}{'variant':<14}{'best_F1_alert':<16}"
       f"{'node_cov':<16}{'alert_P':<16}{'node_F1':<16}")
emit(hdr)
for c in CFGS:
    for v in a1_vars:
        s = sel(c, v)
        if not s:
            emit(f"{c:<15}{v:<14}(缺失)")
            continue
        emit(f"{c:<15}{v:<14}"
             + ms([r["best_F1_alert"] for r in s]).ljust(16)
             + ms([r["node_cov"] for r in s]).ljust(16)
             + ms([r["alert_P"] for r in s]).ljust(16)
             + ms([r["node_best_f1"] for r in s]).ljust(16))

# ---------- 2x2 分解表 ----------
emit("\n=== 表 A1b · 两处升级的 2x2 分解（7 文件 mean best_F1_alert）===")
emit(f"{'config':<15}{'w0.7+agg(旧)':<16}{'w0.7+cmax':<16}{'w1.0+agg':<16}{'w1.0+cmax(新)':<16}")
for c in CFGS:
    row = [c.ljust(15)]
    for v in ["legacy_agg", "legacy_cmax", "legacy_w1", "base"]:
        row.append(f"{mean([r['best_F1_alert'] for r in sel(c, v)]):<16.4f}")
    emit("".join(row))

# ---------- A3 表 ----------
emit("\n=== 表 A3 · 预算 × 排序（7 文件 mean F1_alert@b）===")
a3_vars = ["base", "sort_cmax", "sort_p95", "legacy_agg"]
emit(f"{'config':<15}{'variant':<14}" + "".join(
    f"{'F1@'+b:<9}" for b in ["1", "2", "3", "5", "10", "20"]))
for c in CFGS:
    for v in a3_vars:
        s = sel(c, v)
        if not s:
            continue
        emit(f"{c:<15}{v:<14}" + "".join(
            f"{mean([r['F1_alert@'+b] for r in s]):<9.3f}"
            for b in ["1", "2", "3", "5", "10", "20"]))

# ---------- A4 表 ----------
emit("\n=== 表 A4 · v4 特征逐列消融（7 文件 mean±std）===")
emit(f"{'config':<15}{'variant':<22}{'node_best_f1':<18}"
     f"{'best_F1_alert':<16}{'node_cov':<16}")
for c in CFGS:
    s = sel(c, "base")
    emit(f"{c:<15}{'(full v4 base)':<22}"
         + ms([r["node_best_f1"] for r in s]).ljust(18)
         + ms([r["best_F1_alert"] for r in s]).ljust(16)
         + ms([r["node_cov"] for r in s]).ljust(16))
    for col in ["log_deg", "out_in_ratio", "self_loop"]:
        s = sel(c, f"v4_drop_{col}")
        if not s:
            continue
        emit(f"{c:<15}{('drop ' + col):<22}"
             + ms([r["node_best_f1"] for r in s]).ljust(18)
             + ms([r["best_F1_alert"] for r in s]).ljust(16)
             + ms([r["node_cov"] for r in s]).ljust(16))

# ---------- 逐文件明细（RATE）----------
emit("\n=== 附 · TeRed+RATE 逐文件明细（base new vs legacy_agg old）===")
emit(f"{'file':<16}{'new_F1':<10}{'old_F1':<10}{'delta':<10}"
     f"{'new_cov':<10}{'old_cov':<10}")
for f in FILES:
    nb = [r for r in rows if r["config"] == "TeRed+RATE"
          and r["variant"] == "base" and r["file"] == f]
    ob = [r for r in rows if r["config"] == "TeRed+RATE"
          and r["variant"] == "legacy_agg" and r["file"] == f]
    if not nb or not ob:
        continue
    nv, ov = nb[0]["best_F1_alert"], ob[0]["best_F1_alert"]
    emit(f"{f:<16}{nv:<10.4f}{ov:<10.4f}{nv-ov:<+10.4f}"
         f"{nb[0]['node_cov']:<10.4f}{ob[0]['node_cov']:<10.4f}")

with open(HERE + r"\aggregate_tables_v2.txt", "w", encoding="utf-8") as fp:
    fp.write("\n".join(out))
print("\n已写 aggregate_tables_v2.txt")
