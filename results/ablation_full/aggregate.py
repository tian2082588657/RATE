# -*- coding: utf-8 -*-
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), *[".."] * 2))

"""聚合 e6d_ablation.csv -> A1/A3/A4 论文表 + base 基准核对。"""
import csv
from collections import defaultdict

HERE = _os.path.join(_R, "results", "ablation_full")
rows = list(csv.DictReader(open(HERE + r"\e6d_ablation.csv", encoding="utf-8")))
for r in rows:
    for k in ("best_F1_alert", "node_cov", "node_best_f1", "F1_alert@1",
              "F1_alert@2", "F1_alert@3", "F1_alert@5", "F1_alert@10",
              "F1_alert@20", "alert_P", "first_hit"):
        r[k] = float(r[k]) if r[k] else 0.0

FILES = ["bin.116", "bin.117", "bin.118", "bin.119@500000", "bin.120",
         "bin.6@400000", "bin.7@400000"]
CFGS = ["identity+rate", "TeRed+naive", "TeRed+RATE"]
VARIANTS = ["base", "no_bfs", "no_rank", "bfs_q50", "bfs_q90",
            "w100", "w050", "w000", "sort_cmax", "sort_p95",
            "v4_drop_log_deg", "v4_drop_out_in_ratio", "v4_drop_nbr_type_div",
            "v4_drop_self_loop"]


def ms(vals):
    m = sum(vals) / len(vals)
    s = (sum((x - m) ** 2 for x in vals) / len(vals)) ** 0.5
    return f"{m:.3f}±{s:.3f}"


# ---------- base 基准核对：vs e6_alert_full.csv ----------
try:
    ref = list(csv.DictReader(open(
        _os.path.join(_R, "v4_validation_results", "e6_alert_full.csv"),
        encoding="utf-8")))
    print("=== base 基准核对（vs e6_alert_full.csv, best_F1_alert）===")
    idx = {(r["config"], r["file"]): r for r in ref}
    n_match = 0
    for r in rows:
        if r["variant"] != "base":
            continue
        o = idx.get((r["config"], r["file"]))
        if o is None:
            continue
        d = abs(float(r["best_F1_alert"]) - float(o["best_F1_alert"]))
        ok = d < 0.01
        n_match += ok
        if not ok:
            print(f"  MISMATCH {r['config']} {r['file']}: "
                  f"abl={r['best_F1_alert']} ref={o['best_F1_alert']}")
    print(f"  {n_match}/{sum(1 for r in rows if r['variant']=='base')} 行一致 (容差0.01)")
except FileNotFoundError:
    print("(e6_alert_full.csv 不在本地，跳过基准核对)")

# ---------- A1 表 ----------
print("\n=== 表 A1 · 告警管线解构（7 文件 mean±std）===")
a1_vars = ["base", "no_bfs", "no_rank", "bfs_q50", "bfs_q90", "w100", "w050", "w000"]
hdr = f"{'config':<15}{'variant':<10}{'best_F1_alert':<16}{'node_cov':<16}{'F1@20':<16}{'node_F1':<16}"
print(hdr)
out = [hdr]
for c in CFGS:
    for v in a1_vars:
        sel = [r for r in rows if r["config"] == c and r["variant"] == v]
        line = f"{c:<15}{v:<10}" + ms([r["best_F1_alert"] for r in sel]).ljust(16) \
            + ms([r["node_cov"] for r in sel]).ljust(16) \
            + ms([r["F1_alert@20"] for r in sel]).ljust(16) \
            + ms([r["node_best_f1"] for r in sel]).ljust(16)
        print(line)
        out.append(line)

# ---------- A3 表 ----------
print("\n=== 表 A3 · 预算 × 排序（7 文件 mean, F1_alert@b）===")
a3_vars = ["base", "sort_cmax", "sort_p95"]
hdr = f"{'config':<15}{'variant':<10}" + "".join(f"{'F1@'+b:<9}" for b in
      ["1", "2", "3", "5", "10", "20"])
print(hdr)
out.append("")
out.append(hdr)
for c in CFGS:
    for v in a3_vars:
        sel = [r for r in rows if r["config"] == c and r["variant"] == v]
        line = f"{c:<15}{v:<10}" + "".join(
            f"{sum(r['F1_alert@'+b] for r in sel) / len(sel):<9.3f}"
            for b in ["1", "2", "3", "5", "10", "20"])
        print(line)
        out.append(line)

# ---------- A4 表 ----------
print("\n=== 表 A4 · v4 特征逐列消融（7 文件 mean±std, node_best_f1）===")
hdr = f"{'config':<15}{'variant':<22}{'node_best_f1':<18}{'best_F1_alert':<16}"
print(hdr)
out.append("")
out.append(hdr)
for c in CFGS:
    sel_base = [r for r in rows if r["config"] == c and r["variant"] == "base"]
    line = f"{c:<15}{'(full v4 base)':<22}" \
        + ms([r["node_best_f1"] for r in sel_base]).ljust(18) \
        + ms([r["best_F1_alert"] for r in sel_base]).ljust(16)
    print(line)
    out.append(line)
    for col in ["log_deg", "out_in_ratio", "nbr_type_div", "self_loop"]:
        v = f"v4_drop_{col}"
        sel = [r for r in rows if r["config"] == c and r["variant"] == v]
        line = f"{c:<15}{('drop ' + col):<22}" \
            + ms([r["node_best_f1"] for r in sel]).ljust(18) \
            + ms([r["best_F1_alert"] for r in sel]).ljust(16)
        print(line)
        out.append(line)

with open(HERE + r"\aggregate_tables.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(out))
print("\n已写 aggregate_tables.txt")
