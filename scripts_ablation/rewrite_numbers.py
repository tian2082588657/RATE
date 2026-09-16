# -*- coding: utf-8 -*-
"""rewrite_numbers.py — 定位重构定稿所需的全部数字（含新的 E1 编码矩阵）。

数据源：
  results/e1p/all_e1.csv              E1：7 分区 x 4 编码 x 2 图来源
  results/v3_final/e6d_ablation_v3.csv  原 A1/A3/A4/A5 消融（含 A5 编码消融）
  results/v3_final/e6d_ratestar_v3.csv  RATE*（4 通道）消融
  results/v3_final/e7_origunit.csv      原始单位口径
用法: python scripts_ablation/rewrite_numbers.py
"""
from __future__ import annotations
import csv, os, itertools, statistics as st
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")

FILES7 = ["bin.116", "bin.117", "bin.118", "bin.119@500000", "bin.120",
          "bin.6@400000", "bin.7@400000"]


def rd(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fnum(v):
    try:
        return float(v)
    except Exception:
        return None


def exact_wilcoxon(x, y):
    d = [a - b for a, b in zip(x, y) if abs(a - b) > 1e-12]
    n = len(d)
    if n == 0:
        return 1.0
    order = sorted(range(n), key=lambda i: abs(d[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(abs(d[order[j + 1]]) - abs(d[order[i]])) < 1e-12:
            j += 1
        avg = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    T = sum(ranks[i] for i in range(n) if d[i] > 0)
    tot = sum(ranks)
    obs = min(T, tot - T)
    cnt = 0
    for bits in itertools.product([0, 1], repeat=n):
        s = sum(ranks[i] for i in range(n) if bits[i])
        if min(s, tot - s) <= obs + 1e-9:
            cnt += 1
    return cnt / (2 ** n)


def paired(name, x, y, quiet=False):
    diff = [a - b for a, b in zip(x, y)]
    wins = sum(1 for dd in diff if dd > 1e-12)
    ties = sum(1 for dd in diff if abs(dd) <= 1e-12)
    losses = sum(1 for dd in diff if dd < -1e-12)
    p = exact_wilcoxon(x, y)
    d = np.array(diff, dtype=float)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(d), size=(20000, len(d)))
    mm = d[idx].mean(axis=1)
    lo, hi = np.percentile(mm, 2.5), np.percentile(mm, 97.5)
    sd = st.stdev(diff) if len(diff) > 1 else 0.0
    es = (st.mean(diff) / sd) if sd > 0 else float("nan")
    print("%-44s mean=%+.4f sd=%.4f W/T/L=%d/%d/%d p=%.4f CI=[%+.4f,%+.4f] d=%.2f"
          % (name, st.mean(diff), sd, wins, ties, losses, p, lo, hi, es))


def hdr(t):
    print("\n" + "=" * 96)
    print(t)
    print("=" * 96)


hdr("E1: encoding x graph matrix  (node best-F1, mean over 7 partitions)")
e1 = rd(os.path.join(RES, "e1p", "all_e1.csv"))
M = {(r["fig"], r["encoding"], r["file"]): r for r in e1}
ENCS = ["none", "rate_single", "dual_naive", "rate"]
stock = {}
for enc in ENCS:
    cells = []
    for fig in ["identity", "tered"]:
        vals = [fnum(M[(fig, enc, f)]["node_best_f1"]) for f in FILES7]
        cells.append(vals)
        stock[(fig, enc)] = vals
    print("%-14s unreduced %-18s reduced %-18s" % (
        enc,
        "%.4f +/- %.4f" % (st.mean(cells[0]), st.stdev(cells[0])),
        "%.4f +/- %.4f" % (st.mean(cells[1]), st.stdev(cells[1]))))

print("\nper-partition node best-F1")
print("%-14s %-9s %s" % ("encoding", "graph", " ".join("%8s" % f.split("@")[0] for f in FILES7)))
for enc in ENCS:
    for fig in ["identity", "tered"]:
        print("%-14s %-9s %s" % (enc, fig, " ".join("%8.4f" % v for v in stock[(fig, enc)])))

print("\nnode ROC-AUC means")
for enc in ENCS:
    print("  %-12s unreduced %.4f   reduced %.4f" % (
        enc,
        st.mean([fnum(M[("identity", enc, f)]["node_auc"]) for f in FILES7]),
        st.mean([fnum(M[("tered", enc, f)]["node_auc"]) for f in FILES7])))

print("\nnode coverage means")
for enc in ENCS:
    print("  %-12s unreduced %.4f   reduced %.4f" % (
        enc,
        st.mean([fnum(M[("identity", enc, f)]["node_cov"]) for f in FILES7]),
        st.mean([fnum(M[("tered", enc, f)]["node_cov"]) for f in FILES7])))

print("\n--- necessity: dropping the topology channel ---")
paired("identity: count->none", stock[("identity", "dual_naive")], stock[("identity", "none")])
paired("TeRed:    count->none", stock[("tered", "dual_naive")], stock[("tered", "none")])
paired("none: identity vs TeRed", stock[("identity", "none")], stock[("tered", "none")])

print("\n--- reduction at fixed encoding (2-channel count / RATE) ---")
paired("count: identity vs TeRed", stock[("identity", "dual_naive")], stock[("tered", "dual_naive")])
paired("RATE2ch: identity vs TeRed", stock[("identity", "rate")], stock[("tered", "rate")])
paired("naive-mass tax: rate_single vs dual_naive on TeRed",
       stock[("tered", "rate_single")], stock[("tered", "dual_naive")])

hdr("Main table, own unit (e6d A1/base; RATE* from ratestar)")
d = [r for r in rd(os.path.join(RES, "v3_final", "e6d_ablation_v3.csv"))
     if r["group"] == "A1" and r["variant"] == "base"]
rss = [r for r in rd(os.path.join(RES, "v3_final", "e6d_ratestar_v3.csv"))
       if r["group"] == "A1" and r["variant"] == "base"]
CFG = {"identity": ("e6d", "identity+rate"),
       "TeRed+count": ("e6d", "TeRed+naive"),
       "TeRed+RATE(2ch)": ("e6d", "TeRed+RATE"),
       "TeRed+RATE*": ("rs", "TeRed+RATE*")}
own = {}
for nm, (src, cfg) in CFG.items():
    rows = {r["file"]: r for r in (d if src == "e6d" else rss) if r["config"] == cfg}
    own[nm] = rows
    print("%-16s nodeF1 %.4f  auc %.4f  cov %.4f  @5 %.4f  @10 %.4f  @20 %.4f"
          % (nm,
             st.mean([fnum(rows[f]["node_best_f1"]) for f in FILES7]),
             st.mean([fnum(rows[f]["node_auc"]) for f in FILES7]) if "node_auc" in rows[FILES7[0]] else float("nan"),
             st.mean([fnum(rows[f]["node_cov"]) for f in FILES7]),
             st.mean([fnum(rows[f]["F1_alert@5"]) or 0 for f in FILES7]),
             st.mean([fnum(rows[f]["F1_alert@10"]) or 0 for f in FILES7]),
             st.mean([fnum(rows[f]["F1_alert@20"]) or 0 for f in FILES7])))
print("n_gt (own unit):", [own["TeRed+RATE*"][f]["n_gt"] for f in FILES7])

hdr("Common unit (e7_origunit)")
e7 = rd(os.path.join(RES, "v3_final", "e7_origunit.csv"))
C7 = {(r["config"], r["file"]): r for r in e7}
print("%-16s recall        F1@20        P_alert    n_gt_kept" % "config")
cu = {}
for cfg in ["identity+rate", "TeRed+naive", "TeRed+RATE", "TeRed+RATE*"]:
    rows = [C7[(cfg, f)] for f in FILES7]
    cu[cfg] = rows
    print("%-16s %.4f+-%.4f %.4f+-%.4f %.4f     %s"
          % (cfg,
             st.mean([fnum(r["orig_recall"]) for r in rows]),
             st.stdev([fnum(r["orig_recall"]) for r in rows]),
             st.mean([fnum(r["F1_orig"]) for r in rows]),
             st.stdev([fnum(r["F1_orig"]) for r in rows]),
             st.mean([fnum(r["alert_P_orig"]) for r in rows]),
             [r["n_gt_kept"] for r in rows]))
print("\nGT accounting  orig=%s" % [C7[("identity+rate", f)]["n_orig_gt"] for f in FILES7])
print("               kept=%s" % [C7[("Identity" if False else "identity+rate", f)]["n_gt_kept"] for f in FILES7])
print("            absorbed=%s" % [C7[("identity+rate", f)]["n_gt_absorbed"] for f in FILES7])
print("  sum orig=%d kept=%d absorbed=%d" % (
    sum(int(C7[("identity+rate", f)]["n_orig_gt"]) for f in FILES7),
    sum(int(C7[("identity+rate", f)]["n_gt_kept"]) for f in FILES7),
    sum(int(C7[("identity+rate", f)]["n_gt_absorbed"]) for f in FILES7)))
print("  worst|ddeg|:", [C7[("TeRed+RATE*", f)]["worst_deg_delta"] for f in FILES7])
print("  region/removed:", [(C7[("TeRed+RATE*", f)]["n_regions"], C7[("TeRed+RATE*", f)]["n_removed"]) for f in FILES7])
n_o = [int(C7[("identity+rate", f)]["n_nodes_orig"]) for f in FILES7]
n_r = [int(C7[("TeRed+RATE*", f)]["n_nodes_Gp"]) for f in FILES7]
print("  nodes orig->reduced:", list(zip(n_o, n_r)), " shrink %.1f%%" % (100 * (1 - sum(n_r) / sum(n_o))))

print("\npaired on common unit (n=7):")
paired("recall  RATE* vs identity",
       [fnum(r["orig_recall"]) for r in cu["TeRed+RATE*"]],
       [fnum(r["orig_recall"]) for r in cu["identity+rate"]])
paired("F1@20   RATE* vs identity",
       [fnum(r["F1_orig"]) for r in cu["TeRed+RATE*"]],
       [fnum(r["F1_orig"]) for r in cu["identity+rate"]])
paired("P_alert RATE* vs identity",
       [fnum(r["alert_P_orig"]) for r in cu["TeRed+RATE*"]],
       [fnum(r["alert_P_orig"]) for r in cu["identity+rate"]])
paired("recall  count vs RATE* (both reduced)",
       [fnum(r["orig_recall"]) for r in cu["TeRed+naive"]],
       [fnum(r["orig_recall"]) for r in cu["TeRed+RATE*"]])
paired("F1@20   count vs RATE* (both reduced)",
       [fnum(r["F1_orig"]) for r in cu["TeRed+naive"]],
       [fnum(r["F1_orig"]) for r in cu["TeRed+RATE*"]])

hdr("StreamSpot e4s (server)")
p = os.path.join(RES, "e4s")
print("exists:", os.path.isdir(p))
