# -*- coding: utf-8 -*-
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), *[".."] * 1))

"""等宽对照 (e7_eqwidth) + 端到端成本 (e6g_cost) 汇总：论文可用数字。"""
import csv, os, statistics as st, itertools, math

BASE = _os.path.join(_R, "results", "ablation_full")
OUT = os.path.join(BASE, "_analysis6.txt")
L = []
def P(s=""):
    L.append(str(s))

def load(n):
    p = os.path.join(BASE, n)
    if not os.path.exists(p):
        return []
    return list(csv.DictReader(open(p, encoding="utf-8-sig")))

def num(v):
    try:
        return float(v)
    except Exception:
        return None

# ---------------- 1. 等宽对照 ----------------
P("=" * 78)
P("1. EQUAL-WIDTH CONTROL (original units, rate_ratio encoder both sides)")
P("=" * 78)
eq = load("e7_eqwidth.csv")
CFG_A = "identity+rate_ratio"
CFG_B = "TeRed+RATE*"
files = sorted({r["file"] for r in eq})
def pick(cfg, f):
    for r in eq:
        if r["config"] == cfg and r["file"] == f:
            return r
    return None
keys = ["orig_recall", "F1_orig", "alert_P_orig", "node_cov_red", "F1_red",
        "n_gt_kept", "n_gt_absorbed", "n_nodes_Gp", "n_alerted_orig", "orig_load_prec"]
P(f"{'file':16s} {'recall':>8s} {'F1_orig':>8s} {'P_orig':>7s} {'alerts_orig':>11s}  ||  "
  f"{'recall':>8s} {'F1_orig':>8s} {'P_orig':>7s} {'alerts_orig':>11s}")
P(f"{'':16s} {'--- identity+rate_ratio ---':>36s}  ||  {'--- TeRed+RATE* ---':>36s}")
for f in files:
    a, b = pick(CFG_A, f), pick(CFG_B, f)
    if not a or not b:
        continue
    P(f"{f:16s} {num(a['orig_recall']):8.4f} {num(a['F1_orig']):8.4f} {num(a['alert_P_orig']):7.3f} "
      f"{num(a['n_alerted_orig']):11.0f}  ||  "
      f"{num(b['orig_recall']):8.4f} {num(b['F1_orig']):8.4f} {num(b['alert_P_orig']):7.3f} "
      f"{num(b['n_alerted_orig']):11.0f}")
P()
for k in keys:
    va = [num(pick(CFG_A, f)[k]) for f in files if pick(CFG_A, f) and num(pick(CFG_A, f)[k]) is not None]
    vb = [num(pick(CFG_B, f)[k]) for f in files if pick(CFG_B, f) and num(pick(CFG_B, f)[k]) is not None]
    if not va or not vb:
        continue
    P(f"{k:16s} identity+ratio {st.mean(va):8.4f} +/- {st.pstdev(va):.4f}   "
      f"TeRed+RATE* {st.mean(vb):8.4f} +/- {st.pstdev(vb):.4f}   "
      f"diff(TeR-id) {st.mean(vb)-st.mean(va):+.4f}")

# paired Wilcoxon (exact for n<=10) on orig_recall / F1_orig
def exact_wilcoxon(d):
    d = [x for x in d if abs(x) > 1e-12]
    n = len(d)
    if n == 0:
        return None, None
    rank = {}
    srt = sorted(range(n), key=lambda i: abs(d[i]))
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(abs(d[srt[j + 1]]) - abs(d[srt[i]])) < 1e-12:
            j += 1
        avg = (i + 1 + j + 1) / 2.0
        for t in range(i, j + 1):
            rank[srt[t]] = avg
        i = j + 1
    Wp = sum(rank[i] for i in range(n) if d[i] > 0)
    # exact null distribution
    from itertools import combinations
    tot = 0
    cnt = 0
    ranks = list(rank.values())
    for k in range(n + 1):
        for comb in combinations(range(n), k):
            s = sum(ranks[i] for i in comb)
            if s >= Wp - 1e-9:
                cnt += 1
            tot += 1
    return Wp, cnt / tot

P()
for k in ["orig_recall", "F1_orig"]:
    d = []
    for f in files:
        a, b = pick(CFG_A, f), pick(CFG_B, f)
        if a and b and num(a[k]) is not None and num(b[k]) is not None:
            d.append(num(b[k]) - num(a[k]))
    W, p = exact_wilcoxon(d)
    P(f"paired exact Wilcoxon ({k}, TeRed+RATE* - identity+ratio): n={len(d)} diff_mean={st.mean(d):+.4f} "
      f"win/loss={sum(1 for x in d if x>1e-9)}/{sum(1 for x in d if x<-1e-9)} p={p:.4f}")

# ---------------- 2. 端到端成本 ----------------
P()
P("=" * 78)
P("2. END-TO-END COST PROFILE (e6g_cost, 3 partitions)")
P("=" * 78)
cg = load("e6g_cost.csv")
P(f"{'spec':18s} {'nodes':>7s} {'edges':>8s} | {'id_red':>7s} {'te_red':>8s} | "
  f"{'id_e2e':>7s} {'te_e2e':>8s} {'ratio':>7s}")
for r in cg:
    P(f"{r['spec']:18s} {r['n_nodes_orig']:>7s} {r['n_edges_orig']:>8s} | "
      f"{num(r['identity_reduce_s']):7.2f} {num(r['tered_reduce_s']):8.2f} | "
      f"{num(r['identity_end2end_s']):7.1f} {num(r['tered_end2end_s']):8.1f} "
      f"{num(r['tered_end2end_s'])/num(r['identity_end2end_s']):7.1f}x")
P()
for k in ["identity_reduce_s", "tered_reduce_s", "identity_end2end_s", "tered_end2end_s",
          "identity_score_s", "tered_score_s", "identity_encode_s", "tered_encode_s"]:
    v = [num(r[k]) for r in cg if num(r[k]) is not None]
    P(f"{k:22s} mean {st.mean(v):9.2f}  min {min(v):8.2f}  max {max(v):9.2f}")

open(OUT, "w", encoding="utf-8").write("\n".join(L))
print("\n".join(L))
