# -*- coding: utf-8 -*-
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), *[".."] * 1))

"""为论文生成统一口径表：固定预算、配对检验、分析负载。
纯标准库（无 scipy）。"""
import csv, math, statistics as st
from collections import defaultdict

BASE = _os.path.join(_R, "results", "ablation_full")

def load(fn):
    with open(f"{BASE}/{fn}", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

def fl(x):
    try: return float(x)
    except: return None

def bykey(rows, keyf, fld):
    d = defaultdict(list)
    for r in rows:
        v = fl(r.get(fld, ""))
        if v is not None:
            d[keyf(r)].append(v)
    return d

def ms(v):
    return f"{st.mean(v):.3f}±{st.pstdev(v):.3f}" if len(v) > 1 else f"{v[0]:.3f}"

def wilcoxon(a, b):
    """配对 Wilcoxon 符号秩（双侧）。n<=20 用精确分布，否则正态近似。
    返回 (W, z, p)。零差对按惯例剔除。"""
    from itertools import product
    d = [x - y for x, y in zip(a, b) if x - y != 0]
    n = len(d)
    if n == 0:
        return (0.0, 0.0, 1.0)
    ranks = sorted(range(n), key=lambda i: abs(d[i]))
    rk = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(d[ranks[j + 1]]) == abs(d[ranks[i]]):
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            rk[ranks[k]] = avg
        i = j + 1
    Wp = sum(rk[i] for i in range(n) if d[i] > 0)
    Wm = sum(rk[i] for i in range(n) if d[i] < 0)
    W = min(Wp, Wm)
    if n <= 20:
        # 精确：枚举符号组合，统计 W+ 的分布
        cnt = 0
        tot = 0
        for signs in product((0, 1), repeat=n):
            wp = sum(rk[i] for i in range(n) if signs[i])
            wm = sum(rk[i] for i in range(n) if not signs[i])
            if min(wp, wm) <= W + 1e-9:
                cnt += 1
            tot += 1
        p = cnt / tot
        z = 0.0
    else:
        mu = n * (n + 1) / 4
        sd = math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
        z = (W - mu) / sd if sd > 0 else 0.0
        p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return (W, z, p)

a1 = [r for r in load("e6d_ablation_v2.csv") if r["group"] == "A1"
      and r["variant"] == "base"]
BUD = ["F1_alert@1", "F1_alert@2", "F1_alert@3", "F1_alert@5",
       "F1_alert@10", "F1_alert@20"]
CFG = ["identity+rate", "TeRed+naive", "TeRed+RATE"]

print("### 表 MAIN-1 · 固定预算 alert-F1（mean±std, n=7）")
print(f"{'config':16s} " + " ".join(f"{b:>12s}" for b in
      ["@1", "@2", "@3", "@5", "@10", "@20"]))
store = {}
for c in CFG:
    for b in BUD:
        store[(c, b)] = bykey(a1, lambda r: r["config"], b)[c]
    print(f"{c:16s} " + " ".join(f"{ms(store[(c,b)]):>12s}" for b in BUD))

print("\n### 表 MAIN-2 · RATE vs naive 配对 Wilcoxon（固定预算）")
for b in BUD:
    x = store[("TeRed+RATE", b)]
    y = store[("TeRed+naive", b)]
    W, z, p = wilcoxon(x, y)
    dp = "RATE更好" if st.mean(x) > st.mean(y) else ("naive更好" if st.mean(x) < st.mean(y) else "相同")
    print(f"  {b:12s} RATE={st.mean(x):.3f} naive={st.mean(y):.3f} "
          f"Δ={st.mean(x)-st.mean(y):+.3f}  W={W:.1f} z={z:+.2f} p={p:.3f}  [{dp}]")

print("\n### 表 MAIN-3 · 每文件明细（node_best_f1 / best_F1_alert / F1@5）")
per = defaultdict(dict)
for r in a1:
    per[r["file"]][r["config"]] = (fl(r["node_best_f1"]), fl(r["best_F1_alert"]),
                                   fl(r["F1_alert@5"]))
for f in sorted(per):
    line = f"  {f:16s}"
    for c in CFG:
        if c in per[f]:
            n, a, b5 = per[f][c]
            line += f" | {c[:11]:11s} n={n:.3f} a={a:.3f} @5={b5:.3f}"
    print(line)

print("\n### 表 LOAD · 告警分析负载（覆盖 GT / 告警节点总数 / 节点精度）")
hdr = load("e6_alert_full_v2.csv")
for r in hdr:
    if r["config"] in CFG:
        print(f"  {r['file']:16s} {r['config']:16s} n_gt={r['n_gt']:>3s} "
              f"alerts={r['n_alerts']:>2s} alerted_nodes={r['n_alerted_nodes']:>5s} "
              f"covered_gt={r['covered_gt']:>2s} node_prec={r['node_prec']}")

print("\n### 表 A5 · 编码消融固定预算（mean±std）")
a5 = load("e6d_ablation_a5.csv")
for encfld in ["dual_naive", "rate_single", "none"]:
    rows = [r for r in a5 if r["encoding"] == encfld]
    nf = [fl(r["node_best_f1"]) for r in rows]
    line = f"  {encfld:14s} nodeF1={ms(nf):>12s}"
    for b in BUD[3:]:
        vals = [fl(r[b]) for r in rows if fl(r[b]) is not None]
        if vals:
            line += f" {b}={ms(vals)}"
    print(line)

print("\n### 表 A5b · μ 抢救固定预算 + 配对检验（vs dual_naive）")
a5b = load("e6d_ablation_a5b.csv")
encs = sorted({r["encoding"] for r in a5b})
ref = {}
for r in a5b:
    if r["encoding"] == "dual_naive":
        ref[r["file"]] = r
for e in encs:
    rows = {r["file"]: r for r in a5b if r["encoding"] == e}
    nf = [fl(r["node_best_f1"]) for r in rows.values()]
    line = f"  {e:16s} nodeF1={ms(nf):>12s}"
    for b in ["F1_alert@1", "F1_alert@5", "F1_alert@10"]:
        vals = [fl(r[b]) for r in rows.values() if fl(r[b]) is not None]
        line += f" {b[9:]}={ms(vals)}"
    # 配对检验 vs dual_naive (node_best_f1)
    if e != "dual_naive":
        xs = [fl(rows[f]["node_best_f1"]) for f in sorted(rows)]
        ys = [fl(ref[f]["node_best_f1"]) for f in sorted(rows)]
        W, z, p = wilcoxon(xs, ys)
        line += f" | node配对 p={p:.3f}"
    print(line)
