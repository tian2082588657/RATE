import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), *[".."] * 1))

import csv, os, io, itertools, statistics

BASE = _os.path.join(_R, "results")
OUT = _os.path.join(_R, "scripts_ablation", "_gnn.txt")
buf = io.StringIO()


def P(*a):
    print(*a, file=buf)


def f4(x):
    return "None" if x is None else f"{x:.4f}"


try:
    with open(os.path.join(BASE, "e6e_gnn_baseline.csv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
except FileNotFoundError:
    P("no csv")
    rows = []

FILES = ['bin.116', 'bin.117', 'bin.118', 'bin.119@500000',
         'bin.120', 'bin.6@400000', 'bin.7@400000']


def num(v):
    try:
        return float(v)
    except Exception:
        return None


def m(rs, k):
    v = [num(r[k]) for r in rs]
    v = [x for x in v if x is not None]
    return sum(v) / len(v) if v else None


ORDER = ["identity+rate", "tered+dual_naive", "tered+rate", "tered+rate_ratio",
         "tered+rate_single", "tered+none"]

for det in ["cosine", "gnn"]:
    P(f"\n=== detector={det} ===")
    P(f"{'config':22s} {'n':>3s} {'ROC_AUC':>8s} {'PR_AUC':>8s} {'bestF1':>8s} "
      f"{'FPR':>8s} {'P@100':>8s} {'@5':>8s} {'@10':>8s} {'@20':>8s} {'cov':>8s}")
    for cfg in ORDER:
        rs = [r for r in rows if r['detector'] == det and r['config'] == cfg]
        if not rs:
            continue
        P(f"{cfg:22s} {len(rs):3d} {f4(m(rs,'node_ROC_AUC')):>8s} "
          f"{f4(m(rs,'node_PR_AUC')):>8s} {f4(m(rs,'node_best_F1')):>8s} "
          f"{f4(m(rs,'node_FPR')):>8s} {f4(m(rs,'P@100')):>8s} "
          f"{f4(m(rs,'F1_alert@5')):>8s} {f4(m(rs,'F1_alert@10')):>8s} "
          f"{f4(m(rs,'F1_alert@20')):>8s} {f4(m(rs,'node_cov')):>8s}")

P("\n=== per-file ROC_AUC ===")
for det in ["cosine", "gnn"]:
    P(f"  -- {det} --")
    P("    " + " ".join(f"{f[:11]:>12s}" for f in FILES))
    for cfg in ORDER:
        d = {r['file']: num(r['node_ROC_AUC']) for r in rows
             if r['detector'] == det and r['config'] == cfg}
        if not d:
            continue
        P(f"    {cfg:22s}" + " ".join(
            f"{(d.get(f) if d.get(f) is not None else float('nan')):12.4f}"
            for f in FILES))

# 配对：GNN 下 identity vs tered+dual_naive vs tered+rate_ratio
P("\n=== paired (GNN, node ROC_AUC) ===")


def pf(det, cfg, key):
    return {r['file']: num(r[key]) for r in rows
            if r['detector'] == det and r['config'] == cfg
            and num(r[key]) is not None}


def exact_wilcoxon(d):
    d = [x for x in d if abs(x) > 1e-12]
    n = len(d)
    if n == 0:
        return 1.0
    rr = sorted(range(n), key=lambda i: abs(d[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(abs(d[rr[j + 1]]) - abs(d[rr[i]])) < 1e-12:
            j += 1
        avg = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[rr[k]] = avg
        i = j + 1
    T = sum(ranks[i] for i in range(n) if d[i] > 0)
    tot = sum(ranks)
    cnt = 0
    N = 0
    for signs in itertools.product([0, 1], repeat=n):
        s = sum(ranks[i] for i in range(n) if signs[i])
        N += 1
        if abs(s - tot / 2.0) >= abs(T - tot / 2.0) - 1e-12:
            cnt += 1
    return cnt / N


key = 'node_ROC_AUC'
base = pf("gnn", "identity+rate", key)
for cfg in ["tered+dual_naive", "tered+rate", "tered+rate_ratio",
            "tered+rate_single", "tered+none"]:
    a = pf("gnn", cfg, key)
    fs = [f for f in FILES if f in a and f in base]
    if len(fs) < 2:
        P(f"  gnn identity vs {cfg}: <missing>")
        continue
    d = [a[f] - base[f] for f in fs]
    w = sum(1 for x in d if x > 1e-12)
    l = sum(1 for x in d if x < -1e-12)
    P(f"  gnn identity vs {cfg:20s} meanD={statistics.mean(d):+.4f} "
      f"sd={statistics.pstdev(d):.4f} W/L/T={w}/{l}/{len(fs)-w-l} "
      f"p={exact_wilcoxon(d):.4f}")

P("\n=== cosine vs gnn on the SAME config (node ROC_AUC) ===")
for cfg in ORDER:
    a = pf("cosine", cfg, key)
    b = pf("gnn", cfg, key)
    fs = [f for f in FILES if f in a and f in b]
    if len(fs) < 2:
        continue
    P(f"  {cfg:22s} cosine={statistics.mean([a[f] for f in fs]):.4f} "
      f"gnn={statistics.mean([b[f] for f in fs]):.4f}")

with open(OUT, "w", encoding="utf-8") as f:
    f.write(buf.getvalue())
