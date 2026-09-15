import csv, os, io, itertools, statistics, sys

BASE = "/TeRed+RATE/code/results"
OUT = "/TeRed+RATE/code/logs/_gnn2.txt"
buf = io.StringIO()
P = lambda *a: print(*a, file=buf)
FILES = ['bin.116', 'bin.117', 'bin.118', 'bin.119@500000',
         'bin.120', 'bin.6@400000', 'bin.7@400000']


def load(name):
    p = os.path.join(BASE, name)
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def num(v):
    try:
        return float(v)
    except Exception:
        return None


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
    cnt = N = 0
    for signs in itertools.product([0, 1], repeat=n):
        s = sum(ranks[i] for i in range(n) if signs[i])
        N += 1
        if abs(s - tot / 2.0) >= abs(T - tot / 2.0) - 1e-12:
            cnt += 1
    return cnt / N


rows = load("e6e_gnn_baseline.csv")
ORDER = ["identity+rate", "tered+dual_naive", "tered+rate", "tered+rate_ratio",
         "tered+rate_single", "tered+none"]


def pf(det, cfg, key):
    return {r['file']: num(r[key]) for r in rows
            if r['detector'] == det and r['config'] == cfg
            and num(r[key]) is not None}


def cmp(tag, a, b, key='node_ROC_AUC'):
    fs = [f for f in FILES if f in a and f in b]
    if len(fs) < 2:
        P(f"  {tag:44s} <missing>")
        return
    d = [a[f] - b[f] for f in fs]
    w = sum(1 for x in d if x > 1e-12)
    l = sum(1 for x in d if x < -1e-12)
    P(f"  {tag:44s} meanD={statistics.mean(d):+.4f} sd={statistics.pstdev(d):.4f} "
      f"W/L/T={w}/{l}/{len(fs)-w-l} p={exact_wilcoxon(d):.4f}")


for det in ["cosine", "gnn"]:
    P(f"\n=== {det}: paired node ROC-AUC ===")
    idn = pf(det, "identity+rate", 'node_ROC_AUC')
    for cfg in ORDER[1:]:
        cmp(f"identity vs {cfg}", pf(det, cfg, 'node_ROC_AUC'), idn)
    cmp("rate_ratio vs dual_naive", pf(det, "tered+rate_ratio", 'node_ROC_AUC'),
        pf(det, "tered+dual_naive", 'node_ROC_AUC'))
    cmp("dual_naive vs none", pf(det, "tered+dual_naive", 'node_ROC_AUC'),
        pf(det, "tered+none", 'node_ROC_AUC'))
    cmp("rate_ratio vs rate_single", pf(det, "tered+rate_ratio", 'node_ROC_AUC'),
        pf(det, "tered+rate_single", 'node_ROC_AUC'))
    for k in ['node_best_F1', 'node_PR_AUC']:
        P(f"  -- key={k} --")
        idn2 = pf(det, "identity+rate", k)
        for cfg in ORDER[1:]:
            cmp(f"identity vs {cfg}", pf(det, cfg, k), idn2, k)

# 合并 identity:rate_ratio 若有
extra = load("e6e_identity_ratestar.csv")
if extra:
    P("\n=== identity:rate_ratio (等宽对照，若有) ===")
    for det in ["cosine", "gnn"]:
        rs = [r for r in extra if r['detector'] == det]
        if not rs:
            continue
        auc = [num(r['node_ROC_AUC']) for r in rs]
        P(f"  {det}: node AUC mean={statistics.mean(auc):.4f}  "
          f"per-file={[round(x,4) for x in auc]}")
        a = {r['file']: num(r['node_ROC_AUC']) for r in rs}
        cmp(f"{det} identity:rr vs tered:rr",
            pf(det, "tered+rate_ratio", 'node_ROC_AUC'), a)

with open(OUT, "w", encoding="utf-8") as f:
    f.write(buf.getvalue())
print(buf.getvalue())
