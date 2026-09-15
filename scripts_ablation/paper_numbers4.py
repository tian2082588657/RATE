import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), *[".."] * 1))

"""Local consolidated analysis for the 0912-2 rebuttal experiments.

Reads (all local, under results/ablation_full/):
  - e6e_gnn_baseline.csv      main: cosine + gnn x 6 configs x 7 partitions
  - e6e_identity_ratestar.csv identity:rate_ratio equal-width control (full budget)
  - e6e_cleantrain.csv        clean-train benign variant (cosine)
  - e6g_cost.csv              end-to-end cost profile (if present)

Prints a compact report used to fill the manuscript.
"""
import csv, os, io, itertools, statistics, sys

BASE = _os.path.join(_R, "results", "ablation_full")
OUT = os.path.join(BASE, "_analysis4.txt")
buf = io.StringIO()
P = lambda *a: print(*a, file=buf)
FILES = ['bin.116', 'bin.117', 'bin.118', 'bin.119@500000',
         'bin.120', 'bin.6@400000', 'bin.7@400000']
ORDER = ["identity+rate", "tered+dual_naive", "tered+rate", "tered+rate_ratio",
         "tered+rate_single", "tered+none"]


def load(name):
    p = os.path.join(BASE, name)
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def num(v):
    try:
        return float(v)
    except Exception:
        return None


def boot_ci(d, n=20000, seed=12345):
    import random
    rng = random.Random(seed)
    n = len(d)
    if n == 0:
        return (float('nan'), float('nan'))
    means = []
    for _ in range(n if n > 1 else 1):
        pass
    # 20000 resamples of the paired differences
    means = [statistics.mean([d[rng.randrange(n)] for _ in range(n)]) for _ in range(20000)]
    means.sort()
    return (means[int(0.025 * len(means))], means[int(0.975 * len(means)) - 1])


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


main = load("e6e_gnn_baseline.csv")
extra = load("e6e_identity_ratestar.csv")
clean = load("e6e_cleantrain.csv")


def pf(rows, det, cfg, key, fs=FILES):
    return {r['file']: num(r[key]) for r in rows
            if r['detector'] == det and r['config'] == cfg
            and num(r[key]) is not None}


def cmp(tag, a, b, key='node_ROC_AUC', with_ci=True):
    fs = [f for f in FILES if f in a and f in b]
    if len(fs) < 2:
        P(f"  {tag:46s} <missing>")
        return
    d = [a[f] - b[f] for f in fs]
    w = sum(1 for x in d if x > 1e-12)
    l = sum(1 for x in d if x < -1e-12)
    lo, hi = boot_ci(d)
    ci = f" CI=[{lo:+.4f},{hi:+.4f}]" if with_ci else ""
    P(f"  {tag:46s} meanD={statistics.mean(d):+.4f} sd={statistics.pstdev(d):.4f} "
      f"W/L/T={w}/{l}/{len(fs)-w-l} p={exact_wilcoxon(d):.4f}{ci}")


# ---------------- 1. per-config mean node ROC-AUC / best-F1 ----------------
P("=" * 78)
P("1. Per-config mean node-level ROC-AUC and best-F1 (7 partitions)")
P("=" * 78)
for det in ["cosine", "gnn"]:
    P(f"\n-- detector={det} --")
    for cfg in ORDER:
        a = pf(main, det, cfg, 'node_ROC_AUC')
        if not a:
            continue
        f1 = pf(main, det, cfg, 'node_best_F1')
        pr = pf(main, det, cfg, 'node_PR_AUC')
        P(f"  {cfg:20s} AUC={statistics.mean(a.values()):.4f} "
          f"F1={statistics.mean(f1.values()):.4f} "
          f"PR={statistics.mean(pr.values()):.4f}")

# identity:rate_ratio from the control file
P("\n-- identity+rate_ratio (equal-width control, full budget) --")
for det in ["cosine", "gnn"]:
    a = pf(extra, det, "identity+rate_ratio", 'node_ROC_AUC')
    if a:
        f1 = pf(extra, det, "identity+rate_ratio", 'node_best_F1')
        P(f"  {det:8s} AUC={statistics.mean(a.values()):.4f} "
          f"F1={statistics.mean(f1.values()):.4f}")

# ---------------- 2. paired tests vs identity+rate ----------------
P("\n" + "=" * 78)
P("2. Paired node ROC-AUC: identity+rate  vs  each TeRed config")
P("=" * 78)
for det in ["cosine", "gnn"]:
    P(f"\n-- detector={det} --")
    idn = pf(main, det, "identity+rate", 'node_ROC_AUC')
    for cfg in ORDER[1:]:
        cmp(f"identity vs {cfg}", pf(main, det, cfg, 'node_ROC_AUC'), idn)

P("\n-- key TeRed contrasts --")
for det in ["cosine", "gnn"]:
    cmp(f"[{det}] rate_ratio vs dual_naive",
        pf(main, det, "tered+rate_ratio", 'node_ROC_AUC'),
        pf(main, det, "tered+dual_naive", 'node_ROC_AUC'))
    cmp(f"[{det}] dual_naive vs none",
        pf(main, det, "tered+dual_naive", 'node_ROC_AUC'),
        pf(main, det, "tered+none", 'node_ROC_AUC'))
    cmp(f"[{det}] rate_ratio vs rate_single",
        pf(main, det, "tered+rate_ratio", 'node_ROC_AUC'),
        pf(main, det, "tered+rate_single", 'node_ROC_AUC'))

# ---------------- 3. GNN vs cosine (same config) ----------------
P("\n" + "=" * 78)
P("3. GNN vs cosine on identical features (node ROC-AUC)")
P("=" * 78)
for cfg in ORDER:
    a = pf(main, "gnn", cfg, 'node_ROC_AUC')
    b = pf(main, "cosine", cfg, 'node_ROC_AUC')
    if a and b:
        cmp(f"gnn - cosine [{cfg}]", a, b, with_ci=False)

# ---------------- 4. clean-train vs all (label leakage check) ----------------
P("\n" + "=" * 78)
P("4. clean-train vs all-train (label-leakage check)")
P("=" * 78)
if clean:
    for det in ["cosine"]:
        for cfg in ["identity+rate", "tered+rate_ratio", "tered+dual_naive"]:
            a = pf(clean, det, cfg, 'node_ROC_AUC')
            b = pf(main, det, cfg, 'node_ROC_AUC')
            if a and b:
                fs = [f for f in FILES if f in a and f in b]
                d = [a[f] - b[f] for f in fs]
                P(f"  clean - all [{cfg}] n={len(fs)} "
                  f"max|d|={max(abs(x) for x in d):.2e} "
                  f"identical={all(abs(x) < 1e-9 for x in d)}")
else:
    P("  (e6e_cleantrain.csv 未生成)")

# ---------------- 5. pairing summary used in paper ----------------
P("\n" + "=" * 78)
P("5. Headline numbers (mean over 7 partitions, node ROC-AUC)")
P("=" * 78)
for det in ["cosine", "gnn"]:
    vals = {cfg: statistics.mean(pf(main, det, cfg, 'node_ROC_AUC').values())
            for cfg in ORDER if pf(main, det, cfg, 'node_ROC_AUC')}
    if vals:
        P(f"  {det}: " + "  ".join(f"{k}={v:.4f}" for k, v in vals.items()))

with open(OUT, "w", encoding="utf-8") as f:
    f.write(buf.getvalue())
sys.stdout.reconfigure(encoding="utf-8")
print(buf.getvalue())
