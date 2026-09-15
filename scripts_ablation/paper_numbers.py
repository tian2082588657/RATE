import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), *[".."] * 1))

import csv, os, io, math, itertools, random, statistics

BASE = _os.path.join(_R, "results")
OUT = _os.path.join(_R, "scripts_ablation", "_out.txt")
buf = io.StringIO()
P = lambda *a: print(*a, file=buf)

def load(p):
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))

v3 = load(os.path.join(BASE, "v3_final", "e6d_ablation_v3.csv"))
v3r = load(os.path.join(BASE, "v3_final", "e6d_ratestar_v3.csv"))
e6 = load(os.path.join(BASE, "v3_final", "e6_alert_full_v3.csv"))
e7 = load(os.path.join(BASE, "v3_final", "e7_origunit.csv"))
A = v3 + v3r
FILES = ['bin.116','bin.117','bin.118','bin.119@500000','bin.120','bin.6@400000','bin.7@400000']

def sel(rows, **kw):
    return [r for r in rows if all(r.get(k)==v for k,v in kw.items())]
def col(rows, k):
    return [float(r[k]) for r in rows if r.get(k) not in (None,"")]
def mean(x): return sum(x)/len(x) if x else None
def f4(x): return "None" if x is None else f"{x:.4f}"

# ---------- 1. positives 范围 ----------
P("### 1. positives per partition")
P(f"{'file':16s} {'orig_gt':>8s} {'kept':>6s} {'absorbed':>9s} {'red_gt':>7s} {'nodes_red':>10s}")
for spec in FILES:
    r = sel(e7, file=spec, config="TeRed+RATE*")
    if not r: r = sel(e7, file=spec, config="TeRed+RATE")
    r = r[0]
    P(f"{spec:16s} {r['n_orig_gt']:>8s} {r['n_gt_kept']:>6s} {r['n_gt_absorbed']:>9s} "
      f"{r['n_gt_reduced']:>7s} {r['n_nodes_Gp']:>10s}")
orig = [int(r['n_orig_gt']) for r in e7 if r.get('n_orig_gt')]
red = [int(r['n_gt_reduced']) for r in e7 if r.get('n_gt_reduced')]
P(f"  orig_gt range = {min(orig)}..{max(orig)}  (sum={sum(orig)})")
P(f"  reduced gt range = {min(red)}..{max(red)}")

# ---------- 2. reduced-unit load & 27% ----------
P("\n### 2. reduced-unit load (n_alerted_nodes), e6_alert_full_v3")
for cfg in ["identity+rate","TeRed+naive","TeRed+RATE","TeRed+RATE*"]:
    rows = sel(e6, config=cfg)
    if not rows: continue
    P(f"  {cfg:14s} load_mean={mean(col(rows,'n_alerted_nodes')):.1f} "
      f"cov={f4(mean(col(rows,'node_cov')))} @5={f4(mean(col(rows,'F1_alert'))) if False else ''}")

# ---------- 3. self-loop 消融（RATE*） ----------
P("\n### 3. A4 v4-drop (config=TeRed+RATE*, encoding=rate_ratio)")
for var in ["base","v4_drop_log_deg","v4_drop_out_in_ratio","v4_drop_self_loop"]:
    rows = sel(A, config="TeRed+RATE*", group="A4", variant=var)
    if not rows: rows = sel(A, config="TeRed+RATE*", group="A1", variant=var)
    if not rows: P(f"  {var}: <none>"); continue
    P(f"  {var:22s} nodeF1={f4(mean(col(rows,'node_best_f1')))} "
      f"@5={f4(mean(col(rows,'F1_alert@5')))} @10={f4(mean(col(rows,'F1_alert@10')))} "
      f"cov={f4(mean(col(rows,'node_cov')))} load={mean(col(rows,'n_alerted_nodes')):.0f}")

# ---------- 4. 配对差异 + bootstrap CI + exact Wilcoxon ----------
def exact_wilcoxon(d):
    d = [x for x in d if abs(x) > 1e-12]
    n = len(d)
    if n == 0: return 1.0, 0, 0
    r = sorted(range(n), key=lambda i: abs(d[i]))
    ranks = [0.0]*n
    i = 0
    while i < n:
        j = i
        while j+1 < n and abs(abs(d[r[j+1]])-abs(d[r[i]])) < 1e-12: j += 1
        avg = (i+1+j+1)/2.0
        for k in range(i, j+1): ranks[r[k]] = avg
        i = j+1
    T = sum(ranks[i] for i in range(n) if d[i] > 0)
    tot = sum(ranks)
    cnt = 0; N = 0
    for signs in itertools.product([0,1], repeat=n):
        s = sum(ranks[i] for i in range(n) if signs[i])
        N += 1
        if abs(s - tot/2.0) >= abs(T - tot/2.0) - 1e-12: cnt += 1
    return cnt/N, n, int(T)

def boot_ci(d, iters=20000, seed=0):
    rnd = random.Random(seed)
    n = len(d)
    if n == 0: return (None, None)
    ms = []
    for _ in range(iters):
        ms.append(sum(d[rnd.randrange(n)] for _ in range(n))/n)
    ms.sort()
    return ms[int(0.025*iters)], ms[int(0.975*iters)-1]

P("\n### 4. 关键配对比较（per-file, config=TeRed+RATE*）")
def perfile(cfg, grp, var, enc, key):
    rows = sel(A, config=cfg, group=grp, variant=var, encoding=enc)
    return {r['file']: float(r[key]) for r in rows if r.get(key) not in (None,"")}

comparisons = [
    ("RATE* vs count : nodeF1", perfile("TeRed+RATE*","A5","enc_rate_ratio","rate_ratio","node_best_f1"),
                                perfile("TeRed+RATE","A5","enc_dual_naive","dual_naive","node_best_f1")),
    ("RATE* vs count : alert@10", perfile("TeRed+RATE*","A5","enc_rate_ratio","rate_ratio","F1_alert@10"),
                                  perfile("TeRed+RATE","A5","enc_dual_naive","dual_naive","F1_alert@10")),
    ("RATE* vs scalar: nodeF1", perfile("TeRed+RATE*","A5","enc_rate_ratio","rate_ratio","node_best_f1"),
                               perfile("TeRed+RATE","A5","enc_rate_single","rate_single","node_best_f1")),
    ("RATE* vs none  : nodeF1", perfile("TeRed+RATE*","A5","enc_rate_ratio","rate_ratio","node_best_f1"),
                               perfile("TeRed+RATE","A5","enc_none","none","node_best_f1")),
    ("RATE* vs count : common-unit recall", None, None),
]
for tag, a, b in comparisons[:4]:
    if not a or not b: P(f"  {tag}: <missing>"); continue
    fs = [f for f in FILES if f in a and f in b]
    d = [a[f]-b[f] for f in fs]
    w = sum(1 for x in d if x > 1e-12); l = sum(1 for x in d if x < -1e-12)
    p, n, T = exact_wilcoxon(d)
    lo, hi = boot_ci(d)
    P(f"  {tag}: mean Δ={mean(d):+.4f} sd={statistics.pstdev(d):.4f} "
      f"W/L/T={w}/{l}/{len(d)-w-l} p={p:.4f} boot95=[{lo:+.4f},{hi:+.4f}] d={mean(d)/statistics.pstdev(d):.2f}")

# ---------- 5. entity-level precision / alert ----------
P("\n### 5. alert composition (reduced unit): alerted nodes & GT inside")
for cfg, enc in [("identity+rate","rate"),("TeRed+naive","dual_naive"),("TeRed+RATE*","rate_ratio")]:
    rows = sel(e6, config=cfg)
    if not rows: P(f"  {cfg}: <none>"); continue
    P(f"  {cfg:14s} alerts={mean(col(rows,'n_alerts')):.1f} "
      f"P_alert={f4(mean(col(rows,'alert_P')))} "
      f"alerted_nodes={mean(col(rows,'n_alerted_nodes')):.0f} "
      f"node_prec_in_alerts={f4(mean(col(rows,'node_prec')))} "
      f"cov={f4(mean(col(rows,'node_cov')))}")

# ---------- 6. common-unit per-file recall ----------
P("\n### 6. common-unit per-file")
for spec in FILES:
    row = sel(e7, file=spec)
    P(f"  {spec:16s} " + " ".join(
        f"{r['config']}={float(r['orig_recall']):.3f}/F1={float(r['F1_orig']):.3f}" for r in row))

with open(OUT, "w", encoding="utf-8") as f:
    f.write(buf.getvalue())
print("done")
