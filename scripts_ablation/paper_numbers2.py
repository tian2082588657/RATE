import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), *[".."] * 1))

import csv, os, io, itertools, random, statistics, traceback

BASE = _os.path.join(_R, "results")
OUT = _os.path.join(_R, "scripts_ablation", "_out2.txt")
buf = io.StringIO()


def P(*a):
    print(*a, file=buf)


def load(p):
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def run():
    v3 = load(os.path.join(BASE, "v3_final", "e6d_ablation_v3.csv"))
    v3r = load(os.path.join(BASE, "v3_final", "e6d_ratestar_v3.csv"))
    A = v3 + v3r
    FILES = ['bin.116', 'bin.117', 'bin.118', 'bin.119@500000',
             'bin.120', 'bin.6@400000', 'bin.7@400000']

    def sel(rows, **kw):
        return [r for r in rows if all(r.get(k) == v for k, v in kw.items())]

    def perfile(cfg, grp, var, key, enc=None):
        kw = dict(config=cfg, group=grp, variant=var)
        if enc:
            kw["encoding"] = enc
        rows = sel(A, **kw)
        return {r['file']: float(r[key]) for r in rows
                if r.get(key) not in (None, "")}

    def mean(x):
        return sum(x) / len(x) if x else None

    def fv(rows, key):
        return [float(r[key]) for r in rows if r.get(key) not in (None, "")]

    def f4(x):
        return "None" if x is None else f"{x:.4f}"

    def exact_wilcoxon(d):
        d = [x for x in d if abs(x) > 1e-12]   # 标准约定：零差剔除
        n = len(d)
        if n == 0:
            return 1.0
        r = sorted(range(n), key=lambda i: abs(d[i]))
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and abs(abs(d[r[j + 1]]) - abs(d[r[i]])) < 1e-12:
                j += 1
            avg = (i + 1 + j + 1) / 2.0
            for k in range(i, j + 1):
                ranks[r[k]] = avg
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

    def boot_ci(d, iters=20000, seed=0):
        rnd = random.Random(seed)
        n = len(d)
        if n == 0:
            return (None, None)
        ms = sorted(sum(d[rnd.randrange(n)] for _ in range(n)) / n
                    for _ in range(iters))
        return ms[int(0.025 * iters)], ms[int(0.975 * iters) - 1]

    def report(tag, a, b):
        fs = [f for f in FILES if f in a and f in b]
        if len(fs) < 2:
            P(f"  {tag}: <missing> a={len(a)} b={len(b)}")
            return
        d = [a[f] - b[f] for f in fs]
        w = sum(1 for x in d if x > 1e-12)
        l = sum(1 for x in d if x < -1e-12)
        p = exact_wilcoxon(d)
        lo, hi = boot_ci(d)
        sd = statistics.pstdev(d)
        P(f"  {tag:32s} meanD={mean(d):+.4f} sd={sd:.4f} W/L/T={w}/{l}/{len(fs)-w-l} "
          f"p={p:.4f} boot95=[{lo:+.4f},{hi:+.4f}] d={mean(d)/sd if sd else 0:.2f}")

    RATE_node = perfile("TeRed+RATE*", "A1", "base", "node_best_f1")
    CNT_node = perfile("TeRed+naive", "A1", "base", "node_best_f1")
    RATE_a10 = perfile("TeRed+RATE*", "A1", "base", "F1_alert@10")
    CNT_a10 = perfile("TeRed+naive", "A1", "base", "F1_alert@10")
    RATE_a5 = perfile("TeRed+RATE*", "A1", "base", "F1_alert@5")
    CNT_a5 = perfile("TeRed+naive", "A1", "base", "F1_alert@5")
    SCAL_node = perfile("TeRed+RATE", "A5", "enc_rate_single", "node_best_f1", "rate_single")
    NONE_node = perfile("TeRed+RATE", "A5", "enc_none", "node_best_f1", "none")

    P("### 4. key paired comparisons (n=7, exact Wilcoxon)")
    report("RATE* vs count : nodeF1", RATE_node, CNT_node)
    report("RATE* vs count : alert@5", RATE_a5, CNT_a5)
    report("RATE* vs count : alert@10", RATE_a10, CNT_a10)
    report("RATE* vs scalar: nodeF1", RATE_node, SCAL_node)
    report("RATE* vs none  : nodeF1", RATE_node, NONE_node)

    P("\n### 4b. per-file nodeF1")
    P("  " + " ".join(f"{f[:11]:>12s}" for f in FILES))
    for tag, d in [("RATE*", RATE_node), ("count", CNT_node),
                   ("scalar", SCAL_node), ("none", NONE_node)]:
        P(f"  {tag:8s}" + " ".join(f"{d.get(f, float('nan')):12.4f}" for f in FILES))
    P("  delta   " + " ".join(f"{RATE_node[f]-CNT_node[f]:+12.4f}" for f in FILES))

    P("\n### 3. A4 v4-drop")
    for cfg in ["TeRed+RATE*", "TeRed+naive"]:
        base = sel(A, config=cfg, group="A1", variant="base")
        if base:
            P(f"  {cfg:14s} base                nodeF1={f4(mean(fv(base,'node_best_f1')))} "
              f"@5={f4(mean(fv(base,'F1_alert@5')))} "
              f"@10={f4(mean(fv(base,'F1_alert@10')))}")
        for var in ["v4_drop_log_deg", "v4_drop_out_in_ratio", "v4_drop_self_loop"]:
            rows = sel(A, config=cfg, group="A4", variant=var)
            if not rows:
                continue
            P(f"  {cfg:14s} {var:20s} nodeF1={f4(mean(fv(rows,'node_best_f1')))} "
              f"@5={f4(mean(fv(rows,'F1_alert@5')))} "
              f"@10={f4(mean(fv(rows,'F1_alert@10')))} "
              f"cov={f4(mean(fv(rows,'node_cov')))} "
              f"load={mean(fv(rows,'n_alerted_nodes')):.0f}")

    P("\n### 4c. Bonferroni on the 4 directional-split comparisons")
    for k in [1, 2, 4, 6]:
        P(f"  k={k}: p_adj={min(1.0, 0.015625*k):.4f}")

    P("\n### 5. positives-per-alert / cluster purity (reduced unit)")
    e6 = load(os.path.join(BASE, "v3_final", "e6_alert_full_v3.csv"))
    for cfg in ["identity+rate", "TeRed+naive", "TeRed+RATE", "TeRed+RATE*"]:
        rows = sel(e6, config=cfg)
        if not rows:
            continue
        n_al = fv(rows, 'n_alerts')
        cg = fv(rows, 'covered_gt')
        P(f"  {cfg:14s} alerts={mean(n_al):.1f} covered_gt={mean(cg):.1f} "
          f"gt_per_alert={mean([c/a if a else 0 for c, a in zip(cg, n_al)]):.2f} "
          f"alerted_nodes={mean(fv(rows,'n_alerted_nodes')):.0f} "
          f"node_prec={f4(mean(fv(rows,'node_prec')))} "
          f"P_alert={f4(mean(fv(rows,'alert_P'))) }")


try:
    run()
except Exception:
    P(traceback.format_exc())

with open(OUT, "w", encoding="utf-8") as f:
    f.write(buf.getvalue())
