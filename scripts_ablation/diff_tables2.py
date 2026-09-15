import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), *[".."] * 1))

import csv, os, io

BASE = _os.path.join(_R, "results")
OUT = _os.path.join(_R, "scripts_ablation", "_out.txt")
buf = io.StringIO()

def load(p):
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))

v3 = load(os.path.join(BASE, "v3_final", "e6d_ablation_v3.csv"))
v3r = load(os.path.join(BASE, "v3_final", "e6d_ratestar_v3.csv"))
allv = v3 + v3r

def mean_over(rows, key):
    vals = [float(r[key]) for r in rows if r.get(key) not in (None, "")]
    return sum(vals)/len(vals) if vals else None

def sel(rows, **kw):
    return [r for r in rows if all(r.get(k)==v for k,v in kw.items())]

def fmt(x):
    return "None" if x is None else f"{x:.4f}"

def line(tag, rows):
    if not rows:
        print(f"  {tag:34s} <no rows>", file=buf); return
    print(f"  {tag:34s} n={len(rows)} " +
          " ".join(f"@{b}={fmt(mean_over(rows, f'F1_alert@{b}'))}" for b in [1,2,3,5,10,20]) +
          f"  node={fmt(mean_over(rows,'node_best_f1'))} cov={fmt(mean_over(rows,'node_cov'))}",
          file=buf)

print("=== FULL variant sweep, TeRed+RATE (encoding=rate_ratio) ===", file=buf)
for v in ["base","legacy_agg","legacy_cmax","legacy_w1","w050","w000","bfs_q50","bfs_q90","no_bfs","no_rank","sort_p95"]:
    line(f"RATE*:{v}", sel(allv, config="TeRed+RATE*", group="A1", variant=v))
    line(f"RATE*:{v}", sel(allv, config="TeRed+RATE*", group="A3", variant=v))
print("=== FULL variant sweep, TeRed+RATE (encoding=rate, SUM-mu) ===", file=buf)
for v in ["base","legacy_agg","legacy_w1","w050","w000","bfs_q50","bfs_q90","no_bfs","no_rank","sort_p95"]:
    line(f"SUMmu:{v}", sel(allv, config="TeRed+RATE", group="A1", variant=v, encoding="rate"))
    line(f"SUMmu:{v}", sel(allv, config="TeRed+RATE", group="A3", variant=v, encoding="rate"))
print("=== TeRed+naive (dual_naive) A1/A3 ===", file=buf)
for v in ["base","legacy_agg","legacy_w1","w050","w000","bfs_q50","bfs_q90","no_bfs","no_rank","sort_p95"]:
    line(f"count:{v}", sel(allv, config="TeRed+naive", group="A1", variant=v))
    line(f"count:{v}", sel(allv, config="TeRed+naive", group="A3", variant=v))
print("=== identity ===", file=buf)
for v in ["base","legacy_agg","bfs_q50","bfs_q90","no_bfs","sort_p95"]:
    line(f"id:{v}", sel(allv, config="identity+rate", group="A1", variant=v))
    line(f"id:{v}", sel(allv, config="identity+rate", group="A3", variant=v))

print("\n=== n_gt / nodes per file (RATE* base) ===", file=buf)
for r in sorted(sel(allv, config="TeRed+RATE*", group="A1", variant="base"), key=lambda r: r["file"]):
    print(f"  {r['file']:16s} n_gt={r['n_gt']:>4s} nodes={r['n_nodes_Gp']:>8s} edges={r['n_edges_Gp']:>8s}", file=buf)

with open(OUT, "w", encoding="utf-8") as f:
    f.write(buf.getvalue())
print("done")
