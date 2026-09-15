import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), *[".."] * 1))

import csv, os, io, statistics

BASE = _os.path.join(_R, "results")
OUT = _os.path.join(_R, "scripts_ablation", "_out.txt")
buf = io.StringIO()

def load(p):
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))

v3 = load(os.path.join(BASE, "v3_final", "e6d_ablation_v3.csv"))
v3r = load(os.path.join(BASE, "v3_final", "e6d_ratestar_v3.csv"))
v2 = load(os.path.join(BASE, "ablation_full", "e6d_ablation_v2.csv"))
allv3 = v3 + v3r

FILES = ['bin.116','bin.117','bin.118','bin.119@500000','bin.120','bin.6@400000','bin.7@400000']

def mean_over(rows, key):
    vals = [float(r[key]) for r in rows if r.get(key) not in (None, "")]
    return sum(vals)/len(vals) if vals else None

def sel(rows, **kw):
    return [r for r in rows if all(r.get(k)==v for k,v in kw.items())]

def fmt(x):
    return "None" if x is None else f"{x:.4f}"

print("### A. TeRed+RATE config, A1 base: F1@b by encoding", file=buf)
for enc in ["rate", "rate_ratio", "dual_naive"]:
    rows = sel(allv3, config="TeRed+RATE", group="A1", variant="base", encoding=enc)
    if not rows:
        continue
    print(f" encoding={enc:12s} n={len(rows)} " +
          " ".join(f"@{b}={fmt(mean_over(rows, f'F1_alert@{b}'))}" for b in [1,2,3,5,10,20]),
          file=buf)
    print(f"    node_best_f1={fmt(mean_over(rows,'node_best_f1'))} cov={fmt(mean_over(rows,'node_cov'))}",
          file=buf)

print("\n### B. same from v2 (older run)", file=buf)
for enc in ["rate", "dual_naive"]:
    rows = sel(v2, group="A1", variant="base", encoding=enc)
    if not rows:
        continue
    print(f" encoding={enc:12s} n={len(rows)} " +
          " ".join(f"@{b}={fmt(mean_over(rows, f'F1_alert@{b}'))}" for b in [1,2,3,5,10,20]),
          file=buf)

print("\n### C. tab:abl-budget rows (A1 base, 3 configs)", file=buf)
for cfg, enc in [("identity+rate","rate"),("TeRed+naive","dual_naive"),("TeRed+RATE","rate_ratio")]:
    rows = sel(allv3, config=cfg, group="A1", variant="base", encoding=enc)
    print(f" {cfg:14s}/{enc:11s} n={len(rows)} " +
          " ".join(f"@{b}={fmt(mean_over(rows, f'F1_alert@{b}'))}" for b in [1,2,3,5,10,20]),
          file=buf)

print("\n### D. tab:abl-enc rows (A5 + A1)", file=buf)
# dual-channel edge count = TeRed+naive base (dual_naive)
for label, cfg, grp, var, enc in [
    ("edge-count(TeRed+count)", "TeRed+naive", "A1", "base", "dual_naive"),
    ("sum-mu (TeRed+RATE/rate)", "TeRed+RATE", "A1", "base", "rate"),
    ("count+ratio (RATE*)", "TeRed+RATE", "A1", "base", "rate_ratio"),
]:
    rows = sel(allv3, config=cfg, group=grp, variant=var, encoding=enc)
    print(f" {label:26s} node={fmt(mean_over(rows,'node_best_f1'))} "
          f"@5={fmt(mean_over(rows,'F1_alert@5'))} @10={fmt(mean_over(rows,'F1_alert@10'))} "
          f"cov={fmt(mean_over(rows,'node_cov'))}", file=buf)
# A5 encodings on TeRed+RATE graph
print(" -- A5 encodings (same reduced graph, TeRed+RATE) --", file=buf)
for enc in ["dual_naive","rate","rate_ratio","rate_ratio_only","rate_rank","rate_single","none"]:
    rows = sel(allv3, config="TeRed+RATE", group="A5", variant=f"enc_{enc}")
    if not rows:
        continue
    print(f" {enc:16s} n={len(rows)} node={fmt(mean_over(rows,'node_best_f1'))} "
          f"@5={fmt(mean_over(rows,'F1_alert@5'))} @10={fmt(mean_over(rows,'F1_alert@10'))} "
          f"cov={fmt(mean_over(rows,'node_cov'))}", file=buf)

print("\n### E. per-file node_best_f1 for key encodings (tab:perfile)", file=buf)
perfile = {}
for enc in ["dual_naive","rate_ratio","rate","rate_single","none"]:
    rows = sel(allv3, config="TeRed+RATE", group="A5", variant=f"enc_{enc}")
    d = {r["file"]: float(r["node_best_f1"]) for r in rows}
    perfile[enc] = d
print(" file          " + " ".join(f"{e:>10s}" for e in perfile), file=buf)
for f in FILES:
    print(f" {f:14s}" + " ".join(f"{perfile[e].get(f, float('nan')):10.4f}" for e in perfile), file=buf)

print("\n### F. counts: positives per partition", file=buf)
rows = sel(allv3, config="TeRed+RATE", group="A1", variant="base", encoding="rate_ratio")
for r in sorted(rows, key=lambda r: r["file"]):
    print(f"  {r['file']:16s} n_gt={r['n_gt']:>4s} nodes={r['n_nodes_Gp']:>8s}", file=buf)

with open(OUT, "w", encoding="utf-8") as f:
    f.write(buf.getvalue())
print("done")
