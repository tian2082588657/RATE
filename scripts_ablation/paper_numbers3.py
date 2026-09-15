import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), *[".."] * 1))

import csv, os, io

BASE = _os.path.join(_R, "results")
OUT = _os.path.join(_R, "scripts_ablation", "_out3.txt")
buf = io.StringIO()
P = lambda *a: print(*a, file=buf)

with open(os.path.join(BASE, "v3_final", "e7_origunit.csv"), encoding="utf-8") as f:
    e7 = list(csv.DictReader(f))

P("### common-unit per-config means (e7_origunit)")
P(f"{'config':16s} {'recall':>8s} {'F1':>8s} {'P_orig':>8s} {'P_alert':>8s} {'load':>8s} {'red_cov':>8s} {'red_load':>8s}")
for cfg in ["identity+rate", "TeRed+naive", "TeRed+RATE", "TeRed+RATE*"]:
    rows = [r for r in e7 if r['config'] == cfg]
    if not rows:
        continue
    def m(k):
        v = [float(r[k]) for r in rows if r.get(k) not in (None, "")]
        return sum(v)/len(v) if v else float('nan')
    P(f"{cfg:16s} {m('orig_recall'):8.4f} {m('F1_orig'):8.4f} {m('orig_load_prec'):8.4f} "
      f"{m('alert_P_orig'):8.4f} {m('n_alerted_orig'):8.1f} {m('node_cov_red'):8.4f} {m('n_alerted_red'):8.1f}")

P("\n### 吸收率")
for cfg in ["TeRed+RATE*"]:
    rows = [r for r in e7 if r['config'] == cfg]
    tot = sum(int(r['n_orig_gt']) for r in rows)
    kept = sum(int(r['n_gt_kept']) for r in rows)
    absb = sum(int(r['n_gt_absorbed']) for r in rows)
    bnd = sum(int(r['n_boundary_gt']) for r in rows)
    P(f"  {cfg}: slots={tot} kept={kept} absorbed={absb} ({absb/tot:.0%}) boundary={bnd}")

with open(OUT, "w", encoding="utf-8") as f:
    f.write(buf.getvalue())
