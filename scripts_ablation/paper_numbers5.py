import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), *[".."] * 1))

"""paper_numbers5.py — table-ready numbers for the 0912-2 rebuttal edits."""
import csv, os, io, itertools, statistics, sys

BASE = _os.path.join(_R, "results", "ablation_full")
OUT = os.path.join(BASE, "_analysis5.txt")
buf = io.StringIO()
P = lambda *a: print(*a, file=buf)
FILES = ['bin.116', 'bin.117', 'bin.118', 'bin.119@500000',
         'bin.120', 'bin.6@400000', 'bin.7@400000']
ORDER = ["identity+rate", "identity+rate_ratio", "tered+dual_naive", "tered+rate",
         "tered+rate_ratio", "tered+rate_single", "tered+none"]


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


main = load("e6e_gnn_baseline.csv")
extra = load("e6e_identity_ratestar.csv")
ALL = main + extra


def vals(det, cfg, key):
    return [num(r[key]) for r in ALL if r['detector'] == det and r['config'] == cfg
            and num(r[key]) is not None]


P("=" * 84)
P("COSINE (feature-based ensemble) — mean over 7 partitions")
P("=" * 84)
hdr = f"{'config':22s} {'ROC-AUC':>8s} {'bestF1':>8s} {'PR-AUC':>8s} {'F1@5':>8s} {'F1@10':>8s} {'F1@20':>8s} {'n_alerted':>10s}"
P(hdr)
for cfg in ORDER:
    a = vals("cosine", cfg, "node_ROC_AUC")
    if not a:
        P(f"{cfg:22s} <missing>"); continue
    def g(k):
        v = vals("cosine", cfg, k)
        return statistics.mean(v) if v else float('nan')
    P(f"{cfg:22s} {g('node_ROC_AUC'):8.4f} {g('node_best_F1'):8.4f} "
      f"{g('node_PR_AUC'):8.4f} {g('F1_alert@5'):8.4f} {g('F1_alert@10'):8.4f} "
      f"{g('F1_alert@20'):8.4f} {g('n_alerted_nodes'):10.0f}")

P("")
P("=" * 84)
P("Equal-width control (both unreduced and reduced use the SAME 4-channel")
P("rate_ratio encoding; identical encoder, only reduction differs)")
P("=" * 84)
for k in ["node_ROC_AUC", "node_best_F1", "F1_alert@5", "F1_alert@10"]:
    a = vals("cosine", "identity+rate_ratio", k)
    b = vals("cosine", "tered+rate_ratio", k)
    P(f"  {k:14s} identity+rate_ratio={statistics.mean(a):.4f}  "
      f"tered+rate_ratio={statistics.mean(b):.4f}  diff={statistics.mean(b)-statistics.mean(a):+.4f}")

P("")
P("=" * 84)
P("GNN (unsupervised message-passing) — mean over 7 partitions")
P("=" * 84)
P(f"{'config':22s} {'ROC-AUC':>8s} {'bestF1':>8s} {'PR-AUC':>8s} {'P@100':>8s}")
for cfg in ORDER:
    a = vals("gnn", cfg, "node_ROC_AUC")
    if not a:
        P(f"{cfg:22s} <missing>"); continue
    def g(k):
        v = vals("gnn", cfg, k)
        return statistics.mean(v) if v else float('nan')
    P(f"{cfg:22s} {g('node_ROC_AUC'):8.4f} {g('node_best_F1'):8.4f} "
      f"{g('node_PR_AUC'):8.4f} {g('P@100'):8.4f}")

P("")
P("=" * 84)
P("Dataset facts for the setup paragraph")
P("=" * 84)
a = vals("cosine", "identity+rate", "n_gt")
P(f"  mean n_gt (reduced-unit positives, identity rows) = {statistics.mean(a):.1f}")
b = vals("cosine", "tered+rate_ratio", "n_gt")
P(f"  mean n_gt (tered rows)                            = {statistics.mean(b):.1f}")
c = vals("cosine", "tered+rate_ratio", "covered_gt")
P(f"  mean covered_gt                                   = {statistics.mean(c):.1f}")
d = vals("cosine", "tered+rate_ratio", "node_prec")
P(f"  mean node precision inside alerts (RATE*)          = {statistics.mean(d):.6f}")

with open(OUT, "w", encoding="utf-8") as f:
    f.write(buf.getvalue())
sys.stdout.reconfigure(encoding="utf-8")
print(buf.getvalue())
