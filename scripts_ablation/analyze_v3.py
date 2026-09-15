# -*- coding: utf-8 -*-
"""v3 最终数据分文件分析：e7 原图单元 / e6_alert / e6d A1 base。"""
import csv
import os
import statistics as st

D = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "results", "v3_final")


def load(fn):
    return list(csv.DictReader(open(os.path.join(D, fn), encoding="utf-8-sig")))


def f(r, c, d=float("nan")):
    try:
        return float(r[c])
    except (KeyError, ValueError, TypeError):
        return d


e7 = load("e7_origunit.csv")
e6 = load("e6_alert_full_v3.csv")
e6d = load("e6d_ablation_v3.csv")

print("=" * 100)
print("E7 原图单元 per-file (recall / F1_orig / P_alert / n_alerted_orig)")
print("=" * 100)
cfgs = ["identity+rate", "TeRed+naive", "TeRed+RATE", "TeRed+RATE*"]
short = {"identity+rate": "identity", "TeRed+naive": "count",
         "TeRed+RATE": "RATE(Σμ)", "TeRed+RATE*": "RATE*"}
files = sorted(set(r["file"] for r in e7))
hdr = f"{'file':14s}" + "".join(f"{short[c]:>26s}" for c in cfgs)
print(hdr)
for fl in files:
    line = f"{fl:14s}"
    for c in cfgs:
        rs = [r for r in e7 if r["file"] == fl and r["config"] == c]
        if rs:
            r0 = rs[0]
            line += (f"  {f(r0,'orig_recall'):.3f}/{f(r0,'F1_orig'):.3f}"
                     f"/{f(r0,'alert_P_orig'):.3f}/{int(f(r0,'n_alerted_orig')):5d}")
        else:
            line += f"{'---':>26s}"
    print(line)
print()
for c in cfgs:
    rs = [r for r in e7 if r["config"] == c]
    if rs:
        print(f"{short[c]:10s} mean recall={st.mean(f(r,'orig_recall') for r in rs):.4f} "
              f"F1={st.mean(f(r,'F1_orig') for r in rs):.4f} "
              f"P={st.mean(f(r,'alert_P_orig') for r in rs):.4f} "
              f"load={st.mean(f(r,'n_alerted_orig') for r in rs):.0f}")

print()
print("=" * 100)
print("E6d A1 base per-file (nodeF1 / F1@5 / F1@10 / cov)")
print("=" * 100)
for fl in files:
    line = f"{fl:14s}"
    for c in ["identity+rate", "TeRed+naive", "TeRed+RATE"]:
        rs = [r for r in e6d if r["file"] == fl and r["config"] == c
              and r["group"] == "A1" and r["variant"] == "base"]
        if rs:
            r0 = rs[0]
            line += (f"  {f(r0,'node_best_f1'):.4f}/{f(r0,'F1_alert@5'):.3f}"
                     f"/{f(r0,'F1_alert@10'):.3f}/{f(r0,'node_cov'):.3f}")
        else:
            line += f"{'---':>26s}"
    print(line)

print()
print("=" * 100)
print("E6d A1 base: RATE(Σμ) vs count 配对 (F1@5/@10)")
print("=" * 100)
a = {r["file"]: r for r in e6d if r["config"] == "TeRed+RATE"
     and r["group"] == "A1" and r["variant"] == "base"}
b = {r["file"]: r for r in e6d if r["config"] == "TeRed+naive"
     and r["group"] == "A1" and r["variant"] == "base"}
common = sorted(set(a) & set(b))
for c in ["F1_alert@5", "F1_alert@10", "node_best_f1", "node_cov"]:
    w = sum(1 for fl in common if f(a[fl], c) > f(b[fl], c))
    l = sum(1 for fl in common if f(a[fl], c) < f(b[fl], c))
    t = len(common) - w - l
    ma = st.mean(f(a[fl], c) for fl in common)
    mb = st.mean(f(b[fl], c) for fl in common)
    print(f"{c:14s} RATE={ma:.4f} count={mb:.4f}  胜{w}/负{l}/平{t}")
