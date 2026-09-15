# -*- coding: utf-8 -*-
"""A5 编码消融：@5/@10 固定预算的配对精确 Wilcoxon（n=7，双侧）。"""
import csv
import itertools
import os
from collections import defaultdict

D = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "results", "v3_final")
rows = list(csv.DictReader(open(os.path.join(D, "e6d_ablation_v3.csv"),
                                encoding="utf-8-sig")))


def exact_wilcoxon(a, b):
    d = [x - y for x, y in zip(a, b) if x != y]
    n = len(d)
    if n == 0:
        return 1.0, 0, 0, 0
    w = sum(1 for x in d if x > 0)
    obs = min(w, n - w)
    cnt = 0
    tot = 0
    for signs in itertools.product([0, 1], repeat=n):
        k = sum(signs)
        tot += 1
        if min(k, n - k) <= obs:
            cnt += 1
    return cnt / tot, w, n - w, len(a) - n


data = defaultdict(dict)
for r in rows:
    if r["group"] == "A5":
        data[r["encoding"]][r["file"]] = r
    elif (r["group"] == "A1" and r["variant"] == "base"
          and r["config"] == "TeRed+RATE"):
        data["rate"][r["file"]] = r  # Σμ 对照行（A1 base）

encs = ["dual_naive", "rate", "rate_ratio", "rate_ratio_only",
        "rate_rank", "rate_single"]
files = sorted(set.intersection(*[set(data[e]) for e in encs
                                  if data[e]]) if data else [])
print("common files:", len(files))
for col in ["F1_alert@5", "F1_alert@10", "node_best_f1", "node_cov"]:
    print(f"\n--- {col} ---")
    for i in range(len(encs)):
        for j in range(i + 1, len(encs)):
            e1, e2 = encs[i], encs[j]
            if e1 not in data or e2 not in data:
                continue
            a = [float(data[e1][f][col]) for f in files]
            b = [float(data[e2][f][col]) for f in files]
            p, w, l, t = exact_wilcoxon(a, b)
            m1 = sum(a) / len(a)
            m2 = sum(b) / len(b)
            flag = " *" if p < 0.05 else ""
            print(f"  {e1:16s} vs {e2:16s} {m1:.4f}/{m2:.4f} "
                  f"+{w}/-{l}/={t} p={p:.3f}{flag}")
