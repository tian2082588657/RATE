# -*- coding: utf-8 -*-
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), *[".."] * 1))

"""核验 DeepSeek 对 Manuscript_RATE.tex 的数值指控，并输出统一口径表。"""
import csv, statistics as st
from collections import defaultdict

BASE = _os.path.join(_R, "results", "ablation_full")

def load(fn):
    rows = []
    with open(f"{BASE}/{fn}", newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows

def fl(x):
    try: return float(x)
    except: return None

def agg(rows, keyf, fields):
    """keyf(row)->key; returns dict key -> {field: (mean,std,n)}"""
    buckets = defaultdict(lambda: defaultdict(list))
    for r in rows:
        k = keyf(r)
        for fld in fields:
            v = fl(r.get(fld, ""))
            if v is not None:
                buckets[k][fld].append(v)
    out = {}
    for k, d in buckets.items():
        out[k] = {fld: (st.mean(vs), (st.pstdev(vs) if len(vs) > 1 else 0.0), len(vs))
                  for fld, vs in d.items()}
    return out

def show(title, res, order=None):
    print(f"\n=== {title} ===")
    keys = order or sorted(res.keys())
    for k in keys:
        if k not in res: continue
        d = res[k]
        parts = [f"{fld}={d[fld][0]:.4f}±{d[fld][1]:.4f}" for fld in d]
        ks = k if isinstance(k, str) else "|".join(map(str, k))
        print(f"  {ks:40s} " + "  ".join(parts))

# ---------- 1. A1 base: 主表口径（新） ----------
a1 = load("e6d_ablation_v2.csv")
a1 = [r for r in a1 if r["group"] == "A1"]
flds = ["node_best_f1", "best_F1_alert", "node_cov", "alert_P"]
res = agg(a1, lambda r: (r["config"], r["variant"]), flds)
print("### DeepSeek#3 冗余列检查：node_cov 与 alert 覆盖是否同源")
for cfg in ["identity+rate", "TeRed+naive", "TeRed+RATE"]:
    r = res.get((cfg, "base"))
    if r: print(f"  {cfg}: node_cov={r['node_cov'][0]:.4f}")

# ---------- 2. A1 固定预算 @1..@20（新口径） ----------
bud = ["F1_alert@1", "F1_alert@2", "F1_alert@3", "F1_alert@5", "F1_alert@10", "F1_alert@20"]
resb = agg(a1, lambda r: (r["config"], r["variant"]), bud)
show("DeepSeek#3 固定预算（base 新口径）", resb,
     order=[(c, "base") for c in ["identity+rate", "TeRed+naive", "TeRed+RATE"]])

# ---------- 3. A3 排序键 × 预算 ----------
a3 = [r for r in load("e6d_ablation_v2.csv") if r["group"] == "A3"]
res3 = agg(a3, lambda r: (r["config"], r["variant"]), bud)
show("A3 排序键 × 预算（新口径）", res3)

# ---------- 4. A4 v4 逐列（精确值，验证 0.083 巧合） ----------
a4 = [r for r in load("e6d_ablation_v2.csv") if r["group"] == "A4"]
res4 = agg(a4, lambda r: (r["config"], r["variant"]), ["node_best_f1", "best_F1_alert", "node_cov"])
show("DeepSeek#4 A4 v4 逐列（精确值）", res4)

# ---------- 5. A5 编码消融（旧 a5.csv） ----------
a5 = load("e6d_ablation_a5.csv")
res5 = agg(a5, lambda r: (r["encoding"],), ["node_best_f1", "best_F1_alert", "node_cov"] + bud)
show("A5 编码消融（含固定预算）", res5)
print("\n  [83% 核验]")
if ("dual_naive",) in res5 and ("rate_single",) in res5:
    d = res5[("dual_naive",)]["node_best_f1"][0]
    s = res5[("rate_single",)]["node_best_f1"][0]
    print(f"    dual={d:.4f} single={s:.4f} | 单通道保留={s/d*100:.1f}% 丢失={100-s/d*100:.1f}% | 双通道相对提升={(d-s)/s*100:.1f}%")

# ---------- 6. A5b μ 抢救 ----------
a5b = load("e6d_ablation_a5b.csv")
res5b = agg(a5b, lambda r: (r["encoding"],), ["node_best_f1", "best_F1_alert", "node_cov"] + bud)
show("A5b μ 抢救（含固定预算）", res5b)

print("\n### DeepSeek#7 每文件胜负（A5b: rate_ratio vs dual_naive）")
byf = defaultdict(dict)
for r in a5b:
    byf[r["file"]][r["encoding"]] = (fl(r["node_best_f1"]), fl(r["best_F1_alert"]))
win = 0
for f in sorted(byf):
    d = byf[f]
    if "rate_ratio" in d and "dual_naive" in d:
        dn = d["rate_ratio"][0] - d["dual_naive"][0]
        if dn > 0: win += 1
        print(f"  {f:16s} ratio={d['rate_ratio'][0]:.4f} naive={d['dual_naive'][0]:.4f} Δnode={dn:+.4f}")
print(f"  node 级 rate_ratio 胜出 {win}/7")

# ---------- 7. 每文件明细：主表三配置 ----------
print("\n### 主表三配置逐文件（node_best_f1 / best_F1_alert）")
byf2 = defaultdict(dict)
for r in a1:
    if r["variant"] == "base":
        byf2[r["file"]][r["config"]] = (fl(r["node_best_f1"]), fl(r["best_F1_alert"]),
                                        fl(r["F1_alert@5"]))
for f in sorted(byf2):
    d = byf2[f]
    line = f"  {f:16s}"
    for cfg in ["identity+rate", "TeRed+naive", "TeRed+RATE"]:
        if cfg in d:
            line += f" {cfg}:n={d[cfg][0]:.3f},a={d[cfg][1]:.3f},@5={d[cfg][2]:.3f} |"
    print(line)
