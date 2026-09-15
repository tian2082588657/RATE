# -*- coding: utf-8 -*-
"""对比旧主表 vs 新主表（告警级指标）。

旧: e6_alert_full.csv     口径 wmax=0.7 + agg 排序（0.7max+0.3mean）
新: e6_alert_full_v2.csv  口径 wmax=1.0 + cmax 排序（纯 max）
   且 v4 特征 4 列 -> 3 列（删除 nbr_type_div）

用法:
  python compare_main_v2.py <old.csv> <new.csv> > compare_out.txt
"""
import csv
import os
import sys
from collections import defaultdict

BUDGETS = [1, 2, 3, 5, 10, 20]


def load(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def g(r, k, d=0.0):
    v = r.get(k, "")
    if v in ("", None):
        return d
    try:
        return float(v)
    except ValueError:
        return d


def key(r):
    return (r["config"], r["file"])


def main():
    old_p, new_p = sys.argv[1], sys.argv[2]
    old = {key(r): r for r in load(old_p)}
    new = {key(r): r for r in load(new_p)}
    keys = [k for k in old if k in new]
    missing = [k for k in old if k not in new] + [k for k in new if k not in old]

    out = []
    w = out.append
    w("=" * 100)
    w("主表升级对比：旧口径(w0.7+agg排序) -> 新口径(w1.0+cmax排序) + v4 删 nbr_type_div")
    w("=" * 100)
    if missing:
        w(f"[警告] 不一致行: {missing}")

    # ---- 1. 逐行对比（只列攻击文件；良性对照另列） ----
    w("")
    w("--- 1. 攻击文件逐行（best_F1_alert @ best_b） " + "-" * 40)
    w(f"{'config':<14}{'file':<16}{'旧F1':>8}{'旧b':>5}{'新F1':>8}{'新b':>5}"
      f"{'ΔF1':>9}{'旧cov':>8}{'新cov':>8}{'旧alert_P':>10}{'新alert_P':>10}")
    atk_keys = [k for k in keys if old[k].get("is_attack") == "True"]
    agg_old = defaultdict(list)
    agg_new = defaultdict(list)
    for k in sorted(atk_keys, key=lambda x: (x[0], x[1])):
        o, n = old[k], new[k]
        of, nf = g(o, "best_F1_alert"), g(n, "best_F1_alert")
        w(f"{k[0]:<14}{k[1]:<16}{of:>8.4f}{int(g(o,'best_b')):>5}"
          f"{nf:>8.4f}{int(g(n,'best_b')):>5}{nf-of:>+9.4f}"
          f"{g(o,'node_cov'):>8.3f}{g(n,'node_cov'):>8.3f}"
          f"{g(o,'alert_P'):>10.3f}{g(n,'alert_P'):>10.3f}")
        agg_old[k[0]].append(g(o, "best_F1_alert"))
        agg_new[k[0]].append(g(n, "best_F1_alert"))

    # ---- 2. 按配置汇总 ----
    w("")
    w("--- 2. 按配置汇总（7 文件平均 best_F1_alert） " + "-" * 41)
    w(f"{'config':<14}{'旧均值':>10}{'新均值':>10}{'Δ':>10}{'相对':>9}")
    for cfg in ["identity+rate", "TeRed+naive", "TeRed+RATE"]:
        if cfg not in agg_old:
            continue
        mo = sum(agg_old[cfg]) / len(agg_old[cfg])
        mn = sum(agg_new[cfg]) / len(agg_new[cfg])
        rel = (mn / mo - 1) * 100 if mo else 0.0
        w(f"{cfg:<14}{mo:>10.4f}{mn:>10.4f}{mn-mo:>+10.4f}{rel:>+8.1f}%")

    # ---- 3. 预算曲线（新 vs 旧，按配置平均 F1@b） ----
    w("")
    w("--- 3. 预算曲线 F1_alert@b（按配置·文件平均） " + "-" * 45)
    for cfg in ["identity+rate", "TeRed+naive", "TeRed+RATE"]:
        ks = [k for k in atk_keys if k[0] == cfg]
        if not ks:
            continue
        w(f"\n  [{cfg}]")
        w("    " + "".join(f"{('b'+str(b)):>9}" for b in BUDGETS))
        for tag, tbl in (("旧", old), ("新", new)):
            vals = []
            for b in BUDGETS:
                col = f"F1_alert@{b}"
                xs = [g(tbl[k], col, float("nan")) for k in ks]
                xs = [x for x in xs if x == x]  # drop nan
                vals.append(sum(xs) / len(xs) if xs else float("nan"))
            w(f"  {tag:<2}" + "".join(f"{v:>9.3f}" for v in vals))

    # ---- 4. 良性对照（误报规模） ----
    ben_keys = [k for k in keys if old[k].get("is_attack") != "True"]
    if ben_keys:
        w("")
        w("--- 4. 良性对照文件（n_gt=0）：告警数/覆盖节点 " + "-" * 40)
        w(f"{'config':<14}{'file':<16}{'旧alerts':>10}{'新alerts':>10}"
          f"{'旧alerted_nodes':>18}{'新alerted_nodes':>18}")
        for k in sorted(ben_keys, key=lambda x: (x[0], x[1])):
            o, n = old[k], new[k]
            w(f"{k[0]:<14}{k[1]:<16}{int(g(o,'n_alerts')):>10}"
              f"{int(g(n,'n_alerts')):>10}{int(g(o,'n_alerted_nodes')):>18}"
              f"{int(g(n,'n_alerted_nodes')):>18}")

    # ---- 5. 头部指标（RATE 配置） ----
    w("")
    w("--- 5. RATE 配置细节（cov / alert_P / first_hit） " + "-" * 42)
    w(f"{'file':<16}{'旧cov':>8}{'新cov':>8}{'旧aP':>8}{'新aP':>8}"
      f"{'旧hit':>7}{'新hit':>7}{'旧nodes':>9}{'新nodes':>9}")
    for k in sorted([x for x in atk_keys if x[0] == "TeRed+RATE"], key=lambda x: x[1]):
        o, n = old[k], new[k]
        w(f"{k[1]:<16}{g(o,'node_cov'):>8.3f}{g(n,'node_cov'):>8.3f}"
          f"{g(o,'alert_P'):>8.3f}{g(n,'alert_P'):>8.3f}"
          f"{int(g(o,'first_hit')):>7}{int(g(n,'first_hit')):>7}"
          f"{int(g(o,'n_alerted_nodes')):>9}{int(g(n,'n_alerted_nodes')):>9}")

    print("\n".join(out))


if __name__ == "__main__":
    main()
