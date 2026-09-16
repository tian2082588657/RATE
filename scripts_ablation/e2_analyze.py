# -*- coding: utf-8 -*-
"""scripts_ablation/e2_analyze.py — 槽位级等价性的配对检验。

输入：`e2_slot_equiv.py --slots` 产出的明细 CSV（每行一个 (配置, 分区, 槽位)）。
输出：逐配置的命中率、两配置的配对差、McNemar 精确检验、按分区聚类的 bootstrap
      置信区间，以及 TOST 等价性判定。

用法:
  python scripts_ablation/e2_analyze.py --slots <slots.csv> \
      --a identity+rate --b TeRed+rate_ratio --delta 0.15
"""
from __future__ import annotations
import os, sys, csv, argparse, random
from collections import defaultdict
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)


def binom_two_sided(k, n, p=0.5):
    """精确二项检验（双侧）。n 通常很小（不一致对数）。"""
    if n == 0:
        return 1.0
    pmf = [comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(n + 1)]
    obs = pmf[k]
    return min(1.0, sum(x for x in pmf if x <= obs + 1e-15))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slots", required=True)
    ap.add_argument("--a", default="identity+rate")
    ap.add_argument("--b", default="TeRed+rate_ratio")
    ap.add_argument("--delta", type=float, default=0.15,
                    help="TOST 等价边界（绝对差）")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    hit = {a.a: {}, a.b: {}}      # cfg -> (file, slot) -> 0/1
    files = set()
    with open(a.slots, "r", encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            cfg = r.get("config")
            if cfg not in hit:
                continue
            key = (r.get("file"), r.get("slot"))
            hit[cfg][key] = int(r.get("hit", 0))
            files.add(r.get("file"))

    keys = sorted(set(hit[a.a]) & set(hit[a.b]))
    n = len(keys)
    if n == 0:
        print("没有可配对的槽位：检查 --a/--b 的配置名"); return

    both = onlyA = onlyB = neither = 0
    d = []
    by_file = defaultdict(list)
    for k in keys:
        ha, hb = hit[a.a][k], hit[a.b][k]
        if ha and hb:
            both += 1
        elif ha and not hb:
            onlyA += 1
        elif hb and not ha:
            onlyB += 1
        else:
            neither += 1
        d.append(ha - hb)
        by_file[k[0]].append(ha - hb)

    pa = (both + onlyA) / n
    pb = (both + onlyB) / n
    diff = pa - pb
    nd = onlyA + onlyB
    p_mcnemar = binom_two_sided(onlyA, nd)

    # 按分区聚类的 bootstrap（分区内相关，跨分区独立）
    rng = random.Random(12345)
    fl = sorted(by_file)
    boot = []
    for _ in range(a.boot):
        pick = [fl[rng.randrange(len(fl))] for _ in range(len(fl))]
        vals = [v for f in pick for v in by_file[f]]
        if vals:
            boot.append(sum(vals) / len(vals))
    boot.sort()

    def q(p):
        if not boot:
            return float("nan")
        i = min(len(boot) - 1, max(0, int(p * (len(boot) - 1))))
        return boot[i]

    ci_lo, ci_hi = q(0.05), q(0.95)

    lines = []
    lines.append("=" * 62)
    lines.append("E2 slot-level equivalence test")
    lines.append("=" * 62)
    lines.append("A = %s   B = %s" % (a.a, a.b))
    lines.append("paired slots n = %d  (files = %d)" % (n, len(fl)))
    lines.append("2x2: both=%d  onlyA=%d  onlyB=%d  neither=%d"
                 % (both, onlyA, onlyB, neither))
    lines.append("recall_kept  A = %.4f   B = %.4f   diff(A-B) = %+.4f"
                 % (pa, pb, diff))
    lines.append("McNemar exact p = %.4f  (discordant n = %d)"
                 % (p_mcnemar, nd))
    lines.append("cluster-bootstrap 90%% CI of diff = [%+.4f, %+.4f]"
                 % (ci_lo, ci_hi))
    tost = (ci_lo > -a.delta) and (ci_hi < a.delta)
    lines.append("TOST at delta = %.2f  ->  %s"
                 % (a.delta, "EQUIVALENT (90% CI inside the band)" if tost
                    else "not established at this delta"))
    # 反解：当前分辨率能排除的最小绝对差
    lines.append("minimum absolute difference excluded at 90%%: %.4f"
                 % max(abs(ci_lo), abs(ci_hi)))
    txt = "\n".join(lines)
    print(txt)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            fh.write(txt + "\n")
        print("\nwritten", a.out)


if __name__ == "__main__":
    main()
