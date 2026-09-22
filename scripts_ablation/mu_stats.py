# -*- coding: utf-8 -*-
"""scripts_ablation/mu_stats.py — 归约图上 μ 的分布统计（tab:mechan 用）。

μ(v) = |{u : node_map[u] == v}|，即归约后每个节点承载的原始实体数。
读归约缓存（不重新归约），输出：节点数、μ≡1 的节点数与占比、μ>1 节点数、
最大 μ、边数。

用法:
  python scripts_ablation/mu_stats.py --cache /TeRed+RATE/code2/cache_redo \
      --specs bin.116,bin.117,bin.118,bin.119@500000,bin.120,bin.6@400000,bin.7@400000
"""
from __future__ import annotations
import os, sys, argparse, glob
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from rate_core import load_pickle


def parse_spec(spec):
    if "@" in spec:
        nm, mr = spec.split("@", 1)
        return nm, int(mr)
    return spec, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--specs", required=True)
    ap.add_argument("--tpl", default="b36c705fde93a797")
    a = ap.parse_args()

    rcache = os.path.join(a.cache, "darpa", "e5_f1_reduce")
    tot_n = tot_mu1 = tot_e = 0
    for spec in [x.strip() for x in a.specs.split(",") if x.strip()]:
        nm, mr = parse_spec(spec)
        tag = "at%d" % mr if mr else "atfull"
        pat = os.path.join(rcache, "tered_%s_k-1_%s_*.pkl" % (a.tpl, tag))
        hits = glob.glob(pat)
        if not hits:
            print("%-16s MISS (%s)" % (spec, pat))
            continue
        # 一个 rec_tag 下可能有多个 gid（训练图也在此目录）；按 gid 里含 spec 名筛选
        sel = None
        for p in hits:
            base = os.path.basename(p)
            if nm.replace(".", "_") in base or nm in base:
                sel = p
                break
        if sel is None:
            print("%-16s AMBIGUOUS %d files" % (spec, len(hits)))
            continue
        res = load_pickle(sel)
        nm_map = res.node_map
        mu = Counter(nm_map.values())
        n_gp = res.Gp.n_nodes()
        mu1 = sum(1 for v in res.Gp.nodes if mu.get(v, 0) <= 1)
        # 未被任何原图节点映射到的 Gp 节点（理论上不存在）记 μ=0，一并算入 μ≡1
        zero = sum(1 for v in res.Gp.nodes if mu.get(v, 0) == 0)
        mx = max(mu.values()) if mu else 0
        e = res.Gp.n_edges()
        tot_n += n_gp
        tot_mu1 += mu1
        tot_e += e
        print("%-16s nodes=%d mu1=%d (%.2f%%) mu>1=%d maxmu=%d edges=%d  [%s]"
              % (spec, n_gp, mu1, mu1 * 100.0 / n_gp, n_gp - mu1, mx, e,
                 os.path.basename(sel)))
    if tot_n:
        print("TOTAL nodes=%d mu1=%d (%.2f%%) edges=%d"
              % (tot_n, tot_mu1, tot_mu1 * 100.0 / tot_n, tot_e))


if __name__ == "__main__":
    main()
