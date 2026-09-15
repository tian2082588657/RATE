# -*- coding: utf-8 -*-
"""scripts/e3_probe_potential.py — 每模板独立命中潜力探测（服务器临时诊断用）。"""
from __future__ import annotations
import sys, os, glob, collections, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rate_core import load_graphs
from reduction import template_mining as tmining
from reduction.tered import find_instances, _pick_root, _to_nx_matching


def main():
    wins = load_graphs(glob.glob("cache/darpa/e3cadets_w8_*.jsonl")[0])
    train = wins[:4]
    t0 = time.time()
    templates, info = tmining.mine_templates(
        train, min_support=4, per_graph=400, max_templates=200,
        min_tpl_nodes=3, seed=0, verbose=False)
    print(f"mine {time.time()-t0:.1f}s -> {len(templates)} tpl")

    w0 = wins[0]
    Gnx = _to_nx_matching(w0)
    print("--- 每模板在 win0 独立命中潜力 ---")
    for t in templates:
        Tnx = _to_nx_matching(t)
        root = _pick_root(Tnx, t)
        rtype = Tnx.nodes[root].get("type", "?")
        r_in, r_out = Tnx.in_degree(root), Tnx.out_degree(root)
        cand = [n for n in Gnx.nodes
                if Gnx.nodes[n].get("type", "?") == rtype
                and Gnx.in_degree(n) >= r_in and Gnx.out_degree(n) >= r_out]
        regs = find_instances(w0, [t], max_instances=2000, max_total=8000)
        cover = sum(len(r["nodes"]) for r in regs)
        tc = collections.Counter(
            nd.get("type", "?") for nd in t.nodes.values())
        print(f"  {t.gid}: n={t.n_nodes()} e={t.n_edges()} "
              f"root={root}({rtype},in{r_in},out{r_out}) "
              f"种子={len(cand)} 实例={len(regs)} 覆盖={cover} "
              f"types={dict(tc.most_common(3))}")


if __name__ == "__main__":
    main()
