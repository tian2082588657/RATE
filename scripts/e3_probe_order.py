# -*- coding: utf-8 -*-
"""scripts/e3_probe_order.py — 模板执行顺序对归约率的影响（服务器诊断）。"""
from __future__ import annotations
import sys, os, glob, collections, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rate_core import load_graphs
from reduction import template_mining as tmining
from reduction.tered import TeRedOperator
from reduction.base import check_invariants, node_map_sanity


def reduce_all(w, templates, order, mi=500):
    ts = sorted(templates, key=order)
    op = TeRedOperator(ts, max_instances=mi)
    res = op.reduce(w)
    return res


def main():
    wins = load_graphs(glob.glob("cache/darpa/e3cadets_w8_*.jsonl")[0])
    train = wins[:4]
    templates, info = tmining.mine_templates(
        train, min_support=4, per_graph=400, max_templates=200,
        min_tpl_nodes=3, seed=0, verbose=False)
    print(f"{len(templates)} 模板")

    w0 = wins[0]
    n0 = w0.n_nodes(); e0 = len(w0.edges)
    orders = {
        "原始(小优先)": lambda t: (t.meta.get("support", 0), t.n_nodes()),
        "大优先": lambda t: (-t.n_nodes(), -len(t.edges)),
        "收益优先": lambda t: (-(t.n_nodes() - 2), -len(t.edges)),
        "支持度优先": lambda t: (-t.meta.get("support", 0), -t.n_nodes()),
    }
    for name, key in orders.items():
        res = reduce_all(w0, templates, key)
        n1, e1 = res.Gp.n_nodes(), res.Gp.n_edges()
        print(f"  {name:<12}: nodes {n0}->{n1} ({(1-n1/n0):.1%}) | "
              f"edges {e0}->{e1} ({(1-e1/e0):.1%}) | regions={res.stats.get('n_regions')}")


if __name__ == "__main__":
    main()
