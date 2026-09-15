# -*- coding: utf-8 -*-
"""scripts/smoke_mine_v3.py — v3 挖掘器冒烟：CPR 先挖 + 语义锚 + hub 切片。

用法: python scripts/smoke_mine_v3.py [gids] [per_graph] [min_support]
  gids 默认 "0-3"（StreamSpot 良性训练集），全部走解析缓存（不重新解析）。
输出:
  1) 挖掘耗时/候选数/WL 桶数/模板清单（节点数/边数/type/验证支持度）
  2) 对 gid0 跑一次 tered 归约：节点/边归约率 + INV 全过 + 攻击存活
"""
from __future__ import annotations
import os, sys, glob, time, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rate_core import load_graphs, fingerprint, cache_path
from reduction import template_mining as tmining
from reduction.cpr import CPROperator
from reduction.tered import TeRedOperator
from reduction.base import check_invariants, node_map_sanity
from run_pipeline import parse_gids, attack_survival
import numpy as np


def main():
    gids = parse_gids(sys.argv[1] if len(sys.argv) > 1 else "0-3")
    per_graph = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    min_support = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    cdir = os.path.join("cache", "streamspot", "parsed")
    fs = sorted(glob.glob(os.path.join(cdir, "*.jsonl")))
    graphs = load_graphs(fs[0])
    by = {int(g.gid): g for g in graphs}
    sel = [by[g] for g in gids if g in by]
    print(f"[smoke] 载入 {len(sel)} 图: {sorted(int(g.gid) for g in sel)}")

    t0 = time.time()
    templates, info = tmining.mine_templates(
        sel, min_support=min_support, per_graph=per_graph,
        max_templates=200, min_tpl_nodes=3, seed=0, verbose=True)
    print(f"[smoke] 挖掘耗时 {time.time()-t0:.1f}s | info={info}")
    print(f"[smoke] 模板 {len(templates)} 个：")
    for t in templates:
        tc = collections.Counter(nd.get("type", "?") for nd in t.nodes.values())
        print(f"  {t.gid}: nodes={t.n_nodes()} edges={t.n_edges()} "
              f"anchor={t.meta.get('anchor')} support={t.meta.get('support')} "
              f"types={dict(tc)}")

    # ---- 归约冒烟：gid0 ----
    if not templates:
        print("[smoke] 无模板，跳过归约测试"); return
    G = by[gids[0]]
    op = TeRedOperator(templates, max_instances=200)
    res = op.reduce(G)
    node_map_sanity(res, G)
    inv = check_invariants(G, res)
    ok = all(v is True for k, v in inv.items() if k != "INV2_violations")
    surv = attack_survival(G, res)
    n0, n1 = G.n_nodes(), res.Gp.n_nodes()
    e0, e1 = len(G.edges), res.Gp.n_edges()
    print(f"\n[归约] gid{gids[0]}: nodes {n0}->{n1} "
          f"(归约率 {(1-n1/max(1,n0)):.3f}) | edges {e0}->{e1} "
          f"(归约率 {(1-e1/max(1,e0)):.3f})")
    print(f"[归约] n_regions={res.stats.get('n_regions')} "
          f"templates_used={res.stats.get('templates_used')}")
    print(f"[归约] INV 全过={ok} INV2_violations={len(inv['INV2_violations'])} "
          f"attack_survival={surv:.3f}")


if __name__ == "__main__":
    main()
