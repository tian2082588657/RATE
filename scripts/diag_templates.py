# -*- coding: utf-8 -*-
"""scripts/diag_templates.py — 诊断 StreamSpot 模板挖掘为何只有 7 模板 / 归约率 0.1%。

针对三个问题输出证据：
  Q1 良性图的 1-hop egonet 候选里到底有没有高频同构结构？
  Q2 当前 min_support=5 的绝对门槛是否把有用模式滤掉了？
  Q3 高频模板在图上做 VF2 实例搜索到底能命中多少（归约率上限）？

用法（在 TeRed RATE 根目录）：
  python scripts/diag_templates.py [--archive ../dataset/streamspot/all.tar.gz]
"""
from __future__ import annotations
import os, sys, collections, time

import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), *[".."] * 1))

import networkx as nx
from networkx.algorithms import isomorphism as nxiso

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from rate_core import CanonicalGraph
from reduction import template_mining as tmining
from reduction.tered import _to_nx_matching


def wl_signature(dg, iters=3):
    """整图 WL 色彩多重集签名（对带 node type 的有向图，同构不变、O(V+E)）。"""
    col = {n: str(dg.nodes[n].get("type", "?")) for n in dg.nodes}
    for _ in range(iters):
        new = {}
        for n in dg.nodes:
            nbs = sorted(col[v] for v in dg.successors(n)) + \
                  sorted(col[v] for v in dg.predecessors(n))
            new[n] = col[n] + "/" + ",".join(nbs)
        col = new
    cnt = collections.Counter(col.values())
    return tuple(sorted(cnt.items()))


def _vf2_isomorphic(g1, g2):
    n1, n2 = _to_nx_matching(g1), _to_nx_matching(g2)
    if n1.number_of_nodes() != n2.number_of_nodes() or \
            n1.number_of_edges() != n2.number_of_edges():
        return False
    return nxiso.DiGraphMatcher(n1, n2,
                                node_match=lambda a, b: a.get("type") == b.get("type")
                                ).is_isomorphic()


def main(archive):
    sys.path.insert(0, os.path.dirname(HERE))
    from adapters.streamspot import parse
    from run_pipeline import parse_gids

    gids = parse_gids("0-6,500-501")
    t0 = time.time()
    graphs, _meta = parse(archive, gids=gids, verbose=True)
    print(f"[diag] 解析 {len(graphs)} 图 耗时 {time.time()-t0:.1f}s\n")
    by_gid = {int(g.gid): g for g in graphs}

    print("=== 表1 每图规模 ===")
    for g in graphs:
        types = collections.Counter(nd.get("type", "?") for nd in g.nodes.values())
        print(f"  gid={g.gid:>5}  nodes={g.n_nodes():>6}  edges={len(g.edges):>7}  "
              f"top_types={types.most_common(4)}")

    benign = [by_gid[g] for g in gids if g < 100]
    print(f"\n良性图 {len(benign)} 张: {sorted(int(g.gid) for g in benign)}")

    # ---- Q1: 候选 egonet 的结构重复性 ----
    print("\n=== Q1 候选抽取（khop=1, per_graph=300）===")
    cands = tmining.sample_candidates(benign, khop=1, per_graph=300,
                                      max_tpl_nodes=24, seed=0, verbose=False)
    print(f"  候选总数 = {len(cands)}")
    ns = collections.Counter(c.n_nodes() for c in cands)
    es = collections.Counter(c.n_edges() for c in cands)
    print(f"  候选 n_nodes 分布: {dict(sorted(ns.items()))}")
    print(f"  候选 n_edges 分布(top10): {dict(sorted(es.items())[:10])}")

    # ---- WL 分桶（替代 O(n^2) VF2 聚类，作为高频结构的上界估计）----
    print("\n=== Q2 WL 分桶统计 ===")
    buckets = collections.defaultdict(list)
    for c in cands:
        buckets[wl_signature(_to_nx_matching(c))].append(c)
    sizes = sorted((len(v) for v in buckets.values()), reverse=True)
    hist = collections.Counter(
        1 if s == 1 else 2 if s == 2 else 3 if s == 3 else 4 if s == 4
        else "5-9" if s < 10 else "10+" for s in sizes)
    print(f"  桶数={len(buckets)}  桶大小直方图: {dict(hist)}")
    for th in (2, 3, 5, 8):
        n_b = sum(1 for v in buckets.values() if len(v) >= th)
        print(f"  support>={th}: {n_b} 个模式类")

    # VF2 抽查：对 support>=5 的桶，确认桶内随机两两真同构（WL 假阳性率）
    big = [(k, v) for k, v in buckets.items() if len(v) >= 5]
    if big:
        print(f"\n  VF2 抽查 support>=5 桶的同构真实性（每桶抽 2 个比对）:")
        ok = bad = 0
        for k, v in big[:8]:
            a, b = v[0], v[1]
            iso = _vf2_isomorphic(a, b)
            ok += iso
            bad += (not iso)
            if not iso:
                print(f"    !! 桶内 VF2 判定非同构: {a.n_nodes()}n/{a.n_edges()}e")
        print(f"  抽查 {ok+bad} 对: 同构 {ok} / 假阳性 {bad}")

    # ---- 高频桶代表 -> 真模板，测在真实图上的 VF2 命中 ----
    print("\n=== Q3 高频模板在 9 图上的 VF2 命中上限 ===")
    tpls = []
    for k, v in sorted(big, key=lambda kv: -len(kv[1]))[:30]:
        rep = v[0]
        order = [rep.meta.get("anchor")] + \
                [n for n in sorted(rep.nodes) if n != rep.meta.get("anchor")]
        rid = {nid: str(i) for i, nid in enumerate(order)}
        tp = CanonicalGraph(f"tpl_{len(tpls)}")
        for nid, nd in rep.nodes.items():
            tp.nodes[rid[nid]] = {"type": nd.get("type", "unknown")}
        for e in rep.edges:
            tp.edges.append({"src": rid[e["src"]], "dst": rid[e["dst"]],
                             "etype": e["etype"], "ts": 0, "mu": 1.0})
        tp.meta = {"support": len(v), "anchor": rid[rep.meta.get("anchor")]}
        tpls.append(tp)
    print(f"  用 support>=5 的桶构造 {len(tpls)} 个代表模板 "
          f"(n_nodes: {[t.n_nodes() for t in tpls][:20]})")

    from reduction.tered import find_instances
    t_g = time.time()
    total_inst = 0
    for gi, g in enumerate(graphs):
        regs = find_instances(g, tpls, max_instances=50, max_total=200,
                              verbose=False)
        n_nodes_covered = sum(len(r["nodes"]) for r in regs)
        total_inst += len(regs)
        print(f"  gid={g.gid:>5}: 命中 {len(regs):>4} 实例, "
              f"覆盖 {n_nodes_covered:>6}/{g.n_nodes():>6} 节点 "
              f"({n_nodes_covered/max(1,g.n_nodes()):.1%})")
    print(f"[diag] VF2 命中测试耗时 {time.time()-t_g:.1f}s, 总实例 {total_inst}")


def _resolve_archive(p):
    if p and os.path.exists(p):
        return p
    cands = [
        "../dataset/streamspot/all.tar.gz",
        _os.path.join(_os.path.dirname(_R), "dataset", "streamspot", "all.tar.gz"),
        "/TeRed+RATE/dataset/streamspot/all.tar.gz",
    ]
    for c in cands:
        if os.path.exists(c):
            return c
    return p


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default=None)
    a = ap.parse_args()
    main(_resolve_archive(a.archive))
