# -*- coding: utf-8 -*-
"""scripts/diag_v2.py — 诊断 v2：CPR 压缩 + 定向候选挖掘。

v1 结论：原始图 8~32 万边/图（平均度 20-70），1-hop 均匀候选 70% 是 2 节点单边，
模板挖掘与 VF2 匹配都被"图太密"卡死。

v2 问题：
  A. CPR(同 src,dst,etype 去重) 后每图剩多少边？结构图能压到多大？
  B. CPR 后 1-hop / 2-hop 定向候选（锚点=度适中节点）大小分布如何？
  C. 候选 WL 分桶的高频类大小/构成（模板能不能到 4-10 节点的"结构块"）？
"""
from __future__ import annotations
import os, sys, glob, collections, time
import networkx as nx
from networkx.algorithms import isomorphism as nxiso

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from rate_core import load_graphs, CanonicalGraph
from reduction.cpr import CPROperator
from reduction.tered import _to_nx_matching


def wl_signature(dg, iters=3):
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


def main():
    cdir = os.path.join(ROOT, "cache", "streamspot", "parsed")
    fs = sorted(glob.glob(os.path.join(cdir, "*.jsonl")))
    if not fs:
        print("NO_CACHE"); return
    t0 = time.time()
    graphs = load_graphs(fs[0])
    by = {int(g.gid): g for g in graphs}
    print(f"[v2] 载入 {len(graphs)} 图缓存 {time.time()-t0:.1f}s")
    gids = sorted(by)

    # ---- A. CPR 压缩 ----
    print("\n=== A. CPR 压缩后结构规模 ===")
    cpr = CPROperator()
    cpr_map = {}
    for gi in gids:
        res = cpr.reduce(by[gi])
        cpr_map[gi] = res.Gp
        e0, e1 = res.Gp.n_edges(), len(by[gi].edges)
        print(f"  gid={gi:>5}  edges {len(by[gi].edges):>8} -> {e0:>8} "
              f"({e0/max(1,e1):.1%})  nodes={by[gi].n_nodes()}")

    # 每图唯一边涉及的节点与平均度（CPR 后）
    print("\n  CPR 后度分布抽样 (gid0):")
    G0 = cpr_map[0]
    deg = collections.Counter()
    for e in G0.edges:
        deg[e["src"]] += 1
        deg[e["dst"]] += 1
    dhist = collections.Counter(min(d, 6) if d < 10 else (d // 10 * 10 if d < 100 else 100)
                                for d in deg.values())
    print(f"  gid0 CPR 后唯一边节点={len(deg)}  度直方图: {dict(sorted(dhist.items()))}")

    # ---- B. 候选（CPR 后 gid0, 锚点分层） ----
    print("\n=== B. CPR 后候选邻域（gid0） ===")
    for khop in (1, 2):
        for deg_min, deg_max, tag in ((1, 10**9, "all"), (2, 10, "mid"),
                                      (2, 10**9, "ge2")):
            anchors = [n for n in G0.nodes
                       if deg_min <= sum(1 for e in G0.edges if e["src"] == n or e["dst"] == n) <= deg_max]
            if len(anchors) > 600:
                import random
                random.Random(0).shuffle(anchors)
                anchors = anchors[:600]
            sizes = collections.Counter()
            for a in anchors:
                sub = G0.khop_subgraph(a, khop)
                if sub is None:
                    continue
                sizes[min(sub.n_nodes(), 30)] += 1
            dist = dict(sorted(sizes.items()))
            top = sum(v for k, v in dist.items() if k >= 4)
            print(f"  khop={khop} deg[{deg_min},{deg_max}] anchors={len(anchors)} "
                  f"n_nodes直方图(截30): {dist}  >=4节点候选占比 {top/max(1,len(anchors)):.0%}")

    # ---- C. 候选 WL 分桶（khop1 全锚 + khop2 mid） ----
    print("\n=== C. WL 分桶（模板上限估计） ===")
    def collect(G, khop, deg_max, cap, seed=0):
        import random
        rng = random.Random(seed)
        anchors = [n for n in G.nodes
                   if 1 <= sum(1 for e in G.edges if e["src"] == n or e["dst"] == n) <= deg_max]
        rng.shuffle(anchors)
        out = []
        for a in anchors[:cap]:
            s = G.khop_subgraph(a, khop)
            if s is not None and s.n_nodes() >= 3 and s.n_edges() > 0:
                out.append(s)
        return out

    cands = collect(G0, 1, 10 ** 9, 1200) + collect(G0, 2, 6, 600)
    print(f"  候选 {len(cands)}（khop1 all + khop2 deg<=6）")
    buckets = collections.defaultdict(list)
    for c in cands:
        buckets[wl_signature(_to_nx_matching(c))].append(c)
    sizes = sorted((len(v) for v in buckets.values()), reverse=True)
    hist = collections.Counter(1 if s == 1 else 2 if s == 2 else 3 if s == 3
                               else 4 if s == 4 else "5-9" if s < 10 else "10+" for s in sizes)
    print(f"  桶数={len(buckets)}  桶大小: {dict(hist)}")
    for th in (3, 5, 8):
        print(f"  support>={th}: {sum(1 for v in buckets.values() if len(v) >= th)} 类")

    big = sorted(((len(v), v[0]) for v in buckets.values() if len(v) >= 3),
                 key=lambda x: -x[0])[:10]
    print("\n  top 高频桶代表（3+ support，仅 gid0 一图内）:")
    for s, rep in big:
        n_nodes = rep.n_nodes()
        types = collections.Counter(nd.get("type", "?") for nd in rep.nodes.values())
        n_edges = len(rep.edges)
        print(f"    support={s:>4}  nodes={n_nodes:>3}  edges={n_edges:>4}  "
              f"types={dict(types.most_common(4))}")
    print(f"[v2] 耗时 {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
