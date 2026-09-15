# -*- coding: utf-8 -*-
"""scripts/e3_probe_fanout.py — process 锚扇出块模板 vs egonet 整用（诊断实验）。

验证假设：E3 每窗 ~190 process，若模板 = "1 process + k 个对象"（扇出块），
每 region 只含 1 个 process -> 折叠次数上限 ~190 次，节点归约潜力 ~37%。
对比：egonet 整用挖出的"多 process 共享文件"家族，每 region 含 2 process，
折叠上限被砍半。

用法: python scripts/e3_probe_fanout.py [kmin] [kmax] [min_support]
"""
from __future__ import annotations
import sys, os, glob, collections, time, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rate_core import load_graphs, CanonicalGraph
from reduction import template_mining as tmining
from reduction.cpr import CPROperator
from reduction.tered import (TeRedOperator, find_instances, _to_nx_matching,
                             _MatchIndex, _pick_root, _seeded_match)
from reduction.base import check_invariants, node_map_sanity


def fanout_candidates(wins, kmin=3, kmax=8, per_anchor=8, seed=0,
                      max_tpl_nodes=24):
    """process 锚 -> 直接邻居随机切片成 (process + k 对象) 块。"""
    rng = random.Random(seed)
    cands = []
    for g in wins:
        procs = [n for n, d in g.nodes.items() if d.get("type") == "process"]
        for p in procs:
            nbrs = set()
            for e in g.edges:
                if e["src"] == p:
                    nbrs.add(e["dst"])
                elif e["dst"] == p:
                    nbrs.add(e["src"])
            nbrs = [n for n in nbrs if n in g.nodes]
            if len(nbrs) < kmin:
                continue
            rng.shuffle(nbrs)
            for _ in range(per_anchor):
                k = rng.randint(kmin, min(kmax, len(nbrs)))
                sel = nbrs[:k]
                # 保证至少含一个入边(process 写对象) 或出边
                seen = {p} | set(sel)
                sub = CanonicalGraph(f"{g.gid}:fo:{p}:{len(cands)}")
                for nid in seen:
                    sub.nodes[nid] = {"type": g.nodes[nid].get("type", "unknown"),
                                      "attrs": {}}
                for e in g.edges:
                    if e["src"] in seen and e["dst"] in seen:
                        sub.edges.append({"src": e["src"], "dst": e["dst"],
                                          "etype": e["etype"], "ts": 0, "mu": 1.0})
                if sub.n_edges() == 0:
                    continue
                sub.meta = {"anchor": p, "k": 1, "origin": g.gid,
                            "fanout": True}
                cands.append(sub)
    return cands


def mine_fanout(wins, kmin, kmax, min_support, per_anchor=8):
    cands = fanout_candidates(wins, kmin=kmin, kmax=kmax,
                              per_anchor=per_anchor)
    buckets = collections.defaultdict(list)
    for c in cands:
        buckets[tmining.wl_signature(_to_nx_matching(c))].append(c)
    freq = []
    for sig, subs in buckets.items():
        subs.sort(key=lambda s: (-s.n_nodes(), -s.n_edges()))
        occ, chosen = set(), []
        for s in subs:
            keys = {(s.meta.get("origin"), n) for n in s.nodes}
            if keys & occ:
                continue
            occ |= keys
            chosen.append(s)
        if len(chosen) >= min_support:
            freq.append((len(chosen), chosen[0]))
    freq.sort(key=lambda x: (-x[0], -x[1].n_nodes()))
    templates = []
    for sup, rep in freq:
        tpl = tmining._relabel_rep(rep, len(templates))
        if tpl.n_nodes() < 3:
            continue
        tpl.meta["support"] = sup
        templates.append(tpl)
    return templates, cands, buckets


def main():
    kmin = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    kmax = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    min_support = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    wins = load_graphs(glob.glob("cache/darpa/e3cadets_w8_*.jsonl")[0])
    train = wins[:4]
    t0 = time.time()
    templates, cands, buckets = mine_fanout(
        train, kmin=kmin, kmax=kmax, min_support=min_support)
    print(f"[fanout] 候选 {len(cands)} -> WL桶 {len(buckets)} -> "
          f"模板 {len(templates)} (kmin={kmin} kmax={kmax} sup>={min_support}) "
          f"耗时 {time.time()-t0:.1f}s")

    w0 = wins[0]
    n0 = w0.n_nodes()
    # 逐模板独立潜力 + 一起跑
    tot = 0
    shapes = collections.Counter()
    for t in templates[:20]:
        tc = collections.Counter(nd.get("type", "?") for nd in t.nodes.values())
        shapes[tuple(sorted(tc.items()))] += 1
    print("模板形状分布(top):")
    for sh, c in shapes.most_common(8):
        print(f"  {dict(sh)} x{c}")

    # 一起跑（大优先）
    ts = sorted(templates, key=lambda t: (-t.n_nodes(), -len(t.edges)))
    op = TeRedOperator(ts, max_instances=500)
    res = op.reduce(w0)
    n1 = res.Gp.n_nodes()
    print(f"[fanout] win0 一起跑(大优先): nodes {n0}->{n1} ({(1-n1/n0):.1%}) "
          f"regions={res.stats.get('n_regions')} INV="
          f"{all(v is True for k, v in check_invariants(w0, res).items() if k != 'INV2_violations')}")


if __name__ == "__main__":
    main()
