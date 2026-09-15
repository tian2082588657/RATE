# -*- coding: utf-8 -*-
"""scripts/diag_e6.py — 修复验证实验（在服务器跑）。

假设链（基于 diag_e4/e5 证据）：
  1. E3 窗口里大量进程是"同一父进程 fork 的兄弟/同程序多实例"，1-hop egonet 结构
     高度重复（WL 桶 sup 55/28/22...）——教科书级 TeRed 场景。
  2. 模板挖掘层 bug：WL 桶内按 (origin,node) 全节点去重，兄弟进程共享 系统库
     file + 父进程 节点 -> 互相排斥 -> 大桶被压到 1 -> 模板丢失。
     修复：桶内按 anchor(语义锚进程) 去重，不同锚=不同实例。
  3. 折叠层 bug：吸收时把共享对象(被>=K进程引用的 file/socket)也标 removed，
     第一实例吸收后后续实例无法再匹配 -> 区域数被压到 ~20。
     修复：吸收只标记"进程锚 + 私有对象(共享度<K)"；共享对象不 removed，
     允许后续实例复用（折叠时共享对象作为外部上下文保留）。

本实验: 用 monkey-patch 的挖掘+匹配跑 win0，报告修复前后区域数/吸收节点数/
真实归约率(经 TeRedOperator 变体)与 INV。
"""
from __future__ import annotations
import os, sys, glob, time, collections

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rate_core import load_graphs
from reduction.cpr import CPROperator
from reduction import template_mining as tm
from reduction.tered import (_to_nx_matching, _MatchIndex, _pick_root,
                             _seeded_match, TeRedOperator)
from reduction.base import check_invariants, node_map_sanity


def mine_templates_anchor_dedup(benign_graphs, min_support=4, per_graph=400,
                                max_templates=200, min_tpl_nodes=3, seed=0,
                                hub_fanout=(2, 6), hub_groups=24, verbose=True):
    """mine_templates 的复制，唯一改动：WL 桶内按 anchor 去重（非全节点集）。"""
    from reduction.tered import _to_nx_matching as t2n
    cpr = CPROperator()
    mine_graphs = [cpr.reduce(g).Gp for g in benign_graphs]
    sem_types = tm.auto_semantic_types(mine_graphs, rare_frac=0.05, min_count=4)
    sem_types |= {"process", "subject"}
    cands = tm.sample_candidates(mine_graphs, per_graph=per_graph,
                                 max_tpl_nodes=48, seed=seed,
                                 anchor_types=sem_types, min_tpl_nodes=min_tpl_nodes,
                                 khop_rare=2, hub_fanout=hub_fanout,
                                 hub_groups=hub_groups, verbose=verbose)
    buckets = collections.defaultdict(list)
    for c in cands:
        buckets[tm.wl_signature(t2n(c))].append(c)
    freq = []
    for sig, subs in buckets.items():
        subs.sort(key=lambda s: (-s.n_nodes(), -s.n_edges()))
        seen_anchor, chosen = set(), []
        for s in subs:
            a = s.meta.get("anchor")
            if a in seen_anchor:
                continue
            seen_anchor.add(a)
            chosen.append(s)
        if len(chosen) < min_support:
            continue
        freq.append((len(chosen), chosen[0]))
    freq.sort(key=lambda x: (-x[0], -x[1].n_nodes()))
    templates = []
    for sup, rep in freq:
        tpl = tm._relabel_rep(rep, len(templates))
        if tpl.n_nodes() < min_tpl_nodes or tpl.n_edges() < 2:
            continue
        found = 0
        for g in mine_graphs:
            regs = find_instances_shared(g, [tpl], share_k=999,  # 不限,只数区域
                                         max_instances=min_support,
                                         max_total=min_support)
            found += len(regs)
            if found >= min_support:
                break
        if found < min_support:
            continue
        tpl.meta["support"] = min(sup, found)
        templates.append(tpl)
        if len(templates) >= max_templates:
            break
    if verbose:
        print(f"[diag_e6] {len(buckets)} 桶 -> {len(freq)} 频 -> 模板 {len(templates)}")
    return templates, {"n_candidates": len(cands), "n_clusters": len(buckets)}


def shared_deg(G, k=2):
    """被 >= k 个进程引用的对象集合。"""
    procs = {n for n, nd in G.nodes.items() if nd.get("type") == "process"}
    ref = collections.Counter()
    for e in G.edges:
        if e["src"] in procs and G.nodes.get(e["dst"], {}).get("type") != "process":
            ref[e["dst"]] += 1
    return {o for o, c in ref.items() if c >= k}


def find_instances_shared(G, templates, share_k=2, max_instances=500,
                          max_total=100000, verbose=False):
    """find_instances 的共享感知版：removed 只含 进程锚 + 私有对象。
    共享对象(被>=share_k进程引用)不被吸收 -> 多实例可复用 -> 区域数大增。"""
    shared = shared_deg(G, share_k) if share_k < 999 else set()
    Gnx = _to_nx_matching(G)
    idx = _MatchIndex(Gnx)
    removed = set()
    regions = []
    budget = max_total
    for tpl in templates:
        if budget <= 0:
            break
        Tnx = _to_nx_matching(tpl)
        if Tnx.number_of_nodes() < 2:
            continue
        root_t = _pick_root(Tnx, tpl)
        if root_t is None:
            continue
        rtype = Tnx.nodes[root_t].get("type", "unknown")
        r_in, r_out = Tnx.in_degree(root_t), Tnx.out_degree(root_t)
        seeds = idx.by_type.get(rtype, [])
        seeds = [s for s in seeds
                 if s not in removed and Gnx.in_degree(s) >= r_in
                 and Gnx.out_degree(s) >= r_out]
        n_inst = 0
        for s in seeds:
            if budget <= 0 or n_inst >= max_instances:
                break
            m = _seeded_match(idx, Tnx, root_t, s)
            if m is None:
                continue
            gids = set(m.values())
            if gids & removed:
                continue
            absorb = {x for x in gids
                      if G.nodes[x].get("type") in ("process", "subject")
                      or x not in shared}
            removed |= absorb
            regions.append({"tpl": tpl, "tpl_id": tpl.gid, "nodes": gids,
                            "absorb": absorb})
            n_inst += 1
            budget -= 1
        if verbose:
            print(f"[tered] 模板 {tpl.gid} 命中 {n_inst} 个实例")
    return regions


class TeRedOperatorShared(TeRedOperator):
    """TeRed 折叠但只吸收 absorb 子集（共享对象保留为外部节点）。"""

    def _reduce(self, G):
        regions = find_instances_shared(G, self.templates,
                                        share_k=int(self.cfg.get("share_k", 2)),
                                        max_instances=self.max_instances,
                                        max_total=self.max_total,
                                        verbose=self.cfg.get("verbose", False))
        if not regions:
            Gp = CanonicalGraph(G.gid + ":tered")
            for nid, nd in G.nodes.items():
                Gp.nodes[nid] = copy.deepcopy(nd)
            Gp.labels = dict(G.labels)
            edge_map = {}
            for i, e in enumerate(G.edges):
                Gp.edges.append(dict(e))
                edge_map[len(Gp.edges) - 1] = [i]
            return ReductionResult(Gp, {n: n for n in G.nodes}, edge_map,
                                   {"n_regions": 0}, self.name)
        removed = set()
        for r in regions:
            removed |= r["absorb"]
        kept = set(G.nodes) - removed
        region_of = {}
        for ridx, r in enumerate(regions):
            for nd in r["absorb"]:
                region_of[nd] = ridx
        # 共享对象不在 removed，但被折叠区域引用 -> 需要它们仍在 Gp 中（kept 含）
        node_map = {}
        for ridx, r in enumerate(regions):
            m_id, n_id = f"teredM{ridx}", f"teredN{ridx}"
            # 出口判定: 该 absorb 节点有到"非本区域 absorb 节点/外部"的边
            for nd in sorted(r["absorb"]):
                # 简化: 有出边到本区域外 -> N, 否则 M (与原版一致, 基于 absorb)
                ext_out = any(e["src"] == nd and e["dst"] not in r["absorb"]
                              for e in G.edges)
                node_map[nd] = n_id if ext_out else m_id
        for nid in G.nodes:
            node_map.setdefault(nid, nid)
        Gp = CanonicalGraph(G.gid + ":tered")
        for nid in kept:
            Gp.nodes[nid] = copy.deepcopy(G.nodes[nid])
            if nid in G.labels:
                Gp.labels[nid] = G.labels[nid]
        internal = collections.defaultdict(list)
        # 区域内部边: 两端都属于同一区域 absorb -> 汇总; absorb->共享对象 的边保留?
        # 语义: absorb 节点被折叠成 M/N, 共享对象保留 -> absorb->shared 边 = 跨边保留
        # 原版把 内部边(两端都在removed)折叠; 这里"内部"=两端同区域absorb
        for ridx, r in enumerate(regions):
            m_id, n_id = f"teredM{ridx}", f"teredN{ridx}"
            tpl = r["tpl"]
            members = sorted(r["absorb"])
            att = {"role": "entry", "template": r["tpl_id"], "members": members}
            a0 = tpl.meta.get("anchor", "0")
            ttype = tpl.nodes.get(a0, tpl.nodes.get("0", {"type": "summary"})) \
                        .get("type", "summary")
            Gp.nodes[m_id] = {"type": ttype, "attrs": att}
            Gp.nodes[n_id] = {"type": "summary_node",
                              "attrs": dict(att, role="exit")}
            if any(G.labels.get(x, 0) == 1 for x in members):
                Gp.labels[m_id] = 1
                Gp.labels[n_id] = 1
        edge_map = {}
        for i, e in enumerate(G.edges):
            s, t = e["src"], e["dst"]
            rs, rt = region_of.get(s), region_of.get(t)
            if rs is not None and rs == rt:
                internal[rs].append(i)
                continue
            ns = f"teredN{rs}" if rs is not None else s
            nt = f"teredM{rt}" if rt is not None else t
            pos = len(Gp.edges)
            Gp.edges.append({"src": ns, "dst": nt, "etype": e["etype"],
                             "ts": e["ts"], "mu": 1.0})
            edge_map[pos] = [i]
        for ridx in range(len(regions)):
            if not internal[ridx]:
                continue
            pos = len(Gp.edges)
            Gp.edges.append({"src": f"teredM{ridx}", "dst": f"teredN{ridx}",
                             "etype": "__summary__", "ts": 0,
                             "mu": float(len(internal[ridx]))})
            edge_map[pos] = internal[ridx]
        return ReductionResult(Gp, node_map, edge_map, {
            "nodes_before": G.n_nodes(), "edges_before": G.n_edges(),
            "nodes_after": Gp.n_nodes(), "edges_after": Gp.n_edges(),
            "n_regions": len(regions), "n_removed_nodes": len(removed),
            "templates_used": sorted({r["tpl_id"] for r in regions}),
        }, self.name)


import copy
from rate_core import CanonicalGraph
from reduction.base import ReductionResult


def main():
    f = sorted(glob.glob("cache/darpa/e3cadets_w8_*.jsonl"))[0]
    wins = load_graphs(f)
    g = wins[0]
    print(f"[diag_e6] win0: {g.n_nodes()}n/{len(g.edges)}e")

    t0 = time.time()
    templates, info = mine_templates_anchor_dedup([g], min_support=4,
                                                  per_graph=400, verbose=True)
    print(f"[diag_e6] 挖掘(anchor去重) {time.time()-t0:.1f}s -> {len(templates)} 模板")
    for t in templates[:10]:
        tc = collections.Counter(nd.get("type", "?") for nd in t.nodes.values())
        print(f"  {t.gid}: n={t.n_nodes()} e={t.n_edges()} sup={t.meta.get('support')} "
              f"types={dict(tc.most_common(4))}")

    for share_k in (1, 2, 5):
        op = TeRedOperatorShared(templates, max_instances=500,
                                 max_total=100000, share_k=share_k)
        res = op.reduce(g)
        ok = node_map_sanity(res, g)
        inv = check_invariants(g, res)
        ok2 = all(v is True for k, v in inv.items() if k != "INV2_violations")
        print(f"[diag_e6] share_k={share_k}: 区域={res.stats.get('n_regions')} "
              f"nodes {g.n_nodes()}->{res.Gp.n_nodes()} "
              f"({1-res.Gp.n_nodes()/g.n_nodes():.1%}) | "
              f"edges {len(g.edges)}->{len(res.Gp.edges)} "
              f"({1-len(res.Gp.edges)/len(g.edges):.1%}) | "
              f"map_sanity={ok} INV={ok2} viol={len(inv['INV2_violations'])}")


if __name__ == "__main__":
    main()
