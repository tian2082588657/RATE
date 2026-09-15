# -*- coding: utf-8 -*-
"""reduction/cpr.py — CPR(剪边型归约)：合并重复边。

同 (src, dst, etype) 的 k 条边 -> 1 条，μ=k。这是"μ 恰好还原度数"的最干净实例
（指南 §4.2）：原度数由 k 条边贡献，归约后 1 条边 μ=k，质量加权度不变，
而 naive(数边条数) 会变小 —— naive 掉点、RATE 不掉。
"""
from __future__ import annotations
import copy
from rate_core import CanonicalGraph
from reduction.base import ReductionOperator, ReductionResult


class CPROperator(ReductionOperator):
    name = "cpr"

    def _reduce(self, G):
        Gp = CanonicalGraph(G.gid + ":cpr")
        for nid, nd in G.nodes.items():
            Gp.nodes[nid] = copy.deepcopy(nd)
        Gp.labels = dict(G.labels)
        # bucket: key(src,dst,etype) -> [orig_idx, ...]
        buckets = {}
        for i, e in enumerate(G.edges):
            buckets.setdefault((e["src"], e["dst"], e["etype"]), []).append(i)
        edge_map = {}
        for (src, dst, etype), idxs in buckets.items():
            ts = min(G.edges[j]["ts"] for j in idxs)
            pos = len(Gp.edges)
            Gp.edges.append({"src": src, "dst": dst, "etype": etype,
                             "ts": ts, "mu": float(len(idxs))})
            edge_map[pos] = idxs
        nm = {n: n for n in G.nodes}
        return ReductionResult(Gp, nm, edge_map, {
            "nodes_before": G.n_nodes(), "edges_before": G.n_edges(),
            "edges_after": Gp.n_edges(),
            "merged_groups": len([b for b in buckets.values() if len(b) > 1]),
            "reduction_ratio": (G.n_edges() - Gp.n_edges()) / max(1, G.n_edges()),
        }, self.name)
