# -*- coding: utf-8 -*-
"""reduction/base.py — 归约算子统一接口与 μ 簿记（合并指南 §2.3/§2.4/§8）。

任何算子(TeRed / CPR / NodeMerge / 未来算子)都返回 ReductionResult：
    Gp       归约图(规范格式)
    node_map  π: 原节点id -> 归约节点id（多对一；未吸收节点 -> 自身）
    edge_map  ρ: 新边下标 -> [原始边下标...]（新边质量 μ = len(该列表)）
    stats    归约统计

正确性不变量（INV-1~4 在 tests/ 中用 pytest 验证）：
    INV-1 恒等退化：算子为恒等(μ≡1)时 RATE 特征与 dual_naive 逐位相等
    INV-2 外部节点质量守恒：对 π(v)==v 的原节点，Σ_{新入边进 π(v)} μ == d_in^G(v)
    INV-3 全局边质量守恒：Σ_{e'∈E'} μ(e') == |E(G)|
    INV-4 规模单调：|V'| ≤ |V| 且 |E'| ≤ |E|
"""
from __future__ import annotations
import copy

from rate_core import CanonicalGraph


class ReductionResult:
    __slots__ = ("Gp", "node_map", "edge_map", "stats", "operator")

    def __init__(self, Gp, node_map, edge_map, stats, operator):
        self.Gp = Gp                      # CanonicalGraph
        self.node_map = node_map          # dict orig_nid -> new_nid
        self.edge_map = edge_map          # dict new_idx -> list[orig_idx]
        self.stats = stats or {}
        self.operator = operator

    def mu(self, new_idx):
        return float(len(self.edge_map.get(new_idx, [])))

    def mass_in(self, new_nid, use_mu=True):
        s = 0.0
        for i, e in enumerate(self.Gp.edges):
            if e["dst"] == new_nid:
                s += self.mu(i) if use_mu else 1.0
        return s

    def mass_out(self, new_nid, use_mu=True):
        s = 0.0
        for i, e in enumerate(self.Gp.edges):
            if e["src"] == new_nid:
                s += self.mu(i) if use_mu else 1.0
        return s


class ReductionOperator:
    """算子基类。子类实现 _reduce(G, ...) -> ReductionResult。"""
    name = "base"

    def __init__(self, **cfg):
        self.cfg = cfg

    def reduce(self, G: CanonicalGraph) -> ReductionResult:
        res = self._reduce(G)
        if self.cfg.get("verify", True):
            inv = check_invariants(G, res)
            bad = {k: v for k, v in inv.items() if v is False}
            if bad:
                raise RuntimeError(f"[{self.name}] 不变量失败: {bad} @ {G.gid}")
        return res

    def _reduce(self, G: CanonicalGraph) -> ReductionResult:
        raise NotImplementedError


class IdentityOperator(ReductionOperator):
    """恒等算子（μ≡1）。用于 INV-1 与实验网格的"无归约"列。"""
    name = "identity"

    def _reduce(self, G):
        Gp = CanonicalGraph(G.gid + ":id")
        for nid, nd in G.nodes.items():
            Gp.nodes[nid] = copy.deepcopy(nd)
        Gp.labels = dict(G.labels)
        edge_map = {}
        for i, e in enumerate(G.edges):
            Gp.edges.append(dict(e))
            edge_map[len(Gp.edges) - 1] = [i]
        nm = {n: n for n in G.nodes}
        return ReductionResult(Gp, nm, edge_map,
                               {"nodes_before": G.n_nodes(), "edges_before": G.n_edges()},
                               self.name)


# ---------------- 不变量 ----------------
def check_invariants(G: CanonicalGraph, res: ReductionResult) -> dict:
    out = {}
    # INV-3 全局边质量守恒
    total_mu = res.Gp.total_edge_mu()
    out["INV3_total_mu_eq_edges"] = abs(total_mu - G.n_edges()) < 1e-6
    # INV-4 规模单调
    out["INV4_nodes_monotone"] = res.Gp.n_nodes() <= G.n_nodes()
    out["INV4_edges_monotone"] = res.Gp.n_edges() <= G.n_edges()
    # 每个新边都被映射到 ≥1 条原始边，且无重叠(每条原始边恰好一次) -> 也是 INV-3 的强形式
    seen = []
    ok_cover = True
    for lst in res.edge_map.values():
        seen.extend(lst)
    dup = len(seen) != len(set(seen))
    missing = set(range(G.n_edges())) - set(seen)
    out["INV3_cover_exact"] = (not dup) and (not missing)
    # INV-2 外部节点(π(v)==v)质量守恒
    bad2 = []
    din_o, dout_o = G.degrees(use_mu=False)
    din_n = {nid: 0.0 for nid in res.Gp.nodes}
    dout_n = {nid: 0.0 for nid in res.Gp.nodes}
    for i, e in enumerate(res.Gp.edges):
        m = res.mu(i)
        din_n[e["dst"]] += m
        dout_n[e["src"]] += m
    for v, nv in res.node_map.items():
        if nv == v and v in din_o:      # 未吸收节点
            if abs(din_n.get(v, 0.0) - din_o[v]) > 1e-6:
                bad2.append((v, "in", din_n.get(v), din_o[v]))
            if abs(dout_n.get(v, 0.0) - dout_o[v]) > 1e-6:
                bad2.append((v, "out", dout_n.get(v), dout_o[v]))
    out["INV2_external_conserved"] = len(bad2) == 0
    out["INV2_violations"] = bad2[:10]
    return out


def node_map_sanity(res: ReductionResult, G: CanonicalGraph):
    """node_map 覆盖全部原节点且指向存在的新节点；
    归约图每个节点要么是某原节点的像，要么被至少一条边引用（如汇总边的出口）。"""
    assert set(res.node_map.keys()) == set(G.nodes.keys()), "node_map 未覆盖全部原节点"
    gp_ids = set(res.Gp.nodes.keys())
    for v, nv in res.node_map.items():
        assert nv in gp_ids, f"π({v})={nv} 不存在于归约图"
    referenced = set(res.node_map.values())
    for e in res.Gp.edges:
        referenced.add(e["src"])
        referenced.add(e["dst"])
    assert gp_ids <= referenced, f"存在不可达的新节点: {gp_ids - referenced}"
