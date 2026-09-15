# -*- coding: utf-8 -*-
"""reduction/nodemerge.py — NodeMerge(合并型归约)：合并邻域相同的节点。

入/出邻域（邻居id × 边类型，带方向）完全相同的同类型节点合并为一个超节点；
边随之转移：指向不同成员的同款边去重合并并记账 μ=k，内部成员互连边塌成自环(μ 记账)。
（指南 §4.2。这是"邻域相同 → 一个超节点"的干净实现。）
"""
from __future__ import annotations
import copy
from collections import defaultdict
from rate_core import CanonicalGraph
from reduction.base import ReductionOperator, ReductionResult


class NodeMergeOperator(ReductionOperator):
    name = "nodemerge"

    def __init__(self, min_group=2, **cfg):
        super().__init__(**cfg)
        self.min_group = int(min_group)

    def _reduce(self, G):
        ins = defaultdict(list)
        outs = defaultdict(list)
        for i, e in enumerate(G.edges):
            ins[e["dst"]].append((e["src"], e["etype"], i))
            outs[e["src"]].append((e["dst"], e["etype"], i))

        groups = defaultdict(list)      # signature -> [nid]
        for nid, nd in G.nodes.items():
            key_in = tuple(sorted((s, t) for s, t, _ in ins.get(nid, [])))
            key_out = tuple(sorted((s, t) for s, t, _ in outs.get(nid, [])))
            if not key_in and not key_out:
                continue                # 孤立节点不参与合并（无拓扑信号）
            sig = (nd["type"], key_in, key_out)
            groups[sig].append(nid)
        merged_sets = [v for v in groups.values() if len(v) >= self.min_group]

        # node_map
        nm = {}
        super_of = {}
        for k, members in enumerate(merged_sets):
            sid = f"nm_{k}"
            for m in members:
                nm[m] = sid
                super_of[m] = sid
        for nid in G.nodes:
            nm.setdefault(nid, nid)

        Gp = CanonicalGraph(G.gid + ":nm")
        done = set()
        for nid, nd in G.nodes.items():
            if nid in super_of:
                continue
            Gp.nodes[nid] = copy.deepcopy(nd)
        for members in merged_sets:
            k = list(members)[0]
            sid = nm[k]
            typ = G.nodes[k]["type"]
            attack = 0
            for m in members:
                attack = max(attack, G.labels.get(m, 0))
                done.add(m)
            Gp.nodes[sid] = {"type": typ,
                             "attrs": {"merged_from": sorted(members),
                                       "n_members": len(members)}}
            if attack:
                Gp.labels[sid] = attack
        for nid, lab in G.labels.items():
            if nid not in super_of and nid in Gp.nodes:
                Gp.labels[nid] = lab

        edge_map = {}
        bucket = {}                     # (new_src,new_dst,etype) -> [orig_idx]
        ts_first = {}
        for i, e in enumerate(G.edges):
            ns, nd2 = nm[e["src"]], nm[e["dst"]]
            if ns == e["src"] and nd2 == e["dst"] and e["src"] not in super_of \
                    and e["dst"] not in super_of:
                pos = len(Gp.edges)
                Gp.edges.append(dict(e))
                edge_map[pos] = [i]
            else:
                key = (ns, nd2, e["etype"])
                bucket.setdefault(key, []).append(i)
                ts_first.setdefault(key, e["ts"])
        for (ns, nd2, et), idxs in bucket.items():
            pos = len(Gp.edges)
            Gp.edges.append({"src": ns, "dst": nd2, "etype": et,
                             "ts": ts_first[(ns, nd2, et)], "mu": float(len(idxs))})
            edge_map[pos] = idxs

        return ReductionResult(Gp, nm, edge_map, {
            "nodes_before": G.n_nodes(), "edges_before": G.n_edges(),
            "nodes_after": Gp.n_nodes(), "edges_after": Gp.n_edges(),
            "n_groups": len(merged_sets),
            "n_merged_nodes": sum(len(s) for s in merged_sets),
            "reduction_ratio": (G.n_nodes() - Gp.n_nodes()) / max(1, G.n_nodes()),
        }, self.name)
