# -*- coding: utf-8 -*-
"""e6h3_index_headroom.py — 模板匹配的索引化优化空间测量。

背景：e6h 实测 bin.119 上归约 534s，其中匹配 500s，做了 234,411 次回溯搜索
只找到 874 个区域（成功率 0.37%）。当前预过滤只用「度 >= 模板根度」，
过弱（85% 的候选直接进入完整搜索）。

匹配语义（reduction/tered.py `_node_match`）：只比较 **节点 type**，不看边标签。
因此一个保持语义的强索引是：以**锚点邻居类型签名**剪枝
  节点签名 sig(v)  = { (方向, 邻居type) }        （方向 ∈ {in, out}）
  模板根签名 sig_t = { (方向, 邻居type) }
匹配的必要条件：sig_t ⊆ sig(v)。
分别统计
  deg_only  = sum_t |{v : type(v)=rt, deg(v) >= deg_t(root)}|   （= 当前实现）
  type_idx  = sum_t |{v : type(v)=rt, sig_t ⊆ sig(v), deg 过滤}| （强索引上界）
输出：results/e6h3_index_headroom.json
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--spec", default="bin.119@500000")
    ap.add_argument("--templates", default="")
    ap.add_argument("--cache", default="cache/darpa")
    ap.add_argument("--summary", default="results/e6h3_index_headroom.json")
    a = ap.parse_args()

    from rate_core import load_graphs
    from scripts.e5_f1_eval import parse_one, find_templates
    from scripts.e6_alert_eval import parse_spec
    from reduction.tered import _to_nx_matching, _pick_root

    tpl_path = a.templates or find_templates(
        os.path.join(a.cache, "e5_templates", "*.jsonl"))
    tpls = load_graphs(tpl_path)
    nm, mr = parse_spec(a.data_dir, a.spec)
    g, _meta = parse_one(a.data_dir, nm, mr or None)
    print(f"[e6h3] {a.spec}: {g.n_nodes()} nodes {g.n_edges()} edges, "
          f"{len(tpls)} templates", flush=True)

    ntype = {nid: nd.get("type", "unknown") for nid, nd in g.nodes.items()}
    indeg = defaultdict(int)
    outdeg = defaultdict(int)
    for e in g.edges:
        outdeg[e["src"]] += 1
        indeg[e["dst"]] += 1
    nsig = defaultdict(set)
    for e in g.edges:
        s, t = e["src"], e["dst"]
        nsig[s].add(("out", ntype.get(t, "unknown")))
        nsig[t].add(("in", ntype.get(s, "unknown")))
    by_type = defaultdict(list)
    for nid in g.nodes:
        by_type[ntype[nid]].append(nid)

    roots = []
    for tpl in tpls:
        Tnx = _to_nx_matching(tpl)
        if Tnx.number_of_nodes() < 2:
            continue
        rt = _pick_root(Tnx, tpl)
        if rt is None:
            continue
        tsig = set()
        for nb in Tnx.successors(rt):
            tsig.add(("out", Tnx.nodes[nb].get("type", "unknown")))
        for nb in Tnx.predecessors(rt):
            tsig.add(("in", Tnx.nodes[nb].get("type", "unknown")))
        roots.append((Tnx.nodes[rt].get("type", "unknown"),
                      Tnx.in_degree(rt), Tnx.out_degree(rt),
                      frozenset(tsig), Tnx.number_of_nodes()))

    deg_only = 0
    type_idx = 0
    for (rt, ri, ro, tsig, tn) in roots:
        for v in by_type.get(rt, []):
            if indeg.get(v, 0) < ri or outdeg.get(v, 0) < ro:
                continue
            deg_only += 1
            if tsig.issubset(nsig.get(v, set())):
                type_idx += 1

    out = {
        "spec": a.spec, "n_nodes": g.n_nodes(), "n_edges": g.n_edges(),
        "n_templates": len(roots),
        "deg_only_candidates": deg_only,
        "type_indexed_candidates": type_idx,
        "prune_ratio": round(1.0 - type_idx / deg_only, 4) if deg_only else None,
        "speedup_upper_bound": round(deg_only / type_idx, 2) if type_idx else None,
    }
    with open(a.summary, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("[e6h3]", json.dumps(out, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
