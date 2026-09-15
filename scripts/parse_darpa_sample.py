# -*- coding: utf-8 -*-
"""scripts/parse_darpa_sample.py — DARPA TC 数据集解析冒烟。

用法（在仓库根目录执行）：
    # E3 theia (cdm18 ndjson 小文件，全量解析)
    python scripts/parse_darpa_sample.py --path "../dataset/darpa e3/ta1-theia-e3-official-5m.json.tar.gz" --kind e3_ndjson
    # E3 cadets 官方 (大 tar，只解析前 2e5 条记录 = 早期良性时段)
    python scripts/parse_darpa_sample.py --path "../dataset/darpa e3/ta1-cadets-e3-official.json.tar.gz" --kind e3_ndjson --max-records 200000
    # E5 theia (cdm20 ndjson)
    python scripts/parse_darpa_sample.py --path "../dataset/darpa e5/theia/ta1-theia-1-e5-official-1.json.1.gz" --kind e5_ndjson --max-records 200000
    # E5 cadets (Avro 容器 .bin；需 pip install fastavro)
    python scripts/parse_darpa_sample.py --path "../dataset/darpa e5/cadets/ta1-cadets-1-e5-official-2.bin.1.gz" --kind e5_cadets_bin --max-records 300000

可选 --op cpr|nodemerge|tered：解析后对图跑归约算子，验证不变量在真实溯源图上成立。
"""
from __future__ import annotations
import argparse, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rate_core import CanonicalGraph, cache_path, fingerprint, save_graphs, load_graphs
from adapters import darpa_tc
from reduction.base import check_invariants, node_map_sanity, IdentityOperator
from reduction.cpr import CPROperator
from reduction.nodemerge import NodeMergeOperator
from reduction.tered import TeRedOperator
from reduction import template_mining as tmining


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    ap.add_argument("--kind", default=None,
                    choices=["e3_ndjson", "e5_ndjson", "e5_cadets_bin", None])
    ap.add_argument("--max-records", type=int, default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--op", default=None, choices=["identity", "cpr", "nodemerge", "tered"])
    ap.add_argument("--min-support", type=int, default=3)
    ap.add_argument("--per-graph-anchors", type=int, default=200)
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--no-cache", action="store_true")
    a = ap.parse_args()

    fp = fingerprint({"path": os.path.basename(a.path), "kind": a.kind,
                      "max_records": a.max_records})
    cf = cache_path(a.cache, "darpa", "parsed", fp)
    t0 = time.time()
    if not a.no_cache and os.path.exists(cf):
        graphs = load_graphs(cf)
        print(f"[darpa] 命中缓存 {cf} ({len(graphs)} 图)")
    else:
        graphs, meta = darpa_tc.parse(a.path, name=a.name, kind=a.kind,
                                      max_records=a.max_records, verbose=True)
        if graphs:
            save_graphs(cf, graphs)
    if not graphs:
        print("[darpa] 未解析出任何边。")
        return

    G = graphs[0]
    print(f"[darpa] 图 {G.gid}: {G.n_nodes()} 节点 / {G.n_edges()} 边 | "
          f"类型 {_hist(G)} | 用时 {time.time()-t0:.1f}s")

    if a.op is None:
        return
    if a.op == "tered":
        # 良性时段(前半)挖模板 -> 冻结 -> 归约后半（结构性验证，无真值不打检测分）
        es = sorted(G.edges, key=lambda e: e["ts"])
        cut = int(len(es) * 0.5)
        def _slice(seg):
            sg = CanonicalGraph(G.gid)
            sg.meta = dict(G.meta)
            for e in seg:
                sg.ensure_node(e["src"]); sg.ensure_node(e["dst"])
                sg.edges.append(dict(e))
            return sg
        tG = _slice(es[:cut])
        tpls, info = tmining.mine_templates([tG], min_support=a.min_support,
                                            per_graph=a.per_graph_anchors,
                                            max_templates=100, seed=0, verbose=True)
        op = TeRedOperator(tpls, max_instances=500, verbose=False)
        print(f"[darpa] 挖出 {len(tpls)} 模板，归约后半张图…")
    else:
        op = {"identity": IdentityOperator, "cpr": CPROperator,
              "nodemerge": NodeMergeOperator}[a.op]()
    res = op.reduce(G)
    node_map_sanity(res, G)
    inv = check_invariants(G, res)
    print(f"[darpa] op={a.op}: {G.n_nodes()}→{res.Gp.n_nodes()} 节点, "
          f"{G.n_edges()}→{res.Gp.n_edges()} 边, 区域={res.stats.get('n_regions', '-')}")
    for k, v in inv.items():
        if k != "INV2_violations":
            print(f"    {k}: {v}")
    if inv["INV2_violations"]:
        print("    INV-2 violations(前5):", inv["INV2_violations"][:5])


def _hist(g):
    h = {}
    for n in g.nodes.values():
        h[n.get("type", "?")] = h.get(n.get("type", "?"), 0) + 1
    return dict(list(sorted(h.items()))[:8])


if __name__ == "__main__":
    main()
