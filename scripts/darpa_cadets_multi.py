# -*- coding: utf-8 -*-
"""scripts/darpa_cadets_multi.py — 多文件 E5 cadets 上的 tered 模板挖掘与归约验证。

数据：122 个 bin.*.gz（Avro 容器），每个是一段时间窗口的审计流。
设计（对齐 TeRed"程序行为模板跨运行实例匹配"的语义）：
  - 前 train_n 个文件(按数字序) 当作良性训练图集 -> 挖模板（跨文件 support）；
  - 剩余 held-out 文件用同一冻结模板库归约 -> 报告真实(held-out)归约率 + INV。

用法:
  python scripts/darpa_cadets_multi.py --data-dir ../dataset/darpae5/cadets \
      --n-files 8 --train-files 5 --max-records 400000 --min-support 3 \
      --results results/darpa_e5cadets_tered.json
"""
from __future__ import annotations
import os, sys, glob, json, time, collections, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from rate_core import load_graphs, save_graphs, fingerprint, cache_path
from reduction import template_mining as tmining
from reduction.tered import TeRedOperator
from reduction.base import check_invariants, node_map_sanity
from adapters import darpa_tc


def _num(fn):
    """bin 文件名数字序 key。"""
    import re
    m = re.search(r"\.bin\.(\d+)\.gz$", fn)
    return int(m.group(1)) if m else 0


def parse_files(data_dir, names, max_records):
    graphs = []
    for nm in names:
        p = os.path.join(data_dir, nm)
        gs, _ = darpa_tc.parse(p, max_records=max_records, verbose=False)
        graphs.append(gs[0])
        print(f"    {nm}: {gs[0].n_nodes()} 节点 / {len(gs[0].edges)} 边", flush=True)
    return graphs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/TeRed+RATE/dataset/darpae5/cadets")
    ap.add_argument("--n-files", type=int, default=8)
    ap.add_argument("--train-files", type=int, default=5)
    ap.add_argument("--max-records", type=int, default=400000)
    ap.add_argument("--min-support", type=int, default=3)
    ap.add_argument("--min-tpl-nodes", type=int, default=3)
    ap.add_argument("--max-tpl-nodes", type=int, default=64)
    ap.add_argument("--max-templates", type=int, default=80)
    ap.add_argument("--per-graph", type=int, default=800)
    ap.add_argument("--dedup-key", default="node", choices=("node", "anchor"))
    ap.add_argument("--context-k", type=int, default=0)
    ap.add_argument("--share-k", type=int, default=-1,
                    help="共享上下文阈值。-1 = 论文口径整块吸收(不排除共享节点, v3语义); "
                         ">=0 = v4b 共享感知吸收(排除被 >=share_k 进程引用的节点)")
    ap.add_argument("--cache", default="cache", help="模板持久化目录")
    ap.add_argument("--no-tpl-cache", action="store_true",
                    help="跳过模板缓存命中, 强制重新挖掘")
    ap.add_argument("--results", default=None)
    a = ap.parse_args()

    allf = sorted((f for f in os.listdir(a.data_dir) if f.endswith(".gz")),
                  key=_num)
    use = allf[:a.n_files]
    train_names, test_names = use[:a.train_files], use[a.train_files:]
    print(f"[multi] 共 {len(allf)} 个文件; 用前 {len(use)} 个; "
          f"训练 {len(train_names)} 个 / held-out {len(test_names)} 个", flush=True)

    t0 = time.time()
    train_graphs = parse_files(a.data_dir, train_names, a.max_records)
    print(f"[multi] 训练图解析耗时 {time.time()-t0:.0f}s", flush=True)

    # ---- 语义锚：process/subject 白名单 + 自动低频 ----
    auto = tmining.auto_semantic_types(train_graphs, rare_frac=0.05, min_count=3)
    anchor_types = set(auto) | {"process", "subject"}
    print(f"[multi] 语义锚 type: {sorted(anchor_types)}", flush=True)

    t1 = time.time()
    mine_fp = fingerprint({"ds": "e5cadets", "mine_ver": 8,
                           "n_files": a.n_files, "train_files": a.train_files,
                           "max_records": a.max_records,
                           "min_support": a.min_support,
                           "min_tpl_nodes": a.min_tpl_nodes,
                           "max_tpl_nodes": a.max_tpl_nodes,
                           "max_templates": a.max_templates,
                           "per_graph": a.per_graph,
                           "dedup_key": a.dedup_key, "context_k": a.context_k,
                           "anchors": sorted(anchor_types)})
    tpl_cache = cache_path(a.cache, "darpa", "e5_templates", mine_fp, ext="jsonl")
    if a.no_tpl_cache or not os.path.exists(tpl_cache):
        tpls, info = tmining.mine_templates(
            train_graphs, min_support=a.min_support, per_graph=a.per_graph,
            max_tpl_nodes=a.max_tpl_nodes, max_templates=a.max_templates,
            min_tpl_nodes=a.min_tpl_nodes, seed=0, anchor_types=anchor_types,
            dedup_key=a.dedup_key, context_k=a.context_k, verbose=True)
        save_graphs(tpl_cache, tpls)
        tpl_src = "mined+persisted"
    else:
        tpls = load_graphs(tpl_cache)
        tpl_src = "cache"
    print(f"[multi] 模板来源={tpl_src}: {tpl_cache} -> {len(tpls)} 模板 "
          f"(耗时 {time.time()-t1:.0f}s)", flush=True)
    tc = collections.Counter()
    for t in tpls:
        tc.update(nd.get("type", "?") for nd in t.nodes.values())
    print(f"[multi] 模板节点 type 构成: {dict(tc.most_common(6))}", flush=True)

    # ---- 归约训练图（自身上限）与 held-out 图（真实） ----
    op = TeRedOperator(tpls, max_instances=300, max_total=5000,
                       share_k=a.share_k)
    print(f"[multi] 吸收口径 share_k={a.share_k} "
          f"({'论文整块吸收' if a.share_k < 0 else 'v4b 共享排除'})", flush=True)
    summary = {"n_train": len(train_names), "n_test": len(test_names),
               "n_templates": len(tpls), "templates_used": set(),
               "per_file": []}
    for tag, names in (("train", train_names), ("heldout", test_names)):
        for nm in names:
            gs, _ = darpa_tc.parse(os.path.join(a.data_dir, nm),
                                   max_records=a.max_records, verbose=False)
            G = gs[0]
            t2 = time.time()
            res = op.reduce(G)
            node_map_sanity(res, G)
            inv = check_invariants(G, res)
            bad = {k: v for k, v in inv.items() if v is False}
            nr = 1 - res.Gp.n_nodes() / G.n_nodes()
            er = 1 - len(res.Gp.edges) / len(G.edges)
            summary["templates_used"] |= set(res.stats.get("templates_used", []))
            summary["per_file"].append({
                "file": nm, "tag": tag, "nodes": G.n_nodes(),
                "edges": len(G.edges), "node_red": round(nr, 5),
                "edge_red": round(er, 5), "n_regions": res.stats.get("n_regions"),
                "removed_nodes": res.stats.get("n_removed_nodes"),
                "inv_bad": sorted(bad.keys()), "secs": round(time.time() - t2, 1)})
            print(f"[multi] {tag} {nm}: 节点归约 {nr:.2%} 边归约 {er:.2%} "
                  f"区域 {res.stats.get('n_regions')} INV "
                  f"{'OK' if not bad else bad} ({time.time()-t2:.0f}s)", flush=True)

    summary["templates_used"] = sorted(summary["templates_used"])
    if a.results:
        os.makedirs(os.path.dirname(os.path.abspath(a.results)) or ".", exist_ok=True)
        with open(a.results, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=1)
        print(f"[multi] 结果写入 {a.results}", flush=True)
    print(f"[multi] 总耗时 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
