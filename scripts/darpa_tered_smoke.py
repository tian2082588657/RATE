# -*- coding: utf-8 -*-
"""scripts/darpa_tered_smoke.py — DARPA (E5 cadets) 上验证 tered 模板挖掘与归约率。

流程：载入解析缓存(或现场 parse) -> split_by_time 前 50% 为良性训练子图 ->
mine_templates(语义锚) -> TeRedOperator.reduce(整图) -> 归约率 + INV + 攻击存活。
"""
from __future__ import annotations
import os, sys, glob, time, collections

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from rate_core import load_graphs
from reduction import template_mining as tmining
from reduction.tered import TeRedOperator
from reduction.base import check_invariants, node_map_sanity
from adapters import darpa_tc


def pick_cache(cache_dir, name_hint):
    fs = sorted(glob.glob(os.path.join(cache_dir, "*.jsonl")))
    import json
    for f in fs:
        with open(f, encoding="utf-8") as fh:
            first = json.loads(fh.readline())
        gid = first.get("gid", "")
        if name_hint in gid:
            return f
    return fs[0] if fs else None


def main():
    cache_dir = os.path.join(ROOT, "cache", "darpa", "parsed")
    f = pick_cache(cache_dir, "cadets")
    if f is None:
        print("NO_DARPA_CACHE"); return
    t0 = time.time()
    graphs = load_graphs(f)
    G = graphs[0]
    print(f"[smoke] 载入 {os.path.basename(f)}: {G.n_nodes()} 节点 / "
          f"{len(G.edges)} 边, 耗时 {time.time()-t0:.1f}s")

    # ---- 时间切分：前 50% 为良性训练段 ----
    train, _test = darpa_tc.split_by_time(G, train_frac=0.5)
    tc = collections.Counter(nd.get("type", "?") for nd in train.nodes.values())
    print(f"[smoke] 训练段(前50%时间): {train.n_nodes()} 节点 / "
          f"{len(train.edges)} 边 | types top: {tc.most_common(6)}")

    # ---- 模板挖掘（语义锚 + WL 桶 + 匹配验证） ----
    t1 = time.time()
    # process 是行为主体，必须为锚；rare_frac 放宽到 5% 兜住 2~3% 的 file/socket 类
    auto = tmining.auto_semantic_types([train], rare_frac=0.05, min_count=3)
    anchor_types = set(auto) | {"process", "subject"}
    tpls, info = tmining.mine_templates(
        [train], min_support=3, per_graph=800, max_tpl_nodes=64,
        max_templates=60, min_tpl_nodes=3, seed=0, anchor_types=anchor_types,
        verbose=True)
    print(f"[smoke] 挖掘 {len(tpls)} 模板 耗时 {time.time()-t1:.1f}s")
    for t in tpls[:20]:
        types = collections.Counter(nd.get("type", "?") for nd in t.nodes.values())
        print(f"    tpl support={t.meta.get('support')} nodes={t.n_nodes()} "
              f"edges={t.n_edges()} types={dict(types.most_common(4))}")

    # ---- tered 归约整图 ----
    t2 = time.time()
    op = TeRedOperator(tpls, max_instances=200, max_total=2000)
    res = op.reduce(G)
    node_map_sanity(res, G)
    inv = check_invariants(G, res)
    bad = {k: v for k, v in inv.items() if v is False}
    print(f"[smoke] tered 归约 耗时 {time.time()-t2:.1f}s")
    print(f"[smoke] 归约: 节点 {G.n_nodes()} -> {res.Gp.n_nodes()} "
          f"(率 {1 - res.Gp.n_nodes()/G.n_nodes():.3%}), "
          f"边 {len(G.edges)} -> {len(res.Gp.edges)} "
          f"(率 {1 - len(res.Gp.edges)/len(G.edges):.3%})")
    print(f"[smoke] 区域数 {res.stats.get('n_regions')}, "
          f"吸收节点 {res.stats.get('n_removed_nodes')}, "
          f"模板使用 {res.stats.get('templates_used')}")
    print(f"[smoke] INV bad: {bad if bad else '无（全过）'}")
    # 训练段作为良性代理：攻击存活率=测试段(后50%)节点未被吸收比例
    test_nodes = set(G.nodes) - set(train.nodes)
    absorbed = set(res.node_map.keys()) - {v for v in res.node_map.values() if v == v}
    absorbed = {v for v, nv in res.node_map.items() if nv != v}
    surv = 1 - len(absorbed & test_nodes) / max(1, len(test_nodes))
    print(f"[smoke] 后50%段节点(代理攻击)存活率: {surv:.3f}")
    print(f"[smoke] 总耗时 {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
