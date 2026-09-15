# -*- coding: utf-8 -*-
"""scripts/e3_cadets_rate.py — E3 cadets 时间窗上验证 tered 归约率（M2 go/no-go）。

流程：
  1. 载入 e3cadets_w8 缓存（8 个时间窗，每窗 ~2k 节点 / ~2 万边）；
  2. 前 4 窗 = 良性训练 -> v3 挖掘器挖模板（cpr_first 自动压缩）；
  3. 全部 8 窗过 tered 归约，报告节点/边归约率、INV 全过、命中模板分布。

用法: python scripts/e3_cadets_rate.py [min_support] [per_graph_anchors] [dedup_key] [context_k] [khop_rare] [wprefix]
  dedup_key: node(默认, E3 A/B 实测最优) | anchor(恢复大桶, 端到端略差)
  context_k: >0 时挖矿前剔除被 >=context_k 个进程共享的上下文节点(v5, cadets 上过激)
  khop_rare: 语义锚抽取跳数（默认 2；进程 egonet 建议 1）
  wprefix:   窗口缓存前缀 w8(默认) | w4
  share_k:   共享上下文阈值; -1 = 论文整块吸收(默认) | >=0 = v4b 共享排除
"""
from __future__ import annotations
import os, sys, glob, time, collections

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rate_core import load_graphs
from reduction import template_mining as tmining
from reduction.tered import TeRedOperator
from reduction.base import check_invariants, node_map_sanity
import numpy as np


def main():
    min_support = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    per_graph = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    dedup_key = sys.argv[3] if len(sys.argv) > 3 else "node"
    context_k = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    khop_rare = int(sys.argv[5]) if len(sys.argv) > 5 else 2
    wprefix = sys.argv[6] if len(sys.argv) > 6 else "w8"
    share_k = int(sys.argv[7]) if len(sys.argv) > 7 else -1
    f = sorted(glob.glob(f"cache/darpa/e3cadets_{wprefix}_*.jsonl"))
    if not f:
        print("NO_WINDOW_CACHE: 先跑 scripts/e3_cadets_windows.py"); return
    wins = load_graphs(f[0])
    n = len(wins)
    n_train = max(2, n // 2)
    train, rest = wins[:n_train], wins[n_train:]
    print(f"[e3rate] {n} 窗 | 训练(良性) {n_train} 窗 | 验证 {len(rest)} 窗 | "
          f"min_support={min_support} per_graph={per_graph} dedup_key={dedup_key} "
          f"context_k={context_k} khop_rare={khop_rare} share_k={share_k} "
          f"({'论文整块吸收' if share_k < 0 else 'v4b 共享排除'})")

    t0 = time.time()
    templates, info = tmining.mine_templates(
        train, min_support=min_support, per_graph=per_graph,
        max_templates=200, min_tpl_nodes=3, seed=0, verbose=True,
        dedup_key=dedup_key, context_k=context_k, khop_rare=khop_rare)
    print(f"[e3rate] 挖掘耗时 {time.time()-t0:.1f}s -> {len(templates)} 模板")
    if templates:
        for t in templates[:12]:
            tc = collections.Counter(nd.get("type", "?") for nd in t.nodes.values())
            print(f"  {t.gid}: n={t.n_nodes()} e={t.n_edges()} "
                  f"anchor={t.meta.get('anchor')} sup={t.meta.get('support')} "
                  f"types={dict(tc.most_common(4))}")

    op = TeRedOperator(templates, max_instances=500, verbose=False,
                       share_k=share_k)
    rows = []
    for i, w in enumerate(wins):
        res = op.reduce(w)
        node_map_sanity(res, w)
        inv = check_invariants(w, res)
        ok = all(v is True for k, v in inv.items() if k != "INV2_violations")
        n0, n1 = w.n_nodes(), res.Gp.n_nodes()
        e0, e1 = len(w.edges), res.Gp.n_edges()
        rows.append((i, n0, n1, e0, e1, res.stats.get("n_regions", 0), ok,
                     len(inv["INV2_violations"])))
        tag = "train" if i < n_train else "test"
        print(f"[e3rate] win{i:<2}({tag}) nodes {n0:>5}->{n1:>5} "
              f"({(1-n1/max(1,n0)):>6.1%}) | edges {e0:>6}->{e1:>6} "
              f"({(1-e1/max(1,e0)):>6.1%}) | regions={res.stats.get('n_regions')} "
              f"| INV={ok}")

    nr = np.array([(1 - r[2] / max(1, r[1])) for r in rows])
    er = np.array([(1 - r[4] / max(1, r[3])) for r in rows])
    print(f"\n[e3rate] 平均节点归约率 {nr.mean():.3f} (test窗 {nr[n_train:].mean():.3f}) | "
          f"平均边归约率 {er.mean():.3f} (test窗 {er[n_train:].mean():.3f})")
    print(f"[e3rate] INV 全过={all(r[6] for r in rows)} | "
          f"总区域={sum(r[5] for r in rows)}")


if __name__ == "__main__":
    main()
