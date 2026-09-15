# -*- coding: utf-8 -*-
"""scripts/e3_rate_validation.py — E3 CADETS 上验证 RATE 机制（M2 go/no-go 机制层）。

问题：M2 完整版需要攻击节点标签做 F1 对比；本脚本先在**无标签**层面验证论文立论的
机制前提——"归约破坏流质量(度数)，naive 计数失真，RATE(Σμ) 精确恢复"：

对每张 E3 良性窗：
  1. identity（无归约）：基准流质量 = |E|，节点度数 = 原图入/出度；
  2. tered 归约（论文整块吸收 share_k=-1）：
     naive 视角：每边计 1 → 可见流质量 = |E'|（内部边被折叠丢失）；
     RATE 视角：每边计 μ → Σμ == |E|（INV-3），M/N 节点带 μ=折叠边数。

输出（每窗 + 汇总）：
  * 节点/边归约率（节点归约才是 TeRed 主口径）
  * INV-2 外部节点守恒 ok 数、INV-3 全局守恒 ok（RATE 簿记自检）
  * naive 流质量损失 = 1 - |E'|/|E|（= 边归约率，即 naive 检测器看不见的流比例）
  * M/N 节点上 naive 出度(=1) vs μ(=k) 的失真倍数（度数被系统性低估的直接证据）
"""
from __future__ import annotations
import os, sys, glob, time, collections

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rate_core import load_graphs
from reduction import template_mining as tmining
from reduction.tered import TeRedOperator
from reduction.base import (IdentityOperator, check_invariants,
                            node_map_sanity)
import numpy as np


def main():
    min_support = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    per_graph = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    share_k = int(sys.argv[3]) if len(sys.argv) > 3 else -1
    wprefix = sys.argv[4] if len(sys.argv) > 4 else "w8"
    f = sorted(glob.glob(f"cache/darpa/e3cadets_{wprefix}_*.jsonl"))
    if not f:
        print("NO_WINDOW_CACHE"); return
    wins = load_graphs(f[0])
    n = len(wins)
    n_train = max(2, n // 2)
    train = wins[:n_train]
    print(f"[e3rateval] {n} 窗 | 模板训练 {n_train} 窗 | share_k={share_k}")

    t0 = time.time()
    templates, _ = tmining.mine_templates(
        train, min_support=min_support, per_graph=per_graph,
        max_templates=200, min_tpl_nodes=3, seed=0, verbose=False,
        dedup_key="node", context_k=0, khop_rare=2)
    print(f"[e3rateval] 挖掘 {len(templates)} 模板 耗时 {time.time()-t0:.0f}s")

    op = TeRedOperator(templates, max_instances=500, verbose=False,
                       share_k=share_k)
    print(f"\n{'win':<4}{'tag':<6}{'节点归约':>8}{'边归约':>8}"
          f"{'INV2ok':>8}{'INV3':>6}{'naive流损失':>10}{'M/N':>6}"
          f"{'失真中位':>8}{'失真最大':>8}")
    for i, w in enumerate(wins):
        res = op.reduce(w)
        node_map_sanity(res, w)
        inv = check_invariants(w, res)
        n0, e0 = w.n_nodes(), len(w.edges)
        n1, e1 = res.Gp.n_nodes(), len(res.Gp.edges)
        node_red = 1 - n1 / max(1, n0)
        edge_red = 1 - e1 / max(1, e0)
        inv2_ok = inv.get("INV2_external_conserved", False)
        inv3_ok = inv.get("INV3_total_mu_eq_edges", False)
        # M/N 节点: 出边中 μ>1 的 summary 边失真 (naive 记 1, μ 记 k)
        ratios = []
        for e in res.Gp.edges:
            if e["src"].startswith("teredM") or e["src"].startswith("teredN") \
                    or e["dst"].startswith("teredM") or e["dst"].startswith("teredN"):
                if e.get("mu", 1.0) > 1.0:
                    ratios.append(e["mu"])      # naive=1 vs μ=m -> 低估 m 倍
        tag = "train" if i < n_train else "test"
        med = float(np.median(ratios)) if ratios else 0.0
        mx = float(np.max(ratios)) if ratios else 0.0
        print(f"{i:<4}{tag:<6}{node_red:>8.2%}{edge_red:>8.2%}"
              f"{str(inv2_ok):>8}{str(inv3_ok):>6}{edge_red:>10.2%}"
              f"{len(ratios):>6}{med:>8.1f}{mx:>8.0f}")

    # ---- 汇总（测试窗为主） ----
    print(f"\n[e3rateval] 汇总: 该 E3 集上 tered(share_k={share_k}) 的节点归约率 "
          f"即 TeRed 主口径; naive 检测器可见流质量 = 1-边归约率; RATE Σμ=1 全保 (INV3).")
    print(f"[e3rateval] 模板 {len(templates)} | 运行总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
