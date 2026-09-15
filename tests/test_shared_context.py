# -*- coding: utf-8 -*-
"""tests/test_shared_context.py — 共享上下文剔除 + anchor 去重 端到端回归。

场景（模拟 E3 cadets 的"同程序多实例"）：
  每窗 1 个父进程 P + N 个兄弟进程；每个兄弟进程都连：共享系统库 L(被所有人
  引用)、父进程 P(被所有人引用)、以及 k 个私有 socket（仅自己引用）。
  - v3 (origin,node) 去重 + 模板含 P/L -> 兄弟互相排斥，挖不出模板；
  - anchor 去重恢复大桶，但模板仍含 P/L -> 折叠每窗只能 1 区域；
  - context_k 剔除 P/L 后模板=进程+私有 socket -> 每窗 N 区域全折叠。

断言：
  1. context_k=2 挖出的模板不含共享节点(L/P 之外的共享文件)；
  2. 标准 TeRedOperator 在 held-out 窗上折叠区域数 >= N（兄弟全被折叠）。
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rate_core import CanonicalGraph, load_graphs, save_graphs
from reduction import template_mining as tm
from reduction.tered import TeRedOperator
from reduction.base import check_invariants, node_map_sanity


def _window(gid, n_procs=8, k_sock=3, parent=True, lib=True):
    g = CanonicalGraph(gid)
    P = "P"
    L = "L"
    if parent:
        g.nodes[P] = {"type": "process", "attrs": {}}
    if lib:
        g.nodes[L] = {"type": "file", "attrs": {}}
    for i in range(n_procs):
        p = f"p{i}"
        g.nodes[p] = {"type": "process", "attrs": {}}
        if parent:
            g.edges.append({"src": P, "dst": p, "etype": "fork", "ts": 0, "mu": 1.0})
        if lib:
            g.edges.append({"src": p, "dst": L, "etype": "mmap", "ts": 0, "mu": 1.0})
        for j in range(k_sock):
            s = f"{p}.s{j}"
            g.nodes[s] = {"type": "unix_socket", "attrs": {}}
            g.edges.append({"src": p, "dst": s, "etype": "connect", "ts": 0, "mu": 1.0})
    return g


def _lib_absent(templates):
    for t in templates:
        for nd in t.nodes.values():
            assert nd.get("type") != "file", f"模板 {t.gid} 仍含共享 file 节点"
    return True


def test_context_k_mining_and_folding():
    # 训练 2 窗 + held-out 1 窗，每窗 8 兄弟进程
    train = [_window(f"w0"), _window("w1")]
    test_g = _window("w2")

    tpls, info = tm.mine_templates(train, min_support=4, khop_rare=1,
                                   per_graph=100, max_templates=20,
                                   min_tpl_nodes=3, seed=0, verbose=False,
                                   dedup_key="anchor", context_k=2)
    assert len(tpls) >= 1, f"context_k=2 应挖出 进程+私有socket 模板, got {len(tpls)}"
    _lib_absent(tpls)

    op = TeRedOperator(tpls, max_instances=500)
    res = op.reduce(test_g)
    node_map_sanity(res, test_g)
    inv = check_invariants(test_g, res)
    ok = all(v is True for k, v in inv.items() if k != "INV2_violations")
    assert ok, f"INV 失败: {inv}"
    n_regions = res.stats.get("n_regions", 0)
    # held-out 窗 8 个兄弟应全部被折叠
    assert n_regions >= 8, f"应折叠 >=8 区域, got {n_regions}"


def test_without_context_k_folds_fewer():
    """对照：context_k=0 + anchor 去重 也能挖出模板，但含 P/L -> 折叠被挡。"""
    train = [_window("w0"), _window("w1")]
    test_g = _window("w2")

    tpls, info = tm.mine_templates(train, min_support=4, khop_rare=1,
                                   per_graph=100, max_templates=20,
                                   min_tpl_nodes=3, seed=0, verbose=False,
                                   dedup_key="anchor", context_k=0)
    # 模板含共享 file 或父进程，且跨窗验证时 find_instances 每窗只能取 1 个
    # 非重叠实例 -> min_support=4 (2 窗最多 2) 挖不出；降低阈值到 1 也能挖出但折叠受限
    tpls2, _ = tm.mine_templates(train, min_support=1, khop_rare=1,
                                 per_graph=100, max_templates=20,
                                 min_tpl_nodes=3, seed=0, verbose=False,
                                 dedup_key="anchor", context_k=0)
    has_shared = any(
        any(nd.get("type") == "file" for nd in t.nodes.values())
        for t in tpls2)
    # 模板里应出现共享 file（L 在候选里）
    if has_shared:
        op = TeRedOperator(tpls2, max_instances=500)
        res = op.reduce(test_g)
        # 共享模板每窗至多折叠 1 个区域（P/L 被第一实例吃掉）
        assert res.stats.get("n_regions", 0) < 4, \
            f"含共享节点的模板应只能折叠 <4 区域, got {res.stats.get('n_regions')}"
    # 无论如何 context_k=2 必须比 context_k=0 折得多
    tpls_a, _ = tm.mine_templates(train, min_support=4, khop_rare=1,
                                  per_graph=100, max_templates=20,
                                  min_tpl_nodes=3, seed=0, verbose=False,
                                  dedup_key="anchor", context_k=2)
    op_a = TeRedOperator(tpls_a, max_instances=500)
    res_a = op_a.reduce(test_g)
    op_b = TeRedOperator(tpls2, max_instances=500)
    res_b = op_b.reduce(test_g)
    assert res_a.stats.get("n_regions", 0) >= res_b.stats.get("n_regions", 0)


if __name__ == "__main__":
    test_context_k_mining_and_folding()
    test_without_context_k_folds_fewer()
    print("test_shared_context OK")
