# -*- coding: utf-8 -*-
"""test_invariants.py — 归约正确性不变量（合并指南 §8）。

INV-1 恒等退化 / INV-2 外部节点质量守恒 / INV-3 全局边质量守恒 /
INV-4 规模单调 / INV-5 攻击节点存活率(度量)。
在任何归约算子注册/改动后必须全绿：python -m pytest tests/ -v
"""
from __future__ import annotations
import numpy as np
import pytest

from rate_core import CanonicalGraph, random_graph, star_graph
from reduction.base import (IdentityOperator, ReductionOperator, ReductionResult,
                            check_invariants, node_map_sanity)
from reduction.cpr import CPROperator
from reduction.nodemerge import NodeMergeOperator
from reduction.tered import TeRedOperator, find_instances
from reduction import template_mining as tmining
from features.rate import node_feature_matrix, make_type_vocab


ALL_OPS = []


@pytest.fixture(scope="module")
def templates():
    """从若干"良性"随机星图里挖出的模板（中心 t0 -> 叶子 t1）。"""
    benign = [star_graph(f"b{i}", n_leaves=4, seed=i) for i in range(8)]
    benign += [star_graph(f"b{i}", n_leaves=5, seed=i + 100) for i in range(6)]
    tpls, info = tmining.mine_templates(benign, min_support=5, per_graph=10,
                                        max_tpl_nodes=12, seed=0, verbose=False)
    return tpls


def _check_all(G, op):
    res = op.reduce(G)          # 内部自动校验 INV-2/3/4（verify=True 默认）
    node_map_sanity(res, G)
    return res


# ---------------- INV-1 ----------------
def test_inv1_rate_equals_naive_when_mu_identity():
    """μ≡1（恒等/未归约）时，rate 与 dual_naive 逐位相等。"""
    for seed in range(5):
        G = random_graph(n=40, m=90, seed=seed)
        # 直接在图本身（无归约、边 mu 全 1）上比较
        vocab = make_type_vocab([G])
        X_rate, _, _ = node_feature_matrix(G, "rate", tape_dim=8, vocab=vocab)
        X_naive, _, _ = node_feature_matrix(G, "dual_naive", tape_dim=8, vocab=vocab)
        assert np.allclose(X_rate, X_naive, atol=1e-9), f"seed={seed}"
        # 恒等算子归约后再比
        Gp = IdentityOperator().reduce(G).Gp
        X_rate2, _, _ = node_feature_matrix(Gp, "rate", tape_dim=8, vocab=vocab)
        assert np.allclose(X_rate2, X_naive, atol=1e-9)


def test_inv1_reduced_mu_differs_from_naive():
    """归约后(μ>1 出现) naive 必须与 rate 不同 —— 负面结果的存在性前提。"""
    G = CanonicalGraph("g")
    for i in range(20):
        G.ensure_node(str(i), ntype=f"t{i % 3}")
    G.nodes["0"]["type"] = "proc"
    for i in range(1, 10):
        G.nodes[str(i)]["type"] = "file"
    for _ in range(10):
        G.add_edge("0", "1", "write")       # 10 条并行同型边 -> CPR 去重 μ=10
    res = CPROperator().reduce(G)
    assert res.Gp.n_edges() == 1 and res.Gp.edges[0]["mu"] == 10
    vocab = make_type_vocab([res.Gp])
    X_rate, _, _ = node_feature_matrix(res.Gp, "rate", tape_dim=8, vocab=vocab)
    X_naive, _, _ = node_feature_matrix(res.Gp, "dual_naive", tape_dim=8, vocab=vocab)
    assert not np.allclose(X_rate, X_naive)


# ---------------- INV-2/3/4 随机 + 手工 ----------------
@pytest.mark.parametrize("seed", range(6))
def test_inv234_random_cpr(seed):
    G = random_graph(n=35, m=70, seed=seed)
    _check_all(G, CPROperator())


@pytest.mark.parametrize("seed", range(6))
def test_inv234_random_nodemerge(seed):
    G = random_graph(n=35, m=80, seed=seed)
    _check_all(G, NodeMergeOperator())


def test_inv234_crafted_duplicates():
    """同一文件被同一进程重复读写多次 -> CPR μ 记账 + NodeMerge 邻域合并。"""
    G = CanonicalGraph("dup")
    G.ensure_node("p", ntype="process")
    for i in range(3):
        G.ensure_node(f"f{i}", ntype="file")
    # p 对每个文件 write 两次（并行边）
    for i in range(3):
        for _ in range(2):
            G.add_edge("p", f"f{i}", "write")
            G.add_edge(f"f{i}", "p", "read")
    for op in (CPROperator(), NodeMergeOperator()):
        res = _check_all(G, op)
        assert res.Gp.total_edge_mu() == pytest.approx(G.n_edges())


def test_inv234_tered_random_and_realish(templates):
    G = random_graph(n=30, m=60, seed=7)
    # 手工埋入 2 个良性星型模板实例 + 1 个"攻击"节点（唯一拓扑）
    for c in ("c1", "c2"):
        G.ensure_node(c, ntype="t0")
        for i in range(4):
            G.ensure_node(f"{c}l{i}", ntype="t1")
            G.add_edge(c, f"{c}l{i}", "e0")
    G.ensure_node("evil", ntype="t0")
    for i in range(3):
        G.ensure_node(f"ev{i}", ntype="t9")
        G.add_edge("evil", f"ev{i}", "e9")
    G.labels = {f"ev{i}": 1 for i in range(3)}
    op = TeRedOperator(templates, max_instances=100)
    res = op.reduce(G)
    node_map_sanity(res, G)
    assert res.stats["n_regions"] >= 1
    # INV-5：攻击节点全部存活（未被良性模板吸收）
    surv = sum(1 for v in G.labels if res.node_map.get(v) == v) / len(G.labels)
    assert surv == 1.0


def test_inv234_tered_overlap_nonoverlap(templates):
    """重叠区域：第二个实例若与首个重叠则跳过（贪心非重叠）。"""
    G = CanonicalGraph("ov")
    # 两个几乎重叠的模板实例：共享若干叶子
    G.ensure_node("c1", ntype="t0")
    G.ensure_node("c2", ntype="t0")
    for i in range(4):
        G.ensure_node(f"l{i}", ntype="t1")
        G.add_edge("c1", f"l{i}", "e0")
        if i < 2:                       # c2 共享 l0,l1
            G.add_edge("c2", f"l{i}", "e0")
    G.ensure_node("x", ntype="t1")
    G.add_edge("c2", "x", "e0")
    G.ensure_node("y", ntype="t1")
    G.add_edge("c2", "y", "e0")
    op = TeRedOperator(templates, max_instances=10)
    res = _check_all(G, op)
    # 至少一个区域被塌缩且不变量成立（重叠处理：塌缩区域互不相交）
    regs = find_instances(G, templates, max_instances=10)
    sets = [r["nodes"] for r in regs]
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            assert not (sets[i] & sets[j]), "区域重叠!"


# ---------------- 退化：无匹配模板时算子恒等 ----------------
def test_tered_no_match_identity(templates):
    G = CanonicalGraph("nomatch")
    G.ensure_node("a", ntype="qq")
    G.ensure_node("b", ntype="zz")
    G.add_edge("a", "b", "w")
    op = TeRedOperator(templates, max_instances=10)
    res = op.reduce(G)
    assert res.Gp.n_nodes() == G.n_nodes()
    assert res.Gp.n_edges() == G.n_edges()


# ---------------- INV-5 度量 ----------------
def test_inv5_metric_sensitivity(templates):
    G = random_graph(n=25, m=45, seed=3)
    G.ensure_node("atk1", ntype="t9")
    G.ensure_node("atk2", ntype="t9")
    G.add_edge("atk1", "atk2", "e9")
    G.labels = {"atk1": 1, "atk2": 1}
    res = TeRedOperator(templates, max_instances=10).reduce(G)
    from run_pipeline import attack_survival
    s = attack_survival(G, res)
    assert 0.0 <= s <= 1.0
    assert s == 1.0            # 攻击节点未匹配良性模板


def test_identity_operator_meta():
    G = random_graph(n=20, m=30, seed=1)
    res = IdentityOperator().reduce(G)
    assert res.Gp.n_nodes() == G.n_nodes()
    assert res.Gp.n_edges() == G.n_edges()
    assert res.Gp.total_edge_mu() == G.n_edges()
