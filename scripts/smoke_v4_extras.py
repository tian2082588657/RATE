# -*- coding: utf-8 -*-
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), *[".."] * 1))

"""本地单元冒烟：验证 v4extras + BenignEnsemble 端到端可跑通。

只测试：
  1. extend_with_v4 输出维度/类型正确
  2. BenignEnsemble 在 v4 特征下能 fit/predict/anomaly_scores
  3. 对合成数据：异常节点在 v4 特征下确实可被识别（oracle 检查）

不依赖真实 DARPA 数据。绿了之后再传服务器跑真数据。
"""
import sys
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = _R
sys.path.insert(0, ROOT)

from features.v4extras import extend_with_v4
from models.detector import BenignEnsemble


class FakeProvGraph:
    """最小化 ProvGraph 子集：只有 nodes / edges / degrees。"""
    def __init__(self, nodes, edges, degs):
        # nodes: dict uuid -> {"type": str}
        self.nodes = nodes
        # edges: list of {"src": uuid, "dst": uuid}
        self.edges = edges
        # degs: (din_dict, dout_dict)
        self._din, self._dout = degs

    def degrees(self, use_mu=True):
        return self._din, self._dout


def make_synthetic_graph():
    """造一个简单图：
      - 良性: 节点类型 PROCESS, 邻居多样（连接 FILE/NETFLOW）
      - 攻击: 节点类型 PROCESS 单一, 邻居全部 FILE, 频率异常高
    """
    rng = np.random.default_rng(0)
    n_benign = 50
    n_attack = 5

    nodes = {}
    edges = []

    # 良性节点：B0..B49
    for i in range(n_benign):
        nid = f"B{i}"
        # 良性节点偶有 FILE/NETFLOW 邻居
        if i % 3 == 0:
            t = "FILE_UNIX"
        elif i % 5 == 0:
            t = "NETFLOW"
        else:
            t = "PROCESS"
        nodes[nid] = {"type": t}
    # 良性边：连 2-3 个不同类型
    for i in range(n_benign):
        for _ in range(rng.integers(2, 4)):
            j = rng.integers(0, n_benign)
            if i != j:
                edges.append({"src": f"B{i}", "dst": f"B{j}"})

    # 攻击节点：A0..A4：高频率、邻居单一
    for i in range(n_attack):
        nid = f"A{i}"
        nodes[nid] = {"type": "PROCESS"}  # 单一类型
        # 大量自环 + 大量连接同一目标
        for _ in range(20):
            edges.append({"src": nid, "dst": nid})
        for k in range(n_benign):
            if k % 5 == 0:  # 只连 FILE
                for _ in range(5):
                    edges.append({"src": nid, "dst": f"B{k}"})

    nids = list(nodes.keys())
    # edges_index
    nid2idx = {nid: i for i, nid in enumerate(nids)}
    edges_idx = np.array(
        [[nid2idx[e["src"]], nid2idx[e["dst"]]] for e in edges],
        dtype=np.int64)

    # 度数
    din = {}
    dout = {}
    for nid in nids:
        din[nid] = 0.0
        dout[nid] = 0.0
    for e in edges:
        dout[e["src"]] += 1
        din[e["dst"]] += 1

    return FakeProvGraph(nodes, edges, (din, dout)), nids, edges_idx


def test_extend_dims():
    """v4extras 输出维度正确性。"""
    Gp, nids, edges = make_synthetic_graph()
    # 原始 X：10 维 (假设)
    d0 = 10
    X0 = np.random.RandomState(0).randn(len(nids), d0)
    X1, names = extend_with_v4(X0, nids, Gp, edges)
    assert X1.shape == (len(nids), d0 + 4), f"应为 ({len(nids)}, {d0+4}), 得 {X1.shape}"
    assert names == ["log_deg", "out_in_ratio", "nbr_type_div", "self_loop"]
    assert np.isfinite(X1).all(), "有 inf/nan"
    # 攻击节点 self_loop 应该 = 20（每节点 20 个自环）
    aid_idx = [i for i, nid in enumerate(nids) if nid.startswith("A")]
    self_loop_col = X1[:, d0 + 3]
    for i in aid_idx:
        assert self_loop_col[i] >= 20, \
            f"攻击节点 {nids[i]} 自环数={self_loop_col[i]} 应 >= 20"
    print(f"[ok] extend_with_v4 维度 {X1.shape} + 自环数验证通过")


def test_benign_v4_detects_attack():
    """关键 oracle 测试：训练只良性 → 攻击节点 anomaly_score 排名前列。"""
    Gp, nids, edges = make_synthetic_graph()
    d0 = 10
    rng = np.random.RandomState(0)
    X_all = rng.randn(len(nids), d0)

    X_v4, _ = extend_with_v4(X_all, nids, Gp, edges)

    # 拆：训练只用良性，攻击节点混入测试
    aid_idx = [i for i, nid in enumerate(nids) if nid.startswith("A")]
    bid_idx = [i for i, nid in enumerate(nids) if not nid.startswith("A")]

    X_train = X_v4[bid_idx]
    # 测试 = 全集（看攻击是否高排名）
    X_test = X_v4

    det = BenignEnsemble(radius_q=0.05).fit(
        [X_train], ["benign_only"])
    scores = det.anomaly_scores(X_test)
    # 攻击节点应排前 n_attack 位
    top_k = len(aid_idx)
    top_idx = np.argsort(-scores)[:top_k]
    hit = sum(1 for i in top_idx if i in aid_idx)
    print(f"[oracle] top-{top_k} 排名中攻击节点命中 {hit}/{top_k}")
    print(f"[oracle] 攻击节点 anomaly_score 范围: "
          f"min={scores[aid_idx].min():.4f} max={scores[aid_idx].max():.4f}")
    print(f"[oracle] 良性节点 anomaly_score 范围: "
          f"min={scores[bid_idx].min():.4f} max={scores[bid_idx].max():.4f}")
    return hit, scores, aid_idx, bid_idx


def test_benign_baseline_no_v4():
    """无 v4 特征：oracle 应失败（攻击节点应排在中间/底部）。"""
    Gp, nids, edges = make_synthetic_graph()
    d0 = 10
    rng = np.random.RandomState(0)
    X_all = rng.randn(len(nids), d0)

    aid_idx = [i for i, nid in enumerate(nids) if nid.startswith("A")]
    bid_idx = [i for i, nid in enumerate(nids) if not nid.startswith("A")]

    X_train = X_all[bid_idx]
    X_test = X_all

    det = BenignEnsemble(radius_q=0.05).fit(
        [X_train], ["benign_only"])
    scores = det.anomaly_scores(X_test)
    top_k = len(aid_idx)
    top_idx = np.argsort(-scores)[:top_k]
    hit = sum(1 for i in top_idx if i in aid_idx)
    print(f"[baseline] top-{top_k} 排名中攻击节点命中 {hit}/{top_k}（预期低）")
    return hit


if __name__ == "__main__":
    test_extend_dims()
    print()
    hit_v4, _, _, _ = test_benign_v4_detects_attack()
    print()
    hit_base = test_benign_baseline_no_v4()

    # oracle 断言：v4 命中率严格 > baseline
    assert hit_v4 > hit_base, \
        f"v4 命中率 {hit_v4} 应 > baseline {hit_base}"
    print(f"\n[OK] v4 相对 baseline 攻击命中提升 +{hit_v4-hit_base}")