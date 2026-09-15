# -*- coding: utf-8 -*-
"""features/rate.py — 节点特征矩阵：RATE / dual_naive / rate_single / none × 语义。

编码开关（指南 §5.1 消融表）：
    none            无拓扑特征，纯语义
    dual_naive      双通道、按边条数、无 μ   <- naive 基线（归约后应掉点）
    rate_single     Σμ 不分入/出 单通道      <- 双通道必要性
    rate            双通道 + μ 完整 RATE      <- 主方法（归约后不掉点）

μ 抢救系列（2026-09-11 新增，见 消融实验设计_2026-09-11.md §A5b）：
    诊断：rate = tape(Σμ) 与 dual_naive = tape(count) 在 96.4% 节点上逐位相同，
          余下 3.6% 被少数超级边把 Σμ 拉到 1e6，在 base=1e4 下 tape 高频通道
          角度≈0（失效），仅剩 i=0 通道周期混叠 —— μ 是被淹没，而非无用。
    rate_ratio       双通道计数 tape + 双通道 μ 比值 tape
                     (ratio = Σμ / count ≥ 1，即"平均折叠度")
                     把 μ 从"替代计数的绝对量"改成"正交增量通道"。
    rate_ratio_only  仅双通道 μ 比值 tape（隔离 μ 纯信号，无计数通道）
    rate_rank        Σμ 图内 rank 归一化到 [0,1] 后双通道 tape
                     （抗重尾，消除大取值导致的三角混叠）

μ≡1 时 rate 与 dual_naive 逐位相等（INV-1），由 tests 保证。
"""
from __future__ import annotations
import numpy as np

from features.tape import tape_embedding

MU_ENCODINGS = ("rate_ratio", "rate_ratio_only", "rate_rank")


def make_type_vocab(graphs):
    """跨图一致的节点类型字典（语义 one-hot 需要）。"""
    seen = []
    for g in graphs:
        for n in g.nodes.values():
            t = n.get("type", "unknown")
            if t not in seen:
                seen.append(t)
    return {t: i for i, t in enumerate(sorted(seen))}


def _rank01(a):
    """图内 rank 归一化到 [0,1]（平均秩，抗重尾，与取值分布无关）。"""
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    n = len(a)
    if n <= 1:
        return np.zeros(n, dtype=np.float64)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(n, dtype=np.float64)
    return ranks / (n - 1)


def node_feature_matrix(G, encoding="rate", tape_dim=8, tape_base=10000.0,
                        semantic="onehot", vocab=None, use_mu=True):
    """计算 G 的节点特征。

    use_mu=False 时即使 encoding='rate' 也按边条数算（用于跨编码一致性检查）；
    正常调用保持 True。返回 (X, node_ids, names)。
    """
    node_ids = list(G.nodes.keys())
    if vocab is None:
        vocab = make_type_vocab([G])
    parts = []
    names = []
    if semantic == "onehot":
        Xs = np.zeros((len(node_ids), len(vocab)), dtype=np.float64)
        for r, nid in enumerate(node_ids):
            t = G.nodes[nid].get("type", "unknown")
            if t in vocab:
                Xs[r, vocab[t]] = 1.0
        parts.append(Xs)
        names.append(f"semantic_onehot{len(vocab)}")

    if encoding != "none":
        if encoding in MU_ENCODINGS:
            # ---- μ 抢救系列：显式分离「计数拓扑」与「折叠比值」信号 ----
            din_c, dout_c = G.degrees(use_mu=False)   # 边条数
            din_m, dout_m = G.degrees(use_mu=True)    # Σμ 质量度
            arr_in_c = np.array([din_c[n] for n in node_ids])
            arr_out_c = np.array([dout_c[n] for n in node_ids])
            arr_in_m = np.array([din_m[n] for n in node_ids])
            arr_out_m = np.array([dout_m[n] for n in node_ids])
            if encoding in ("rate_ratio", "rate_ratio_only"):
                ratio_in = arr_in_m / np.maximum(arr_in_c, 1.0)
                ratio_out = arr_out_m / np.maximum(arr_out_c, 1.0)
                sub = []
                if encoding == "rate_ratio":
                    sub.append(tape_embedding(arr_in_c, tape_dim, tape_base))
                    sub.append(tape_embedding(arr_out_c, tape_dim, tape_base))
                sub.append(tape_embedding(ratio_in, tape_dim, tape_base))
                sub.append(tape_embedding(ratio_out, tape_dim, tape_base))
                p = np.concatenate(sub, axis=1)
            else:  # rate_rank
                p = np.concatenate([tape_embedding(_rank01(arr_in_m), tape_dim, tape_base),
                                    tape_embedding(_rank01(arr_out_m), tape_dim, tape_base)],
                                   axis=1)
            names.append(f"{encoding}_d{tape_dim}b{tape_base}")
        else:
            din, dout = G.degrees(use_mu=use_mu and encoding != "dual_naive")
            arr_in = np.array([din[n] for n in node_ids])
            arr_out = np.array([dout[n] for n in node_ids])
            if encoding == "rate":
                p = np.concatenate([tape_embedding(arr_in, tape_dim, tape_base),
                                    tape_embedding(arr_out, tape_dim, tape_base)], axis=1)
                names.append(f"rate_d{tape_dim}b{tape_base}")
            elif encoding == "dual_naive":
                p = np.concatenate([tape_embedding(arr_in, tape_dim, tape_base),
                                    tape_embedding(arr_out, tape_dim, tape_base)], axis=1)
                names.append(f"dual_naive_d{tape_dim}b{tape_base}")
            elif encoding == "rate_single":
                p = tape_embedding(arr_in + arr_out, tape_dim, tape_base)
                names.append(f"rate_single_d{tape_dim}b{tape_base}")
            else:
                raise ValueError(f"未知编码: {encoding}")
        parts.append(p)

    X = np.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]
    return X, node_ids, "+".join(names)


def label_vector(G, node_ids):
    return np.array([G.labels.get(n, 0) for n in node_ids], dtype=np.int64)
