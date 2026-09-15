# -*- coding: utf-8 -*-
"""features/v4extras.py — v4 节点特征扩展（频次 / log度数 / 邻居多样性）。

动机：v3 GraphSAGE 全量节点级 F1 < 0.04。BenignEnsemble 余弦质心对
"频率型"信号不敏感（tape 编码后频次信息被压缩成相位），而 DARPA
攻击节点的真实区分信号往往在结构层（度数量级、邻居单一性、自环）。
v4 直接补这 4 列：log_deg / out_in_ratio / nbr_type_div / self_loop。

接入：node_feature_matrix(...) → extend_with_v4(X, nids, Gp, edges_idx) →
新 (X_v4, names_v4)。归约产物不变，缓存照旧命中。
"""
from __future__ import annotations
import numpy as np


def extend_with_v4(X, nids, Gp, edges_idx, scale="raw"):
    """在已有 X 上追加 4 列（log_deg / out_in_ratio / nbr_type_div /
    self_loop），返回 (X_new, names_new)。

    X:    (n, d) ndarray — 已由 node_feature_matrix 生成
    nids: list of node UUID（与 X 行序一致）
    Gp:   ProvGraph（提供 nodes[uid]['type'] 与 degrees）
    edges_idx: (E, 2) ndarray 节点索引；None 则邻居相关列填 0
    scale: "raw" 原值；"robust" 图内 robust z-score（median/IQR, 防零除加 eps）

    v4b 动机：raw 形式下 out_in_ratio 范围 0~1000+ 与 log_deg (0~10) 尺度
    差两个数量级，余弦相似度被 out_in_ratio 主导。robust z-score 把每列映射到
    (s - median)/IQR，与 baseline 特征量级对齐，FP 漂移可控。
    """
    X = np.asarray(X, dtype=np.float64)
    n = len(nids)
    names_new = ["log_deg", "out_in_ratio", "nbr_type_div", "self_loop"]
    if n == 0:
        return X, names_new

    # ---- 1. 度数统计（μ-加权）----
    din, dout = Gp.degrees(use_mu=True)
    din_arr = np.array([din.get(nid, 0.0) for nid in nids])
    dout_arr = np.array([dout.get(nid, 0.0) for nid in nids])
    deg_total = din_arr + dout_arr

    # log_deg：log(1 + total_deg) — 频次信号
    log_deg = np.log1p(np.maximum(deg_total, 0.0))

    # out_in_ratio：out / (in + 1) — 攻击节点常 src-heavy
    out_in_ratio = dout_arr / (din_arr + 1.0)

    # ---- 2. 邻居类型多样性 ----
    type_set_per_node = [set() for _ in range(n)]
    self_loop = np.zeros(n, dtype=np.float64)
    if edges_idx is not None and len(edges_idx) > 0:
        # 本图节点类型 vocab
        all_types = set()
        for nid in nids:
            all_types.add(Gp.nodes[nid].get("type", "unknown"))
        vocab_size = max(len(all_types), 1)

        for s, d in edges_idx:
            s, d = int(s), int(d)
            if 0 <= s < n:
                type_set_per_node[s].add(
                    Gp.nodes[nids[s]].get("type", "unknown"))
                if s == d:
                    self_loop[s] += 1.0
            if 0 <= d < n and s != d:
                type_set_per_node[d].add(
                    Gp.nodes[nids[d]].get("type", "unknown"))

        # 归一化到 [0, 1]：多样性 / vocab_size
        nbr_type_div = np.array([len(s) / vocab_size
                                  for s in type_set_per_node])
    else:
        nbr_type_div = np.zeros(n, dtype=np.float64)

    cols = [log_deg, out_in_ratio, nbr_type_div, self_loop]

    # ---- 3. 可选 robust z-score 缩放（图内，避免量级失衡）----
    if scale == "robust":
        scaled_cols = []
        for c in cols:
            med = float(np.median(c))
            iqr = float(np.percentile(c, 75) - np.percentile(c, 25))
            if iqr < 1e-9:
                iqr = 1.0
            scaled_cols.append((c - med) / iqr)
        cols = scaled_cols

    X_new = np.concatenate([X] + [c.reshape(-1, 1) for c in cols], axis=1)
    return X_new, names_new