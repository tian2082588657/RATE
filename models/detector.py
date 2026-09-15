# -*- coding: utf-8 -*-
"""models/detector.py — 良性单类识别器集成（Presumption of Innocence 推理）。

对齐指南 §2.2 的最后两环：
    良性单类训练(只用良性子图) -> 残差驱动多识别器集成 -> 推理：任一识别器判良 -> 良；
    全部判异 -> 告警。

当前实现（MVP，可跑可出真实数字）：
    每个良性组（StreamSpot 的场景 / DARPA 的时间窗子图）训练一个"良性识别器"=
    该组良性节点特征的中心向量 + 组内 5% 分位相似度半径。
    推理：test 节点特征与任一识别器的余弦相似度 ≥ 半径 -> 判良；全部低于 -> 告警。
    异常分 score = 1 - max_sim。

注意：这是占位实现，论文级精度需要残差集成迭代 + GraphSAGE 语义特征（预留接口 fit_residual）。
"""
from __future__ import annotations
import numpy as np


def _l2norm(M):
    n = np.linalg.norm(M, axis=1, keepdims=True)
    return M / np.maximum(n, 1e-12)


class BenignEnsemble:
    def __init__(self, radius_q=0.05, min_radius=0.50):
        self.centroids = []      # 每识别器单位中心
        self.radii = []          # 每识别器半径(相似度下界)
        self.group_names = []
        self.radius_q = radius_q
        self.min_radius = min_radius

    def fit(self, X_list, group_names=None):
        """X_list: 每个良性组一个 (n_i, d) 特征矩阵；只喂良性数据。"""
        self.centroids, self.radii, self.group_names = [], [], []
        for gi, X in enumerate(X_list):
            X = np.asarray(X, dtype=np.float64)
            if X.shape[0] == 0:
                continue
            Xn = _l2norm(X)
            c = Xn.mean(axis=0)
            cn = c / max(np.linalg.norm(c), 1e-12)
            sims = Xn @ cn
            r = float(np.percentile(sims, self.radius_q * 100)) if len(sims) > 1 \
                else float(sims[0])
            self.centroids.append(cn)
            self.radii.append(max(r, self.min_radius))
            self.group_names.append(str(group_names[gi]) if group_names else f"g{gi}")
        return self

    # ---- 残差专家接口（预留：挖误判良性样本 -> 训练专家模型，迭代）----
    def fit_residual(self, X_fp_benign_like, X_fp_attack_like, **kw):
        """MVP 占位：把误判严重的节点再聚一个识别器。论文阶段扩展。"""
        if len(X_fp_benign_like):
            self.fit([np.asarray(X_fp_benign_like, dtype=np.float64)],
                     group_names=["residual"])
        return self

    def anomaly_scores(self, X):
        Xn = _l2norm(np.asarray(X, dtype=np.float64))
        sim = Xn @ np.array(self.centroids).T      # (n, n_id)
        maxsim = sim.max(axis=1) if sim.ndim == 2 else sim
        return 1.0 - maxsim

    def predict(self, X):
        """True=良性（任一识别器判良）；False=告警。"""
        Xn = _l2norm(np.asarray(X, dtype=np.float64))
        sim = Xn @ np.array(self.centroids).T
        if sim.ndim == 2:
            ok = (sim >= np.array(self.radii)[None, :]).any(axis=1)
        else:
            ok = sim >= self.radii[0]
        return ok

    def state_dict(self):
        return {"centroids": [c.tolist() for c in self.centroids],
                "radii": self.radii, "group_names": self.group_names}

    def load_state_dict(self, d):
        self.centroids = [np.array(c) for c in d["centroids"]]
        self.radii = d["radii"]
        self.group_names = d["group_names"]
        return self
