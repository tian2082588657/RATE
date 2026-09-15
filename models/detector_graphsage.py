# -*- coding: utf-8 -*-
"""models/detector_graphsage.py — GraphSAGE 自监督节点异常检测器（轻量版）。

对齐指南 §2.2 占位 MVP（BenignEnsemble）的语义升级：
    2 层 GraphSAGE (mean 聚合) → 节点嵌入 → 两种评分模式。
    训练：只在良性图上，真边 (u,v) → sigmoid(sim)→1，随机负对 (u,·) → 0。

评分模式（score_mode）：
    recon  v2：异常分 = 1 - mean(sigmoid(sim)) 邻域重建误差。
           已证在全量跨图场景失效（图级分布偏移淹没节点级信号，
           攻击节点挂高频枢纽反而重建得好 → ROC<0.5），保留用于对照。
    mahal  v3：异常分 = 节点嵌入到良性嵌入流形的（平方）Mahalanobis 距离。
           训练后用最终模型对全训练图前向收集嵌入 → mu + 收缩协方差逆。
           对图级偏移的鲁棒性好于重建误差（距离有方向性，且可归一化）。

评分归一化（score_norm）：
    zscore  每图内用鲁棒统计 (median, IQR) 做标准化后再与训练阈值比较。
            动机：攻击图整体相对训练图是 OOD，绝对分数不可跨图比较；
            图内标准化消掉图级平移，让训练分位阈值可迁移。

设计要点（v2, 2026-09-05）：纯正对 BCE 会让嵌入塌缩（所有内积→1，
训练异常分全 0，阈值失效）。引入负采样后模型必须区分真边/假边，
嵌入被分散开，训练异常分呈非退化分布，分位阈值恢复区分力。

接口兼容 BenignEnsemble（同 .fit / .predict / .anomaly_scores / .state_dict / .load_state_dict）。

差异：fit / anomaly_scores / predict 多接一个 edges（ndarray (E, 2), 节点索引空间）参数。
"""
from __future__ import annotations
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import SAGEConv
    from torch_geometric.data import Data
    _HAS_PYG = True
except ImportError:           # pragma: no cover
    _HAS_PYG = False


def _build_adj(edge_index_np, num_nodes):
    """edge_index_np: (2, E) ndarray; 返回 adj: list of lists."""
    adj = [[] for _ in range(num_nodes)]
    for s, d in zip(edge_index_np[0].tolist(), edge_index_np[1].tolist()):
        adj[s].append(d)
    return adj


def _sample_neighbors(adj, n, k, rng):
    """为每个节点采样 k 个邻居（不足则重采样放回；孤立节点自环）。"""
    rows = np.empty(n * k, dtype=np.int64)
    cols = np.empty(n * k, dtype=np.int64)
    for i in range(n):
        nbrs = adj[i]
        if not nbrs:
            # 自环
            sampled = np.full(k, i, dtype=np.int64)
        elif len(nbrs) >= k:
            idx = rng.choice(len(nbrs), size=k, replace=False)
            sampled = np.array([nbrs[j] for j in idx], dtype=np.int64)
        else:
            idx = rng.integers(0, len(nbrs), size=k)
            sampled = np.array([nbrs[j] for j in idx], dtype=np.int64)
        rows[i * k:(i + 1) * k] = i
        cols[i * k:(i + 1) * k] = sampled
    return rows, cols


class _SAGE(nn.Module):
    def __init__(self, d_in, hidden=64):
        super().__init__()
        self.conv1 = SAGEConv(d_in, hidden, aggr="mean")
        self.conv2 = SAGEConv(hidden, hidden, aggr="mean")

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = self.conv2(h, edge_index)
        return h


class GraphSAGEDetector:
    """GraphSAGE 自监督节点异常检测器（GPU 优先，PyG 不可用时退化为特征均值）。"""

    def __init__(self, hidden=64, n_neighbors=25, lr=0.01, epochs=20,
                 radius_q=0.99, neg_ratio=1.0, score_mode="mahal",
                 score_norm="zscore", shrink=0.1, seed=0, device=None):
        # radius_q 默认 0.99：负采样打破嵌入塌缩后，训练异常分为非退化
        # 分布，高分位分位阈值即有区分力（v1 纯正对训练时曾全 0 失效）。
        # score_mode 默认 mahal（v3）：recon 已在全量场景证伪，仅留对照。
        # score_norm 默认 zscore（v3）：图内鲁棒标准化，训练阈值可跨图迁移。
        self.hidden = hidden
        self.k = n_neighbors
        self.lr = lr
        self.epochs = epochs
        self.radius_q = radius_q
        self.neg_ratio = neg_ratio
        self.score_mode = score_mode
        self.score_norm = score_norm
        self.shrink = shrink
        self.seed = seed
        self.device = device or ("cuda" if (_HAS_PYG and torch.cuda.is_available())
                                 else "cpu" if _HAS_PYG else "cpu")
        self.model = None
        self.threshold = 0.0
        self.group_names = []
        self.d_in = None
        # mahal 模式状态
        self.mu = None       # (d,) 均值
        self.Cinv = None     # (d, d) 收缩协方差逆

    # ---------- 私有工具 ----------
    def _make_data(self, X, edges):
        """X: (n, d), edges: (E, 2) ndarray 索引空间。返回 PyG Data + adj + n。"""
        x = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        if edges.shape[0] == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=self.device)
        else:
            ei = np.concatenate([edges, edges[:, [1, 0]]], axis=0)
            ei = np.unique(ei, axis=0)
            edge_index = torch.as_tensor(ei.T, dtype=torch.long, device=self.device)
        data = Data(x=x, edge_index=edge_index)
        adj = _build_adj(edge_index.cpu().numpy(), X.shape[0])
        return data, adj, X.shape[0]

    def _forward(self, data):
        return self.model(data.x, data.edge_index)

    def _per_node_score(self, h, adj, n, rng):
        """对每节点采样 k 邻居 → 计算 -mean(sigmoid(sim)) 当异常分。返回 (n,)。"""
        if n * self.k == 0:
            return np.zeros(n, dtype=np.float64)
        rows, cols = _sample_neighbors(adj, n, self.k, rng)
        rows_t = torch.as_tensor(rows, dtype=torch.long, device=self.device)
        cols_t = torch.as_tensor(cols, dtype=torch.long, device=self.device)
        sim = (h[rows_t] * h[cols_t]).sum(dim=-1)
        # 异常分：sigmoid(sim) 低 = 邻居距离 = 异常
        s = torch.sigmoid(sim).cpu().numpy()
        per = np.zeros(n, dtype=np.float64)
        cnt = np.zeros(n, dtype=np.int64)
        np.add.at(per, rows, 1.0 - s)        # 1 - sigmoid = anomaly-ish
        np.add.at(cnt, rows, 1)
        cnt = np.maximum(cnt, 1)
        return per / cnt

    @staticmethod
    def _robust_z(s):
        """图内鲁棒标准化：(s - median) / (IQR + eps)。消除图级平移/尺度。"""
        med = float(np.median(s))
        q75, q25 = np.percentile(s, [75, 25])
        iqr = max(float(q75 - q25), 1e-9)
        return (s - med) / iqr

    def _mahal_sq(self, h):
        """平方 Mahalanobis 距离 (n,)：嵌入到良性流形的距离，越大越异常。"""
        diff = h - self.mu
        return ((diff @ self.Cinv) * diff).sum(dim=-1)

    # ---------- 训练 ----------
    def fit(self, X_list, edges_list, group_names=None):
        assert len(X_list) == len(edges_list)
        if not _HAS_PYG:
            raise RuntimeError("PyG 未安装, GraphSAGEDetector 不可用")
        rng = np.random.default_rng(self.seed)
        d_in = X_list[0].shape[1]
        self.d_in = d_in
        self.model = _SAGE(d_in, self.hidden).to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # ---- 阶段 1: 顺序训练（负采样链接预测，打破嵌入塌缩） ----
        fitted = []            # (data, adj, n) 供阶段 2 统一评分
        for gi, (X, edges) in enumerate(zip(X_list, edges_list)):
            n = X.shape[0]
            if n < 2:
                self.group_names.append(str(group_names[gi]) if group_names else f"g{gi}")
                continue
            data, adj, n = self._make_data(X, edges)
            n_neg = max(1, int(self.k * self.neg_ratio))
            self.model.train()
            for ep in range(self.epochs):
                opt.zero_grad()
                h = self._forward(data)
                rows, cols = _sample_neighbors(adj, n, self.k, rng)
                neg_rows = np.repeat(np.arange(n, dtype=np.int64), n_neg)
                neg_cols = rng.integers(0, n, size=n * n_neg, dtype=np.int64)
                rows_t = torch.as_tensor(np.concatenate([rows, neg_rows]),
                                          dtype=torch.long, device=self.device)
                cols_t = torch.as_tensor(np.concatenate([cols, neg_cols]),
                                         dtype=torch.long, device=self.device)
                labels = torch.cat([
                    torch.ones(len(rows), device=self.device),
                    torch.zeros(len(neg_rows), device=self.device)])
                logits = (h[rows_t] * h[cols_t]).sum(dim=-1)
                loss = F.binary_cross_entropy_with_logits(logits, labels)
                loss.backward()
                opt.step()
            fitted.append((data, adj, n))
            self.group_names.append(str(group_names[gi]) if group_names else f"g{gi}")

        self.model.eval()

        # ---- 阶段 2: 用最终模型统一评分（训练分布校准） ----
        all_train_scores = []
        if self.score_mode == "mahal":
            H_all = []
            for data, adj, n in fitted:
                with torch.no_grad():
                    H_all.append(self._forward(data).cpu().numpy())
            H = np.vstack(H_all)
            mu = H.mean(axis=0)
            C = np.cov(H.T)
            d = H.shape[1]
            # 收缩协方差：抗小样本/病态；加 eps 防奇异
            C = (1 - self.shrink) * C + self.shrink * np.diag(np.diag(C))
            C += 1e-6 * np.eye(d)
            self.mu = torch.as_tensor(mu, dtype=torch.float32, device=self.device)
            self.Cinv = torch.as_tensor(np.linalg.inv(C), dtype=torch.float32,
                                       device=self.device)
            for data, adj, n in fitted:
                with torch.no_grad():
                    s = self._mahal_sq(self._forward(data)).cpu().numpy()
                if self.score_norm == "zscore":
                    s = self._robust_z(s)
                all_train_scores.extend(s.tolist())
            print(f"[gs-detector] mahal: N={H.shape[0]} d={d} "
                  f"shrink={self.shrink}", flush=True)
        else:  # recon（v2 对照）
            for data, adj, n in fitted:
                with torch.no_grad():
                    h = self._forward(data)
                    s = self._per_node_score(h, adj, n, rng)
                if self.score_norm == "zscore":
                    s = self._robust_z(s)
                all_train_scores.extend(s.tolist())

        if all_train_scores:
            self.threshold = float(np.percentile(all_train_scores,
                                                self.radius_q * 100))
            arr = np.asarray(all_train_scores)
            mu_t, sd = float(arr.mean()), float(arr.std())
            print(f"[gs-detector] train 异常分({self.score_mode}"
                  f"{'+z' if self.score_norm == 'zscore' else ''}) "
                  f"min={arr.min():.4f} mu={mu_t:.4f} sigma={sd:.4f} "
                  f"max={arr.max():.4f} mu+3s={mu_t + 3 * sd:.4f} "
                  f"-> threshold={self.threshold:.4f} (q={self.radius_q})",
                  flush=True)
        return self

    # ---------- 推理 ----------
    def anomaly_scores(self, X, edges):
        """返回 (n,) 异常分, 越大越异常。"""
        if not _HAS_PYG or self.model is None:
            return np.zeros(X.shape[0], dtype=np.float64)
        rng = np.random.default_rng(self.seed + 1)
        data, adj, n = self._make_data(X, edges)
        self.model.eval()
        with torch.no_grad():
            h = self._forward(data)
            if self.score_mode == "mahal":
                if self.mu is None or self.Cinv is None:
                    raise RuntimeError("mahal 模式需先 fit（mu/Cinv 未初始化）")
                s = self._mahal_sq(h).cpu().numpy()
            else:
                s = self._per_node_score(h, adj, n, rng)
        if self.score_norm == "zscore":
            s = self._robust_z(s)
        return s

    def predict(self, X, edges):
        """True = 良性（异常分 ≤ 阈值）, False=告警。"""
        s = self.anomaly_scores(X, edges)
        return s <= self.threshold

    # ---------- 持久化（最小） ----------
    def state_dict(self):
        d = {"threshold": self.threshold, "d_in": self.d_in,
             "group_names": self.group_names,
             "score_mode": self.score_mode, "score_norm": self.score_norm}
        if self.mu is not None:
            d["mu"] = self.mu.cpu().numpy()
            d["Cinv"] = self.Cinv.cpu().numpy()
        return d

    def load_state_dict(self, d):
        self.threshold = d["threshold"]
        self.d_in = d["d_in"]
        self.group_names = d.get("group_names", [])
        self.score_mode = d.get("score_mode", "recon")
        self.score_norm = d.get("score_norm", "none")
        if _HAS_PYG and self.d_in is not None:
            self.model = _SAGE(self.d_in, self.hidden).to(self.device)
        if d.get("mu") is not None and _HAS_PYG:
            self.mu = torch.as_tensor(d["mu"], dtype=torch.float32,
                                      device=self.device)
            self.Cinv = torch.as_tensor(d["Cinv"], dtype=torch.float32,
                                        device=self.device)
        return self