# -*- coding: utf-8 -*-
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), *[".."] * 1))

"""C 阶段：告警级口径评估（无需 GT 的本地原型）。
- 输入：v4 raw 异常分 + 节点 + 边索引
- 流程：top-K 异常节点 → 连通子图 → 子图聚合异常分 → 子图级告警
- 评估：告警覆盖率 vs 误报率（用攻击图已知 GT 节点数作代理）

本机冒烟：bin.1 训练（良性），bin.118 测试（攻击图，含已知 GT）。
如果攻击图能产生告警、良性图告警数为 0 → C 阶段评估可行。

不依赖服务器，GT 用一个常驻 csv（如果本地有）或解析日志自标。
"""
import sys
import os
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = _R
sys.path.insert(0, ROOT)

from scripts.e5_f1_eval import parse_one, _edges_index
from reduction.base import IdentityOperator
from features.rate import node_feature_matrix, make_type_vocab, label_vector
from features.v4extras import extend_with_v4
from models.detector import BenignEnsemble

DATA_DIR = _os.path.join(_os.path.dirname(_R), "dataset", "darpa e5", "cadets")
MAX_TR = 15000
MAX_TE = 80000


def fit_v4(G_train, G_test):
    """训练 BenignEnsemble on v4 raw 特征。"""
    vocab = make_type_vocab([G_train, G_test])  # 全局 vocab 防维度不一致
    Xt, nids_t, _ = node_feature_matrix(G_train, encoding="rate",
                                          tape_dim=8, tape_base=10000.0,
                                          semantic="onehot", vocab=vocab)
    et = _edges_index(G_train, nids_t)
    Xt, _ = extend_with_v4(Xt, nids_t, G_train, et, scale="raw")
    det = BenignEnsemble(radius_q=0.05).fit([Xt], ["tr0"])
    return det, vocab


def alert_pipeline(G_test, det, vocab, top_k=50, max_alerts=5, min_cluster_size=3):
    """C 阶段告警级：
       1. 算异常分
       2. 选 top_k 个异常节点
       3. BFS 扩展连通子图（保留 min_cluster_size 阈值）
       4. 每个子图算聚合异常分（max + mean）
       5. 取聚合分 top max_alerts 子图作为告警
       返回 alerts: [(subgraph_nodes, agg_score), ...]
    """
    X, nids, _ = node_feature_matrix(G_test, encoding="rate",
                                      tape_dim=8, tape_base=10000.0,
                                      semantic="onehot", vocab=vocab)
    e = _edges_index(G_test, nids)
    X, _ = extend_with_v4(X, nids, G_test, e, scale="raw")

    score = det.anomaly_scores(X)
    n = len(nids)

    # 1. top_k 异常节点
    order = np.argsort(-score)
    top_idx = order[:top_k]
    top_score = score[top_idx]

    # 2. 建邻接
    adj = [[] for _ in range(n)]
    for s, d in e:
        s, d = int(s), int(d)
        if 0 <= s < n:
            adj[s].append(d)
        if 0 <= d < n and s != d:
            adj[d].append(s)

    # 3. 对每个 top 节点 BFS 找连通子图
    visited = set()
    clusters = []
    for seed in top_idx:
        if int(seed) in visited:
            continue
        cluster = []
        stack = [int(seed)]
        while stack:
            u = stack.pop()
            if u in visited:
                continue
            visited.add(u)
            cluster.append(u)
            for v in adj[u]:
                if v not in visited and score[v] >= np.percentile(score, 50):
                    stack.append(v)
        if len(cluster) >= min_cluster_size:
            agg = float(np.max([score[u] for u in cluster]))
            clusters.append((cluster, agg))
    clusters.sort(key=lambda x: -x[1])

    alerts = clusters[:max_alerts]
    return alerts, score, nids


def main():
    print("[C 阶段] 训练 + 三场景对比（bin.118 / bin.2 / bin.116）")
    t0 = time.time()
    Gt, _ = parse_one(DATA_DIR, "bin.1", MAX_TR)
    print(f"[parse] train bin.1 = {Gt.n_nodes()} 节点 ({time.time()-t0:.0f}s)")

    for atk_file in ["bin.118", "bin.2", "bin.116"]:
        t1 = time.time()
        try:
            G, _ = parse_one(DATA_DIR, atk_file, MAX_TE)
        except Exception as e:
            print(f"  [{atk_file}] 解析失败: {e}")
            continue
        n_nodes = G.n_nodes()
        det, vocab = fit_v4(Gt, G)
        alerts, score, nids = alert_pipeline(
            G, det, vocab, top_k=50, max_alerts=5, min_cluster_size=3)

        # 告警覆盖率（占节点比例）
        alerted_nodes = sum(len(c) for c, _ in alerts)
        coverage = alerted_nodes / n_nodes if n_nodes > 0 else 0
        agg_scores = [a for _, a in alerts]

        print(f"\n  [{atk_file}] n={n_nodes} ({time.time()-t1:.0f}s)")
        print(f"    score max={score.max():.3f} p99={np.percentile(score, 99):.3f} "
              f"p95={np.percentile(score, 95):.3f}")
        print(f"    alerts: {len(alerts)} 个, 覆盖 {alerted_nodes} 节点 "
              f"({100*coverage:.1f}% of {n_nodes})")
        for i, (cl, agg) in enumerate(alerts):
            print(f"      alert#{i+1}: |cluster|={len(cl)} agg={agg:.3f}")
        if alerts:
            print(f"    top alert agg_score={agg_scores[0]:.3f}")


if __name__ == "__main__":
    main()