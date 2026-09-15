# -*- coding: utf-8 -*-
"""scripts_ablation/e6e_gnn_baseline.py — 消息传递 GNN 检测器上的编码消融（救援实验）。

动机（DeepSeek 0912-2 第 1/3 条）：论文动机是 topology-aware GNN，主实验却是
label-free cosine one-class 检测器。本脚本在同一批 reduced graph、同一 ground
truth、同一告警预算下，跑一个**消息传递 GraphSAGE** 无监督检测器，回答：
   (a) 归约是否破坏 GNN 依赖的结构信号？
   (b) 拓扑编码（双通道计数 / Σμ / collapse-ratio）能否为 GNN 恢复？
   (c) RATE* 是否优于 reduction-blind 的边计数编码？

同时（第 2 条）对 cosine 检测器补**节点级阈值无关指标**（ROC-AUC / PR-AUC /
best-F1 / 固定操作点 FPR / precision / recall），给出可比较的误报口径；并跑
**剥离 ground-truth 正例后的良性训练变体**，验证结论不依赖被污染的训练集。

GNN 设计：2 层 mean 聚合 SAGE，链接预测自监督（正边 + 负采样）训练，
Mahalanobis 距离打分（对图级分布偏移稳健），图内鲁棒 z-score 归一化。
训练图与测试文件无关 → 每个 (config, encoding) 只训练一次，对 7 个分区推理。

用法:
  python scripts_ablation/e6e_gnn_baseline.py \
      --detectors cosine,gnn --configs identity:rate,tered:dual_naive,tered:rate,tered:rate_ratio \
      --results results/e6e_gnn_baseline.csv
"""
from __future__ import annotations
import os, sys, time, argparse, gc

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import numpy as np

from rate_core import load_pickle
from features.rate import node_feature_matrix, make_type_vocab, label_vector
from features.v4extras import extend_with_v4
from models.detector import BenignEnsemble
from eval.metrics import append_csv, binary_metrics

from scripts.e5_f1_eval import _edges_index, resolve_name, find_templates, load_gt
from scripts.e6_alert_eval import build_adj, parse_spec
from scripts.e6b_alert_replay import load_res

TAU = "TeRed+RATE*"          # 论文报告的方法口径
BUDGETS = [5, 10, 20]


# ------------------------------------------------------------------ 告警管线
def alert_pipeline2(score, edges, n, top_k=100, bfs_q=75, min_cluster=3,
                    max_cluster=200, max_alerts=20, wmax=1.0, sort_mode="cmax"):
    """与 e6d_ablation_replay.alert_pipeline2 完全一致（base 口径）。"""
    order = np.argsort(-score, kind="mergesort")
    top_idx = order[:top_k]
    thresh = float(np.percentile(score, bfs_q))
    adj = build_adj(n, edges)
    visited, clusters = set(), []
    for seed in top_idx:
        seed = int(seed)
        if seed in visited:
            continue
        cluster, stack = [], [seed]
        while stack and len(cluster) < max_cluster:
            u = stack.pop()
            if u in visited:
                continue
            visited.add(u)
            cluster.append(u)
            for v in adj[u]:
                if v not in visited and score[v] >= thresh:
                    stack.append(v)
        if len(cluster) >= min_cluster:
            cs = score[np.array(cluster, dtype=np.int64)]
            clusters.append((cluster, float(cs.max()) * wmax +
                             float(cs.mean()) * (1.0 - wmax),
                             float(cs.max()), float(np.percentile(cs, 95))))
    if sort_mode == "cmax":
        clusters.sort(key=lambda x: -x[2])
    elif sort_mode == "p95":
        clusters.sort(key=lambda x: -x[3])
    elif sort_mode == "seed":
        pass
    else:
        clusters.sort(key=lambda x: -x[1])
    return [(c, s) for c, s, _, _ in clusters][:max_alerts]


def alert_metrics2(clusters, y, score, top_k):
    """与 e6d_ablation_replay.alert_metrics2 同口径（累计覆盖回映射全局下标）。"""
    n_gt = int(y.sum())
    alerted = [np.array(c, dtype=np.int64) for c, _ in clusters]
    tp_flags = [int(y[c].sum() > 0) for c in alerted]
    n_alerts = len(clusters)
    cum_cov, seen = [], set()
    for c in alerted:
        seen.update(c[np.nonzero(y[c])[0]].tolist())
        cum_cov.append(len(seen))
    alert_TP = int(sum(tp_flags))
    alert_P = alert_TP / n_alerts if n_alerts else 0.0
    node_cov = cum_cov[-1] / n_gt if (n_gt and cum_cov) else 0.0
    f1_alert = (2 * alert_P * node_cov / (alert_P + node_cov)
                if (alert_P + node_cov) else 0.0)
    f1_at = {}
    for b in range(1, n_alerts + 1):
        p = sum(tp_flags[:b]) / b
        r = cum_cov[b - 1] / n_gt if n_gt else 0.0
        f1_at[b] = round(2 * p * r / (p + r), 4) if (p + r) else 0.0
    n_alerted_nodes = int(sum(len(c) for c in alerted))
    covered_gt = cum_cov[-1] if cum_cov else 0
    m = {"n_gt": n_gt, "n_alerts": n_alerts, "alert_TP": alert_TP,
         "alert_P": round(alert_P, 4), "node_cov": round(node_cov, 4),
         "F1_alert": round(f1_alert, 4),
         "n_alerted_nodes": n_alerted_nodes, "covered_gt": int(covered_gt),
         "node_prec": round(covered_gt / n_alerted_nodes, 4) if n_alerted_nodes else 0.0}
    for b in BUDGETS:
        m[f"F1_alert@{b}"] = f1_at.get(b, None)
    return m


# --------------------------------------------------- 向量化 GraphSAGE 采样器
class SAGEDetectorFast:
    """与 models/detector_graphsage.GraphSAGEDetector 同架构/同训练目标，
    仅把逐节点 Python 邻居采样替换为向量化实现（同分布），供大数据量消融使用。"""

    def __init__(self, hidden=64, k=25, lr=0.01, epochs=20, radius_q=0.95,
                 neg_ratio=1.0, shrink=0.1, seed=0, device=None, op_quantile=95.0):
        import torch
        self.torch = torch
        self.hidden, self.k, self.lr, self.epochs = hidden, k, lr, epochs
        self.radius_q, self.neg_ratio, self.shrink, self.seed = (
            radius_q, neg_ratio, shrink, seed)
        self.op_quantile = op_quantile
        self.op_thr = None
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.mu = None
        self.Cinv = None

    # -- 构建 CSR 邻居索引 --
    @staticmethod
    def _csr(edges, n):
        if len(edges) == 0:
            return (np.zeros(0, dtype=np.int64),
                    np.zeros(n + 1, dtype=np.int64),
                    np.zeros(n, dtype=np.int64))
        src = np.concatenate([edges[:, 0], edges[:, 1]])
        dst = np.concatenate([edges[:, 1], edges[:, 0]])
        order = np.argsort(src, kind="stable")
        src_s, dst_s = src[order], dst[order]
        deg = np.bincount(src_s, minlength=n).astype(np.int64)
        ptr = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(deg, out=ptr[1:])
        return dst_s, ptr, deg

    @staticmethod
    def _sample(dst_s, ptr, deg, n, k, rng):
        rows = np.repeat(np.arange(n, dtype=np.int64), k)
        if len(dst_s) == 0:
            return rows, rows.copy()
        off = rng.integers(0, np.maximum(deg, 1))
        idx = ptr[rows] + np.minimum(off[rows], np.maximum(deg[rows] - 1, 0))
        iso = deg[rows] == 0
        idx = np.where(iso, 0, np.minimum(idx, len(dst_s) - 1))
        cols = dst_s[idx]
        cols = np.where(iso, rows, cols)
        return rows, cols

    def _build(self, torch, d_in):
        import torch.nn as nn
        import torch.nn.functional as F
        from torch_geometric.nn import SAGEConv

        class M(nn.Module):
            def __init__(s):
                super().__init__()
                s.c1 = SAGEConv(d_in, self.hidden, aggr="mean")
                s.c2 = SAGEConv(self.hidden, self.hidden, aggr="mean")

            def forward(s, x, ei):
                h = F.relu(s.c1(x, ei))
                return s.c2(h, ei)
        return M().to(self.device)

    def fit(self, X_list, edges_list, log=print):
        torch = self.torch
        import torch.nn.functional as F
        rng = np.random.default_rng(self.seed)
        self.model = self._build(torch, X_list[0].shape[1])
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        fitted = []
        for gi, (X, edges) in enumerate(zip(X_list, edges_list)):
            n = X.shape[0]
            if n < 2:
                continue
            data, ei = self._mk(torch, X, edges)
            dst_s, ptr, deg = self._csr(edges, n)
            n_neg = max(1, int(self.k * self.neg_ratio))
            self.model.train()
            for ep in range(self.epochs):
                opt.zero_grad()
                h = self.model(data.x, data.edge_index)
                rows, cols = self._sample(dst_s, ptr, deg, n, self.k, rng)
                neg_rows = np.repeat(np.arange(n, dtype=np.int64), n_neg)
                neg_cols = rng.integers(0, n, size=n * n_neg)
                rt = torch.as_tensor(np.concatenate([rows, neg_rows]),
                                     dtype=torch.long, device=self.device)
                ct = torch.as_tensor(np.concatenate([cols, neg_cols]),
                                     dtype=torch.long, device=self.device)
                lab = torch.cat([torch.ones(len(rows), device=self.device),
                                 torch.zeros(len(neg_rows), device=self.device)])
                loss = F.binary_cross_entropy_with_logits(
                    (h[rt] * h[ct]).sum(-1), lab)
                loss.backward()
                opt.step()
            log(f"    [gnn] g{gi}: n={n} e={len(edges)} ep={self.epochs} "
                f"loss={float(loss):.4f}")
            fitted.append((data, n))
            del data, ei, h, loss
            gc.collect()
        self.model.eval()
        with torch.no_grad():
            H = np.vstack([self.model(d.x, d.edge_index).cpu().numpy()
                           for d, _ in fitted])
        mu = H.mean(axis=0)
        C = np.cov(H.T)
        d = H.shape[1]
        C = (1 - self.shrink) * C + self.shrink * np.diag(np.diag(C))
        C += 1e-6 * np.eye(d)
        self.mu = torch.as_tensor(mu, dtype=torch.float32, device=self.device)
        self.Cinv = torch.as_tensor(np.linalg.inv(C), dtype=torch.float32,
                                    device=self.device)
        del H
        gc.collect()
        # 训练分布操作点（每图内鲁棒 z 后池化，与推理口径一致）
        pooled = []
        with torch.no_grad():
            for d, _ in fitted:
                h = self.model(d.x, d.edge_index)
                diff = h - self.mu
                pooled.append(self._rz(((diff @ self.Cinv) * diff)
                                       .sum(-1).cpu().numpy()))
        pooled = np.concatenate(pooled)
        self.op_thr = float(np.percentile(pooled, self.op_quantile))
        log(f"    [gnn] train z-score N={len(pooled)} "
            f"op_thr(q{self.op_quantile})={self.op_thr:.4f}")
        del pooled
        gc.collect()
        return self

    @staticmethod
    def _rz(s):
        med = float(np.median(s))
        q75, q25 = np.percentile(s, [75, 25])
        return (s - med) / max(float(q75 - q25), 1e-9)

    def score(self, X, edges):
        torch = self.torch
        try:
            return self._score_on(X, edges, self.device)
        except RuntimeError as e:
            if "out of memory" not in str(e):
                raise
            print(f"    [gnn] CUDA OOM on score ({X.shape[0]} nodes) -> CPU 回退",
                  flush=True)
            torch.cuda.empty_cache()
            self._cpu = True
            return self._score_on(X, edges, "cpu")

    def _score_on(self, X, edges, device):
        torch = self.torch
        data, _ = self._mk(torch, X, edges, device=device)
        self.model.to(device).eval()
        mu = self.mu.to(device)
        Cinv = self.Cinv.to(device)
        with torch.no_grad():
            h = self.model(data.x, data.edge_index)
            diff = h - mu
            s = ((diff @ Cinv) * diff).sum(-1).cpu().numpy()
        del data, h, diff
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        return self._rz(s)

    # ---------- 构建 PyG Data ----------
    def _mk(self, torch, X, edges, device=None):
        device = device or self.device
        ei = np.concatenate([edges, edges[:, [1, 0]]], axis=0)
        ei = np.unique(ei, axis=0)
        from torch_geometric.data import Data
        return Data(x=torch.as_tensor(X, dtype=torch.float32, device=device),
                    edge_index=torch.as_tensor(ei.T, dtype=torch.long,
                                               device=device)), ei


# ------------------------------------------------------------------- 主流程
def subgraph_drop(X, edges, keep_mask):
    """按 keep_mask 保留节点并重映射边索引（用于剥离 GT 正例的良性训练变体）。"""
    idx = np.nonzero(keep_mask)[0]
    remap = -np.ones(len(keep_mask), dtype=np.int64)
    remap[idx] = np.arange(len(idx), dtype=np.int64)
    if len(edges):
        e = edges[(keep_mask[edges[:, 0]]) & (keep_mask[edges[:, 1]])]
        e = remap[e]
    else:
        e = edges.reshape(0, 2)
    return X[idx], e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/TeRed+RATE/dataset/darpae5/cadets")
    ap.add_argument("--train-files", default="bin.1,bin.2,bin.3,bin.4,bin.5")
    ap.add_argument("--max-records-train", type=int, default=400000)
    ap.add_argument("--test-files",
                    default="bin.116,bin.117,bin.118,bin.119@500000,bin.120,"
                            "bin.6@400000,bin.7@400000")
    ap.add_argument("--gt", default="/TeRed+RATE/code/ground_truth")
    ap.add_argument("--templates", default=None)
    ap.add_argument("--semantic", default="onehot")
    ap.add_argument("--tape-dim", type=int, default=8)
    ap.add_argument("--tape-base", type=float, default=10000.0)
    ap.add_argument("--v4-scale", default="raw", choices=["raw", "robust"])
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--bfs-q", type=int, default=75)
    ap.add_argument("--max-cluster", type=int, default=200)
    ap.add_argument("--max-alerts", type=int, default=20)
    ap.add_argument("--cache", default="/TeRed+RATE/code/cache")
    ap.add_argument("--results", required=True)
    ap.add_argument("--detectors", default="cosine,gnn")
    ap.add_argument("--configs",
                    default="identity:rate,tered:dual_naive,tered:rate,"
                            "tered:rate_ratio,tered:rate_single,tered:none")
    ap.add_argument("--gnn-epochs", type=int, default=20)
    ap.add_argument("--gnn-k", type=int, default=25)
    ap.add_argument("--gnn-hidden", type=int, default=64)
    ap.add_argument("--op-quantile", type=float, default=95.0,
                    help="固定操作点：训练分位阈值（节点级 FPR 口径）")
    ap.add_argument("--clean-train", action="store_true",
                    help="剥离 GT 正例后的良性训练变体（cosine）")
    a = ap.parse_args()

    gt = load_gt(a.gt)
    detectors = [s.strip() for s in a.detectors.split(",") if s.strip()]
    specs = [s.strip() for s in a.test_files.split(",") if s.strip()]

    tpl_path = a.templates or find_templates(
        os.path.join(a.cache, "darpa", "e5_templates", "*.jsonl"))
    tpl_stem = os.path.splitext(os.path.basename(tpl_path))[0]
    rcache = os.path.join(a.cache, "darpa", "e5_f1_reduce")
    rec_tag_tr = f"tr{a.max_records_train}"
    print(f"[e6e] templates={tpl_stem} gt={len(gt)}", flush=True)

    # ---------- 加载训练图（id / tered） ----------
    Gp_tr = {"identity": [], "tered": []}
    for nm in [s.strip() for s in a.train_files.split(",") if s.strip()]:
        gid = resolve_name(a.data_dir, nm)
        Gp_tr["identity"].append(load_res(rcache, "id", tpl_stem, rec_tag_tr, gid).Gp)
        Gp_tr["tered"].append(load_res(rcache, "tered", tpl_stem, rec_tag_tr, gid).Gp)
    print(f"[e6e] train graphs: id={len(Gp_tr['identity'])} "
          f"tered={len(Gp_tr['tered'])}", flush=True)

    # ---------- 加载测试图 + 边 ----------
    tests = []
    for spec in specs:
        nm, mr = parse_spec(a.data_dir, spec)
        rec_tag = f"at{mr}" if mr else "atfull"
        gid = resolve_name(a.data_dir, nm)
        tests.append({"spec": spec, "rec_tag": rec_tag,
                      "identity": load_res(rcache, "id", tpl_stem, rec_tag, gid).Gp,
                      "tered": load_res(rcache, "tered", tpl_stem, rec_tag, gid).Gp})
        print(f"[e6e] test {spec}: id={tests[-1]['identity'].n_nodes()} "
              f"tered={tests[-1]['tered'].n_nodes()}", flush=True)

    parsed = [(s.strip().split(":")[0], s.strip().split(":")[1])
              for s in a.configs.split(",") if s.strip()]

    rows_written = 0
    for op_name, enc in parsed:
        # 统一 vocab：训练图 + 全部测试图（保证跨文件特征维一致）
        vgraphs = list(Gp_tr[op_name]) + [t[op_name] for t in tests]
        vocab = make_type_vocab(vgraphs)
        print(f"\n[e6e] ===== {op_name} / {enc} (vocab={len(vocab)}) =====", flush=True)

        # ---- 训练侧特征 ----
        t0 = time.time()
        Xtr, Etr, labels_tr = [], [], []
        n_dropped = 0
        for Gp in Gp_tr[op_name]:
            X, nids, _ = node_feature_matrix(
                Gp, encoding=enc, tape_dim=a.tape_dim, tape_base=a.tape_base,
                semantic=a.semantic, vocab=vocab)
            e = _edges_index(Gp, nids)
            X, _ = extend_with_v4(X, nids, Gp, e, scale=a.v4_scale)
            y_tr = label_vector(Gp, nids)
            labels_tr.append(y_tr)
            if a.clean_train and y_tr.sum() > 0:
                n_dropped += int(y_tr.sum())
                X, e = subgraph_drop(X, e, y_tr == 0)
            Xtr.append(X)
            Etr.append(e)
        if a.clean_train:
            print(f"[e6e] clean-train: 剥离训练图 GT 正例 {n_dropped} 个节点",
                  flush=True)
        print(f"[e6e] train features {time.time()-t0:.0f}s", flush=True)

        # ---- 测试侧特征/标签 ----
        test_feat = []
        for t in tests:
            Gp = t[op_name]
            X, nids, _ = node_feature_matrix(
                Gp, encoding=enc, tape_dim=a.tape_dim, tape_base=a.tape_base,
                semantic=a.semantic, vocab=vocab)
            e = _edges_index(Gp, nids)
            X, _ = extend_with_v4(X, nids, Gp, e, scale=a.v4_scale)
            test_feat.append({"spec": t["spec"], "rec_tag": t["rec_tag"],
                              "X": X, "e": e, "y": label_vector(Gp, nids),
                              "n": len(nids), "nodes": Gp.n_nodes()})

        # ---------------- cosine 检测器 ----------------
        if "cosine" in detectors:
            t0 = time.time()
            det = BenignEnsemble().fit(Xtr, [f"tr{i}" for i in range(len(Xtr))])
            train_scores = np.concatenate([det.anomaly_scores(X) for X in Xtr])
            op_thr = float(np.percentile(train_scores, a.op_quantile))
            print(f"[e6e] cosine fit {time.time()-t0:.0f}s op_thr={op_thr:.4f}",
                  flush=True)
            for tf in test_feat:
                s = det.anomaly_scores(tf["X"])
                m = binary_metrics(tf["y"], (s > op_thr).astype(np.int64), s)
                am = alert_metrics2(alert_pipeline2(
                    s, tf["e"], tf["n"], top_k=a.top_k, bfs_q=a.bfs_q,
                    max_cluster=a.max_cluster, max_alerts=a.max_alerts),
                    tf["y"], s, a.top_k)
                row = {"detector": "cosine", "operator": op_name, "config": f"{op_name}+{enc}",
                       "encoding": enc, "file": tf["spec"], "rec_tag": tf["rec_tag"],
                       "train_scheme": "clean" if a.clean_train else "all",
                       "n_nodes_Gp": tf["nodes"], "n_edges_Gp": len(tf["e"]),
                       "op_quantile": a.op_quantile, "op_threshold": round(op_thr, 4),
                       "node_ROC_AUC": round(m["ROC_AUC"], 4),
                       "node_PR_AUC": round(m["PR_AUC"], 4),
                       "node_best_F1": round(m["best_F1"], 4),
                       "node_FPR": round(m["FPR"], 6),
                       "node_Precision": round(m["Precision"], 4),
                       "node_Recall": round(m["Recall"], 4),
                       "P@100": round(m["P@100"], 4)}
                row.update(am)
                append_csv(a.results, row)
                rows_written += 1
                print(f"  [cosine {op_name}/{enc}] {tf['spec']:16s} "
                      f"AUC={m['ROC_AUC']:.3f} PR={m['PR_AUC']:.3f} "
                      f"bF1={m['best_F1']:.3f} FPR={m['FPR']:.4f} "
                      f"@5={am['F1_alert@5']} @10={am['F1_alert@10']}", flush=True)
            del det, train_scores
            gc.collect()

        # ---------------- GNN 检测器 ----------------
        if "gnn" in detectors:
            t0 = time.time()
            gdet = SAGEDetectorFast(hidden=a.gnn_hidden, k=a.gnn_k,
                                    epochs=a.gnn_epochs,
                                    radius_q=0.95, seed=0,
                                    op_quantile=a.op_quantile).fit(Xtr, Etr)
            print(f"[e6e] gnn fit {time.time()-t0:.0f}s", flush=True)
            for tf in test_feat:
                t1 = time.time()
                s = gdet.score(tf["X"], tf["e"])
                m = binary_metrics(tf["y"], (s > gdet.op_thr).astype(np.int64), s)
                am = alert_metrics2(alert_pipeline2(
                    s, tf["e"], tf["n"], top_k=a.top_k, bfs_q=a.bfs_q,
                    max_cluster=a.max_cluster, max_alerts=a.max_alerts),
                    tf["y"], s, a.top_k)
                row = {"detector": "gnn", "operator": op_name, "config": f"{op_name}+{enc}",
                       "encoding": enc, "file": tf["spec"], "rec_tag": tf["rec_tag"],
                       "train_scheme": "all",
                       "n_nodes_Gp": tf["nodes"], "n_edges_Gp": len(tf["e"]),
                       "op_quantile": a.op_quantile,
                       "op_threshold": round(gdet.op_thr, 4),
                       "node_ROC_AUC": round(m["ROC_AUC"], 4),
                       "node_PR_AUC": round(m["PR_AUC"], 4),
                       "node_best_F1": round(m["best_F1"], 4),
                       "node_FPR": round(m["FPR"], 6),
                       "node_Precision": round(m["Precision"], 4),
                       "node_Recall": round(m["Recall"], 4),
                       "P@100": round(m["P@100"], 4)}
                row.update(am)
                append_csv(a.results, row)
                rows_written += 1
                print(f"  [gnn {op_name}/{enc}] {tf['spec']:16s} "
                      f"AUC={m['ROC_AUC']:.3f} PR={m['PR_AUC']:.3f} "
                      f"bF1={m['best_F1']:.3f} P@100={m['P@100']:.3f} "
                      f"@5={am['F1_alert@5']} @10={am['F1_alert@10']} "
                      f"({time.time()-t1:.0f}s)", flush=True)
                del s
                gc.collect()
            del gdet
            gc.collect()

        del Xtr, Etr, test_feat
        gc.collect()

    print(f"\n[e6e] 完成，{rows_written} 行 -> {a.results}", flush=True)


if __name__ == "__main__":
    main()
