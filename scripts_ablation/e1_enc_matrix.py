# -*- coding: utf-8 -*-
"""scripts_ablation/e1_enc_matrix.py — E1：图来源 x 拓扑编码 的检测质量矩阵。

目的（补审稿人一定会问的对照）：现有 A5 编码消融只在【归约图】上跑过，缺
"未归约图 + 无拓扑描述子"这一格。两种可能的结果都直接决定论文叙事：

  (a) identity + none ~= tered + none ~= 0.007
      -> 可检测性由【编码】决定，与是否归约无关；RATE 的价值是同一个编码器
         同时适用于两种图（deployment-agnostic）。
  (b) identity + none >> 0.007
      -> 归约确实破坏了结构信号，是 TeRed 在拓扑检测器上的直接反例。

矩阵：fig ∈ {identity, tered} × enc ∈ {none, rate_single, dual_naive, rate, rate_ratio}
每格输出节点级 best-F1 / coverage / 告警 F1@b / alert_P / n_alerts。

用法:
  python scripts_ablation/e1_enc_matrix.py \
    --test-files bin.116,bin.117,bin.118,bin.119@500000,bin.120,bin.6@400000,bin.7@400000 \
    --figs identity,tered --encs none,rate_single,dual_naive,rate \
    --results results/e1_enc_matrix.csv
"""
from __future__ import annotations
import os, sys, time, argparse, gc, random
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import numpy as np

from rate_core import load_pickle, save_pickle
from reduction.tered import TeRedOperator
from reduction.base import IdentityOperator, node_map_sanity
from features.rate import node_feature_matrix, make_type_vocab, label_vector
from features.v4extras import extend_with_v4
from models.detector import BenignEnsemble
from eval.metrics import append_csv

from scripts.e5_f1_eval import (_edges_index, parse_one, label_graph,
                               resolve_name, find_templates, load_gt)
from scripts.e6_alert_eval import build_adj, parse_spec
from scripts_ablation.graphcache import parse_cached

BUDGETS = [1, 2, 3, 5, 10, 20]

# E1b 语义-only 对照（问题-0916-2 P0）：encoding=none 的 X 只剩语义 one-hot，
# 但 extend_with_v4 还会追加 log_deg / out_in_ratio / nbr_type_div / self_loop
# 四列结构特征——其中三列是度拓扑。真正的"无拓扑"对照必须连 v4 一起跳过，
# 否则它证明的仍是"正弦编码 > 标量度特征"，而非"拓扑信息必要"。
V4_SKIP_ENCODINGS = {"semantic_only"}


def alert_pipeline(score, edges, n, top_k=100, bfs_q=75, min_cluster=3,
                   max_cluster=200, max_alerts=20, wmax=1.0, sort_mode="cmax",
                   rng_seed=0):
    """BFS 告警聚合（与主表口径一致）。"""
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
            agg = float(cs.max()) * wmax + float(cs.mean()) * (1.0 - wmax)
            clusters.append((cluster, agg, float(cs.max()),
                             float(np.percentile(cs, 95))))
    if sort_mode == "cmax":
        clusters.sort(key=lambda x: -x[2])
    elif sort_mode == "p95":
        clusters.sort(key=lambda x: -x[3])
    elif sort_mode == "seed":
        pass
    elif sort_mode == "rand":
        random.Random(rng_seed).shuffle(clusters)
    else:
        clusters.sort(key=lambda x: -x[1])
    return [(c, s) for c, s, _, _ in clusters][:max_alerts]


def node_best_f1(y, score):
    """节点级 best-F1（PR 扫描）。"""
    n_pos = int(y.sum())
    if n_pos == 0:
        return 0.0
    order = np.argsort(-score, kind="mergesort")
    ys = y[order]
    tp = np.cumsum(ys).astype(np.float64)
    fp = np.cumsum(1 - ys).astype(np.float64)
    prec = tp / np.maximum(tp + fp, 1.0)
    rec = tp / n_pos
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    return round(float(np.max(f1)), 4)


def node_auc(y, score):
    """节点级 ROC-AUC（阈值无关）。"""
    n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1, dtype=np.float64)
    # 处理并列：用平均秩
    s_sorted = score[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            avg = (i + j) / 2.0 + 1.0
            ranks[order[i:j + 1]] = avg
        i = j + 1
    return round(float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0)
                       / (n_pos * n_neg)), 4)


def alert_metrics(clusters, y, score, top_k):
    """告警级指标 + F1@b 预算列。"""
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
    best_f1, best_b = 0.0, 0
    f1_at = {}
    for b in range(1, n_alerts + 1):
        p = sum(tp_flags[:b]) / b
        r = cum_cov[b - 1] / n_gt if n_gt else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        f1_at[b] = round(f, 4)
        if f > best_f1:
            best_f1, best_b = f, b
    m = {
        "n_gt": n_gt, "n_alerts": n_alerts, "alert_TP": alert_TP,
        "alert_P": round(alert_P, 4), "node_cov": round(node_cov, 4),
        "F1_alert": round(f1_alert, 4),
        "best_F1_alert": round(best_f1, 4), "best_b": best_b,
        "node_best_f1": node_best_f1(y, score),
        "node_auc": node_auc(y, score),
    }
    for b in BUDGETS:
        m["F1_alert@%d" % b] = f1_at.get(b, None)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/TeRed+RATE/dataset/darpae5/cadets")
    ap.add_argument("--train-files", default="bin.1,bin.2,bin.3,bin.4,bin.5")
    ap.add_argument("--max-records-train", type=int, default=400000)
    ap.add_argument("--test-files", required=True)
    ap.add_argument("--gt", default="/TeRed+RATE/code/ground_truth")
    ap.add_argument("--templates", default=None)
    ap.add_argument("--semantic", default="onehot")
    ap.add_argument("--tape-dim", type=int, default=8)
    ap.add_argument("--tape-base", type=float, default=10000.0)
    ap.add_argument("--v4-scale", default="raw", choices=["raw", "robust"])
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--bfs-q", type=int, default=75)
    ap.add_argument("--min-cluster", type=int, default=3)
    ap.add_argument("--max-cluster", type=int, default=200)
    ap.add_argument("--max-alerts", type=int, default=20)
    ap.add_argument("--share-k", type=int, default=-1)
    ap.add_argument("--max-instances", type=int, default=300)
    ap.add_argument("--max-total", type=int, default=5000)
    ap.add_argument("--figs", default="identity,tered",
                    help="图来源: identity(未归约) / tered(归约)")
    ap.add_argument("--encs", default="none,rate_single,dual_naive,rate",
                    help="拓扑编码列表；semantic_only=仅语义one-hot、无任何"
                         "度/结构列（E1b 无拓扑对照）")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--graph-cache", default="cache/darpa/graphcache",
                    help="规范图解析缓存目录（避免重复解 gz）")
    ap.add_argument("--results", required=True)
    ap.add_argument("--no-reduce-cache", action="store_true")
    ap.add_argument("--reduce-cache-dir", default=None,
                    help="覆盖归约缓存目录（默认 <cache>/darpa/e5_f1_reduce）")
    a = ap.parse_args()

    figs = [x.strip() for x in a.figs.split(",") if x.strip()]
    encs = [x.strip() for x in a.encs.split(",") if x.strip()]

    tpl_path = a.templates or find_templates(
        os.path.join(a.cache, "darpa", "e5_templates", "*.jsonl"))
    tpls = [g for g in _load_tpls(tpl_path)]
    tpl_stem = os.path.splitext(os.path.basename(tpl_path))[0]
    rcache_dir = a.reduce_cache_dir or os.path.join(a.cache, "darpa", "e5_f1_reduce")
    print("[e1] 模板 %s -> %d 个 (share_k=%d)" % (tpl_path, len(tpls), a.share_k),
          flush=True)

    use_rcache = (not a.no_reduce_cache) and a.share_k == -1
    ops = {}
    if "identity" in figs:
        ops["identity"] = IdentityOperator()
    if "tered" in figs:
        ops["tered"] = TeRedOperator(tpls, max_instances=a.max_instances,
                                     max_total=a.max_total, share_k=a.share_k)

    def _n_pos(res):
        return sum(1 for v in res.Gp.labels.values() if v == 1)

    def reduce_one(fig, g, rec_tag, expect_pos=True):
        op_tag = "id" if fig == "identity" else "tered"
        # 缓存键与 e5_f1_eval/e7_origunit 保持一致（不含 max_instances），
        # 以便复用既有 12G 归约缓存；扫描归约强度时改用独立 --cache 目录。
        cpath = os.path.join(
            rcache_dir, "%s_%s_k%d_%s_%s.pkl" % (op_tag, tpl_stem, a.share_k,
                                                 rec_tag, g.gid))
        if use_rcache and os.path.exists(cpath):
            res = load_pickle(cpath)
            node_map_sanity(res, g)
            # 缓存可能是在"图未打标签"的状态下生成的：那样 Gp.labels 全 0，
            # 整张表会静默变成 0。这里做一次硬校验，绝不放过。
            if expect_pos and _n_pos(res) == 0:
                print("[e1] 警告：缓存 %s 无正例标签，重新归约"
                      % os.path.basename(cpath), flush=True)
                res = ops[fig].reduce(g)
                node_map_sanity(res, g)
                save_pickle(cpath, res)
            return res
        res = ops[fig].reduce(g)
        node_map_sanity(res, g)
        if use_rcache:
            save_pickle(cpath, res)
        return res

    train_names = [s.strip() for s in a.train_files.split(",") if s.strip()]
    gt = load_gt(a.gt)
    print("[e1] GT %s -> %d UUIDs" % (a.gt, len(gt)), flush=True)
    assert gt, "ground truth 为空，检查 --gt 指向的目录"
    train_res = {}
    for fig in figs:
        train_res[fig] = []
        for nm in train_names:
            g = parse_cached(a.data_dir, nm, a.max_records_train or None,
                             a.graph_cache)[0]
            label_graph(g, gt)
            train_res[fig].append(reduce_one(fig, g, "tr%d" % a.max_records_train,
                                             expect_pos=False))
            print("[e1] %s 训练图 %s 归约完成" % (fig, nm), flush=True)

    t_start = time.time()
    specs = [s.strip() for s in a.test_files.split(",") if s.strip()]
    for spec in specs:
        nm, mr = parse_spec(a.data_dir, spec)
        rec_tag = "at%d" % mr if mr else "atfull"
        g, _meta = parse_cached(a.data_dir, nm, mr or None, a.graph_cache)
        label_graph(g, gt)
        print("\n[e1] ===== %s (%s): %d 节点 =====" % (spec, rec_tag, g.n_nodes()),
              flush=True)
        for fig in figs:
            res_test = reduce_one(fig, g, rec_tag)
            Gp_test = res_test.Gp
            gpt = [r.Gp for r in train_res[fig]]
            print("[e1] %s: %d -> %d 节点" % (fig, g.n_nodes(), Gp_test.n_nodes()),
                  flush=True)
            for enc in encs:
                tc = time.time()
                # semantic_only：node_feature_matrix 用 none（仅语义 one-hot），
                # 且跳过 extend_with_v4（不加任何度/结构列）。
                nfm_enc = "none" if enc in V4_SKIP_ENCODINGS else enc
                vocab = make_type_vocab(gpt + [Gp_test])
                Xs_train = []
                for Gp in gpt:
                    X, nids_t, _ = node_feature_matrix(
                        Gp, encoding=nfm_enc, tape_dim=a.tape_dim,
                        tape_base=a.tape_base, semantic=a.semantic, vocab=vocab)
                    if enc not in V4_SKIP_ENCODINGS:
                        et = _edges_index(Gp, nids_t)
                        X, _ = extend_with_v4(X, nids_t, Gp, et, scale=a.v4_scale)
                    Xs_train.append(X)
                X, nids, _ = node_feature_matrix(
                    Gp_test, encoding=nfm_enc, tape_dim=a.tape_dim,
                    tape_base=a.tape_base, semantic=a.semantic, vocab=vocab)
                edges = _edges_index(Gp_test, nids)
                if enc not in V4_SKIP_ENCODINGS:
                    X, _ = extend_with_v4(X, nids, Gp_test, edges,
                                          scale=a.v4_scale)
                y = label_vector(Gp_test, nids)
                det = BenignEnsemble().fit(Xs_train,
                                           ["tr%d" % i for i in range(len(Xs_train))])
                score = det.anomaly_scores(X)
                clusters = alert_pipeline(
                    score, edges, len(nids), top_k=a.top_k, bfs_q=a.bfs_q,
                    min_cluster=a.min_cluster, max_cluster=a.max_cluster,
                    max_alerts=a.max_alerts)
                m = alert_metrics(clusters, y, score, a.top_k)
                row = dict(dataset="e5_cadets", fig=fig, encoding=enc, file=spec,
                           rec_tag=rec_tag, n_nodes_orig=g.n_nodes(),
                           n_edges_orig=g.n_edges(),
                           n_nodes_Gp=Gp_test.n_nodes(), n_edges_Gp=len(edges),
                           n_dim=X.shape[1], top_k=a.top_k,
                           max_alerts=a.max_alerts, v4_scale=a.v4_scale,
                           runtime_s=round(time.time() - tc, 1))
                row.update(m)
                append_csv(a.results, row)
                print("  [%s][%s] nodeF1=%.4f auc=%.4f cov=%.4f alerts=%d "
                      "P=%.3f best=%.3f@b%d F1=%.3f (%.0fs)"
                      % (fig, enc, m["node_best_f1"], m["node_auc"],
                         m["node_cov"], m["n_alerts"], m["alert_P"],
                         m["best_F1_alert"], m["best_b"], m["F1_alert"],
                         row["runtime_s"]), flush=True)
                del X, Xs_train, det, score, edges
                gc.collect()
            del res_test, Gp_test, gpt
            gc.collect()
        del g
        gc.collect()
    print("\n[e1] 完成 (%.0fs) -> %s" % (time.time() - t_start, a.results),
          flush=True)


def _load_tpls(path):
    from rate_core import load_graphs
    return load_graphs(path)


if __name__ == "__main__":
    main()
