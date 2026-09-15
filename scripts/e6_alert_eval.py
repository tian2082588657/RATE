# -*- coding: utf-8 -*-
"""scripts/e6_alert_eval.py — C 阶段告警级口径评测（M2）。

在 e5_f1_eval.py（节点级）基础上，把节点异常分聚合成"子图告警"再评测：

  1. 良性训练图 bin.1-5 -> BenignEnsemble（v4 特征，与 v4 全量验证同口径）
  2. 测试图（攻击 / 良性对照）-> 三配置（identity / TeRed+naive / TeRed+RATE）异常分
  3. 告警聚合：top-K 异常种子 -> BFS 连通扩展（分数 >= 图内分位数门槛，
     簇大小封顶）-> 簇聚合分（max）排序 -> top-B 簇 = 告警
  4. 告警级指标：
       alert_P    = 含 >=1 GT 节点的告警占比（告警精确率）
       node_cov   = 告警覆盖的 GT 节点 / 图内 GT 节点（节点级召回）
       F1_alert   = harmonic(alert_P, node_cov)
       best 版本  按告警预算 b=1..B 扫描取最优
       flat_topk  不聚合、直接 top-K 节点作告警的对照（验证聚合价值）
     良性对照图：n_gt=0，告警数即误报规模（false alerts）

复用 e5_f1_eval 的归约缓存（share_k=-1 逐图独立），攻击文件 atfull/at500k
口径与 v4 全量验证完全一致，保证结果可直接对比。

用法:
  python scripts/e6_alert_eval.py \
      --test-files bin.116,bin.117,bin.118,bin.119@500000,bin.120,bin.6@400000 \
      --results results/e6_alert.csv
"""
from __future__ import annotations
import os, sys, glob, time, argparse, gc

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import numpy as np

from rate_core import load_graphs, load_pickle, save_pickle
from reduction.tered import TeRedOperator
from reduction.base import IdentityOperator, node_map_sanity
from features.rate import node_feature_matrix, make_type_vocab, label_vector
from features.v4extras import extend_with_v4
from models.detector import BenignEnsemble
from eval.metrics import append_csv
from adapters import darpa_tc

from scripts.e5_f1_eval import (_edges_index, load_gt, parse_one, label_graph,
                                resolve_name, find_templates)


# ---------------------------------------------------------------- 告警聚合

def build_adj(n, edges):
    """无向邻接表（跳过自环；端点越界丢弃）。"""
    adj = [[] for _ in range(n)]
    for s, d in edges:
        s, d = int(s), int(d)
        if s == d or not (0 <= s < n) or not (0 <= d < n):
            continue
        adj[s].append(d)
        adj[d].append(s)
    return adj


def alert_pipeline(score, edges, n, top_k=100, bfs_q=75, min_cluster=3,
                   max_cluster=200, max_alerts=20):
    """top-K 异常种子 -> 受限 BFS 连通簇 -> 聚合分排序 -> top-B 告警。

    返回 [(node_idx_list, agg_score), ...]（按 agg 降序，最多 max_alerts 个）。
    """
    order = np.argsort(-score, kind="mergesort")
    top_idx = order[:top_k]
    thresh = float(np.percentile(score, bfs_q))
    adj = build_adj(n, edges)

    visited = set()
    clusters = []
    for seed in top_idx:
        seed = int(seed)
        if seed in visited:
            continue
        cluster = []
        stack = [seed]
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
            agg = float(cs.max()) * 0.7 + float(cs.mean()) * 0.3
            clusters.append((cluster, agg))
    clusters.sort(key=lambda x: -x[1])
    return clusters[:max_alerts]


def alert_metrics(clusters, y, score, top_k):
    """告警级指标 + flat top-K 对照。y: 1=攻击（与 score 同序）。"""
    n_gt = int(y.sum())
    alerted = [np.array(c, dtype=np.int64) for c, _ in clusters]
    tp_flags = [int(y[c].sum() > 0) for c in alerted]
    n_alerts = len(clusters)

    # 累计覆盖（按告警顺序）
    # 注意：必须回映射到全局节点下标。np.nonzero(y[c])[0] 是簇内局部下标，
    # 直接累计会把不同簇里处于同一局部位置的 GT 误判为重复（已修 2026-09-11）。
    cum_cov, seen = [], set()
    for c in alerted:
        seen.update(c[np.nonzero(y[c])[0]].tolist())
        cum_cov.append(len(seen))

    # 全预算指标
    alert_TP = int(sum(tp_flags))
    alert_P = alert_TP / n_alerts if n_alerts else 0.0
    node_cov = cum_cov[-1] / n_gt if (n_gt and cum_cov) else 0.0
    f1_alert = (2 * alert_P * node_cov / (alert_P + node_cov)
                if (alert_P + node_cov) else 0.0)

    # 预算扫描 best F1
    best_f1, best_b = 0.0, 0
    first_hit = 0
    for b in range(1, n_alerts + 1):
        p = sum(tp_flags[:b]) / b
        r = cum_cov[b - 1] / n_gt if n_gt else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        if f > best_f1:
            best_f1, best_b = f, b
        if first_hit == 0 and tp_flags[b - 1]:
            first_hit = b

    # flat top-K（不聚合对照）
    k = min(top_k, len(y))
    flat_idx = np.argsort(-score, kind="mergesort")[:k]
    flat_cov = float(y[flat_idx].sum()) / n_gt if n_gt else 0.0
    flat_prec = float(y[flat_idx].sum()) / k if k else 0.0

    n_alerted_nodes = int(sum(len(c) for c in alerted))
    covered_gt = cum_cov[-1] if cum_cov else 0
    return {
        "n_gt": n_gt, "n_alerts": n_alerts, "alert_TP": alert_TP,
        "alert_P": round(alert_P, 4), "node_cov": round(node_cov, 4),
        "F1_alert": round(f1_alert, 4),
        "best_F1_alert": round(best_f1, 4), "best_b": best_b,
        "first_hit": first_hit, "top1_hit": tp_flags[0] if tp_flags else 0,
        "n_alerted_nodes": n_alerted_nodes,
        "covered_gt": int(covered_gt),
        "node_prec": round(covered_gt / n_alerted_nodes, 4) if n_alerted_nodes else 0.0,
        "flat_topk_cov": round(flat_cov, 4), "flat_topk_prec": round(flat_prec, 4),
    }


# ---------------------------------------------------------------- 主流程

def parse_spec(data_dir, spec):
    """'bin.119@500000' -> (短名, max_records)；无 @ 后缀 = 全量。"""
    if "@" in spec:
        nm, mr = spec.split("@", 1)
        return nm.strip(), int(mr)
    return spec.strip(), 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/TeRed+RATE/dataset/darpae5/cadets")
    ap.add_argument("--train-files", default="bin.1,bin.2,bin.3,bin.4,bin.5")
    ap.add_argument("--test-files", required=True,
                    help="逗号分隔; 'bin.119@500000' 表示截断 50 万记录; 无 @ = 全量")
    ap.add_argument("--gt", default="ground_truth")
    ap.add_argument("--max-records-train", type=int, default=400000)
    ap.add_argument("--templates", default=None)
    ap.add_argument("--share-k", type=int, default=-1)
    ap.add_argument("--semantic", default="onehot")
    ap.add_argument("--tape-dim", type=int, default=8)
    ap.add_argument("--tape-base", type=float, default=10000.0)
    ap.add_argument("--v4-scale", default="raw", choices=["raw", "robust"])
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--bfs-q", type=int, default=75,
                    help="BFS 扩展分数门槛分位数(0-100)")
    ap.add_argument("--min-cluster", type=int, default=3)
    ap.add_argument("--max-cluster", type=int, default=200)
    ap.add_argument("--max-alerts", type=int, default=20)
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--results", default="results/e6_alert.csv")
    a = ap.parse_args()

    gt = load_gt(a.gt)
    print(f"[alert] ground truth: {len(gt)} UUIDs", flush=True)

    # ---------- 1. 良性训练图（解析 + 归约缓存） ----------
    train_names = [s.strip() for s in a.train_files.split(",") if s.strip()]
    train_graphs = []
    t0 = time.time()
    for nm in train_names:
        g, _ = parse_one(a.data_dir, nm, a.max_records_train or None)
        train_graphs.append(g)
    print(f"[alert] 训练图 {len(train_graphs)} 张解析完成 ({time.time()-t0:.0f}s)",
          flush=True)

    tpl_path = a.templates or find_templates(
        os.path.join(a.cache, "darpa", "e5_templates", "*.jsonl"))
    tpls = load_graphs(tpl_path)
    print(f"[alert] 模板: {os.path.basename(tpl_path)} -> {len(tpls)} 个 "
          f"(share_k={a.share_k})", flush=True)

    use_rcache = a.share_k == -1
    tpl_stem = os.path.splitext(os.path.basename(tpl_path))[0]
    rcache_dir = os.path.join(a.cache, "darpa", "e5_f1_reduce")
    os.makedirs(rcache_dir, exist_ok=True)

    def reduce_one(op, g, op_tag, rec_tag):
        cpath = (os.path.join(
            rcache_dir, f"{op_tag}_{tpl_stem}_k{a.share_k}_{rec_tag}_{g.gid}.pkl")
            if use_rcache else None)
        if cpath and os.path.exists(cpath):
            res = load_pickle(cpath)
            node_map_sanity(res, g)
            return res
        res = op.reduce(g)
        node_map_sanity(res, g)
        if cpath:
            save_pickle(cpath, res)
        return res

    t1 = time.time()
    res_train_id = [reduce_one(IdentityOperator(), g, "id",
                               f"tr{a.max_records_train}") for g in train_graphs]
    tered = TeRedOperator(tpls, max_instances=300, max_total=5000,
                          share_k=a.share_k)
    res_train_te = [reduce_one(tered, g, "tered",
                               f"tr{a.max_records_train}") for g in train_graphs]
    print(f"[alert] 训练图归约完成 ({time.time()-t1:.0f}s, 缓存命中)", flush=True)

    # ---------- 2. 逐测试文件 ----------
    specs = [s.strip() for s in a.test_files.split(",") if s.strip()]
    t_start = time.time()
    for spec in specs:
        nm, mr = parse_spec(a.data_dir, spec)
        rec_tag = f"at{mr}" if mr else "atfull"
        tf = time.time()
        g, meta = parse_one(a.data_dir, nm, mr or None)
        hit, _ = label_graph(g, gt)
        is_attack = hit > 0
        print(f"\n[alert] ===== {spec}: {g.n_nodes()} 节点 {g.n_edges()} 边 "
              f"records={meta['records']} GT命中={hit} "
              f"({'攻击' if is_attack else '良性对照'}) =====", flush=True)

        res_id = reduce_one(IdentityOperator(), g, "id", rec_tag)
        res_te = reduce_one(tered, g, "tered", rec_tag)
        print(f"[alert] {nm} 归约: id {g.n_nodes()}->{res_id.Gp.n_nodes()}, "
              f"tered {g.n_nodes()}->{res_te.Gp.n_nodes()} "
              f"({time.time()-tf:.0f}s)", flush=True)

        configs = [
            ("identity+rate", "identity", [r.Gp for r in res_train_id],
             res_id.Gp, "rate"),
            ("TeRed+naive", "tered", [r.Gp for r in res_train_te],
             res_te.Gp, "dual_naive"),
            ("TeRed+RATE", "tered", [r.Gp for r in res_train_te],
             res_te.Gp, "rate"),
            ("TeRed+RATE*", "tered", [r.Gp for r in res_train_te],
             res_te.Gp, "rate_ratio"),
        ]

        for cname, op_name, gpt, Gp_test, enc in configs:
            tc = time.time()
            vocab = make_type_vocab(gpt + [Gp_test])
            Xs_train = []
            for Gp in gpt:
                X, nids_t, _ = node_feature_matrix(
                    Gp, encoding=enc, tape_dim=a.tape_dim, tape_base=a.tape_base,
                    semantic=a.semantic, vocab=vocab)
                et = _edges_index(Gp, nids_t)
                X, _ = extend_with_v4(X, nids_t, Gp, et, scale=a.v4_scale)
                Xs_train.append(X)
            det = BenignEnsemble().fit(Xs_train,
                                       [f"tr{i}" for i in range(len(Xs_train))])

            X, nids, _ = node_feature_matrix(
                Gp_test, encoding=enc, tape_dim=a.tape_dim, tape_base=a.tape_base,
                semantic=a.semantic, vocab=vocab)
            edges = _edges_index(Gp_test, nids)
            X, _ = extend_with_v4(X, nids, Gp_test, edges, scale=a.v4_scale)
            score = det.anomaly_scores(X)
            y = label_vector(Gp_test, nids)

            clusters = alert_pipeline(score, edges, len(nids),
                                      top_k=a.top_k, bfs_q=a.bfs_q,
                                      min_cluster=a.min_cluster,
                                      max_cluster=a.max_cluster,
                                      max_alerts=a.max_alerts)
            m = alert_metrics(clusters, y, score, a.top_k)

            row = {"dataset": "e5_cadets", "config": cname, "operator": op_name,
                   "encoding": enc, "file": spec, "rec_tag": rec_tag,
                   "is_attack": is_attack,
                   "n_nodes_Gp": Gp_test.n_nodes(), "n_edges_Gp": len(edges),
                   "top_k": a.top_k, "bfs_q": a.bfs_q,
                   "min_cluster": a.min_cluster, "max_cluster": a.max_cluster,
                   "max_alerts": a.max_alerts,
                   "v4_scale": a.v4_scale,
                   "runtime_s": round(time.time() - tc, 1)}
            row.update(m)
            append_csv(a.results, row)

            tag = "ATK" if is_attack else "BENIGN"
            print(f"  [{cname}] n_gt={m['n_gt']} alerts={m['n_alerts']} "
                  f"TP={m['alert_TP']} alert_P={m['alert_P']:.2f} "
                  f"cov={m['node_cov']:.2f} F1={m['F1_alert']:.3f} "
                  f"bestF1={m['best_F1_alert']:.3f}@b{m['best_b']} "
                  f"first_hit={m['first_hit']} top1={m['top1_hit']} "
                  f"nodes={m['n_alerted_nodes']} prec={m['node_prec']:.3f} "
                  f"flat_cov={m['flat_topk_cov']:.2f} ({row['runtime_s']}s)",
                  flush=True)
            del X, Xs_train, det, edges
            gc.collect()

        del g, res_id, res_te, Gp_test
        gc.collect()

    print(f"\n[alert] 全部完成 ({time.time()-t_start:.0f}s), "
          f"结果追加至 {a.results}", flush=True)


if __name__ == "__main__":
    main()
