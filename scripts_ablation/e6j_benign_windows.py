# -*- coding: utf-8 -*-
"""e6j_benign_windows.py — 在「无标注正例」窗口上的留一预测（良性替代实验）。

动机（DeepSeek 第三轮 #6）：CADETS 无 attack-free 分区，无法估计 FAR。
已做的两个反事实（抹标签 / 删实体）都证明「攻击行为还在图里」，
不能给 FAR。本脚本做第三种、也是最接近 benign replay 的替代：

  五个训练窗口（bin.1-5 前 4e5 条记录）在全量扫描下 **不含任何已标注 GT 实体**。
  对每个窗口 i：用其余四个窗口拟合检测器，在窗口 i 上打分 + 走同一告警管线
  （top_k=100, bfs_q=75, min_cluster=3, max_cluster=200, max_alerts=20）。
  由于窗口 i 无已标注攻击，其告警量即「在公开标注下」的误报体积上界。

⚠️ 必须诚实标注的三点局限：
  1. 公开标注不完整，未标注攻击可能存在 → 不是真 FAR，只是上界；
  2. 窗口来自训练期（时间上早于测试分区），分布与测试期不同；
  3. 5% 分位半径使被标记节点比例约为 5%，告警体积本身就受预算控制。

输出：results/e6j_benign_windows.csv
"""
from __future__ import annotations
import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--windows", default="bin.1,bin.2,bin.3,bin.4,bin.5")
    ap.add_argument("--max-records", type=int, default=400000)
    ap.add_argument("--templates", default="")
    ap.add_argument("--cache", default="cache/darpa")
    ap.add_argument("--share-k", type=int, default=-1)
    ap.add_argument("--tape-dim", type=int, default=8)
    ap.add_argument("--tape-base", type=float, default=10000.0)
    ap.add_argument("--semantic", default="onehot")
    ap.add_argument("--v4-scale", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--bfs-q", type=float, default=75.0)
    ap.add_argument("--min-cluster", type=int, default=3)
    ap.add_argument("--max-cluster", type=int, default=200)
    ap.add_argument("--max-alerts", type=int, default=20)
    ap.add_argument("--results", default="results/e6j_benign_windows.csv")
    a = ap.parse_args()

    from rate_core import load_graphs, load_pickle
    from scripts.e5_f1_eval import (parse_one, find_templates, load_gt,
                                    label_graph, _edges_index)
    from scripts.e6_alert_eval import alert_pipeline
    from reduction.tered import TeRedOperator
    from reduction.base import IdentityOperator
    from features.rate import node_feature_matrix, make_type_vocab, label_vector
    from features.v4extras import extend_with_v4
    from models.detector import BenignEnsemble

    gt = load_gt(a.gt)
    semantic = str(a.semantic).lower() not in ("0", "false", "no")
    names = [s.strip() for s in a.windows.split(",") if s.strip()]

    tpl_path = a.templates or find_templates(
        os.path.join(a.cache, "darpa", "e5_templates", "*.jsonl"))
    tpls = load_graphs(tpl_path)
    tered = TeRedOperator(tpls, max_instances=300, max_total=5000,
                          share_k=a.share_k)
    tpl_stem = os.path.splitext(os.path.basename(tpl_path))[0]
    rcache = os.path.join(a.cache, "darpa", "e5_f1_reduce")

    graphs = []
    for nm in names:
        g, _ = parse_one(a.data_dir, nm, a.max_records)
        hit, _ = label_graph(g, gt)
        print(f"[e6j] {nm}: {g.n_nodes()} nodes {g.n_edges()} edges, GT hit={hit}",
              flush=True)
        graphs.append((nm, g))

    def reduce_one(op, g, op_tag):
        cp = os.path.join(
            rcache, f"{op_tag}_{tpl_stem}_k{a.share_k}_tr{a.max_records}_{g.gid}.pkl")
        if os.path.exists(cp):
            return load_pickle(cp)
        return op.reduce(g)

    # 归约/身份图 + 特征（每个窗口各算一次，供留一复用）
    CACHE = {}
    for nm, g in graphs:
        res_id = reduce_one(IdentityOperator(), g, "id")
        res_te = reduce_one(tered, g, "tered")
        CACHE[nm] = {"id": res_id.Gp, "tered": res_te.Gp}
    print("[e6j] graphs reduced/loaded", flush=True)

    def feats(Gp, enc, vocab):
        X, nids, _ = node_feature_matrix(
            Gp, encoding=enc, tape_dim=a.tape_dim, tape_base=a.tape_base,
            semantic=semantic, vocab=vocab)
        et = _edges_index(Gp, nids)
        X, _ = extend_with_v4(X, nids, Gp, et, scale=a.v4_scale)
        return X, nids, et

    CONFIGS = [("identity+rate", "id", "rate"),
               ("TeRed+RATE*", "tered", "rate_ratio")]

    rows = []
    for i, (nm, _g) in enumerate(graphs):
        held = [n for n, _ in graphs if n != nm]
        for cname, optag, enc in CONFIGS:
            Gp_tr = [CACHE[n][optag] for n in held]
            vocab = make_type_vocab(Gp_tr)
            Xs = [feats(G, enc, vocab)[0] for G in Gp_tr]
            det = BenignEnsemble().fit(Xs, [f"w{j}" for j in range(len(Xs))])

            Gp_te = CACHE[nm][optag]
            label_graph(Gp_te, gt)
            X, nids, et = feats(Gp_te, enc, vocab)
            y = np.asarray(label_vector(Gp_te, nids), dtype=np.int64)
            score = det.anomaly_scores(X)
            clusters = alert_pipeline(
                score, et, len(nids), top_k=a.top_k, bfs_q=a.bfs_q,
                min_cluster=a.min_cluster, max_cluster=a.max_cluster,
                max_alerts=a.max_alerts)
            alerted = set()
            for c, _s in clusters:
                alerted |= set(c)
            inlier = det.predict(X)
            flagged = int((~np.asarray(inlier, dtype=bool)).sum())
            cs = [len(c) for c, _ in clusters]
            # 该窗口在公开标注下含 5-7 个 GT 实体：统计告警中有多少锚定到它们
            tp = sum(1 for c, _ in clusters if y[np.asarray(c, dtype=int)].sum() > 0)
            cov = len({int(u) for c, _ in clusters
                       for u in np.asarray(c, dtype=int) if y[u] == 1})
            rows.append({
                "held_out": nm, "config": cname,
                "n_nodes": Gp_te.n_nodes(), "n_edges": Gp_te.n_edges(),
                "flagged_nodes": flagged,
                "n_alerts": len(clusters),
                "n_alerted_nodes": len(alerted),
                "gt_in_window": int(y.sum()),
                "alerts_with_gt": tp,
                "alerts_without_gt": len(clusters) - tp,
                "gt_covered": cov,
                "max_cluster": max(cs) if cs else 0,
                "mean_cluster": round(sum(cs) / len(cs), 1) if cs else 0.0,
            })
            print(f"  [{nm} held out | {cname}] alerts={len(clusters)} "
                  f"with_gt={tp} without_gt={len(clusters)-tp} "
                  f"windows_gt={int(y.sum())} alerted={len(alerted)}",
                  flush=True)

    os.makedirs(os.path.dirname(a.results) or ".", exist_ok=True)
    with open(a.results, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[e6j] done -> {a.results}", flush=True)


if __name__ == "__main__":
    main()
