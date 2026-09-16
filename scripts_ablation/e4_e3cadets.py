# -*- coding: utf-8 -*-
"""scripts_ablation/e4_e3cadets.py — E4a：DARPA TC E3 CADETS 第二数据集。

数据复用服务器已有的 8 个时间窗缓存（`cache/darpa/e3cadets_w8_*.jsonl`）与在
同一批窗口上挖出的模板库（`cache/darpa/e3cadets_templates_w8.jsonl`）。
评估口径与 E5 实验一致：前若干窗训练良性单类识别器，其余窗作为测试分区，
在 identity / tered 两种图上、对每种拓扑编码各跑一遍完整管线。

用法:
  python scripts_ablation/e4_e3cadets.py \
      --windows cache/darpa/e3cadets_w8_9d122ad1298a59ed.jsonl \
      --templates cache/darpa/e3cadets_templates_w8.jsonl \
      --gt /home/huguangze/pidsmaker/Ground_Truth/threatrace/E3-CADETS/ground_truth.txt \
      --train-wins 0,1,2,3 --test-wins 4,5,6,7 \
      --figs identity,tered --encs none,dual_naive,rate --results results/e4a.csv
"""
from __future__ import annotations
import os, sys, time, argparse, gc
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import numpy as np

from rate_core import load_graphs
from reduction.tered import TeRedOperator
from reduction.base import IdentityOperator, node_map_sanity
from features.rate import node_feature_matrix, make_type_vocab, label_vector
from features.v4extras import extend_with_v4
from models.detector import BenignEnsemble
from eval.metrics import append_csv
from scripts.e5_f1_eval import _edges_index
from scripts_ablation.e1_enc_matrix import alert_pipeline, alert_metrics


def load_gt_file(path):
    gt = set()
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            tok = line.split(",")[0].strip()
            # orthrus 风格 CSV 也兼容：第一列即 uuid
            if tok and not tok.startswith("#"):
                gt.add(tok)
    return gt


def label(g, gt):
    hit = 0
    for nid in g.nodes:
        if nid in gt:
            g.labels[nid] = 1
            hit += 1
    return hit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", required=True)
    ap.add_argument("--templates", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--train-wins", default="0,1,2,3")
    ap.add_argument("--test-wins", default="4,5,6,7")
    ap.add_argument("--figs", default="identity,tered")
    ap.add_argument("--encs", default="none,dual_naive,rate")
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
    ap.add_argument("--results", required=True)
    a = ap.parse_args()

    figs = [x.strip() for x in a.figs.split(",") if x.strip()]
    encs = [x.strip() for x in a.encs.split(",") if x.strip()]
    tr_w = [int(x) for x in a.train_wins.split(",") if x.strip() != ""]
    te_w = [int(x) for x in a.test_wins.split(",") if x.strip() != ""]

    gt = load_gt_file(a.gt)
    wins = load_graphs(a.windows)
    print("[e4a] 窗口 %d 个, GT %d 个 uuid" % (len(wins), len(gt)), flush=True)
    for i, g in enumerate(wins):
        n = label(g, gt)
        print("   win%d: %d 节点 / %d 边, GT 命中 %d"
              % (i, g.n_nodes(), g.n_edges(), n), flush=True)

    tpls = load_graphs(a.templates)
    print("[e4a] 模板 %d 个" % len(tpls), flush=True)

    ops = {}
    if "identity" in figs:
        ops["identity"] = IdentityOperator()
    if "tered" in figs:
        ops["tered"] = TeRedOperator(tpls, max_instances=a.max_instances,
                                     max_total=a.max_total, share_k=a.share_k)

    reduced = {}
    for fig in figs:
        reduced[fig] = []
        for i in tr_w:
            t0 = time.time()
            r = ops[fig].reduce(wins[i])
            node_map_sanity(r, wins[i])
            reduced[fig].append(r)
            st = r.stats
            print("[e4a] %s 训练窗 win%d: %d -> %d 节点 (%.1fs)"
                  % (fig, i, wins[i].n_nodes(), r.Gp.n_nodes(), time.time() - t0),
                  flush=True)

    t_all = time.time()
    for i in te_w:
        g = wins[i]
        print("\n[e4a] ===== 测试窗 win%d: %d 节点 =====" % (i, g.n_nodes()),
              flush=True)
        for fig in figs:
            t0 = time.time()
            r_te = ops[fig].reduce(g)
            node_map_sanity(r_te, g)
            Gp = r_te.Gp
            gpt = [r.Gp for r in reduced[fig]]
            print("  [%s] %d -> %d 节点 (%.1fs)"
                  % (fig, g.n_nodes(), Gp.n_nodes(), time.time() - t0), flush=True)
            for enc in encs:
                tc = time.time()
                vocab = make_type_vocab(gpt + [Gp])
                Xs = []
                for Gt in gpt:
                    X, nids_t, _ = node_feature_matrix(
                        Gt, encoding=enc, tape_dim=a.tape_dim,
                        tape_base=a.tape_base, semantic=a.semantic, vocab=vocab)
                    et = _edges_index(Gt, nids_t)
                    X, _ = extend_with_v4(X, nids_t, Gt, et, scale=a.v4_scale)
                    Xs.append(X)
                X, nids, _ = node_feature_matrix(
                    Gp, encoding=enc, tape_dim=a.tape_dim,
                    tape_base=a.tape_base, semantic=a.semantic, vocab=vocab)
                edges = _edges_index(Gp, nids)
                X, _ = extend_with_v4(X, nids, Gp, edges, scale=a.v4_scale)
                y = label_vector(Gp, nids)
                det = BenignEnsemble().fit(Xs, ["tr%d" % k for k in range(len(Xs))])
                score = det.anomaly_scores(X)
                clusters = alert_pipeline(
                    score, edges, len(nids), top_k=a.top_k, bfs_q=a.bfs_q,
                    min_cluster=a.min_cluster, max_cluster=a.max_cluster,
                    max_alerts=a.max_alerts)
                m = alert_metrics(clusters, y, score, a.top_k)
                row = dict(dataset="e3_cadets", window="win%d" % i, fig=fig,
                           encoding=enc, n_nodes_Gp=Gp.n_nodes(),
                           n_edges_Gp=len(edges), n_gt=int(y.sum()),
                           n_regions=r_te.stats.get("n_regions", 0),
                           runtime_s=round(time.time() - tc, 1))
                row.update(m)
                append_csv(a.results, row)
                print("      (%s) nodeF1=%.4f auc=%.4f cov=%.4f alerts=%d "
                      "best=%.3f@b%d" % (enc, m["node_best_f1"], m["node_auc"],
                                         m["node_cov"], m["n_alerts"],
                                         m["best_F1_alert"], m["best_b"]),
                      flush=True)
                del X, Xs, det, score, edges
                gc.collect()
            del r_te, Gp, gpt
            gc.collect()
    print("\n[e4a] 完成 (%.0fs) -> %s" % (time.time() - t_all, a.results),
          flush=True)


if __name__ == "__main__":
    main()
