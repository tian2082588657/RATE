# -*- coding: utf-8 -*-
"""scripts/e7_origunit_eval.py — 统一评估单元（映射回原图）公平评测。

背景（应对审稿意见）：
  identity 在【未归约图】上评估，TeRed 在【归约图】上评估；且归约塌缩含 GT 的
  区域后，入口 M / 出口 N 都会继承正例标签 -> 归约图正例数被放大（如 bin.116
  34->46）。于是 coverage 的分母在两配置间不一致，直接比较 F1 不公平。

本脚本把两种配置的告警都【映射回原始节点集】：
  - 对归约图：节点 u(原始) 的代表节点 = node_map[u]（未吸收则 u 自身）
  - orig_recall = |{u in GT_orig : node_map[u] 落在某告警簇内}| / |GT_orig|
  - orig_P      = 含 >=1 原始 GT 的告警占比
  - F1_orig     = harmonic(orig_P, orig_recall)   <-- 统一单元下的可比 F1
另输出：GT 塌缩存活率、边界 GT 节点度失真、归约规模统计。

用法:
  python scripts/e7_origunit_eval.py \
    --test-files bin.116,bin.117,bin.118,bin.119@500000,bin.120,bin.6@400000,bin.7@400000 \
    --results results/e7_origunit.csv
"""
from __future__ import annotations
import os, sys, glob, time, argparse, gc
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import numpy as np

from rate_core import load_graphs, save_pickle, load_pickle
from reduction.tered import TeRedOperator
from reduction.base import IdentityOperator, node_map_sanity
from features.rate import node_feature_matrix, make_type_vocab, label_vector
from features.v4extras import extend_with_v4
from models.detector import BenignEnsemble
from eval.metrics import append_csv

from scripts.e5_f1_eval import (_edges_index, load_gt, parse_one, label_graph,
                                resolve_name, find_templates)
from scripts.e6_alert_eval import alert_pipeline, parse_spec


def orig_metrics(alerted_node_ids, G, gt_ids, node_map):
    """在原始 GT 实体集上评估。

    alerted_node_ids: 归约图(或原图)中被告警覆盖的节点 id 集合
    G: 原始图；gt_ids: 原始 GT 节点 id 集合；node_map: π
    """
    n_orig = len(gt_ids)
    if n_orig == 0:
        return dict(n_orig=0, n_kept=0, n_absorbed=0, orig_covered=0,
                    orig_recall=0.0)
    kept = [u for u in gt_ids if node_map.get(u, u) == u]
    covered = [u for u in gt_ids if node_map.get(u, u) in alerted_node_ids]
    return dict(n_orig=n_orig, n_kept=len(kept),
                n_absorbed=n_orig - len(kept),
                orig_covered=len(covered),
                orig_recall=len(covered) / n_orig)


def degree_distortion(G, Gp, node_map, gt_ids):
    """GT 节点的度失真：未吸收 GT 节点 原始 count 度 vs 归约图 count 度。

    单次遍历归约图预计算入/出度（O(|E'|)），再做 GT 查表。
    """
    din_p = defaultdict(int)
    dout_p = defaultdict(int)
    for e in Gp.edges:
        dout_p[e["src"]] += 1
        din_p[e["dst"]] += 1
    din, dout = G.degrees(use_mu=False)     # 原图 count 度
    n_boundary = 0
    worst = 0.0
    for u in gt_ids:
        nu = node_map.get(u, u)
        if nu != u:
            continue                        # 被吸收，无对应节点
        ci, co = din.get(u, 0), dout.get(u, 0)
        mi, mo = din_p.get(nu, 0), dout_p.get(nu, 0)
        if mi != ci or mo != co:
            n_boundary += 1
            worst = max(worst, abs(mi - ci), abs(mo - co))
    return n_boundary, worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/TeRed+RATE/dataset/darpae5/cadets")
    ap.add_argument("--train-files", default="bin.1,bin.2,bin.3,bin.4,bin.5")
    ap.add_argument("--test-files", required=True)
    ap.add_argument("--gt", default="ground_truth")
    ap.add_argument("--max-records-train", type=int, default=400000)
    ap.add_argument("--templates", default=None)
    ap.add_argument("--share-k", type=int, default=-1)
    ap.add_argument("--semantic", default="onehot")
    ap.add_argument("--tape-dim", type=int, default=8)
    ap.add_argument("--tape-base", type=float, default=10000.0)
    ap.add_argument("--v4-scale", default="raw", choices=["raw", "robust"])
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--bfs-q", type=int, default=75)
    ap.add_argument("--min-cluster", type=int, default=3)
    ap.add_argument("--max-cluster", type=int, default=200)
    ap.add_argument("--max-alerts", type=int, default=20)
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--results", default="results/e7_origunit.csv")
    a = ap.parse_args()

    gt = load_gt(a.gt)
    print(f"[e7] ground truth: {len(gt)} UUIDs", flush=True)

    train_names = [s.strip() for s in a.train_files.split(",") if s.strip()]
    train_graphs = [parse_one(a.data_dir, nm, a.max_records_train or None)[0]
                    for nm in train_names]
    print(f"[e7] 训练图解析完成", flush=True)

    tpl_path = a.templates or find_templates(
        os.path.join(a.cache, "darpa", "e5_templates", "*.jsonl"))
    tpls = load_graphs(tpl_path)
    use_rcache = a.share_k == -1
    tpl_stem = os.path.splitext(os.path.basename(tpl_path))[0]
    rcache_dir = os.path.join(a.cache, "darpa", "e5_f1_reduce")

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

    res_train_id = [reduce_one(IdentityOperator(), g, "id",
                               f"tr{a.max_records_train}") for g in train_graphs]
    tered = TeRedOperator(tpls, max_instances=300, max_total=5000,
                          share_k=a.share_k)
    res_train_te = [reduce_one(tered, g, "tered",
                               f"tr{a.max_records_train}") for g in train_graphs]
    print(f"[e7] 训练图归约完成", flush=True)

    specs = [s.strip() for s in a.test_files.split(",") if s.strip()]
    for spec in specs:
        nm, mr = parse_spec(a.data_dir, spec)
        rec_tag = f"at{mr}" if mr else "atfull"
        g, meta = parse_one(a.data_dir, nm, mr or None)
        hit, _ = label_graph(g, gt)
        gt_ids = {nid for nid, l in g.labels.items() if l == 1}
        print(f"\n[e7] ===== {spec}: {g.n_nodes()} 节点, 原始GT={hit} =====", flush=True)

        res_id = reduce_one(IdentityOperator(), g, "id", rec_tag)
        res_te = reduce_one(tered, g, "tered", rec_tag)

        # 塌缩统计
        n_gt_kept = sum(1 for u in gt_ids if res_te.node_map.get(u, u) == u)
        n_boundary, worst_dd = degree_distortion(
            g, res_te.Gp, res_te.node_map, gt_ids)
        print(f"[e7] GT 存活 {n_gt_kept}/{hit}, 边界 GT={n_boundary} "
              f"最大度差={worst_dd:.0f}; 归约节点 {g.n_nodes()}->{res_te.Gp.n_nodes()}, "
              f"区域数={res_te.stats.get('n_regions')}, "
              f"移除={res_te.stats.get('n_removed_nodes')}", flush=True)

        configs = [
            ("identity+rate", [r.Gp for r in res_train_id], res_id, "rate"),
            ("TeRed+naive", [r.Gp for r in res_train_te], res_te, "dual_naive"),
            ("TeRed+RATE", [r.Gp for r in res_train_te], res_te, "rate"),
        ]
        for cname, gpt, res_test, enc in configs:
            Gp_test = res_test.Gp
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

            clusters = alert_pipeline(score, edges, len(nids), top_k=a.top_k,
                                      bfs_q=a.bfs_q, min_cluster=a.min_cluster,
                                      max_cluster=a.max_cluster,
                                      max_alerts=a.max_alerts)
            alerted_ids = set()
            for c, _ in clusters:
                for i in c:
                    alerted_ids.add(nids[int(i)])
            n_gt_reduced = int(y.sum())
            # 归约图单元指标
            tp_flags = [int(y[c].sum() > 0) for c, _ in clusters]
            alert_P_red = (sum(tp_flags) / len(clusters)) if clusters else 0.0
            seen = set()
            for c, _ in clusters:
                seen.update(c[np.nonzero(y[c])[0]].tolist())   # 回映射全局下标
            node_cov_red = len(seen) / n_gt_reduced if n_gt_reduced else 0.0
            # ---- 原始单元指标 ----
            om = orig_metrics(alerted_ids, g, gt_ids, res_test.node_map)
            # 告警精确率（原始单元）：含 >=1 原始 GT 的告警
            tp_orig = 0
            for c, _ in clusters:
                ids = {nids[int(i)] for i in c}
                if any(res_test.node_map.get(u, u) in ids for u in gt_ids):
                    tp_orig += 1
            alert_P_orig = tp_orig / len(clusters) if clusters else 0.0
            f1_orig = (2 * alert_P_orig * om["orig_recall"]
                       / (alert_P_orig + om["orig_recall"])
                       if (alert_P_orig + om["orig_recall"]) else 0.0)
            f1_red = (2 * alert_P_red * node_cov_red
                      / (alert_P_red + node_cov_red)
                      if (alert_P_red + node_cov_red) else 0.0)
            # 分析负载：告警覆盖的原始节点数
            n_alerted_orig = 0
            if res_test.node_map is not None:
                inv = defaultdict(list)
                for u, nu in res_test.node_map.items():
                    inv[nu].append(u)
                for nid in alerted_ids:
                    n_alerted_orig += len(inv.get(nid, ()))
            n_alerted_red = sum(len(c) for c, _ in clusters)
            row = dict(dataset="e5_cadets", config=cname, encoding=enc, file=spec,
                       n_orig_gt=hit, n_gt_kept=n_gt_kept,
                       n_gt_absorbed=hit - n_gt_kept, n_gt_reduced=n_gt_reduced,
                       n_regions=res_test.stats.get("n_regions", 0),
                       n_removed=res_test.stats.get("n_removed_nodes", 0),
                       n_nodes_orig=g.n_nodes(), n_nodes_Gp=Gp_test.n_nodes(),
                       n_boundary_gt=n_boundary, worst_deg_delta=round(worst_dd, 1),
                       n_alerts=len(clusters), alert_TP_orig=tp_orig,
                       alert_P_orig=round(alert_P_orig, 4),
                       orig_covered=om["orig_covered"],
                       orig_recall=round(om["orig_recall"], 4),
                       F1_orig=round(f1_orig, 4),
                       alert_P_red=round(alert_P_red, 4),
                       node_cov_red=round(node_cov_red, 4),
                       F1_red=round(f1_red, 4),
                       n_alerted_red=n_alerted_red,
                       n_alerted_orig=n_alerted_orig,
                       orig_load_prec=round(om["orig_covered"] /
                                            max(1, n_alerted_orig), 5))
            append_csv(a.results, row)
            print(f"  [{cname}] orig_GT={hit} kept={n_gt_kept} "
                  f"red_pos={n_gt_reduced} | orig_recall={om['orig_recall']:.3f} "
                  f"({om['orig_covered']}/{hit}) alert_P_orig={alert_P_orig:.3f} "
                  f"F1_orig={f1_orig:.3f} || red_cov={node_cov_red:.3f} "
                  f"F1_red={f1_red:.3f} | alerts={len(clusters)} "
                  f"load_orig={n_alerted_orig}", flush=True)
            del X, Xs_train, det, edges
            gc.collect()
        del g, res_id, res_te
        gc.collect()

    print(f"\n[e7] 完成 -> {a.results}", flush=True)


if __name__ == "__main__":
    main()
