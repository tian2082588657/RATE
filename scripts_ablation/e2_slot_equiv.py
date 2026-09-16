# -*- coding: utf-8 -*-
"""scripts_ablation/e2_slot_equiv.py — E2：把评估单元从"分区"换成"归约未触碰的
真值槽位"，对"归约零代价"做有功效的等价性检验。

动机
----
e7_origunit.csv 的逐分区配对只有 n=7、sd=0.161 -> 90% CI ~ [-0.157, +0.080]，
任何 delta<0.15 的 TOST 都过不了。改为在"未被归约吸收的 GT 实体"上配对后，两种
图的**正例集与分母完全相同**（同一批 uuid），配对单元从 7 升到 82，功效提升一个
数量级；而且"在存活下来的真值实体上，归约图不劣于未归约图"本身就是一句干净、
可检验、有分量的话。

输出
----
--results  每 (分区, 配置) 一行：n_kept / covered_kept / recall_kept /
           alert_P_kept / F1_kept / n_alerts
--slots    每 (分区, 配置, 槽位) 一行：slot_uuid / hit(0|1) —— 供配对检验

用法
----
  python scripts_ablation/e2_slot_equiv.py \
    --test-files bin.116,bin.117,bin.118,bin.119@500000,bin.120,bin.6@400000,bin.7@400000 \
    --figs identity,tered --results results/e2_slot.csv --slots results/e2_slots.csv
"""
from __future__ import annotations
import os, sys, time, argparse, gc
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
                               find_templates, load_gt)
from scripts.e6_alert_eval import parse_spec
from scripts_ablation.graphcache import parse_cached

# 复用 E1 的告警管线（口径与主表一致）
from scripts_ablation.e1_enc_matrix import alert_pipeline, alert_metrics

BUDGETS = [1, 2, 3, 5, 10, 20]


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
    ap.add_argument("--figs", default="identity,tered")
    ap.add_argument("--encs", default="rate,rate_ratio",
                    help="identity 侧用 rate，tered 侧用 rate_ratio(论文方法) —— "
                         "按 --figs 顺序对应")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--graph-cache", default="cache/darpa/graphcache",
                    help="规范图解析缓存目录")
    ap.add_argument("--no-reduce-cache", action="store_true")
    ap.add_argument("--results", required=True)
    ap.add_argument("--slots", default=None)
    a = ap.parse_args()

    figs = [x.strip() for x in a.figs.split(",") if x.strip()]
    encs = [x.strip() for x in a.encs.split(",") if x.strip()]
    if len(encs) == 1:
        encs = encs * len(figs)
    assert len(encs) == len(figs), "--encs 数量须等于 --figs 数量（或给 1 个）"

    from rate_core import load_graphs
    tpl_path = a.templates or find_templates(
        os.path.join(a.cache, "darpa", "e5_templates", "*.jsonl"))
    tpls = load_graphs(tpl_path)
    tpl_stem = os.path.splitext(os.path.basename(tpl_path))[0]
    rcache_dir = os.path.join(a.cache, "darpa", "e5_f1_reduce")
    use_rcache = (not a.no_reduce_cache) and a.share_k == -1

    ops = {"identity": IdentityOperator(),
           "tered": TeRedOperator(tpls, max_instances=a.max_instances,
                                  max_total=a.max_total, share_k=a.share_k)}

    def _n_pos(res):
        return sum(1 for v in res.Gp.labels.values() if v == 1)

    def reduce_one(fig, g, rec_tag, expect_pos=True):
        op_tag = "id" if fig == "identity" else "tered"
        cpath = os.path.join(
            rcache_dir, "%s_%s_k%d_%s_%s.pkl" % (op_tag, tpl_stem, a.share_k,
                                                 rec_tag, g.gid))
        if use_rcache and os.path.exists(cpath):
            res = load_pickle(cpath)
            node_map_sanity(res, g)
            if expect_pos and _n_pos(res) == 0:
                print("[e2] 警告：缓存 %s 无正例标签，重新归约"
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
    train_res = {}
    for fig in figs:
        train_res[fig] = [reduce_one(fig,
                                     parse_cached(a.data_dir, nm,
                                                  a.max_records_train or None,
                                                  a.graph_cache)[0],
                                     "tr%d" % a.max_records_train,
                                     expect_pos=False)
                          for nm in train_names]
        print("[e2] %s 训练图归约完成 (%d)" % (fig, len(train_names)), flush=True)

    t0 = time.time()
    for spec in [s.strip() for s in a.test_files.split(",") if s.strip()]:
        nm, mr = parse_spec(a.data_dir, spec)
        rec_tag = "at%d" % mr if mr else "atfull"
        g, _m = parse_cached(a.data_dir, nm, mr or None, a.graph_cache)
        gt = load_gt(a.gt)
        assert gt, "ground truth 为空，检查 --gt 指向的目录: %s" % a.gt
        label_graph(g, gt)
        gt_ids = {nid for nid, l in g.labels.items() if l == 1}
        res_te = reduce_one("tered", g, rec_tag)
        # 未触碰槽位：归约图里仍以自身 id 存在的 GT 实体
        gt_kept = sorted(u for u in gt_ids if res_te.node_map.get(u, u) == u)
        print("\n[e2] ===== %s: 原始GT=%d, 未触碰槽位=%d ====="
              % (spec, len(gt_ids), len(gt_kept)), flush=True)

        for fig, enc in zip(figs, encs):
            Gp = reduce_one(fig, g, rec_tag).Gp
            gpt = [r.Gp for r in train_res[fig]]
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
                Gp, encoding=enc, tape_dim=a.tape_dim, tape_base=a.tape_base,
                semantic=a.semantic, vocab=vocab)
            edges = _edges_index(Gp, nids)
            X, _ = extend_with_v4(X, nids, Gp, edges, scale=a.v4_scale)
            y = label_vector(Gp, nids)
            det = BenignEnsemble().fit(Xs, ["tr%d" % i for i in range(len(Xs))])
            score = det.anomaly_scores(X)
            clusters = alert_pipeline(score, edges, len(nids), top_k=a.top_k,
                                      bfs_q=a.bfs_q, min_cluster=a.min_cluster,
                                      max_cluster=a.max_cluster,
                                      max_alerts=a.max_alerts)
            alerted_ids = set()
            for c, _sc in clusters:
                for i in c:
                    alerted_ids.add(nids[int(i)])
            cfg = "%s+%s" % (fig, enc)
            hits = [1 if u in alerted_ids else 0 for u in gt_kept]
            covered = int(sum(hits))
            recall = covered / len(gt_kept) if gt_kept else 0.0
            # 告警精度（槽位口径）：含 >=1 未触碰 GT 的告警占比
            tp_alert = 0
            for c, _sc in clusters:
                ids = {nids[int(i)] for i in c}
                if any(u in ids for u in gt_kept):
                    tp_alert += 1
            alert_P = tp_alert / len(clusters) if clusters else 0.0
            f1 = (2 * alert_P * recall / (alert_P + recall)
                  if (alert_P + recall) else 0.0)
            m = alert_metrics(clusters, y, score, a.top_k)
            row = dict(dataset="e5_cadets", config=cfg, fig=fig, encoding=enc,
                       file=spec, n_gt_orig=len(gt_ids),
                       n_kept=len(gt_kept), covered_kept=covered,
                       recall_kept=round(recall, 4),
                       alert_P_kept=round(alert_P, 4),
                       F1_kept=round(f1, 4),
                       n_alerts=len(clusters), alert_TP_kept=tp_alert,
                       n_nodes_Gp=Gp.n_nodes())
            row.update({k: m[k] for k in
                        ("node_best_f1", "node_cov", "F1_alert",
                         "best_F1_alert", "best_b", "alert_P")})
            append_csv(a.results, row)
            if a.slots:
                for u, h in zip(gt_kept, hits):
                    append_csv(a.slots, dict(dataset="e5_cadets", config=cfg,
                                             fig=fig, encoding=enc, file=spec,
                                             slot=u, hit=h))
            print("  [%s] kept=%d covered=%d recall=%.3f P_kept=%.3f F1_kept=%.3f "
                  "| alerts=%d nodeF1=%.4f" % (cfg, len(gt_kept), covered,
                                               recall, alert_P, f1,
                                               len(clusters),
                                               m["node_best_f1"]), flush=True)
            del X, Xs, det, score, edges
            gc.collect()
        del g, res_te
        gc.collect()
    print("\n[e2] 完成 (%.0fs) -> %s" % (time.time() - t0, a.results), flush=True)


if __name__ == "__main__":
    main()
