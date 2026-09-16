# -*- coding: utf-8 -*-
"""scripts_ablation/e3_sweep.py — E3：归约强度扫描（主图）。

问题
----
`find_instances` 的成本几乎只由"模板 x seeds"的遍历决定，与区域预算无关
（实测 max_total=200 -> 189.8s，max_total=1000 -> 194.3s）。所以直接对每个强度
各跑一次归约，成本是 O(强度数 x 匹配时间)，大图上不可行。

做法
----
对每张图只做 **一次完整匹配**（无预算限制），把区域列表缓存下来；之后每个强度
只做"按 (max_instances, max_total) 截断 + 塌缩"。由于 `find_instances` 本身是
顺序贪心（接受一个区域就 budget-=1），全量结果的前 k 个满足约束的区域与
"直接以 k 为预算"跑出来的结果一致，因此截断是等价的近似。

用法
----
  python scripts_ablation/e3_sweep.py \
    --test-files bin.118,bin.119@500000,bin.6@400000 \
    --strengths 300:250,300:1000,300:4000,300:16000,300:100000 \
    --encs none,dual_naive,rate --results results/e3_sweep.csv
"""
from __future__ import annotations
import os, sys, time, argparse, gc
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import numpy as np

import reduction.tered as tered_mod
from rate_core import load_pickle, save_pickle
from reduction.tered import TeRedOperator
from reduction.base import node_map_sanity
from features.rate import node_feature_matrix, make_type_vocab, label_vector
from features.v4extras import extend_with_v4
from models.detector import BenignEnsemble
from eval.metrics import append_csv
from scripts.e5_f1_eval import _edges_index, find_templates
from scripts.e6_alert_eval import parse_spec
from scripts_ablation.graphcache import parse_cached
from scripts_ablation.e1_enc_matrix import alert_pipeline, alert_metrics

_ORIG_FIND = tered_mod.find_instances
_FULL = {}


def _install_region_cache():
    """让 find_instances 对每张图只做一次完整匹配，之后按强度截断。"""

    def patched(G, templates, max_instances=100, max_total=1000, **kw):
        key = (G.gid, id(templates))
        if key not in _FULL:
            t0 = time.time()
            full = _ORIG_FIND(G, templates, max_instances=10 ** 9,
                              max_total=10 ** 9, **kw)
            _FULL[key] = full
            print("    [match] %s 完整匹配 %d 个区域 (%.0fs)"
                  % (G.gid, len(full), time.time() - t0), flush=True)
        full = _FULL[key]
        out, per, budget = [], defaultdict(int), max_total
        for r in full:
            if budget <= 0:
                break
            tid = r["tpl_id"]
            if per[tid] >= max_instances:
                continue
            out.append(r)
            per[tid] += 1
            budget -= 1
        return out

    tered_mod.find_instances = patched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/TeRed+RATE/dataset/darpae5/cadets")
    ap.add_argument("--train-files", default="bin.1,bin.2,bin.3,bin.4,bin.5")
    ap.add_argument("--max-records-train", type=int, default=400000)
    ap.add_argument("--test-files", required=True)
    ap.add_argument("--gt", default="ground_truth")
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
    ap.add_argument("--strengths", default="300:250,300:1000,300:4000,300:16000",
                    help="逗号分隔的 max_instances:max_total 强度点")
    ap.add_argument("--encs", default="none,dual_naive,rate")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--graph-cache", default="cache/darpa/graphcache")
    ap.add_argument("--results", required=True)
    a = ap.parse_args()

    strengths = []
    for s in a.strengths.split(","):
        mi, mt = s.split(":")
        strengths.append((int(mi), int(mt)))
    encs = [x.strip() for x in a.encs.split(",") if x.strip()]

    _install_region_cache()

    tpl_path = a.templates or find_templates(
        os.path.join(a.cache, "darpa", "e5_templates", "*.jsonl"))
    from rate_core import load_graphs
    tpls = load_graphs(tpl_path)
    print("[e3] 模板 %d 个; 强度点 %s" % (len(tpls), strengths), flush=True)

    train_names = [s.strip() for s in a.train_files.split(",") if s.strip()]
    train_graphs = [parse_cached(a.data_dir, nm, a.max_records_train or None,
                                 a.graph_cache)[0] for nm in train_names]
    print("[e3] 训练图 %d 张就绪" % len(train_graphs), flush=True)

    t_all = time.time()
    for spec in [s.strip() for s in a.test_files.split(",") if s.strip()]:
        nm, mr = parse_spec(a.data_dir, spec)
        g, _m = parse_cached(a.data_dir, nm, mr or None, a.graph_cache)
        print("\n[e3] ===== %s: %d 节点 / %d 边 ====="
              % (spec, g.n_nodes(), g.n_edges()), flush=True)

        for (mi, mt) in strengths:
            op = TeRedOperator(tpls, max_instances=mi, max_total=mt,
                               share_k=a.share_k)
            t0 = time.time()
            res_tr = [op.reduce(G) for G in train_graphs]
            res_te = op.reduce(g)
            red_s = time.time() - t0
            n_cut = 1 - res_te.Gp.n_nodes() / max(1, g.n_nodes())
            e_cut = 1 - res_te.Gp.n_edges() / max(1, g.n_edges())
            print("  [s mi=%d mt=%d] 区域=%d 节点-%.1f%% 边-%.1f%% (%.0fs)"
                  % (mi, mt, res_te.stats.get("n_regions", 0), 100 * n_cut,
                     100 * e_cut, red_s), flush=True)

            gpt = [r.Gp for r in res_tr]
            Gp = res_te.Gp
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
                det = BenignEnsemble().fit(Xs, ["tr%d" % i for i in range(len(Xs))])
                score = det.anomaly_scores(X)
                clusters = alert_pipeline(
                    score, edges, len(nids), top_k=a.top_k, bfs_q=a.bfs_q,
                    min_cluster=a.min_cluster, max_cluster=a.max_cluster,
                    max_alerts=a.max_alerts)
                m = alert_metrics(clusters, y, score, a.top_k)
                row = dict(dataset="e5_cadets", file=spec, encoding=enc,
                           max_instances=mi, max_total=mt,
                           n_nodes_orig=g.n_nodes(), n_edges_orig=g.n_edges(),
                           n_nodes_red=Gp.n_nodes(), n_edges_red=Gp.n_edges(),
                           node_cut=round(n_cut, 4), edge_cut=round(e_cut, 4),
                           n_regions=res_te.stats.get("n_regions", 0),
                           n_gt_orig=len({n for n, l in g.labels.items() if l == 1}),
                           reduce_s=round(red_s, 1),
                           runtime_s=round(time.time() - tc, 1))
                row.update(m)
                append_csv(a.results, row)
                print("      (%s) nodeF1=%.4f auc=%.4f cov=%.4f best=%.3f@b%d"
                      % (enc, m["node_best_f1"], m["node_auc"], m["node_cov"],
                         m["best_F1_alert"], m["best_b"]), flush=True)
                del X, Xs, det, score, edges
                gc.collect()
            del res_tr, res_te, gpt, Gp
            gc.collect()
        del g
        gc.collect()
    print("\n[e3] 完成 (%.0fs) -> %s" % (time.time() - t_all, a.results),
          flush=True)


if __name__ == "__main__":
    main()
