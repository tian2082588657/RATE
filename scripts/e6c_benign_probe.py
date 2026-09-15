# -*- coding: utf-8 -*-
"""scripts/e6c_benign_probe.py — 纯良性/反事实误报探针（FPR 口径，M2 表 4）。

E5 cadets 是长时程 APT 战役，几乎所有文件都命中若干"贯穿基础设施"GT
节点，不存在严格 0 命中的纯良性文件。为补"纯良性日期的每日假告警数"
这一论文口径，本脚本提供两种探针：

  A. 真实低命中文件：parse -> identity 管线 -> n_alerts / alert_TP；
     FP_real = n_alerts - alert_TP（真实 GT 下被判定为假的告警数）。
  B. 反事实纯良性（--as-benign）：把 GT 向量 y 全置 0 再跑同管线，
     此时 n_alerts 即"假设该日无任何已知攻击，分析师要看的假告警数"
     （同域同分布的最干净 FPR 估计）。

只跑 identity 配置（FPR 主口径不需要 tered 归约增强；tered 在纯良性
文件上需数小时归约，成本不成比例）。训练侧复用 cache/darpa/e5_f1_reduce/
bin.1-5 identity 归约缓存，与 e6b_alert_replay.py 同款加载。

用法:
  python scripts/e6c_benign_probe.py \
      --test-files bin.52@400000,bin.90@400000 \
      --max-alerts 5 --as-benign --results results/e6c_benign_probe.csv
"""
from __future__ import annotations
import os, sys, time, argparse, gc

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import numpy as np

from rate_core import load_pickle
from reduction.base import IdentityOperator
from features.rate import node_feature_matrix, make_type_vocab, label_vector
from features.v4extras import extend_with_v4
from models.detector import BenignEnsemble
from eval.metrics import append_csv
from adapters import darpa_tc

from scripts.e5_f1_eval import (_edges_index, load_gt, parse_one, label_graph,
                                resolve_name, find_templates)
from scripts.e6_alert_eval import alert_pipeline, alert_metrics, parse_spec


def load_res(cache_dir, op_tag, tpl_stem, rec_tag, gid):
    cpath = os.path.join(cache_dir,
                         f"{op_tag}_{tpl_stem}_k-1_{rec_tag}_{gid}.pkl")
    if not os.path.exists(cpath):
        raise FileNotFoundError(f"归约缓存缺失: {cpath}")
    return load_pickle(cpath)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/TeRed+RATE/dataset/darpae5/cadets")
    ap.add_argument("--train-files", default="bin.1,bin.2,bin.3,bin.4,bin.5")
    ap.add_argument("--max-records-train", type=int, default=400000)
    ap.add_argument("--test-files", required=True,
                    help="逗号分隔; 'bin.52@400000' 截断, 无 @ = atfull")
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
    ap.add_argument("--max-alerts", type=int, default=5)
    ap.add_argument("--as-benign", action="store_true",
                    help="反事实模式: y 全置 0（把该日当作无攻击）")
    ap.add_argument("--cache", default="/TeRed+RATE/code/cache")
    ap.add_argument("--results", required=True)
    a = ap.parse_args()

    gt = load_gt(a.gt)
    print(f"[probe] ground truth: {len(gt)} UUIDs; "
          f"mode={'as-benign(反事实)' if a.as_benign else 'real-GT'}; "
          f"max_alerts={a.max_alerts}", flush=True)

    # ---------- 训练 identity 产物（缓存，与 replay 同款） ----------
    tpl_path = a.templates or find_templates(
        os.path.join(a.cache, "darpa", "e5_templates", "*.jsonl"))
    tpl_stem = os.path.splitext(os.path.basename(tpl_path))[0]
    rcache_dir = os.path.join(a.cache, "darpa", "e5_f1_reduce")
    train_names = [s.strip() for s in a.train_files.split(",") if s.strip()]
    rec_tag_tr = f"tr{a.max_records_train}"
    Gp_tr_id = []
    for nm in train_names:
        gid = resolve_name(a.data_dir, nm)
        Gp_tr_id.append(load_res(rcache_dir, "id", tpl_stem,
                                 rec_tag_tr, gid).Gp)
    print(f"[probe] 训练 identity 产物 {len(Gp_tr_id)} 张 ({rec_tag_tr})",
          flush=True)

    # ---------- 逐测试文件 ----------
    specs = [s.strip() for s in a.test_files.split(",") if s.strip()]
    t_start = time.time()
    for spec in specs:
        nm, mr = parse_spec(a.data_dir, spec)
        rec_tag = f"at{mr}" if mr else "atfull"
        tf = time.time()
        g, meta = parse_one(a.data_dir, nm, mr or None)
        hit, _ = label_graph(g, gt)
        print(f"\n[probe] ===== {spec}: {g.n_nodes()} 节点 {g.n_edges()} 边 "
              f"records={meta['records']} 真实GT命中={hit} "
              f"({time.time()-tf:.0f}s parse) =====", flush=True)

        res_id = IdentityOperator().reduce(g)   # 不写缓存（一次性探针）
        Gp = res_id.Gp
        print(f"[probe] identity 归约: {g.n_nodes()}->{Gp.n_nodes()}",
              flush=True)

        # 训练（identity 口径）
        vocab = make_type_vocab(Gp_tr_id + [Gp])
        Xs_train = []
        for Gpt in Gp_tr_id:
            Xt, nidst, _ = node_feature_matrix(
                Gpt, encoding="rate", tape_dim=a.tape_dim,
                tape_base=a.tape_base, semantic=a.semantic, vocab=vocab)
            et = _edges_index(Gpt, nidst)
            Xt, _ = extend_with_v4(Xt, nidst, Gpt, et, scale=a.v4_scale)
            Xs_train.append(Xt)
        det = BenignEnsemble().fit(Xs_train,
                                   [f"tr{i}" for i in range(len(Xs_train))])

        X, nids, _ = node_feature_matrix(
            Gp, encoding="rate", tape_dim=a.tape_dim, tape_base=a.tape_base,
            semantic=a.semantic, vocab=vocab)
        edges = _edges_index(Gp, nids)
        X, _ = extend_with_v4(X, nids, Gp, edges, scale=a.v4_scale)
        score = det.anomaly_scores(X)

        y = label_vector(Gp, nids)
        if a.as_benign:
            y = np.zeros_like(y)          # 反事实：假设该日无攻击
        clusters = alert_pipeline(score, edges, len(nids),
                                  top_k=a.top_k, bfs_q=a.bfs_q,
                                  min_cluster=a.min_cluster,
                                  max_cluster=a.max_cluster,
                                  max_alerts=a.max_alerts)
        m = alert_metrics(clusters, y, score, a.top_k)

        n_gt_real = int(hit)
        fp = m["n_alerts"] - m["alert_TP"]
        row = {"dataset": "e5_cadets", "mode": "as_benign" if a.as_benign
               else "real_gt", "config": "identity+rate", "encoding": "rate",
               "file": spec, "rec_tag": rec_tag,
               "n_gt_real": n_gt_real, "n_gt_used": int(m["n_gt"]),
               "n_alerts": m["n_alerts"], "alert_TP": m["alert_TP"],
               "FP": fp, "alert_P": m["alert_P"],
               "node_cov": m["node_cov"], "F1_alert": m["F1_alert"],
               "n_alerted_nodes": m["n_alerted_nodes"],
               "runtime_s": round(time.time() - tf, 1)}
        append_csv(a.results, row)

        print(f"  [{'as-benign' if a.as_benign else 'real'}] "
              f"真实GT={n_gt_real} alerts={m['n_alerts']} "
              f"TP={m['alert_TP']} FP={fp} alert_P={m['alert_P']:.2f} "
              f"cov={m['node_cov']:.2f} nodes={m['n_alerted_nodes']} "
              f"({row['runtime_s']}s)", flush=True)
        del X, Xs_train, det, edges, g, res_id, Gp
        gc.collect()

    print(f"\n[probe] 完成 ({time.time()-t_start:.0f}s), "
          f"结果追加至 {a.results}", flush=True)


if __name__ == "__main__":
    main()
