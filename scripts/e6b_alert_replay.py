# -*- coding: utf-8 -*-
"""scripts/e6b_alert_replay.py — 从归约缓存直接重放告警级评测段（省 parse/归约）。

e6_alert_eval.py 每次运行都要重新 parse 训练/测试文件（atfull 一个 ~6 分钟），
只为了改 max_alerts / bfs_q / top_k 等告警参数太浪费。本脚本直接 load
cache/darpa/e5_f1_reduce/ 下的归约产物（res.Gp 含 labels/类型），跳过
parse+reduce，只跑：特征 -> BenignEnsemble -> score -> alert_pipeline ->
alert_metrics，与 e6 评测段逐位一致，可任意调告警参数。

用法:
  python scripts/e6b_alert_replay.py \
      --test-files bin.116,bin.117,bin.118,bin.119@500000,bin.120,bin.6@400000,bin.7@400000 \
      --max-alerts 5 --results results/e6_alert_max5.csv
"""
from __future__ import annotations
import os, sys, glob, time, argparse, gc

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import numpy as np

from rate_core import load_pickle
from features.rate import node_feature_matrix, make_type_vocab, label_vector
from features.v4extras import extend_with_v4
from models.detector import BenignEnsemble
from eval.metrics import append_csv

from scripts.e5_f1_eval import _edges_index, resolve_name, find_templates
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
                    help="逗号分隔; 'bin.119@500000' 截断口径, 无 @ = atfull")
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
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--results", required=True)
    a = ap.parse_args()

    # ---------- 模板 stem（仅用于定位缓存文件名） ----------
    tpl_path = a.templates or find_templates(
        os.path.join(a.cache, "darpa", "e5_templates", "*.jsonl"))
    tpl_stem = os.path.splitext(os.path.basename(tpl_path))[0]
    print(f"[replay] 模板 stem: {tpl_stem}", flush=True)
    rcache_dir = os.path.join(a.cache, "darpa", "e5_f1_reduce")

    # ---------- 训练归约产物（id / tered 各 5 张） ----------
    train_names = [s.strip() for s in a.train_files.split(",") if s.strip()]
    rec_tag_tr = f"tr{a.max_records_train}"
    Gp_tr_id, Gp_tr_te = [], []
    for nm in train_names:
        gid = resolve_name(a.data_dir, nm)
        Gp_tr_id.append(load_res(rcache_dir, "id", tpl_stem,
                                 rec_tag_tr, gid).Gp)
        Gp_tr_te.append(load_res(rcache_dir, "tered", tpl_stem,
                                 rec_tag_tr, gid).Gp)
    print(f"[replay] 训练产物: id {len(Gp_tr_id)} 张, tered {len(Gp_tr_te)} 张 "
          f"({rec_tag_tr})", flush=True)

    # ---------- 逐测试文件重放 ----------
    specs = [s.strip() for s in a.test_files.split(",") if s.strip()]
    t_start = time.time()
    for spec in specs:
        nm, mr = parse_spec(a.data_dir, spec)
        rec_tag = f"at{mr}" if mr else "atfull"
        gid = resolve_name(a.data_dir, nm)
        res_id = load_res(rcache_dir, "id", tpl_stem, rec_tag, gid)
        res_te = load_res(rcache_dir, "tered", tpl_stem, rec_tag, gid)
        Gp_id, Gp_te = res_id.Gp, res_te.Gp
        print(f"\n[replay] ===== {spec} ({rec_tag}): "
              f"id {Gp_id.n_nodes()} 节点 / tered {Gp_te.n_nodes()} 节点 =====",
              flush=True)

        configs = [
            ("identity+rate", [r for r in Gp_tr_id], Gp_id, "rate"),
            ("TeRed+naive", [r for r in Gp_tr_te], Gp_te, "dual_naive"),
            ("TeRed+RATE", [r for r in Gp_tr_te], Gp_te, "rate"),
        ]
        for cname, gpt, Gp_test, enc in configs:
            tc = time.time()
            vocab = make_type_vocab(gpt + [Gp_test])
            Xs_train = []
            for Gp in gpt:
                X, nids_t, _ = node_feature_matrix(
                    Gp, encoding=enc, tape_dim=a.tape_dim,
                    tape_base=a.tape_base, semantic=a.semantic, vocab=vocab)
                et = _edges_index(Gp, nids_t)
                X, _ = extend_with_v4(X, nids_t, Gp, et, scale=a.v4_scale)
                Xs_train.append(X)
            det = BenignEnsemble().fit(Xs_train,
                                       [f"tr{i}" for i in range(len(Xs_train))])

            X, nids, _ = node_feature_matrix(
                Gp_test, encoding=enc, tape_dim=a.tape_dim,
                tape_base=a.tape_base, semantic=a.semantic, vocab=vocab)
            edges = _edges_index(Gp_test, nids)
            X, _ = extend_with_v4(X, nids, Gp_test, edges, scale=a.v4_scale)
            score = det.anomaly_scores(X)
            y = label_vector(Gp_test, nids)
            n_gt = int(y.sum())

            clusters = alert_pipeline(score, edges, len(nids),
                                      top_k=a.top_k, bfs_q=a.bfs_q,
                                      min_cluster=a.min_cluster,
                                      max_cluster=a.max_cluster,
                                      max_alerts=a.max_alerts)
            m = alert_metrics(clusters, y, score, a.top_k)

            row = {"dataset": "e5_cadets", "config": cname,
                   "operator": "identity" if cname == "identity+rate"
                   else "tered",
                   "encoding": enc, "file": spec, "rec_tag": rec_tag,
                   "is_attack": n_gt > 0,
                   "n_nodes_Gp": Gp_test.n_nodes(), "n_edges_Gp": len(edges),
                   "top_k": a.top_k, "bfs_q": a.bfs_q,
                   "min_cluster": a.min_cluster, "max_cluster": a.max_cluster,
                   "max_alerts": a.max_alerts, "v4_scale": a.v4_scale,
                   "runtime_s": round(time.time() - tc, 1)}
            row.update(m)
            append_csv(a.results, row)

            tag = "ATK" if n_gt > 0 else "BENIGN"
            print(f"  [{cname}] n_gt={n_gt} alerts={m['n_alerts']} "
                  f"TP={m['alert_TP']} alert_P={m['alert_P']:.2f} "
                  f"cov={m['node_cov']:.2f} F1={m['F1_alert']:.3f} "
                  f"bestF1={m['best_F1_alert']:.3f}@b{m['best_b']} "
                  f"first_hit={m['first_hit']} top1={m['top1_hit']} "
                  f"nodes={m['n_alerted_nodes']} flat_cov={m['flat_topk_cov']:.2f} "
                  f"({row['runtime_s']}s)", flush=True)
            del X, Xs_train, det, edges
            gc.collect()
        del Gp_id, Gp_te, res_id, res_te
        gc.collect()

    print(f"\n[replay] 完成 ({time.time()-t_start:.0f}s), "
          f"结果追加至 {a.results}", flush=True)


if __name__ == "__main__":
    main()
