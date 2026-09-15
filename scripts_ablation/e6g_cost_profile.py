# -*- coding: utf-8 -*-
"""scripts_ablation/e6g_cost_profile.py — 端到端成本画像（DeepSeek 0912-2 第 9 条）。

Table 10 只报了 feature extraction / scoring / alert aggregation，排除了
ingestion（解析）与 reduction 算子本身。本脚本逐阶段计时，给出端到端口径：

  parse      : 事件流 -> 原始 provenance graph（ingestion）
  reduce     : 模板匹配 + 折叠（reduction operator，模板预先挖掘）
  encode     : 节点拓扑编码 + 结构特征（feature extraction）
  fit        : BenignEnsemble 训练（每个训练窗口一次）
  score      : 测试窗口打分
  alert      : top-K 种子 + BFS 扩展 + 预算截断
并记录各阶段 RSS 峰值（resource.getrusage）与该分区的规模。

用法:
  python scripts_ablation/e6g_cost_profile.py --specs bin.116,bin.6@400000 \
      --results results/e6g_cost.csv
"""
from __future__ import annotations
import os, sys, time, argparse, gc, resource

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import numpy as np

from rate_core import load_graphs, load_pickle
from reduction.tered import TeRedOperator
from reduction.base import IdentityOperator
from features.rate import node_feature_matrix, make_type_vocab
from features.v4extras import extend_with_v4
from models.detector import BenignEnsemble
from eval.metrics import append_csv

from scripts.e5_f1_eval import _edges_index, resolve_name, find_templates, parse_one
from scripts.e6_alert_eval import parse_spec
from scripts.e6b_alert_replay import load_res

sys.path.insert(0, HERE)
from e6e_gnn_baseline import alert_pipeline2


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 / 1024.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/TeRed+RATE/dataset/darpae5/cadets")
    ap.add_argument("--train-files", default="bin.1,bin.2,bin.3,bin.4,bin.5")
    ap.add_argument("--max-records-train", type=int, default=400000)
    ap.add_argument("--specs", default="bin.116,bin.6@400000")
    ap.add_argument("--templates", default=None)
    ap.add_argument("--semantic", default="onehot")
    ap.add_argument("--tape-dim", type=int, default=8)
    ap.add_argument("--tape-base", type=float, default=10000.0)
    ap.add_argument("--v4-scale", default="raw")
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--bfs-q", type=int, default=75)
    ap.add_argument("--max-cluster", type=int, default=200)
    ap.add_argument("--max-alerts", type=int, default=20)
    ap.add_argument("--cache", default="/TeRed+RATE/code/cache")
    ap.add_argument("--results", required=True)
    a = ap.parse_args()

    tpl_path = a.templates or find_templates(
        os.path.join(a.cache, "darpa", "e5_templates", "*.jsonl"))
    tpl_stem = os.path.splitext(os.path.basename(tpl_path))[0]
    rcache = os.path.join(a.cache, "darpa", "e5_f1_reduce")
    rec_tag_tr = f"tr{a.max_records_train}"

    # 训练侧（缓存）：与主表同口径的五个训练分区
    Gp_tr_id = []
    for nm in [s.strip() for s in a.train_files.split(",") if s.strip()]:
        gid = resolve_name(a.data_dir, nm)
        Gp_tr_id.append(load_res(rcache, "id", tpl_stem, rec_tag_tr, gid).Gp)
    vocab = make_type_vocab(Gp_tr_id)
    print(f"[cost] train graphs loaded: {len(Gp_tr_id)} vocab={len(vocab)}",
          flush=True)

    for spec in [s.strip() for s in a.specs.split(",") if s.strip()]:
        nm, mr = parse_spec(a.data_dir, spec)
        row = {"spec": spec, "n_nodes_orig": "", "n_edges_orig": "",
               "n_nodes_red": "", "n_edges_red": ""}
        t0 = time.time()
        g, meta = parse_one(a.data_dir, nm, mr or None)
        t_parse = time.time() - t0
        row["n_nodes_orig"] = g.n_nodes()
        row["n_edges_orig"] = g.n_edges()
        print(f"[cost] {spec}: parse {t_parse:.1f}s  "
              f"{g.n_nodes()} nodes {g.n_edges()} edges rss={rss_gb():.1f}G",
              flush=True)

        # identity + reduction，各测一次（verify=False：不变量已在缓存阶段验证过，
        # 避免 O(E) 级 set 构造污染计时）
        t0 = time.time(); res_id = IdentityOperator(verify=False).reduce(g)
        t_id = time.time() - t0
        tpls = load_graphs(tpl_path)
        tered = TeRedOperator(tpls, max_instances=300, max_total=5000,
                              share_k=-1, verify=False)
        t0 = time.time(); res_te = tered.reduce(g); t_te = time.time() - t0
        row["n_nodes_red"] = res_te.Gp.n_nodes()
        row["n_edges_red"] = len(res_te.Gp.edges)
        print(f"[cost]   identity {t_id:.1f}s | tered {t_te:.1f}s "
              f"-> {res_te.Gp.n_nodes()} nodes", flush=True)

        for op, res in [("identity", res_id), ("tered", res_te)]:
            Gp = res.Gp
            vocab2 = make_type_vocab(Gp_tr_id + [Gp])
            t0 = time.time()
            X, nids, _ = node_feature_matrix(
                Gp, encoding="rate", tape_dim=a.tape_dim, tape_base=a.tape_base,
                semantic=a.semantic, vocab=vocab2)
            e = _edges_index(Gp, nids)
            X, _ = extend_with_v4(X, nids, Gp, e, scale=a.v4_scale)
            t_enc = time.time() - t0
            # 训练（用对应口径的训练图特征）
            Xs2 = []
            for Gpt in Gp_tr_id:
                Xt, nt, _ = node_feature_matrix(
                    Gpt, encoding="rate", tape_dim=a.tape_dim,
                    tape_base=a.tape_base, semantic=a.semantic, vocab=vocab2)
                et = _edges_index(Gpt, nt)
                Xt, _ = extend_with_v4(Xt, nt, Gpt, et, scale=a.v4_scale)
                Xs2.append(Xt)
            t0 = time.time()
            det = BenignEnsemble().fit(Xs2, [f"tr{i}" for i in range(len(Xs2))])
            t_fit = time.time() - t0
            t0 = time.time(); s = det.anomaly_scores(X); t_score = time.time() - t0
            t0 = time.time()
            alert_pipeline2(s, e, len(nids), top_k=a.top_k, bfs_q=a.bfs_q,
                            max_cluster=a.max_cluster, max_alerts=a.max_alerts)
            t_alert = time.time() - t0
            row[f"{op}_parse_s"] = round(t_parse, 1)
            row[f"{op}_reduce_s"] = round(t_id if op == "identity" else t_te, 1)
            row[f"{op}_encode_s"] = round(t_enc, 1)
            row[f"{op}_fit_s"] = round(t_fit, 1)
            row[f"{op}_score_s"] = round(t_score, 1)
            row[f"{op}_alert_s"] = round(t_alert, 1)
            row[f"{op}_stage_total_s"] = round(t_enc + t_score + t_alert, 1)
            row[f"{op}_end2end_s"] = round(t_parse + (t_id if op == "identity"
                                                      else t_te) + t_enc
                                          + t_score + t_alert, 1)
            row[f"rss_gb"] = round(rss_gb(), 2)
            print(f"[cost]   [{op}] parse={t_parse:.1f} reduce="
                  f"{(t_id if op == 'identity' else t_te):.1f} enc={t_enc:.1f} "
                  f"fit={t_fit:.1f} score={t_score:.1f} alert={t_alert:.1f} "
                  f"| end2end={row[f'{op}_end2end_s']:.1f}s", flush=True)
            del X, e, s, det, Xs2
            gc.collect()
        append_csv(a.results, row)
        del g, res_id, res_te, tpls
        gc.collect()

    print(f"[cost] done -> {a.results}", flush=True)


if __name__ == "__main__":
    main()
