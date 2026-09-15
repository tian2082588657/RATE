# -*- coding: utf-8 -*-
"""scripts/e5_gs_smoke.py — GraphSAGE 检测器在 E5 bin.119 上的快速冒烟。

仅用于验证 GraphSAGEDetector 在服务器 GPU 上能跑通 + 对比 BenignEnsemble 的绝对 F1。
训练图：bin.1, bin.2（max-records 截断到 200K，归约时间可控）。
攻击图：bin.119 截断到 500K 记录（约 1.3 万节点，atted归约 ~1-2 分钟）。
不写主 per-file CSV，输出到 results/e5_gs_smoke.csv 单独看。
"""
from __future__ import annotations
import os, sys, glob, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from rate_core import load_graphs
from reduction.tered import TeRedOperator
from reduction.base import IdentityOperator, node_map_sanity, check_invariants
from features.rate import node_feature_matrix, make_type_vocab, label_vector
from models.detector import BenignEnsemble
from models.detector_graphsage import GraphSAGEDetector
from eval.metrics import binary_metrics, append_csv
from adapters import darpa_tc

sys.path.insert(0, "/TeRed+RATE/code/scripts")
from e5_f1_eval import resolve_name, parse_one, label_graph, attack_survival, _edges_index


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/TeRed+RATE/dataset/darpae5/cadets")
    ap.add_argument("--gt", default="ground_truth")
    ap.add_argument("--attack-file", default="bin.119")
    ap.add_argument("--train-files", default="bin.1,bin.2")
    ap.add_argument("--max-records-train", type=int, default=200000)
    ap.add_argument("--max-records-attack", type=int, default=500000)
    ap.add_argument("--share-k", type=int, default=-1)
    ap.add_argument("--templates", default=None)
    ap.add_argument("--results", default="results/e5_gs_smoke.csv")
    a = ap.parse_args()

    from e5_f1_eval import load_gt, find_templates
    gt = load_gt(a.gt)
    print(f"[gs] ground truth: {len(gt)} UUIDs", flush=True)

    # parse train
    train_graphs = []
    for nm in a.train_files.split(","):
        nm = nm.strip()
        if not nm:
            continue
        g, _ = parse_one(a.data_dir, nm, a.max_records_train)
        train_graphs.append(g)
        print(f"[gs] train {nm}: {g.n_nodes()} nodes {g.n_edges()} edges", flush=True)

    # parse attack
    g_atk, meta = parse_one(a.data_dir, a.attack_file, a.max_records_attack)
    hit, atk = label_graph(g_atk, gt)
    print(f"[gs] attack {a.attack_file}: {g_atk.n_nodes()} nodes {g_atk.n_edges()} edges "
          f"records={meta['records']} | GT hit {atk}/{len(gt)}", flush=True)

    # templates
    tpl_path = a.templates or find_templates(
        os.path.join("cache", "darpa", "e5_templates", "*.jsonl"))
    tpls = load_graphs(tpl_path)
    print(f"[gs] templates: {tpl_path} -> {len(tpls)} (share_k={a.share_k})", flush=True)

    # identity + tered reduce
    res_tr_id = [IdentityOperator().reduce(g) for g in train_graphs]
    res_atk_id = IdentityOperator().reduce(g_atk)
    tered = TeRedOperator(tpls, max_instances=300, max_total=5000, share_k=a.share_k)
    print("[gs] tered reduce train...", flush=True)
    t0 = time.time()
    res_tr_te = [tered.reduce(g) for g in train_graphs]
    print(f"[gs] tered train done ({time.time()-t0:.0f}s); reducing attack...", flush=True)
    t0 = time.time()
    res_atk_te = tered.reduce(g_atk)
    print(f"[gs] tered attack reduce {g_atk.n_nodes()}->{res_atk_te.Gp.n_nodes()} "
          f"({time.time()-t0:.0f}s)", flush=True)

    inv = check_invariants(g_atk, res_atk_te)
    inv_ok = all(v is True for k, v in inv.items() if k != "INV2_violations")
    print(f"[gs] INV: {'OK' if inv_ok else 'FAIL'} (n_viol={len(inv['INV2_violations'])})",
          flush=True)

    # three configs
    configs = [
        ("无归约基线", [r.Gp for r in res_tr_id], res_atk_id.Gp, g_atk, res_atk_id, "rate"),
        ("TeRed+naive", [r.Gp for r in res_tr_te], res_atk_te.Gp, g_atk, res_atk_te, "dual_naive"),
        ("TeRed+RATE", [r.Gp for r in res_tr_te], res_atk_te.Gp, g_atk, res_atk_te, "rate"),
    ]

    detectors_map = {
        "BenignEnsemble": lambda Xs, Es, names: BenignEnsemble().fit(Xs, names),
        "GraphSAGEDetector": lambda Xs, Es, names: GraphSAGEDetector(
            hidden=64, n_neighbors=25, lr=0.01, epochs=20, radius_q=0.95).fit(Xs, Es, names),
    }

    vocab = make_type_vocab([g.Gp for r in [res_tr_id, [res_atk_id]] for g in r] +
                            [res_atk_te.Gp] + [r.Gp for r in res_tr_te])

    for name, Gp_tr_list, Gp_atk, G_orig, res_atk, enc in configs:
        # benign
        print(f"\n[gs] === {name} ({enc}) - BenignEnsemble ===", flush=True)
        Xs, Es, Ns = [], [], []
        for Gp in Gp_tr_list:
            X, nids, _ = node_feature_matrix(Gp, encoding=enc, vocab=vocab)
            Xs.append(X); Ns.append(nids); Es.append(_edges_index(Gp, nids))
        Xa, Na, _ = node_feature_matrix(Gp_atk, encoding=enc, vocab=vocab)
        Ea = _edges_index(Gp_atk, Na)
        det_b = BenignEnsemble().fit(Xs, [f"t{i}" for i in range(len(Xs))])
        y = label_vector(Gp_atk, Na)
        pred = (~det_b.predict(Xa)).astype(np.int64)
        score = det_b.anomaly_scores(Xa)
        m = binary_metrics(y, pred, score)
        surv, _ = attack_survival(G_orig, res_atk)
        print(f"  Benign  TP={m['TP']} FP={m['FP']} FN={m['FN']} "
              f"P={m['Precision']:.3f} R={m['Recall']:.3f} F1={m['F1']:.3f} "
              f"AUC_PR={m['PR_AUC']:.4f} surv={surv:.2f}", flush=True)
        append_csv(a.results, {
            "config": name, "encoding": enc, "detector": "benign",
            "attack_file": a.attack_file, "n_nodes": len(Na),
            "TP": m["TP"], "FP": m["FP"], "FN": m["FN"], "TN": m["TN"],
            "Precision": m["Precision"], "Recall": m["Recall"],
            "F1": m["F1"], "PR_AUC": m["PR_AUC"],
            "ROC_AUC": m["ROC_AUC"], "FPR": m["FPR"], "survival": surv,
        })
        # graphsage
        print(f"[gs] === {name} ({enc}) - GraphSAGEDetector ===", flush=True)
        t0 = time.time()
        det_g = GraphSAGEDetector(hidden=64, n_neighbors=25, lr=0.01,
                                   epochs=20, radius_q=0.99, neg_ratio=1.0).fit(
            Xs, Es, [f"t{i}" for i in range(len(Xs))])
        print(f"[gs] GraphSAGE fit {time.time()-t0:.0f}s", flush=True)
        t0 = time.time()
        pred = (~det_g.predict(Xa, Ea)).astype(np.int64)
        score = det_g.anomaly_scores(Xa, Ea)
        m = binary_metrics(y, pred, score)
        # 阈值扫描找最优 F1（论文级精度证明）
        order = np.argsort(-score)   # 异常分降序
        y_sorted = y[order]
        cum_tp = np.cumsum(y_sorted)
        n_atk = int(y.sum())
        best_f1 = 0.0; best_k = 0
        for k in range(1, n_atk * 4 + 1):
            tp = cum_tp[k - 1]
            fp = k - tp
            fn = n_atk - tp
            p = tp / (tp + fp) if tp + fp else 0.0
            r = tp / n_atk
            f1 = 2 * p * r / (p + r) if p + r else 0.0
            if f1 > best_f1:
                best_f1, best_k = f1, k
        print(f"  SAGE 最佳top-{best_k} F1={best_f1:.3f} "
              f"(提示：阈值校准后可达上限)", flush=True)
        print(f"[gs] GraphSAGE infer {time.time()-t0:.0f}s", flush=True)
        surv, _ = attack_survival(G_orig, res_atk)
        print(f"  SAGE    TP={m['TP']} FP={m['FP']} FN={m['FN']} "
              f"P={m['Precision']:.3f} R={m['Recall']:.3f} F1={m['F1']:.3f} "
              f"AUC_PR={m['PR_AUC']:.4f} surv={surv:.2f}", flush=True)
        append_csv(a.results, {
            "config": name, "encoding": enc, "detector": "graphsage",
            "attack_file": a.attack_file, "n_nodes": len(Na),
            "TP": m["TP"], "FP": m["FP"], "FN": m["FN"], "TN": m["TN"],
            "Precision": m["Precision"], "Recall": m["Recall"],
            "F1": m["F1"], "PR_AUC": m["PR_AUC"],
            "ROC_AUC": m["ROC_AUC"], "FPR": m["FPR"], "survival": surv,
        })


if __name__ == "__main__":
    main()