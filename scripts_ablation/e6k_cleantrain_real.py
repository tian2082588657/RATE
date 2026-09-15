# -*- coding: utf-8 -*-
"""e6k_cleantrain_real.py — 真实版「剥离训练集 GT 正例」实验。

背景（bug）：e6e_gnn_baseline.py 的 --clean-train 用 `label_vector(Gp, nids)`
读取缓存图的 labels，但 tr400000 归约缓存 **不含 labels** → 训练标签向量恒 0
→ "剥离 0 个节点、结果逐位相同" 是缓存缺 labels 的假象，不是实验证据。

本脚本对缓存训练图重新用 GT 标注（label_graph），再真实剥离 GT 正例：

  方案 all   : 5 个训练窗口原样（每个窗口实测含 5-7 个 GT 实体）
  方案 clean : 同样的 5 个窗口，但删除 GT 正例节点后重新拟合检测器

在 7 个测试分区上按主协议（top_k=100, bfs_q=75, min_cluster=3,
max_cluster=200, max_alerts=20）评估节点级与告警级指标，并报告配对差。

输出：results/e6k_cleantrain_real.csv
"""
from __future__ import annotations
import argparse
import gc
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--gt", default="ground_truth")
    ap.add_argument("--train-files", default="bin.1,bin.2,bin.3,bin.4,bin.5")
    ap.add_argument("--max-records-train", type=int, default=400000)
    ap.add_argument("--test-files", default="bin.116,bin.117,bin.118,"
                                            "bin.119@500000,bin.120,"
                                            "bin.6@400000,bin.7@400000")
    ap.add_argument("--templates", default="")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--share-k", type=int, default=-1)
    ap.add_argument("--tape-dim", type=int, default=8)
    ap.add_argument("--tape-base", type=float, default=10000.0)
    ap.add_argument("--semantic", default="onehot")
    ap.add_argument("--v4-scale", type=float, default=1.0)
    ap.add_argument("--op-quantile", type=float, default=5.0)
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--bfs-q", type=float, default=75.0)
    ap.add_argument("--min-cluster", type=int, default=3)
    ap.add_argument("--max-cluster", type=int, default=200)
    ap.add_argument("--max-alerts", type=int, default=20)
    ap.add_argument("--results", default="results/e6k_cleantrain_real.csv")
    a = ap.parse_args()

    from scripts_ablation.e6e_gnn_baseline import (alert_pipeline2,
                                                   alert_metrics2,
                                                   subgraph_drop)
    from eval.metrics import append_csv, binary_metrics
    from scripts.e5_f1_eval import (_edges_index, resolve_name, find_templates,
                                    load_gt, label_graph)
    from scripts.e6_alert_eval import parse_spec
    from scripts.e6b_alert_replay import load_res
    from features.rate import node_feature_matrix, make_type_vocab, label_vector
    from features.v4extras import extend_with_v4
    from models.detector import BenignEnsemble

    gt = load_gt(a.gt)
    print(f"[e6k] GT uuids = {len(gt)}", flush=True)
    tpl_path = a.templates or find_templates(
        os.path.join(a.cache, "darpa", "e5_templates", "*.jsonl"))
    tpl_stem = os.path.splitext(os.path.basename(tpl_path))[0]
    rcache = os.path.join(a.cache, "darpa", "e5_f1_reduce")
    rec_tr = f"tr{a.max_records_train}"

    # ---------- 训练图（缓存）+ 重新标注 ----------
    train = {"identity": [], "tered": []}
    gt_counts = {}
    for nm in [s.strip() for s in a.train_files.split(",") if s.strip()]:
        gid = resolve_name(a.data_dir, nm)
        for op in ("identity", "tered"):
            Gp = load_res(rcache, "id" if op == "identity" else "tered",
                          tpl_stem, rec_tr, gid).Gp
            hit, _ = label_graph(Gp, gt)
            gt_counts[f"{nm}:{op}"] = hit
            train[op].append(Gp)
        print(f"[e6k] {nm}@{a.max_records_train}: GT hit "
              f"id={gt_counts[f'{nm}:identity']} tered={gt_counts[f'{nm}:tered']}",
              flush=True)

    # ---------- 测试图 ----------
    tests = []
    for spec in [s.strip() for s in a.test_files.split(",") if s.strip()]:
        nm, mr = parse_spec(a.data_dir, spec)
        rec_tag = f"at{mr}" if mr else "atfull"
        gid = resolve_name(a.data_dir, nm)
        tests.append({"spec": spec, "rec_tag": rec_tag,
                      "identity": load_res(rcache, "id", tpl_stem, rec_tag, gid).Gp,
                      "tered": load_res(rcache, "tered", tpl_stem, rec_tag, gid).Gp})
    print("[e6k] test graphs loaded", flush=True)

    CONFIGS = [("identity", "rate", "identity+rate"),
               ("tered", "dual_naive", "TeRed+count"),
               ("tered", "rate_ratio", "TeRed+RATE*")]

    summary = {}
    for op, enc, cname in CONFIGS:
        vocab = make_type_vocab(list(train[op]) + [t[op] for t in tests])

        def build(Gp):
            X, nids, _ = node_feature_matrix(
                Gp, encoding=enc, tape_dim=a.tape_dim, tape_base=a.tape_base,
                semantic=a.semantic, vocab=vocab)
            e = _edges_index(Gp, nids)
            X, _ = extend_with_v4(X, nids, Gp, e, scale=a.v4_scale)
            return X, e, label_vector(Gp, nids)

        Xtr_all, Xtr_clean, n_rm = [], [], 0
        for Gp in train[op]:
            X, e, y = build(Gp)
            Xtr_all.append(X)
            if int(y.sum()) > 0:
                n_rm += int(y.sum())
                X, _e = subgraph_drop(X, e, y == 0)
            Xtr_clean.append(X)
        print(f"\n[e6k] ===== {cname} (vocab={len(vocab)}) "
              f"train positives removed={n_rm} =====", flush=True)

        det_all = BenignEnsemble().fit(Xtr_all, [f"a{i}" for i in range(len(Xtr_all))])
        sa = np.concatenate([det_all.anomaly_scores(X) for X in Xtr_all])
        thr_all = float(np.percentile(sa, a.op_quantile))
        det_cln = BenignEnsemble().fit(Xtr_clean, [f"c{i}" for i in range(len(Xtr_clean))])
        sc = np.concatenate([det_cln.anomaly_scores(X) for X in Xtr_clean])
        thr_cln = float(np.percentile(sc, a.op_quantile))
        print(f"[e6k] thresholds: all={thr_all:.4f} clean={thr_cln:.4f}", flush=True)

        del sa, sc
        gc.collect()

        per = {"all": {}, "clean": {}}
        for t in tests:
            Gp = t[op]
            X, e, y = build(Gp)
            for scheme, det, thr in (("all", det_all, thr_all),
                                     ("clean", det_cln, thr_cln)):
                s = det.anomaly_scores(X)
                m = binary_metrics(y, (s > thr).astype(np.int64), s)
                am = alert_metrics2(alert_pipeline2(
                    s, e, len(y), top_k=a.top_k, bfs_q=a.bfs_q,
                    min_cluster=a.min_cluster, max_cluster=a.max_cluster,
                    max_alerts=a.max_alerts), y, s, a.top_k)
                row = {"detector": "cosine", "operator": op, "config": cname,
                       "encoding": enc, "file": t["spec"], "train_scheme": scheme,
                       "n_train_gt_removed": n_rm,
                       "n_nodes_Gp": Gp.n_nodes(), "n_gt": int(y.sum()),
                       "node_ROC_AUC": round(m["ROC_AUC"], 4),
                       "node_PR_AUC": round(m["PR_AUC"], 4),
                       "node_best_F1": round(m["best_F1"], 4),
                       "op_threshold": round(thr, 4)}
                row.update(am)
                append_csv(a.results, row)
                per[scheme][t["spec"]] = {"auc": m["ROC_AUC"],
                                          "bf1": m["best_F1"],
                                          "a5": am.get("F1_alert@5"),
                                          "a10": am.get("F1_alert@10")}
                print(f"  [{scheme:5s}] {t['spec']:16s} AUC={m['ROC_AUC']:.4f} "
                      f"bF1={m['best_F1']:.4f} @5={am.get('F1_alert@5')} "
                      f"@10={am.get('F1_alert@10')}", flush=True)

        def mean(k, scheme):
            v = [per[scheme][s][k] for s in per[scheme]
                 if per[scheme][s][k] is not None]
            return float(np.mean(v)) if v else float("nan")

        def paired(k):
            d = [per["all"][s][k] - per["clean"][s][k] for s in per["all"]
                 if per["all"][s][k] is not None and per["clean"][s][k] is not None]
            return (float(np.mean(d)), float(np.max(np.abs(d))) if d else 0.0)

        summary[cname] = {"n_train_gt_removed": n_rm}
        for k, lab in (("auc", "node_ROC_AUC"), ("bf1", "node_best_F1"),
                       ("a5", "F1_alert@5"), ("a10", "F1_alert@10")):
            md, mx = paired(k)
            summary[cname][lab] = {
                "all_mean": round(mean(k, "all"), 4),
                "clean_mean": round(mean(k, "clean"), 4),
                "mean_delta_all_minus_clean": round(md, 4),
                "max_abs_delta": round(mx, 4)}
            print(f"[e6k] {cname} {lab}: all={mean(k,'all'):.4f} "
                  f"clean={mean(k,'clean'):.4f} Δ={md:+.4f} (max|Δ|={mx:.4f})",
                  flush=True)

        del det_all, det_cln
        gc.collect()

    with open(os.path.splitext(a.results)[0] + "_summary.json", "w",
              encoding="utf-8") as f:
        json.dump({"gt_counts": gt_counts, "configs": summary}, f,
                  ensure_ascii=False, indent=2)
    print(f"[e6k] done -> {a.results}", flush=True)


if __name__ == "__main__":
    main()
