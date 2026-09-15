# -*- coding: utf-8 -*-
"""e6i_benign_counterfactual.py — 攻击移除反事实：误报率代理指标。

与论文 §4.10 批判的「清标签反事实」的关键区别：
  清标签只抹掉 annotation，攻击**行为**仍在图里 —— 因此无效。
  本实验把 **所有 ground-truth 实体及其关联结构从图中删除**，
  攻击行为随之消失；在剩余图上产生的告警，按已发布标签就是误报。

实现（两条路径）：
  * identity：直接在原始图上删除 GT 节点及其入射边；
  * reduced ：用归约的 node_map 把 GT 实体映射到归约图节点，
              删除这些节点（含吸收了 GT 的区域汇总节点）及其入射边。
  两个路径都不重新归约（归约在原始图上做，删点后重归约成本不可接受），
  因此剩余图的结构是「归约时攻击仍在场」的结果 —— 这是本实验的
  caveat，写入论文。
  训练图在缓存中不带 labels，本脚本不做依赖；拟合与 e6k 的 all 方案一致。
  测试图直接读归约缓存（含 labels），不重新 parse，逐分区释放内存。

输出：results/e6i_benign_far.csv（每分区每配置一行）
"""
from __future__ import annotations
import argparse
import csv
import gc
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def clone_minus(G, del_ids):
    """返回 G 去掉 del_ids 及其入射边后的 CanonicalGraph 副本。"""
    from rate_core import CanonicalGraph
    H = CanonicalGraph(G.gid + ":minus")
    for nid, nd in G.nodes.items():
        if nid in del_ids:
            continue
        H.nodes[nid] = nd
        if nid in G.labels:
            H.labels[nid] = G.labels[nid]
    for e in G.edges:
        if e["src"] in del_ids or e["dst"] in del_ids:
            continue
        H.edges.append(dict(e))
    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--train-files", default="bin.1,bin.2,bin.3,bin.4,bin.5")
    ap.add_argument("--max-records-train", type=int, default=400000)
    ap.add_argument("--test-files", default="bin.116,bin.117,bin.118,"
                                           "bin.119@500000,bin.120,bin.6@400000,bin.7@400000")
    ap.add_argument("--templates", default="")
    ap.add_argument("--cache", default="cache/darpa")
    ap.add_argument("--share-k", type=int, default=-1)
    ap.add_argument("--tape-dim", type=int, default=8)
    ap.add_argument("--tape-base", type=float, default=10000.0)
    ap.add_argument("--semantic", default="onehot")
    ap.add_argument("--v4-scale", type=float, default=1.0)
    # 必须与主实验 scripts/e6_alert_eval.py 的默认协议一致：
    #   top_k=100, bfs_q=75(百分位), min_cluster=3, max_cluster=200, max_alerts=20
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--bfs-q", type=float, default=75.0)
    ap.add_argument("--min-cluster", type=int, default=3)
    ap.add_argument("--max-cluster", type=int, default=200)
    ap.add_argument("--max-alerts", type=int, default=20)
    ap.add_argument("--results", default="results/e6i_benign_far.csv")
    a = ap.parse_args()

    from rate_core import load_graphs, load_pickle
    from scripts.e5_f1_eval import (parse_one, find_templates, load_gt,
                                    label_graph, _edges_index, resolve_name)
    from scripts.e6_alert_eval import alert_pipeline, parse_spec
    from scripts.e6b_alert_replay import load_res
    from reduction.tered import TeRedOperator
    from reduction.base import IdentityOperator, node_map_sanity
    from features.rate import node_feature_matrix, make_type_vocab, label_vector
    from features.v4extras import extend_with_v4
    from models.detector import BenignEnsemble

    gt = load_gt(a.gt)
    print(f"[e6i] ground truth: {len(gt)} UUIDs", flush=True)
    semantic = str(a.semantic).lower() not in ("0", "false", "no")

    train_names = [s.strip() for s in a.train_files.split(",") if s.strip()]
    train_graphs = [parse_one(a.data_dir, nm, a.max_records_train or None)[0]
                    for nm in train_names]
    print("[e6i] train parsed", flush=True)

    tpl_path = a.templates or find_templates(
        os.path.join(a.cache, "darpa", "e5_templates", "*.jsonl"))
    tpls = load_graphs(tpl_path)
    tered = TeRedOperator(tpls, max_instances=300, max_total=5000,
                          share_k=a.share_k)

    tpl_stem = os.path.splitext(os.path.basename(tpl_path))[0]
    rcache = os.path.join(a.cache, "darpa", "e5_f1_reduce")

    def reduce_one(op, g, op_tag, rec_tag):
        """优先读缓存（与 e7 同一命名约定），避免重跑昂贵的归约。"""
        cp = os.path.join(
            rcache, f"{op_tag}_{tpl_stem}_k{a.share_k}_{rec_tag}_{g.gid}.pkl")
        if os.path.exists(cp):
            res = load_pickle(cp)
            try:
                node_map_sanity(res, g)
            except Exception:
                pass
            return res
        res = op.reduce(g)
        return res

    real_train = [reduce_one(IdentityOperator(), g, "id",
                             f"tr{a.max_records_train}").Gp for g in train_graphs]
    red_train = [reduce_one(tered, g, "tered",
                            f"tr{a.max_records_train}").Gp for g in train_graphs]
    print("[e6i] train graphs loaded/reduced", flush=True)

    # 配置: (名称, 训练图列表, 算子 or None(identity), 编码)
    CONFIGS = [
        ("identity+rate", real_train, None, "rate"),
        ("TeRed+count", red_train, tered, "dual_naive"),
        ("TeRed+RATE*", red_train, tered, "rate_ratio"),
    ]

    def build_X(Gp, vocab, enc):
        X, nids, _ = node_feature_matrix(
            Gp, encoding=enc, tape_dim=a.tape_dim, tape_base=a.tape_base,
            semantic=semantic, vocab=vocab)
        edges = _edges_index(Gp, nids)
        X, _ = extend_with_v4(X, nids, Gp, edges, scale=a.v4_scale)
        return X, nids, edges

    def fit_and_run(cname, gpt, enc, G_real, G_benign):
        """vocab 覆盖 train+两张测试图，拟合（无监督，不用标签）后两边各跑一次。"""
        vocab = make_type_vocab(gpt + [G_real, G_benign])
        Xs = [build_X(Gp, vocab, enc)[0] for Gp in gpt]
        det = BenignEnsemble().fit(Xs, [f"tr{i}" for i in range(len(Xs))])
        out = []
        for Gp in (G_real, G_benign):
            X, nids, edges = build_X(Gp, vocab, enc)
            score = det.anomaly_scores(X)
            clusters = alert_pipeline(score, edges, len(nids), top_k=a.top_k,
                                      bfs_q=a.bfs_q, min_cluster=a.min_cluster,
                                      max_cluster=a.max_cluster,
                                      max_alerts=a.max_alerts)
            alerted = set()
            for c, _s in clusters:
                alerted |= set(c)
            out.append((len(clusters), len(alerted)))
        return out

    rows = []
    specs = [s.strip() for s in a.test_files.split(",") if s.strip()]
    tpl_stem = os.path.splitext(os.path.basename(tpl_path))[0]
    for spec in specs:
        nm, mr = parse_spec(a.data_dir, spec)
        rec_tag = f"at{mr}" if mr else "atfull"
        gid = resolve_name(a.data_dir, nm)
        # 直接读归约缓存（含 labels），不再重新 parse 全量图：省内存、省时间
        res_id = load_res(rcache, "id", tpl_stem, rec_tag, gid)
        res_te = load_res(rcache, "tered", tpl_stem, rec_tag, gid)
        ids_id = {nid for nid, l in (res_id.Gp.labels or {}).items() if l == 1}
        ids_te = {nid for nid, l in (res_te.Gp.labels or {}).items() if l == 1}
        hit = len(ids_id)

        g_minus = clone_minus(res_id.Gp, ids_id)
        gp_minus_te = clone_minus(res_te.Gp, ids_te)

        print(f"\n[e6i] ===== {spec}: {res_id.Gp.n_nodes()} nodes, "
              f"GT id={hit} te={len(ids_te)} =====", flush=True)
        for cname, gpt, op, enc in CONFIGS:
            if op is None:
                G_real, G_benign = res_id.Gp, g_minus
            else:
                G_real, G_benign = res_te.Gp, gp_minus_te
            (n_real, a_real), (n_ben, a_ben) = fit_and_run(
                cname, gpt, enc, G_real, G_benign)
            rows.append({
                "spec": spec, "config": cname,
                "gt_entities": hit, "gt_entities_reduced": len(ids_te),
                "n_nodes_real": G_real.n_nodes(),
                "n_nodes_benign": G_benign.n_nodes(),
                "alerts_real": n_real, "alerted_nodes_real": a_real,
                "alerts_benign": n_ben, "alerted_nodes_benign": a_ben,
                "false_alerts": n_ben,
            })
            print(f"  [{cname}] real alerts={n_real} | "
                  f"attack-removed alerts={n_ben} "
                  f"(nodes {G_real.n_nodes()}->{G_benign.n_nodes()})", flush=True)

        del g_minus, gp_minus_te, res_id, res_te
        gc.collect()

    keep = list(rows[0].keys())
    write_header = not os.path.exists(a.results)
    with open(a.results, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keep)
        if write_header:
            w.writeheader()
        w.writerows(rows)
    print(f"[e6i] done -> {a.results}", flush=True)


if __name__ == "__main__":
    main()
