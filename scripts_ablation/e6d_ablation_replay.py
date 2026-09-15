# -*- coding: utf-8 -*-
"""scripts/e6d_ablation_replay.py — A1/A3/A4 消融重放（分数只算一次，变体全量重聚合）。

设计：对每个 (test file × config) 只算一次 特征->BenignEnsemble->score，
然后所有消融变体从同一份 score/edges 数组重跑告警管线（秒级），避免重复特征化。

变体清单：
  A1（管线解构，base=新主表口径 BFS@P75 + 纯max聚合 + cmax排序）:
    base      新主表口径（作 sanity 对照，应与 e6_alert_full_v2 数字一致）
    legacy_agg 旧主表口径（wmax=0.7 + agg 排序）——量化升级带来的增益
    no_bfs    去 BFS 扩展：top-100 种子各成一条告警（阈值失效）
    no_rank   去 top-B 排序：成簇照常，但按种子出现顺序出告警
    bfs_q50 / bfs_q90   BFS 阈值分位扫描
    w050 / w000  聚合权重（0.5 / 纯mean；w100 即 base，不再单列）
  A3（预算×排序）:
    sort_cmax / sort_p95  按簇内最高分 / 节点分 P95 排序
    （预算维度：所有变体输出 F1@{1,2,3,5,10,20} 列，无需单独跑）
  A4（v4 特征逐列消融）:
    v4_drop_log_deg / v4_drop_out_in_ratio / v4_drop_self_loop
    （对应列在 train+test 同时置零后重训重评，余弦评分下等价于删列；管线参数=base）
    注：nbr_type_div 已于 2026-09-11 从 v4 删除（消融证明为死重列），故不再作为变体。
  A5（RATE 拓扑编码消融，仅 TeRed+RATE 配置）:
    enc_dual_naive  双通道但按边条数（无 μ）—— 等价于 TeRed+naive 的编码
    enc_rate_single 单通道 Σμ（不分入/出）—— 检验双通道必要性
    enc_none        无拓扑特征（仅语义 one-hot + v4 结构列）
    （同一归约图 Gp 上换编码重训重评；管线参数=base。回答"RATE 编码的 μ/双通道各贡献多少"）

用法:
  python scripts/e6d_ablation_replay.py \
      --test-files bin.116,bin.117,bin.118,bin.119@500000,bin.120,bin.6@400000,bin.7@400000 \
      --groups A1,A3,A4 --results results/e6d_ablation.csv
"""
from __future__ import annotations
import os, sys, time, argparse, gc, random

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
from scripts.e6_alert_eval import build_adj, parse_spec
from scripts.e6b_alert_replay import load_res

V4_COLS = ["log_deg", "out_in_ratio", "self_loop"]  # nbr_type_div 已于 2026-09-11 删除（死重列）
BUDGETS = [1, 2, 3, 5, 10, 20]


def alert_pipeline2(score, edges, n, top_k=100, bfs_mode="bfs", bfs_q=75,
                    min_cluster=3, max_cluster=200, max_alerts=20,
                    wmax=1.0, sort_mode="cmax", rng_seed=0):
    """泛化版告警管线。返回 [(node_idx_list, agg_score), ...] 按 sort_mode 排序。"""
    order = np.argsort(-score, kind="mergesort")
    top_idx = order[:top_k]
    if bfs_mode == "off":
        clusters = [([int(s)], float(score[int(s)])) for s in top_idx]
        return clusters[:max_alerts]
    thresh = float(np.percentile(score, bfs_q))
    adj = build_adj(n, edges)
    visited, clusters = set(), []
    for seed in top_idx:
        seed = int(seed)
        if seed in visited:
            continue
        cluster, stack = [], [seed]
        while stack and len(cluster) < max_cluster:
            u = stack.pop()
            if u in visited:
                continue
            visited.add(u)
            cluster.append(u)
            for v in adj[u]:
                if v not in visited and score[v] >= thresh:
                    stack.append(v)
        if len(cluster) >= min_cluster:
            cs = score[np.array(cluster, dtype=np.int64)]
            agg = float(cs.max()) * wmax + float(cs.mean()) * (1.0 - wmax)
            clusters.append((cluster, agg, float(cs.max()),
                             float(np.percentile(cs, 95))))
    if sort_mode == "cmax":
        clusters.sort(key=lambda x: -x[2])
    elif sort_mode == "p95":
        clusters.sort(key=lambda x: -x[3])
    elif sort_mode == "seed":
        pass
    elif sort_mode == "rand":
        random.Random(rng_seed).shuffle(clusters)
    else:
        clusters.sort(key=lambda x: -x[1])
    return [(c, s) for c, s, _, _ in clusters][:max_alerts]


def alert_metrics2(clusters, y, score, top_k):
    """e6_alert_eval.alert_metrics 的超集：追加 F1@b 预算列与 node_best_f1。"""
    n_gt = int(y.sum())
    alerted = [np.array(c, dtype=np.int64) for c, _ in clusters]
    tp_flags = [int(y[c].sum() > 0) for c in alerted]
    n_alerts = len(clusters)
    cum_cov, seen = [], set()
    for c in alerted:
        # 回映射到全局下标（修复：局部下标累计会误判跨簇重复）
        seen.update(c[np.nonzero(y[c])[0]].tolist())
        cum_cov.append(len(seen))
    alert_TP = int(sum(tp_flags))
    alert_P = alert_TP / n_alerts if n_alerts else 0.0
    node_cov = cum_cov[-1] / n_gt if (n_gt and cum_cov) else 0.0
    f1_alert = (2 * alert_P * node_cov / (alert_P + node_cov)
                if (alert_P + node_cov) else 0.0)
    best_f1, best_b, first_hit = 0.0, 0, 0
    f1_at = {}
    for b in range(1, n_alerts + 1):
        p = sum(tp_flags[:b]) / b
        r = cum_cov[b - 1] / n_gt if n_gt else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        f1_at[b] = round(f, 4)
        if f > best_f1:
            best_f1, best_b = f, b
        if first_hit == 0 and tp_flags[b - 1]:
            first_hit = b
    k = min(top_k, len(y))
    flat_idx = np.argsort(-score, kind="mergesort")[:k]
    flat_cov = float(y[flat_idx].sum()) / n_gt if n_gt else 0.0
    flat_prec = float(y[flat_idx].sum()) / k if k else 0.0
    n_alerted_nodes = int(sum(len(c) for c in alerted))
    covered_gt = cum_cov[-1] if cum_cov else 0
    m = {
        "n_gt": n_gt, "n_alerts": n_alerts, "alert_TP": alert_TP,
        "alert_P": round(alert_P, 4), "node_cov": round(node_cov, 4),
        "F1_alert": round(f1_alert, 4),
        "best_F1_alert": round(best_f1, 4), "best_b": best_b,
        "first_hit": first_hit, "top1_hit": tp_flags[0] if tp_flags else 0,
        "n_alerted_nodes": n_alerted_nodes,
        "covered_gt": int(covered_gt),
        "node_prec": round(covered_gt / n_alerted_nodes, 4) if n_alerted_nodes else 0.0,
        "flat_topk_cov": round(flat_cov, 4), "flat_topk_prec": round(flat_prec, 4),
        "node_best_f1": node_best_f1(y, score),
    }
    for b in BUDGETS:
        m[f"F1_alert@{b}"] = f1_at.get(b, None)
    return m


def node_best_f1(y, score):
    """节点级 best_F1（PR 扫描，M2 A 阶段口径）。"""
    n_pos = int(y.sum())
    if n_pos == 0:
        return 0.0
    order = np.argsort(-score, kind="mergesort")
    ys = y[order]
    tp = np.cumsum(ys).astype(np.float64)
    fp = np.cumsum(1 - ys).astype(np.float64)
    prec = tp / np.maximum(tp + fp, 1.0)
    rec = tp / n_pos
    denom = np.maximum(prec + rec, 1e-12)
    f1 = 2 * prec * rec / denom
    return round(float(np.max(f1)), 4)


def run_variant(name, group, score, edges, n, y, top_k, a, extra=None,
                max_alerts=20):
    clusters = alert_pipeline2(score, edges, n, top_k=top_k,
                               bfs_mode=a.get("bfs_mode", "bfs"),
                               bfs_q=a.get("bfs_q", 75),
                               min_cluster=a.get("min_cluster", 3),
                               max_cluster=a.get("max_cluster", 200),
                               max_alerts=max_alerts,
                               wmax=a.get("wmax", 1.0),
                               sort_mode=a.get("sort_mode", "cmax"))
    m = alert_metrics2(clusters, y, score, top_k)
    m["group"] = group
    m["variant"] = name
    if extra:
        m.update(extra)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/TeRed+RATE/dataset/darpae5/cadets")
    ap.add_argument("--train-files", default="bin.1,bin.2,bin.3,bin.4,bin.5")
    ap.add_argument("--max-records-train", type=int, default=400000)
    ap.add_argument("--test-files", required=True)
    ap.add_argument("--templates", default=None)
    ap.add_argument("--semantic", default="onehot")
    ap.add_argument("--tape-dim", type=int, default=8)
    ap.add_argument("--tape-base", type=float, default=10000.0)
    ap.add_argument("--v4-scale", default="raw", choices=["raw", "robust"])
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--max-cluster", type=int, default=200)
    ap.add_argument("--max-alerts", type=int, default=20)
    ap.add_argument("--groups", default="A1,A3,A4")
    ap.add_argument("--configs", default="identity+rate,TeRed+naive,TeRed+RATE")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--results", required=True)
    ap.add_argument("--a5-enc",
                    default="dual_naive,rate_single,none",
                    help="A5 组逐个重训的拓扑编码列表（逗号分隔）")
    a = ap.parse_args()
    groups = [g.strip() for g in a.groups.split(",") if g.strip()]
    want_cfgs = [c.strip() for c in a.configs.split(",") if c.strip()]

    tpl_path = a.templates or find_templates(
        os.path.join(a.cache, "darpa", "e5_templates", "*.jsonl"))
    tpl_stem = os.path.splitext(os.path.basename(tpl_path))[0]
    print(f"[abl] 模板 stem: {tpl_stem}", flush=True)
    rcache_dir = os.path.join(a.cache, "darpa", "e5_f1_reduce")

    rec_tag_tr = f"tr{a.max_records_train}"
    Gp_tr_id, Gp_tr_te = [], []
    for nm in [s.strip() for s in a.train_files.split(",") if s.strip()]:
        gid = resolve_name(a.data_dir, nm)
        Gp_tr_id.append(load_res(rcache_dir, "id", tpl_stem, rec_tag_tr, gid).Gp)
        Gp_tr_te.append(load_res(rcache_dir, "tered", tpl_stem, rec_tag_tr, gid).Gp)
    print(f"[abl] 训练产物: id {len(Gp_tr_id)}, tered {len(Gp_tr_te)} ({rec_tag_tr})",
          flush=True)

    # ---------- 变体定义（A1/A3 从缓存 score 重聚合；A4 需重训） ----------
    A1A3 = [
        # A1 · 主口径 2×2 分解（量化"排序升级"与"聚合升级"各自贡献）
        ("base",        dict(bfs_mode="bfs", bfs_q=75, wmax=1.0, sort_mode="cmax")),
        ("legacy_agg",  dict(bfs_mode="bfs", bfs_q=75, wmax=0.7, sort_mode="agg")),
        ("legacy_cmax", dict(bfs_mode="bfs", bfs_q=75, wmax=0.7, sort_mode="cmax")),
        ("legacy_w1",   dict(bfs_mode="bfs", bfs_q=75, wmax=1.0, sort_mode="agg")),
        # A1 · 管线部件消融
        ("no_bfs",      dict(bfs_mode="off")),
        ("no_rank",     dict(bfs_mode="bfs", bfs_q=75, wmax=1.0, sort_mode="seed")),
        ("bfs_q50",     dict(bfs_mode="bfs", bfs_q=50, wmax=1.0, sort_mode="cmax")),
        ("bfs_q90",     dict(bfs_mode="bfs", bfs_q=90, wmax=1.0, sort_mode="cmax")),
        # A1 · wmax 扫描（0.0 / 0.5 / 0.7=legacy / 1.0=base）
        ("w050",        dict(bfs_mode="bfs", bfs_q=75, wmax=0.5, sort_mode="cmax")),
        ("w000",        dict(bfs_mode="bfs", bfs_q=75, wmax=0.0, sort_mode="cmax")),
        # A3 · 预算×排序策略（预算维度由 F1@b 列覆盖）
        ("sort_p95",    dict(bfs_mode="bfs", bfs_q=75, wmax=1.0, sort_mode="p95")),
    ]
    groups_run = []
    if "A1" in groups:
        groups_run += [("A1", n, p) for n, p in A1A3 if n != "sort_p95"]
    if "A3" in groups:
        groups_run += [("A3", n, p) for n, p in A1A3 if n == "sort_p95"]

    t_start = time.time()
    specs = [s.strip() for s in a.test_files.split(",") if s.strip()]
    for spec in specs:
        nm, mr = parse_spec(a.data_dir, spec)
        rec_tag = f"at{mr}" if mr else "atfull"
        gid = resolve_name(a.data_dir, nm)
        res_id = load_res(rcache_dir, "id", tpl_stem, rec_tag, gid)
        res_te = load_res(rcache_dir, "tered", tpl_stem, rec_tag, gid)
        print(f"\n[abl] ===== {spec} ({rec_tag}): "
              f"id {res_id.Gp.n_nodes()} / tered {res_te.Gp.n_nodes()} 节点 =====",
              flush=True)

        configs = [
            ("identity+rate", [r for r in Gp_tr_id], res_id.Gp, "rate"),
            ("TeRed+naive", [r for r in Gp_tr_te], res_te.Gp, "dual_naive"),
            ("TeRed+RATE", [r for r in Gp_tr_te], res_te.Gp, "rate"),
            # 论文方法口径：RATE* = 计数双通道 + collapse-ratio 双通道（Eq.7）
            ("TeRed+RATE*", [r for r in Gp_tr_te], res_te.Gp, "rate_ratio"),
        ]
        for cname, gpt, Gp_test, enc in configs:
            if cname not in want_cfgs:
                continue
            tc = time.time()
            vocab = make_type_vocab(gpt + [Gp_test])
            Xs_train, ets_tr = [], []
            for Gp in gpt:
                X, nids_t, _ = node_feature_matrix(
                    Gp, encoding=enc, tape_dim=a.tape_dim,
                    tape_base=a.tape_base, semantic=a.semantic, vocab=vocab)
                et = _edges_index(Gp, nids_t)
                X, _ = extend_with_v4(X, nids_t, Gp, et, scale=a.v4_scale)
                Xs_train.append(X)
                ets_tr.append(et)

            X, nids, _ = node_feature_matrix(
                Gp_test, encoding=enc, tape_dim=a.tape_dim,
                tape_base=a.tape_base, semantic=a.semantic, vocab=vocab)
            edges = _edges_index(Gp_test, nids)
            X, _ = extend_with_v4(X, nids, Gp_test, edges, scale=a.v4_scale)
            y = label_vector(Gp_test, nids)
            n = len(nids)
            base_extra = {"config": cname, "encoding": enc, "file": spec,
                          "rec_tag": rec_tag, "top_k": a.top_k,
                          "max_alerts": a.max_alerts, "v4_scale": a.v4_scale,
                          "n_nodes_Gp": Gp_test.n_nodes(),
                          "n_edges_Gp": len(edges)}

            # ---- A1/A3：同一份 score 跑全部管线变体 ----
            if "A1" in groups or "A3" in groups:
                det = BenignEnsemble().fit(
                    Xs_train, [f"tr{i}" for i in range(len(Xs_train))])
                score = det.anomaly_scores(X)
                for gname, vname, prm in groups_run:
                    tv = time.time()
                    m = run_variant(vname, gname, score, edges, n, y,
                                    a.top_k, prm, extra=dict(base_extra),
                                    max_alerts=a.max_alerts)
                    m["runtime_s"] = round(time.time() - tv, 1)
                    append_csv(a.results, m)
                    print(f"  [{cname}][{gname}:{vname}] alerts={m['n_alerts']} "
                          f"TP={m['alert_TP']} cov={m['node_cov']:.2f} "
                          f"F1={m['F1_alert']:.3f} best={m['best_F1_alert']:.3f}"
                          f"@b{m['best_b']} nodeF1={m['node_best_f1']:.3f} "
                          f"({m['runtime_s']}s)", flush=True)
                del det, score
                gc.collect()

            # ---- A4：v4 逐列置零 -> 重训重评（管线参数=base） ----
            if "A4" in groups:
                for j, col in enumerate(V4_COLS):
                    tv = time.time()
                    Xs_z = []
                    for Xt in Xs_train:
                        Xz = np.array(Xt, copy=True)
                        Xz[:, -len(V4_COLS) + j] = 0.0
                        Xs_z.append(Xz)
                    Xz_te = np.array(X, copy=True)
                    Xz_te[:, -len(V4_COLS) + j] = 0.0
                    det = BenignEnsemble().fit(
                        Xs_z, [f"tr{i}" for i in range(len(Xs_z))])
                    score_z = det.anomaly_scores(Xz_te)
                    m = run_variant(f"v4_drop_{col}", "A4", score_z, edges, n,
                                    y, a.top_k, dict(bfs_mode="bfs", bfs_q=75,
                                                     wmax=1.0, sort_mode="cmax"),
                                    extra=dict(base_extra, v4_dropped=col),
                                    max_alerts=a.max_alerts)
                    m["runtime_s"] = round(time.time() - tv, 1)
                    append_csv(a.results, m)
                    print(f"  [{cname}][A4:drop_{col}] nodeF1={m['node_best_f1']:.3f} "
                          f"cov={m['node_cov']:.2f} best={m['best_F1_alert']:.3f}"
                          f"@b{m['best_b']} ({m['runtime_s']}s)", flush=True)
                    del Xs_z, Xz_te, det, score_z
                    gc.collect()

            # ---- A5：RATE 拓扑编码消融（同一归约图换编码 -> 重训重评） ----
            if "A5" in groups and cname == "TeRed+RATE":
                a5_encs = [e.strip() for e in a.a5_enc.split(",") if e.strip()]
                for enc2 in a5_encs:
                    tv = time.time()
                    vocab2 = make_type_vocab(gpt + [Gp_test])
                    Xs2 = []
                    for Gp in gpt:
                        X2, nids2, _ = node_feature_matrix(
                            Gp, encoding=enc2, tape_dim=a.tape_dim,
                            tape_base=a.tape_base, semantic=a.semantic,
                            vocab=vocab2)
                        et2 = _edges_index(Gp, nids2)
                        X2, _ = extend_with_v4(X2, nids2, Gp, et2,
                                               scale=a.v4_scale)
                        Xs2.append(X2)
                    X2, nids2, _ = node_feature_matrix(
                        Gp_test, encoding=enc2, tape_dim=a.tape_dim,
                        tape_base=a.tape_base, semantic=a.semantic, vocab=vocab2)
                    et2 = _edges_index(Gp_test, nids2)
                    X2, _ = extend_with_v4(X2, nids2, Gp_test, et2,
                                           scale=a.v4_scale)
                    det2 = BenignEnsemble().fit(
                        Xs2, [f"tr{i}" for i in range(len(Xs2))])
                    s2 = det2.anomaly_scores(X2)
                    m = run_variant(f"enc_{enc2}", "A5", s2, et2, len(nids2), y,
                                    a.top_k,
                                    dict(bfs_mode="bfs", bfs_q=75,
                                         wmax=1.0, sort_mode="cmax"),
                                    extra=dict(base_extra, encoding=enc2),
                                    max_alerts=a.max_alerts)
                    m["runtime_s"] = round(time.time() - tv, 1)
                    append_csv(a.results, m)
                    print(f"  [{cname}][A5:enc_{enc2}] nodeF1="
                          f"{m['node_best_f1']:.3f} cov={m['node_cov']:.2f} "
                          f"best={m['best_F1_alert']:.3f}@b{m['best_b']} "
                          f"({m['runtime_s']}s)", flush=True)
                    del Xs2, X2, det2, s2
                    gc.collect()

            del X, Xs_train, ets_tr, edges
            gc.collect()
            print(f"  [{cname}] 累计 {time.time()-tc:.0f}s", flush=True)
        del res_id, res_te
        gc.collect()

    print(f"\n[abl] 完成 ({time.time()-t_start:.0f}s), 结果追加至 {a.results}",
          flush=True)


if __name__ == "__main__":
    main()
