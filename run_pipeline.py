# -*- coding: utf-8 -*-
"""run_pipeline.py — RATE 端到端流水线（StreamSpot 主冒烟入口）。

用法示例：
    # 1) 全链路冒烟：TeRed 归约 + RATE 编码 + 良性单类检测
    python run_pipeline.py --operator tered --encoding rate --gids "0-6,500-501"
    # 2) naive 对比（同一归约图、按边条数）
    python run_pipeline.py --operator tered --encoding dual_naive --gids "0-6,500-501"
    # 3) 无归约基线
    python run_pipeline.py --operator identity --encoding rate --gids "0-6,500-501"

流水线（指南 §2.2）：数据 -> 适配器 -> 规范图 -> 归约R(G' + π/ρ/μ) -> 特征
(semantic ‖ RATE/naive) -> 良性单类训练 -> Presumption-of-Innocence 推理 -> 指标。
所有中间产物按配置指纹缓存到 cache/；每次运行一行真实结果写入 results/。
"""
from __future__ import annotations
import argparse, os, random, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from rate_core import (CanonicalGraph, fingerprint, save_graphs, load_graphs,
                       save_pickle, load_pickle, cache_path)
from adapters import streamspot
from reduction.base import IdentityOperator, check_invariants, node_map_sanity
from reduction.cpr import CPROperator
from reduction.nodemerge import NodeMergeOperator
from reduction.tered import TeRedOperator
from reduction import template_mining as tmining
from features.rate import node_feature_matrix, make_type_vocab, label_vector
from models.detector import BenignEnsemble
from eval.metrics import binary_metrics, append_csv


def parse_gids(spec):
    """'0-6,500-501' -> [0,1,...,6,500,501]"""
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def pick(args, meta_key):
    return args


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default=os.path.join("dataset", "streamspot", "all.tar.gz"))
    ap.add_argument("--gids", default="0-6,500-501",
                    help="要解析的图 id 集合（文件按 gid 有序）")
    ap.add_argument("--split", default="scenes", choices=["scenes", "threatrace"])
    ap.add_argument("--operator", default="identity",
                    choices=["identity", "tered", "cpr", "nodemerge"])
    ap.add_argument("--encoding", default="rate",
                    choices=["none", "dual_naive", "rate_single", "rate"])
    ap.add_argument("--semantic", default="onehot")
    ap.add_argument("--tape-dim", type=int, default=8)
    ap.add_argument("--tape-base", type=float, default=10000.0)
    ap.add_argument("--min-support", type=int, default=5, help="模板最小支持度")
    ap.add_argument("--min-tpl-nodes", type=int, default=3)
    ap.add_argument("--coverage", type=float, default=1.0, help="模板覆盖率抽样(归约强度)")
    ap.add_argument("--max-instances", type=int, default=200, help="每模板最大归约实例数")
    ap.add_argument("--per-graph-anchors", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--results", default="results/streamspot.csv")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    t0 = time.time()
    rng = random.Random(a.seed)
    gids = parse_gids(a.gids)
    label_fn = (lambda g: 1 if 125 <= int(g) < 150 else 0) if a.split == "threatrace" \
        else (lambda g: 1 if int(g) // 100 == 5 else 0)
    # train / val(良性) / test(攻击) 划分
    benign_gids = [g for g in gids if label_fn(g) == 0]
    attack_gids = [g for g in gids if label_fn(g) == 1]
    rng.shuffle(benign_gids)
    n_train = max(1, int(len(benign_gids) * 0.7))
    train_gids = benign_gids[:n_train]
    val_gids = benign_gids[n_train:]
    print(f"[pipeline] 解析 {len(gids)} 图 | 良性训练 {train_gids} | "
          f"良性验证 {val_gids} | 攻击 {attack_gids} | operator={a.operator} "
          f"encoding={a.encoding}")

    # ---------- 1. 解析 + 缓存 ----------
    # ver: 解析器/标签逻辑一旦变更必须递增，否则会命中旧代码的脏缓存。
    data_fp = fingerprint({"ds": "streamspot", "ver": 2, "gids": gids,
                           "split": a.split, "seed": a.seed})
    cache_f = cache_path(a.cache, "streamspot", "parsed", data_fp)
    if a.no_cache or not os.path.exists(cache_f):
        graphs, meta = streamspot.parse(a.archive, gids=gids, split=a.split,
                                        verbose=True)
        save_graphs(cache_f, graphs)
    else:
        graphs = load_graphs(cache_f)
        print(f"[pipeline] 命中解析缓存 {cache_f} ({len(graphs)} 图)")
    all_gids = sorted(int(g.gid) for g in graphs)
    by_gid = {int(g.gid): g for g in graphs}

    # ---------- 2. 模板（仅 teRed 需要，从良性训练图挖掘并冻结） ----------
    templates = []
    if a.operator == "tered":
        tpl_fp = fingerprint({"mine": True, "mine_ver": 7,  # v6: node去重默认+共享感知吸收v4b(2026-09-03, A/B: node最优)
                              "train_gids": train_gids,
                              "min_support": a.min_support,
                              "anchors": a.per_graph_anchors, "seed": a.seed})
        tpl_cache = cache_path(a.cache, "streamspot", "templates", tpl_fp, ext="jsonl")
        if a.no_cache or not os.path.exists(tpl_cache):
            train_benign = [by_gid[g] for g in train_gids]
            templates, info = tmining.mine_templates(
                train_benign, min_support=a.min_support,
                per_graph=a.per_graph_anchors, max_templates=200,
                min_tpl_nodes=a.min_tpl_nodes, seed=a.seed, verbose=True)
            save_graphs(tpl_cache, templates)
        else:
            templates = load_graphs(tpl_cache)
        templates = tmining.load_subset(templates, coverage=a.coverage, seed=a.seed)
        print(f"[pipeline] 模板库 {len(templates)} 个（coverage={a.coverage}）")

    # ---------- 3. 归约 ----------
    op_map = {
        "identity": lambda: IdentityOperator(),
        "cpr": lambda: CPROperator(),
        "nodemerge": lambda: NodeMergeOperator(),
        "tered": lambda: TeRedOperator(templates, max_instances=a.max_instances,
                                       verbose=a.verbose),
    }
    red_fp = fingerprint({"operator": a.operator, "templates_fp": fingerprint(
        {"t": [t.gid for t in templates], "mi": a.max_instances})})
    cache_r = cache_path(a.cache, "streamspot", "reduced", data_fp + "_" + red_fp)
    if a.no_cache or not os.path.exists(cache_r):
        op = op_map[a.operator]()
        reduced = {}
        for g in graphs:
            res = op.reduce(g)
            node_map_sanity(res, g)
            reduced[int(g.gid)] = (res.Gp, res)
        save_pickle(cache_r, reduced)
    else:
        reduced = load_pickle(cache_r)
        print(f"[pipeline] 命中归约缓存 {cache_r}")

    # ---------- 4. 特征 ----------
    vocab = make_type_vocab([r[0] for r in reduced.values()])
    feats = {}
    for gid_int, (Gp, res) in reduced.items():
        X, nids, fname = node_feature_matrix(Gp, encoding=a.encoding,
                                             tape_dim=a.tape_dim,
                                             tape_base=a.tape_base,
                                             semantic=a.semantic, vocab=vocab)
        feats[gid_int] = (X, nids, Gp, res)
    dim = feats[all_gids[0]][0].shape[1]

    # ---------- 5. 训练（只用良性训练图） ----------
    groups_X, group_names = [], []
    for g in train_gids:
        X, nids, Gp, res = feats[g]
        groups_X.append(X)
        group_names.append(f"scene{g // 100}:gid{g}")
    det = BenignEnsemble().fit(groups_X, group_names)

    # ---------- 6. 推理 + 评测（验证良性 + 攻击） ----------
    all_y, all_pred, all_score = [], [], []
    per_graph = []
    for gid_int in val_gids + attack_gids:
        X, nids, Gp, res = feats[gid_int]
        y = label_vector(Gp, nids)
        pred = det.predict(X).astype(np.int64)
        score = det.anomaly_scores(X)
        m = binary_metrics(y, pred, score)
        per_graph.append((gid_int, m))
        all_y += y.tolist()
        all_pred += pred.tolist()
        all_score += score.tolist()

    overall = binary_metrics(np.array(all_y), np.array(all_pred),
                             np.array(all_score))
    print(f"\n[pipeline] 总指标(节点级, n={len(all_y)}, dim={dim}, {fname}):")
    for k, v in overall.items():
        if isinstance(v, float):
            print(f"    {k:<10} {v:.4f}")
        else:
            print(f"    {k:<10} {v}")
    for gid_int, m in per_graph:
        print(f"    graph {gid_int}: "
              f"TP={m['TP']} FP={m['FP']} FN={m['FN']} "
              f"P={m['Precision']:.3f} R={m['Recall']:.3f} F1={m['F1']:.3f} "
              f"FPR={m['FPR']:.3f}")

    # ---------- 7. 归约统计 + 不变量摘要 ----------
    red_stats = []
    for gid_int, (Gp, res) in reduced.items():
        inv = check_invariants(by_gid[gid_int], res)
        ok = all(v is True for k, v in inv.items() if k != "INV2_violations")
        surv = attack_survival(by_gid[gid_int], res)
        red_stats.append({"gid": gid_int, "op": a.operator,
                          "n0": by_gid[gid_int].n_nodes(),
                          "n1": Gp.n_nodes(),
                          "e0": by_gid[gid_int].n_edges(), "e1": Gp.n_edges(),
                          "inv_ok": ok, "inv2_bad": len(inv["INV2_violations"]),
                          "mu_sum": Gp.total_edge_mu(), "attack_survival": surv})
    if a.operator != "identity":
        import json as _json
        print(f"\n[归约] 平均归约率(节点)="
              f"{np.mean([(1 - r['n1'] / max(1, r['n0'])) for r in red_stats]):.3f}  "
              f"边="
              f"{np.mean([(1 - r['e1'] / max(1, r['e0'])) for r in red_stats]):.3f}  "
              f"INV 全过={all(r['inv_ok'] for r in red_stats)}  "
              f"攻击节点存活={np.mean([r['attack_survival'] for r in red_stats]):.3f}")

    # ---------- 8. 落盘 ----------
    row = {"dataset": "streamspot", "split": a.split, "gids": a.gids,
           "operator": a.operator, "encoding": a.encoding, "semantic": a.semantic,
           "tape_dim": a.tape_dim, "tape_base": a.tape_base,
           "min_support": a.min_support, "coverage": a.coverage, "seed": a.seed,
           "n_train": len(train_gids), "n_val": len(val_gids),
           "n_attack": len(attack_gids),
           "n_templates": len(templates), "dim": dim,
           "runtime_s": round(time.time() - t0, 1)}
    for k, v in overall.items():
        row[k] = v
    append_csv(a.results, row)
    print(f"\n[pipeline] 结果已追加: {a.results} (用时 {row['runtime_s']}s)")


def attack_survival(G, res):
    """INV-5：ground-truth 攻击节点中未被模板吸收(π(v)==v)的比例。"""
    atk = [v for v, l in G.labels.items() if l == 1]
    if not atk:
        return 1.0
    kept = [v for v in atk if res.node_map.get(v) == v]
    return len(kept) / len(atk)


if __name__ == "__main__":
    main()
