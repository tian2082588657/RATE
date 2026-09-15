# -*- coding: utf-8 -*-
"""scripts/e5_f1_eval.py — E5 cadets 三配置节点级 F1 评测（M2 裁决实验）。

三配置（对齐合并指南 §9 M2）：
  identity+rate      无归约基线（μ≡1 时与 dual_naive 逐位相等, INV-1）
  tered+dual_naive   TeRed 归约 + naive 边数特征（预期掉点）
  tered+rate         TeRed 归约 + μ 簿记特征（预期收复）

数据：
  良性训练 = 若干早段 bin 文件（与模板挖掘同口径, max-records 截断）；
  攻击测试 = 含 ground-truth 攻击节点的 bin 文件（全量解析）；
  标签 = ubc-provenance/ground-truth E5-CADETS UUID（Nginx Drakon APT）。

用法:
  python scripts/e5_f1_eval.py --attack-file bin.100 \
      --train-files bin.1,bin.2,bin.3,bin.4,bin.5 \
      --gt ground_truth --results results/e5_f1.csv
"""
from __future__ import annotations
import os, sys, glob, json, time, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import numpy as np

from rate_core import load_graphs, fingerprint, cache_path, save_pickle, load_pickle
from reduction.tered import TeRedOperator
from reduction.base import IdentityOperator, check_invariants, node_map_sanity
from features.rate import node_feature_matrix, make_type_vocab, label_vector
from features.v4extras import extend_with_v4
from models.detector import BenignEnsemble
try:
    from models.detector_graphsage import GraphSAGEDetector
except Exception:               # pragma: no cover
    GraphSAGEDetector = None
from eval.metrics import binary_metrics, append_csv
from adapters import darpa_tc


def _edges_index(Gp, node_ids):
    """把 Gp.edges 的 UUID 端点转为对齐 node_ids 列表的整数索引 (E, 2)。"""
    nid2idx = {nid: i for i, nid in enumerate(node_ids)}
    out = np.empty((len(Gp.edges), 2), dtype=np.int64)
    n_drop = 0
    for i, e in enumerate(Gp.edges):
        s = nid2idx.get(e["src"])
        d = nid2idx.get(e["dst"])
        if s is None or d is None:
            out[i] = 0
            n_drop += 1
        else:
            out[i, 0] = s
            out[i, 1] = d
    if n_drop:
        print(f"[edges] 丢弃 {n_drop}/{len(Gp.edges)} 端点不在 node_ids 的边", flush=True)
    return out


def load_gt(gt_dir):
    gt = {}
    for f in sorted(glob.glob(os.path.join(gt_dir, "node_*.csv"))):
        for line in open(f, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            parts = line.split(",", 2)
            u = parts[0].strip().lower()
            attr = parts[1] if len(parts) > 1 else ""
            if u:
                gt[u] = attr
    return gt


def resolve_name(data_dir, nm):
    """支持短名 bin.N / bin.gz -> 完整 ta1-*.bin.N.gz；不存在则原样返回。"""
    p = os.path.join(data_dir, nm)
    if os.path.exists(p):
        return nm
    if not nm.endswith(".gz"):
        nm2 = nm + ".gz"
        if os.path.exists(os.path.join(data_dir, nm2)):
            return nm2
    import glob as _g
    if nm.endswith(".gz"):
        cand = _g.glob(os.path.join(data_dir, f"*{nm}"))
    else:
        cand = _g.glob(os.path.join(data_dir, f"*.{nm}.gz"))
    if len(cand) == 1:
        return os.path.basename(cand[0])
    raise FileNotFoundError(f"{p} (短名也无法唯一定位: {cand})")


def parse_one(data_dir, nm, max_records):
    nm = resolve_name(data_dir, nm)
    p = os.path.join(data_dir, nm)
    gs, meta = darpa_tc.parse(p, max_records=max_records, verbose=True)
    if not gs:
        raise RuntimeError(f"解析为空图: {nm}")
    return gs[0], meta


def label_graph(G, gt):
    """按 GT UUID 打节点标签；返回 (n_hit, n_atk)。"""
    hit = 0
    for nid in G.nodes:
        if nid in gt:
            G.labels[nid] = 1
            hit += 1
    return hit, hit


def attack_survival(G, res):
    atk = [v for v, l in G.labels.items() if l == 1]
    if not atk:
        return 1.0, 0
    kept = [v for v in atk if res.node_map.get(v) == v]
    return len(kept) / len(atk), len(kept)


def find_templates(cache_glob):
    cands = sorted(glob.glob(cache_glob), key=os.path.getmtime)
    if not cands:
        raise RuntimeError(f"未找到模板缓存: {cache_glob}")
    return cands[-1]


def run_config(name, op_name, Gp_train_list, Gp_attack_list, G_attack_list,
               res_attack_list, encoding, a):
    """三配置统一评测：特征(算一次) -> 多检测器(良性训练) -> 攻击图节点级指标。

    --detector=benign     BenignEnsemble（余弦质心+MVP）。
    --detector=graphsage  GraphSAGEDetector（PyG 负采样链接预测重建）。
    --detector=both       同一次归约/特征下双检测器评测（复用 5-8h 归约产物）。

    返回 {detector_name: (overall, per_file)}。
    """
    detectors = (["benign", "graphsage"] if a.detector == "both"
                 else [a.detector])
    if "graphsage" in detectors and GraphSAGEDetector is None:
        raise RuntimeError("GraphSAGE 不可用: PyG / torch 缺失")

    vocab = make_type_vocab(Gp_train_list + Gp_attack_list)
    Xs_train, Nids_train = [], []
    for Gp in Gp_train_list:
        X, nids, _ = node_feature_matrix(
            Gp, encoding=encoding, tape_dim=a.tape_dim, tape_base=a.tape_base,
            semantic=a.semantic, vocab=vocab)
        Edges_t = _edges_index(Gp, nids)
        if getattr(a, "v4_extras", True):
            X, _ = extend_with_v4(X, nids, Gp, Edges_t, scale=a.v4_scale)
        Xs_train.append(X)
        Nids_train.append(nids)
    Edges_train = [_edges_index(Gp, nids)
                   for Gp, nids in zip(Gp_train_list, Nids_train)]

    Xs_attack, ys_attack = [], []
    for Gp in Gp_attack_list:
        X, nids, fname = node_feature_matrix(
            Gp, encoding=encoding, tape_dim=a.tape_dim, tape_base=a.tape_base,
            semantic=a.semantic, vocab=vocab)
        Edges_a = _edges_index(Gp, nids)
        if getattr(a, "v4_extras", True):
            X, _ = extend_with_v4(X, nids, Gp, Edges_a, scale=a.v4_scale)
        Xs_attack.append((X, Edges_a))
        ys_attack.append(label_vector(Gp, nids))

    out = {}
    for det_name in detectors:
        if det_name == "graphsage":
            det = GraphSAGEDetector(hidden=a.gs_hidden, n_neighbors=a.gs_k,
                                    lr=a.gs_lr, epochs=a.gs_epochs,
                                    radius_q=a.gs_q, score_mode=a.gs_score,
                                    score_norm=a.gs_norm).fit(
                Xs_train, Edges_train,
                [f"train{i}" for i in range(len(Xs_train))])
        else:
            det = BenignEnsemble().fit(
                Xs_train, [f"train{i}" for i in range(len(Xs_train))])

        all_y, all_pred, all_score = [], [], []
        per_file = []
        for (X, edges), y, G, res in zip(Xs_attack, ys_attack,
                                        G_attack_list, res_attack_list):
            if det_name == "graphsage":
                pred = (~det.predict(X, edges)).astype(np.int64)
                score = det.anomaly_scores(X, edges)
            else:
                pred = (~det.predict(X)).astype(np.int64)
                score = det.anomaly_scores(X)
            m = binary_metrics(y, pred, score)
            surv, n_kept = attack_survival(G, res)
            m["attack_survival"] = surv
            m["n_atk_in_Gp"] = int(y.sum())
            per_file.append((G.gid, m))
            all_y += y.tolist(); all_pred += pred.tolist()
            all_score += score.tolist()

        overall = binary_metrics(np.array(all_y), np.array(all_pred),
                                 np.array(all_score))
        print(f"\n=== 配置 {name} ({op_name}+{encoding}) 检测器={det_name} "
              f"总指标 (n={len(all_y)}, atk={int(np.sum(all_y))}) ===",
              flush=True)
        for k, v in overall.items():
            if isinstance(v, float):
                print(f"    {k:<10} {v:.4f}")
            else:
                print(f"    {k:<10} {v}")
        for gid, m in per_file:
            print(f"    [{gid}] TP={m['TP']} FP={m['FP']} FN={m['FN']} "
                  f"P={m['Precision']:.3f} R={m['Recall']:.3f} F1={m['F1']:.3f} "
                  f"P@50={m.get('P@50', float('nan')):.3f} "
                  f"bestF1={m.get('best_F1', float('nan')):.3f} "
                  f"surv={m['attack_survival']:.2f} atk_in_Gp={m['n_atk_in_Gp']}",
                  flush=True)
        out[det_name] = (overall, per_file)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/TeRed+RATE/dataset/darpae5/cadets")
    ap.add_argument("--train-files", default="bin.1,bin.2,bin.3,bin.4,bin.5")
    ap.add_argument("--attack-file", required=True,
                    help="逗号分隔的含攻击文件名, 如 bin.110,bin.111")
    ap.add_argument("--gt", default="ground_truth")
    ap.add_argument("--max-records-train", type=int, default=400000)
    ap.add_argument("--max-records-attack", type=int, default=0,
                    help="0 = 攻击文件全量解析")
    ap.add_argument("--templates", default=None,
                    help="模板 jsonl 路径; 缺省取 cache/darpa/e5_templates/ 最新")
    ap.add_argument("--share-k", type=int, default=-1)
    ap.add_argument("--semantic", default="onehot")
    ap.add_argument("--tape-dim", type=int, default=8)
    ap.add_argument("--tape-base", type=float, default=10000.0)
    ap.add_argument("--detector", default="both",
                    choices=["benign", "graphsage", "both"],
                    help="benign=BenignEnsemble 余弦质心 (MVP); "
                         "graphsage=GraphSAGE 自监督重建 (PyG); "
                         "both=同一次归约下双检测器（复用归约产物）")
    ap.add_argument("--gs-hidden", type=int, default=64)
    ap.add_argument("--gs-k", type=int, default=25)
    ap.add_argument("--gs-lr", type=float, default=0.01)
    ap.add_argument("--gs-epochs", type=int, default=20)
    ap.add_argument("--gs-q", type=float, default=0.99,
                    help="训练异常分分位数阈值(>判异常); 负采样后默认 0.99")
    ap.add_argument("--gs-score", default="mahal", choices=["recon", "mahal"],
                    help="v3 默认 mahal=嵌入 Mahalanobis 距离; "
                         "recon=v2 邻域重建误差(全量已证伪, 仅对照)")
    ap.add_argument("--gs-norm", default="zscore", choices=["none", "zscore"],
                    help="zscore=图内鲁棒标准化(消图级偏移, 阈值可跨图迁移)")
    ap.add_argument("--no-reduce-cache", action="store_true",
                    help="禁用归约产物 pickle 缓存（默认启用）")
    ap.add_argument("--v4-extras", action="store_true", default=True,
                    help="启用 v4 扩展特征(log_deg / out_in_ratio / "
                         "nbr_type_div / self_loop)。默认开，--no-v4-extras 关闭")
    ap.add_argument("--no-v4-extras", dest="v4_extras", action="store_false")
    ap.add_argument("--v4-scale", default="raw", choices=["raw", "robust"],
                    help="v4 特征缩放：raw 原值；robust 图内 z-score(median/IQR)")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--results", default="results/e5_f1.csv")
    a = ap.parse_args()

    gt = load_gt(a.gt)
    print(f"[f1] ground truth: {len(gt)} UUIDs (E5-CADETS Nginx Drakon APT)",
          flush=True)

    # ---------- 1. 良性训练图 ----------
    train_names = [s.strip() for s in a.train_files.split(",") if s.strip()]
    train_graphs = []
    t0 = time.time()
    for nm in train_names:
        g, _ = parse_one(a.data_dir, nm, a.max_records_train or None)
        train_graphs.append(g)
    print(f"[f1] 良性训练图 {len(train_graphs)} 张解析完成 "
          f"({time.time()-t0:.0f}s)", flush=True)

    # ---------- 2. 攻击图（全量 + 标签） ----------
    atk_names = [s.strip() for s in a.attack_file.split(",") if s.strip()]
    atk_graphs = []
    total_atk = 0
    for nm in atk_names:
        g, meta = parse_one(a.data_dir, nm,
                            (a.max_records_attack or None))
        hit, atk = label_graph(g, gt)
        total_atk += atk
        print(f"[f1] 攻击图 {nm}: {g.n_nodes()} 节点 {g.n_edges()} 边 "
              f"records={meta['records']} | GT 命中 {atk}/{len(gt)} "
              f"({100*atk/max(1,len(gt)):.1f}%)", flush=True)
        atk_graphs.append(g)
    if total_atk == 0:
        print("[f1] !! 攻击图 0 命中 —— 文件选择错误或解析截断, 终止", flush=True)
        sys.exit(2)

    # ---------- 3. 归约（identity / tered 各一次, 编码复用） ----------
    tpl_path = a.templates or find_templates(
        os.path.join(a.cache, "darpa", "e5_templates", "*.jsonl"))
    tpls = load_graphs(tpl_path)
    print(f"[f1] 模板: {tpl_path} -> {len(tpls)} 个 (share_k={a.share_k})",
          flush=True)

    # 归约缓存：share_k=-1 时 TeRed 逐图独立，可按 (算子,模板,图) 缓存；
    # share_k>=0 时共享预算跨图全局，缓存会破坏语义 → 自动禁用。
    use_rcache = (not a.no_reduce_cache) and a.share_k == -1
    tpl_stem = os.path.splitext(os.path.basename(tpl_path))[0]
    rcache_dir = os.path.join(a.cache, "darpa", "e5_f1_reduce")
    os.makedirs(rcache_dir, exist_ok=True)

    def reduce_all(op, graphs, tag, op_tag, rec_tag):
        out = []
        t = time.time()
        for i, g in enumerate(graphs):
            cpath = (os.path.join(
                rcache_dir, f"{op_tag}_{tpl_stem}_k{a.share_k}_{rec_tag}_{g.gid}.pkl")
                if use_rcache else None)
            if cpath and os.path.exists(cpath):
                res = load_pickle(cpath)
                node_map_sanity(res, g)
                out.append(res)
                print(f"[f1] {tag} 归约 {g.gid}: 命中缓存 "
                      f"({g.n_nodes()}->{res.Gp.n_nodes()} 节点, "
                      f"{time.time()-t:.0f}s)", flush=True)
                continue
            res = op.reduce(g)
            node_map_sanity(res, g)
            out.append(res)
            if cpath:
                save_pickle(cpath, res)
            print(f"[f1] {tag} 归约 {g.gid}: {g.n_nodes()}->{res.Gp.n_nodes()} "
                  f"节点 / {g.n_edges()}->{res.Gp.n_edges()} 边 "
                  f"({time.time()-t:.0f}s)", flush=True)
        return out

    t1 = time.time()
    res_train_id = reduce_all(IdentityOperator(), train_graphs, "id/train",
                              "id", f"tr{a.max_records_train}")
    res_atk_id = reduce_all(IdentityOperator(), atk_graphs, "id/atk",
                            "id", f"at{a.max_records_attack or 'full'}")
    print(f"[f1] identity 归约完成 {time.time()-t1:.0f}s", flush=True)

    t2 = time.time()
    tered = TeRedOperator(tpls, max_instances=300, max_total=5000,
                          share_k=a.share_k)
    res_train_te = reduce_all(tered, train_graphs, "tered/train",
                              "tered", f"tr{a.max_records_train}")
    res_atk_te = reduce_all(tered, atk_graphs, "tered/atk",
                            "tered", f"at{a.max_records_attack or 'full'}")
    print(f"[f1] tered 归约完成 {time.time()-t2:.0f}s", flush=True)

    # INV（攻击图）
    inv = check_invariants(atk_graphs[0], res_atk_te[0])
    inv_ok = all(v is True for k, v in inv.items() if k != "INV2_violations")
    print(f"[f1] 攻击图 INV(tered): {'全过' if inv_ok else 'FAIL'} "
          f"(INV2_violations={len(inv['INV2_violations'])})", flush=True)

    # ---------- 4. 三配置评测 ----------
    configs = [
        ("无归约基线", "identity", [r.Gp for r in res_train_id],
         [r.Gp for r in res_atk_id], atk_graphs, res_atk_id, "rate"),
        ("TeRed+naive", "tered", [r.Gp for r in res_train_te],
         [r.Gp for r in res_atk_te], atk_graphs, res_atk_te, "dual_naive"),
        ("TeRed+RATE", "tered", [r.Gp for r in res_train_te],
         [r.Gp for r in res_atk_te], atk_graphs, res_atk_te, "rate"),
    ]
    t3 = time.time()
    for name, op_name, gpt, gpa, ga, ra, enc in configs:
        results_by_det = run_config(name, op_name, gpt, gpa, ga, ra, enc, a)
        for det_name, (overall, per_file) in results_by_det.items():
            row = {"dataset": "e5_cadets", "config": name, "operator": op_name,
                   "encoding": enc, "semantic": a.semantic,
                   "tape_dim": a.tape_dim, "tape_base": a.tape_base,
                   "share_k": a.share_k, "n_templates": len(tpls),
                   "train_files": a.train_files, "attack_files": a.attack_file,
                   "gt_nodes": len(gt), "inv_ok": inv_ok,
                   "runtime_s": round(time.time() - t3, 1)}
            for k, v in overall.items():
                row[k] = v
            row["detector"] = det_name
            if det_name == "graphsage":
                row["gs_hidden"] = a.gs_hidden
                row["gs_k"] = a.gs_k
                row["gs_epochs"] = a.gs_epochs
                row["gs_score"] = a.gs_score
                row["gs_norm"] = a.gs_norm
            for gid, m in per_file:
                row[f"surv_{gid}"] = m["attack_survival"]
            append_csv(a.results, row)
    print(f"\n[f1] 完成, 结果追加至 {a.results} (评测段 {time.time()-t3:.0f}s)",
          flush=True)


if __name__ == "__main__":
    main()
