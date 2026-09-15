# -*- coding: utf-8 -*-
"""train/train_torch.py — GPU 版 GraphSAGE 训练/推理路线（threatrace 语义特征环节）。

与 run_pipeline.py（纯 numpy 良性单类）互补：本脚本把 models/torch_net.py 的
SAGENet 接上同一套「解析 -> 归约 -> RATE 特征」前端，在 GPU 上做监督式
节点二分类（良性 0 / 攻击 1），评测口径与 run_pipeline 完全一致
（binary_metrics 节点级指标 + results CSV 落盘）。

用法（服务器，venv 已装 torch+torch_geometric）：
    python train/train_torch.py --archive ../dataset/streamspot/all.tar.gz \
        --gids "0-6,500-501" --operator identity --encoding rate \
        --results results/streamspot_sage.csv

设计要点：
  * 解析/归约/模板缓存与 run_pipeline 共用（同一 fingerprint 口径，ver=2）；
  * 攻击图对半分：一半进训练（监督信号），一半留测试；
  * 类不平衡用逆频率类别权重（threatrace 用 focal loss，MVP 先用权重版 NLL）；
  * 边做双向化（SAGE 消息传递口径与 threatrace 一致）。
"""
from __future__ import annotations
import argparse, os, random, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from rate_core import fingerprint, save_graphs, load_graphs, save_pickle, load_pickle, cache_path
from adapters import streamspot
from reduction.base import IdentityOperator
from reduction.cpr import CPROperator
from reduction.nodemerge import NodeMergeOperator
from reduction.tered import TeRedOperator
from reduction import template_mining as tmining
from features.rate import node_feature_matrix, make_type_vocab, label_vector
from models.torch_net import SAGENet
from eval.metrics import binary_metrics, append_csv


def parse_gids(spec):
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


def graph_to_data(X, nids, Gp, y):
    """规范图 + 特征矩阵 -> PyG Data（边双向化）。"""
    idx = {nid: i for i, nid in enumerate(nids)}
    src = [idx[e["src"]] for e in Gp.edges if e["src"] in idx and e["dst"] in idx]
    dst = [idx[e["dst"]] for e in Gp.edges if e["src"] in idx and e["dst"] in idx]
    if src:
        edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)
    else:  # 无边图：自环保底，避免 SAGEConv 空邻域
        edge_index = torch.tensor([[], []], dtype=torch.long)
    return Data(x=torch.as_tensor(X, dtype=torch.float32),
                edge_index=edge_index,
                y=torch.as_tensor(y, dtype=torch.long))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default=os.path.join("dataset", "streamspot", "all.tar.gz"))
    ap.add_argument("--gids", default="0-6,500-501")
    ap.add_argument("--split", default="scenes", choices=["scenes", "threatrace"])
    ap.add_argument("--operator", default="identity",
                    choices=["identity", "tered", "cpr", "nodemerge"])
    ap.add_argument("--encoding", default="rate",
                    choices=["none", "dual_naive", "rate_single", "rate"])
    ap.add_argument("--semantic", default="onehot")
    ap.add_argument("--tape-dim", type=int, default=8)
    ap.add_argument("--tape-base", type=float, default=10000.0)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--attack-train-frac", type=float, default=0.5,
                    help="攻击图进入训练集的比例（其余留测试）")
    # tered 归约参数（与 run_pipeline 同名同默认）
    ap.add_argument("--min-support", type=int, default=5)
    ap.add_argument("--min-tpl-nodes", type=int, default=3)
    ap.add_argument("--coverage", type=float, default=1.0)
    ap.add_argument("--max-instances", type=int, default=200)
    ap.add_argument("--per-graph-anchors", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--results", default="results/streamspot_sage.csv")
    ap.add_argument("--no-cache", action="store_true")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[sage] device={device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    t0 = time.time()
    rng = random.Random(a.seed)
    gids = parse_gids(a.gids)
    label_fn = (lambda g: 1 if 125 <= int(g) < 150 else 0) if a.split == "threatrace" \
        else (lambda g: 1 if int(g) // 100 == 5 else 0)
    benign_gids = [g for g in gids if label_fn(g) == 0]
    attack_gids = [g for g in gids if label_fn(g) == 1]
    rng.shuffle(benign_gids)
    n_train = max(1, int(len(benign_gids) * 0.7))
    train_benign = benign_gids[:n_train]
    val_benign = benign_gids[n_train:]
    # 攻击图对半：一半进训练（监督信号），一半留测试
    rng.shuffle(attack_gids)
    n_atk_train = int(len(attack_gids) * a.attack_train_frac) if attack_gids else 0
    train_attack = attack_gids[:n_atk_train]
    test_attack = attack_gids[n_atk_train:]
    train_gids = train_benign + train_attack
    test_gids = val_benign + test_attack
    print(f"[sage] 训练图 {train_gids} | 测试图 {test_gids} | "
          f"operator={a.operator} encoding={a.encoding}")

    # ---------- 1. 解析（缓存口径与 run_pipeline 完全一致） ----------
    data_fp = fingerprint({"ds": "streamspot", "ver": 2, "gids": gids,
                           "split": a.split, "seed": a.seed})
    cache_f = cache_path(a.cache, "streamspot", "parsed", data_fp)
    if a.no_cache or not os.path.exists(cache_f):
        graphs, meta = streamspot.parse(a.archive, gids=gids, split=a.split,
                                        verbose=True)
        save_graphs(cache_f, graphs)
    else:
        graphs = load_graphs(cache_f)
        print(f"[sage] 命中解析缓存 {cache_f} ({len(graphs)} 图)")
    by_gid = {int(g.gid): g for g in graphs}

    # ---------- 2. 模板（仅 tered） ----------
    templates = []
    if a.operator == "tered":
        tpl_fp = fingerprint({"mine": True, "train_gids": train_benign,
                              "min_support": a.min_support,
                              "anchors": a.per_graph_anchors, "seed": a.seed})
        tpl_cache = cache_path(a.cache, "streamspot", "templates", tpl_fp, ext="jsonl")
        if a.no_cache or not os.path.exists(tpl_cache):
            train_benign_graphs = [by_gid[g] for g in train_benign]
            templates, info = tmining.mine_templates(
                train_benign_graphs, min_support=a.min_support,
                per_graph=a.per_graph_anchors, max_templates=200,
                min_tpl_nodes=a.min_tpl_nodes, seed=a.seed, verbose=True)
            save_graphs(tpl_cache, templates)
        else:
            templates = load_graphs(tpl_cache)
        templates = tmining.load_subset(templates, coverage=a.coverage, seed=a.seed)
        print(f"[sage] 模板库 {len(templates)} 个（coverage={a.coverage}）")

    # ---------- 3. 归约（缓存口径与 run_pipeline 一致） ----------
    op_map = {
        "identity": lambda: IdentityOperator(),
        "cpr": lambda: CPROperator(),
        "nodemerge": lambda: NodeMergeOperator(),
        "tered": lambda: TeRedOperator(templates, max_instances=a.max_instances),
    }
    red_fp = fingerprint({"operator": a.operator, "templates_fp": fingerprint(
        {"t": [t.gid for t in templates], "mi": a.max_instances})})
    cache_r = cache_path(a.cache, "streamspot", "reduced", data_fp + "_" + red_fp)
    if a.no_cache or not os.path.exists(cache_r):
        op = op_map[a.operator]()
        reduced = {}
        for g in graphs:
            res = op.reduce(g)
            reduced[int(g.gid)] = (res.Gp, res)
        save_pickle(cache_r, reduced)
    else:
        reduced = load_pickle(cache_r)
        print(f"[sage] 命中归约缓存 {cache_r}")

    # ---------- 4. 特征 + PyG Data ----------
    vocab = make_type_vocab([r[0] for r in reduced.values()])
    datas, dims = {}, None
    for gid_int, (Gp, res) in reduced.items():
        X, nids, fname = node_feature_matrix(Gp, encoding=a.encoding,
                                             tape_dim=a.tape_dim,
                                             tape_base=a.tape_base,
                                             semantic=a.semantic, vocab=vocab)
        y = label_vector(Gp, nids)
        datas[gid_int] = graph_to_data(X, nids, Gp, y).to(device)
        dims = X.shape[1]
    print(f"[sage] 特征 dim={dims} ({fname})")

    # ---------- 5. 训练（监督式，类别逆频率加权） ----------
    model = SAGENet(dims, hidden=a.hidden, label_num=2, dropout=a.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr,
                           weight_decay=a.weight_decay)
    train_y_all = torch.cat([datas[g].y for g in train_gids])
    n_pos = int((train_y_all == 1).sum())
    n_neg = int((train_y_all == 0).sum())
    if n_pos and n_neg:
        w = torch.tensor([1.0 / n_neg, 1.0 / n_pos], dtype=torch.float32,
                         device=device)
        w = w / w.sum() * 2.0
    else:
        w = None
        print("[sage] 警告：训练集只有单类节点")

    model.train()
    for ep in range(1, a.epochs + 1):
        opt.zero_grad()
        loss = 0.0
        for g in train_gids:
            d = datas[g]
            out = model(d.x, d.edge_index)
            loss = loss + F.nll_loss(out, d.y, weight=w)
        loss.backward()
        opt.step()
        if ep % 10 == 0 or ep == 1:
            print(f"[sage] epoch {ep:>4} loss={float(loss) / len(train_gids):.4f}")

    # ---------- 6. 推理 + 评测（与 run_pipeline 同口径） ----------
    model.eval()
    all_y, all_pred, all_score = [], [], []
    per_graph = []
    with torch.no_grad():
        for g in test_gids:
            d = datas[g]
            logp = model(d.x, d.edge_index)
            prob = logp.exp()
            pred = logp.argmax(dim=1).cpu().numpy()
            score = prob[:, 1].cpu().numpy()          # 攻击类概率做异常分
            y = d.y.cpu().numpy()
            m = binary_metrics(y, pred, score)
            per_graph.append((g, m))
            all_y += y.tolist()
            all_pred += pred.tolist()
            all_score += score.tolist()

    overall = binary_metrics(np.array(all_y), np.array(all_pred),
                             np.array(all_score))
    print(f"\n[sage] 总指标(节点级, n={len(all_y)}, dim={dims}, {fname}):")
    for k, v in overall.items():
        if isinstance(v, float):
            print(f"    {k:<10} {v:.4f}")
        else:
            print(f"    {k:<10} {v}")
    for g, m in per_graph:
        print(f"    graph {g}: TP={m['TP']} FP={m['FP']} FN={m['FN']} "
              f"P={m['Precision']:.3f} R={m['Recall']:.3f} F1={m['F1']:.3f} "
              f"FPR={m['FPR']:.3f}")

    # ---------- 7. 落盘 ----------
    row = {"dataset": "streamspot", "model": "sage", "split": a.split,
           "gids": a.gids, "operator": a.operator, "encoding": a.encoding,
           "semantic": a.semantic, "tape_dim": a.tape_dim,
           "tape_base": a.tape_base, "hidden": a.hidden, "epochs": a.epochs,
           "lr": a.lr, "attack_train_frac": a.attack_train_frac,
           "seed": a.seed, "n_train": len(train_gids), "n_test": len(test_gids),
           "dim": dims, "device": device.type,
           "runtime_s": round(time.time() - t0, 1)}
    for k, v in overall.items():
        row[k] = v
    append_csv(a.results, row)
    print(f"\n[sage] 结果已追加: {a.results} (用时 {row['runtime_s']}s)")


if __name__ == "__main__":
    main()
