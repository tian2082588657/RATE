#!/usr/bin/env python3
"""e10: paper's own detector (BenignEnsemble on RATE features) under the *same*
protocol as e9.

Purpose
-------
e9 already reports protocol-matched alert-level scores for three *published*
detectors (Kairos/FLASH/ThreaTrace). This script fills the missing fourth
column: the paper's own node-level detector, evaluated on the *same* 154 test
windows, *same* alert-aggregation pipeline, *same* fixed budget, *same* GT
(node2attacks from Kairos's epoch-3 score pkl).

Configurations (encoding switch only; no reduction applied here so the four
columns stay directly comparable on identical graphs):
    identity     -- semantic one-hot, no topology feature      (paper's "none" mode)
    count        -- semantic + dual_naive (count-based in/out)  (paper's "TeRed+count" without reduction)
    rate         -- semantic + rate (mu-based in/out)           (paper's "TeRed+RATE" without reduction)
    rate_ratio   -- semantic + count tape + mu-ratio tape       (A5b rescued encoding)

Training data
-------------
BenignEnsemble centroids are fit on the *benign* training windows
(2019-05-08, 05-09, 05-11, 05-12) provided by PIDSMaker's CADETS_E5
transformation.  Each window contributes one centroid (== group); the
ensemble radius is the 5th-percentile cosine similarity (same as paper).
This matches the paper's "every benign group -> one recogniser" practice.

Output
------
Per-config json with overall alert_precision / gt_coverage / alert_f1,
plus optional per-window debug rows for sanity checks.
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch
import networkx as nx  # noqa: F401  (graph unpickling)

# ----- paper modules (uploaded next to this script) -----
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rate_core import CanonicalGraph
from features.rate import node_feature_matrix, make_type_vocab
from models.detector import BenignEnsemble

TOP_K, BFS_Q, MIN_CLUSTER, MAX_CLUSTER, MAX_ALERTS = 100, 75.0, 3, 200, 20
TEST_DATES = ["2019-05-16", "2019-05-17"]
TRAIN_DATES = ["2019-05-08", "2019-05-09", "2019-05-11", "2019-05-12"]
TAPE_DIM = 8
TAPE_BASE = 10000.0

# encoding name -> internal switch in features/rate.node_feature_matrix
CONFIGS = (("identity", "none"),
           ("count", "dual_naive"),
           ("rate", "rate"),
           ("rate_ratio", "rate_ratio"))


def nx_to_canonical(G_nx):
    """PIDSMaker MultiDiGraph -> CanonicalGraph, mu=1 (no reduction)."""
    g = CanonicalGraph("pidsnx")
    for nid, d in G_nx.nodes(data=True):
        nid = str(nid)
        g.nodes[nid] = {"type": d.get("node_type", "unknown"), "attrs": {}}
    for u, v, ed in G_nx.edges(data=True):
        g.add_edge(str(u), str(v),
                   ed.get("label", "?"),
                   ts=int(ed.get("time", 0) or 0),
                   mu=1.0)
    return g


def build_adj(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj


def alert_pipeline(score, adj, top_k, bfs_q, min_cluster, max_cluster, max_alerts):
    """Identical to e9."""
    order = np.argsort(-score, kind="mergesort")
    top_idx = order[:top_k]
    thresh = float(np.percentile(score, bfs_q))
    visited = set()
    clusters = []
    for seed in top_idx:
        seed = int(seed)
        if seed in visited:
            continue
        cluster = []
        stack = [seed]
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
            clusters.append((cluster,
                             float(cs.max()) * 0.7 + float(cs.mean()) * 0.3))
    clusters.sort(key=lambda x: -x[1])
    return clusters[:max_alerts]


def load_gt(path):
    """node2attacks from a score pkl (Kairos epoch 3 here, but identical for all)."""
    o = torch.load(path, map_location="cpu")
    return {int(k) for k in o["node2attacks"].keys()}


def iter_windows(tf_root, dates):
    for d in dates:
        dd = os.path.join(tf_root, "graph_" + d)
        if not os.path.isdir(dd):
            continue
        for w in sorted(os.listdir(dd)):
            yield d, w, os.path.join(dd, w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoding", default="all",
                    choices=[c[0] for c in CONFIGS] + ["all"],
                    help="feature encoding; 'all' runs the four configs")
    ap.add_argument("--gt-pkl", default=None,
                    help="path to a score pkl for GT (defaults to Kairos epoch 3)")
    ap.add_argument("--out", default="e10_paper_matched.json")
    ap.add_argument("--debug-windows", type=int, default=0,
                    help="emit per-window detail for first N windows")
    a = ap.parse_args()

    H = os.path.expanduser("~")
    tf_root = glob.glob(os.path.join(
        H, "pids_artifacts_full38/transformation/CADETS_E5/transformation/*/nx"))[0]
    if a.gt_pkl is None:
        a.gt_pkl = glob.glob(os.path.join(
            H, "pids_artifacts_full38/evaluation/evaluation/*/CADETS_E5/"
               "precision_recall_dir/scores_model_epoch_3.pkl"))[0]
    gt = load_gt(a.gt_pkl)
    print(f"[gt] {len(gt)} positive nodes from {a.gt_pkl}", flush=True)

    # ---- enumerate windows ----
    train_paths = [(d, w, p) for d, w, p in iter_windows(tf_root, TRAIN_DATES)]
    test_paths = [(d, w, p) for d, w, p in iter_windows(tf_root, TEST_DATES)]
    print(f"[win] train={len(train_paths)} test={len(test_paths)}", flush=True)

    # ---- global node-type vocab across ALL windows ----
    print("[vocab] building type vocabulary from training+test windows...",
          flush=True)
    types = set()
    for _, _, p in train_paths + test_paths:
        G = torch.load(p, map_location="cpu")
        for _, d in G.nodes(data=True):
            types.add(d.get("node_type", "unknown"))
    vocab = {t: i for i, t in enumerate(sorted(types))}
    print(f"[vocab] {vocab}", flush=True)

    # ---- build BenignEnsemble per encoding ----
    encs = CONFIGS if a.encoding == "all" else (
        next(c for c in CONFIGS if c[0] == a.encoding),)
    ensembles = {}
    print("[fit] training BenignEnsemble (one centroid per benign window)...",
          flush=True)
    t0 = time.time()
    for name, enc in encs:
        Xs, names = [], []
        for i, (d, w, p) in enumerate(train_paths):
            G = torch.load(p, map_location="cpu")
            g = nx_to_canonical(G)
            X, node_ids, _ = node_feature_matrix(
                g, encoding=enc, tape_dim=TAPE_DIM, tape_base=TAPE_BASE,
                semantic="onehot", vocab=vocab)
            Xs.append(X)
            names.append(f"{d}/{w}")
            del G, g
        ens = BenignEnsemble(radius_q=0.05, min_radius=0.50)
        ens.fit(Xs, group_names=names)
        ensembles[name] = ens
        print(f"  [{name}/{enc}] {len(Xs)} groups, "
              f"{sum(x.shape[0] for x in Xs)} nodes, "
              f"{time.time() - t0:.1f}s elapsed", flush=True)

    # ---- evaluate each test window under each encoding ----
    agg = {name: {"n_win": 0, "n_win_gt": 0, "alerts": 0, "alerts_hit": 0,
                  "gt_total": 0, "gt_cov": 0, "nodes": 0}
           for name, _ in encs}
    debug = []

    for wi, (d, w, wp) in enumerate(test_paths):
        G = torch.load(wp, map_location="cpu")
        nids = [int(x) for x in G.nodes()]
        idx = {nd: i for i, nd in enumerate(nids)}
        n = len(nids)
        edges = []
        for u, v in G.edges():
            iu, iv = idx.get(int(u)), idx.get(int(v))
            if iu is not None and iv is not None:
                edges.append((iu, iv))
        adj = build_adj(n, edges)
        g = nx_to_canonical(G)
        for name, enc in encs:
            X, _, _ = node_feature_matrix(
                g, encoding=enc, tape_dim=TAPE_DIM, tape_base=TAPE_BASE,
                semantic="onehot", vocab=vocab)
            sc = ensembles[name].anomaly_scores(X).astype(np.float64)
            y = np.array([1 if nd in gt else 0 for nd in nids], dtype=np.int64)
            clusters = alert_pipeline(sc, adj, TOP_K, BFS_Q,
                                      MIN_CLUSTER, MAX_CLUSTER, MAX_ALERTS)
            hit, cov = 0, set()
            for c, _ in clusters:
                cc = np.array(c, dtype=np.int64)
                pos = cc[y[cc] == 1]
                if len(pos):
                    hit += 1
                    cov.update(pos.tolist())
            r = agg[name]
            r["n_win"] += 1
            r["nodes"] += n
            r["alerts"] += len(clusters)
            r["alerts_hit"] += hit
            r["gt_total"] += int(y.sum())
            r["gt_cov"] += len(cov)
            if int(y.sum()) > 0:
                r["n_win_gt"] += 1
            if a.debug_windows and wi < a.debug_windows:
                debug.append({
                    "enc": name, "date": d, "window": w, "n": n,
                    "n_gt": int(y.sum()),
                    "alerts": len(clusters), "hit": hit,
                    "cov": len(cov),
                })
        del G, g, adj
        if (wi + 1) % 20 == 0:
            print(f"  [{wi + 1}/{len(test_paths)}] done", flush=True)

    out = {"test_dates": TEST_DATES, "train_dates": TRAIN_DATES,
           "windows": len(test_paths),
           "config": {"tape_dim": TAPE_DIM, "tape_base": TAPE_BASE,
                      "top_k": TOP_K, "bfs_q": BFS_Q,
                      "min_cluster": MIN_CLUSTER, "max_cluster": MAX_CLUSTER,
                      "max_alerts": MAX_ALERTS, "mu": 1.0,
                      "reduction": "none (identity graphs)"}}
    for name, _ in encs:
        r = agg[name]
        nal, nh = r["alerts"], r["alerts_hit"]
        P = nh / nal if nal else 0.0
        cov = r["gt_cov"] / r["gt_total"] if r["gt_total"] else 0.0
        F1 = 2 * P * cov / (P + cov) if (P + cov) else 0.0
        out[name] = {**r, "alert_precision": round(P, 4),
                     "gt_coverage": round(cov, 4),
                     "alert_f1": round(F1, 4),
                     "alerts_per_window": round(nal / r["n_win"], 3)}
        print(f"[{name}] alerts={nal} hit={nh} P={P:.4f} cov={cov:.4f} "
              f"F1={F1:.4f}", flush=True)
    if debug:
        out["_debug"] = debug
    json.dump(out, open(a.out, "w"), indent=2)
    print("[saved]", a.out, flush=True)


if __name__ == "__main__":
    main()