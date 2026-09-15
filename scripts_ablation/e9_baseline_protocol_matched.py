#!/usr/bin/env python3
"""e9: protocol-matched alert-level evaluation of three published detectors.

Systems : Kairos (TGN), FLASH (semantic+temporal GNN), THREATRACE (node GraphSAGE)
Data    : PIDSMaker CADETS_E5 38-bin rebuild.  Each 15-min test window is one
          partition.  Graphs are nx.MultiDiGraph saved with torch.save; node ids
          are the *string* form of the int ids used by the score pkl (verified
          100% overlap).
Protocol: identical to the paper's main protocol --
          top_k=100 seeds, BFS expansion with score >= P75, min_cluster=3,
          max_cluster=200, max_alerts=20, fixed threshold (no test-set selection).
GT      : node2attacks from the score pkl (node level, published labels).
Output  : per-system alert precision, GT coverage and alert F1 over all windows.

This is the protocol-matched comparison the paper previously declared missing:
every system is scored on the same graph, the same positive set and the same
alert-aggregation pipeline, with the same budget.
"""
import argparse
import glob
import json
import os

import numpy as np
import torch
import networkx as nx  # noqa: F401  (needed to unpickle the graphs)

TOP_K, BFS_Q, MIN_CLUSTER, MAX_CLUSTER, MAX_ALERTS = 100, 75.0, 3, 200, 20
TEST_DATES = ["2019-05-16", "2019-05-17"]


def build_adj(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj


def alert_pipeline(score, adj, top_k, bfs_q, min_cluster, max_cluster, max_alerts):
    """top-K seeds -> threshold-limited BFS clusters -> aggregate -> top-B alerts."""
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
            clusters.append((cluster, float(cs.max()) * 0.7 + float(cs.mean()) * 0.3))
    clusters.sort(key=lambda x: -x[1])
    return clusters[:max_alerts]


def load_score_map(path):
    o = torch.load(path, map_location="cpu")
    nodes = [int(x) for x in o["nodes"]]
    sc = np.asarray(o["pred_scores"], dtype=np.float64)
    gt = {int(k) for k in o["node2attacks"].keys()}
    return dict(zip(nodes, sc)), gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epoch", type=int, default=3)
    ap.add_argument("--out", default="e9_baseline_matched.json")
    a = ap.parse_args()
    H = os.path.expanduser("~")
    tf_root = glob.glob(os.path.join(
        H, "pids_artifacts_full38/transformation/CADETS_E5/transformation/*/nx"))[0]
    spaths = {
        "Kairos": glob.glob(os.path.join(
            H, "pids_artifacts_full38/evaluation/evaluation/*/CADETS_E5/"
               "precision_recall_dir/scores_model_epoch_%d.pkl" % a.epoch))[0],
        "FLASH": os.path.join(
            H, "thr_flash_loss/pr_dir/scores_model_epoch_%d.pkl" % a.epoch),
        "ThreaTrace": glob.glob(os.path.join(
            H, "pids_artifacts_threatrace/evaluation/evaluation/*/CADETS_E5/"
               "precision_recall_dir/scores_model_epoch_%d.pkl" % a.epoch))[0],
    }

    smaps, gts = {}, {}
    for k, p in spaths.items():
        sm, gt = load_score_map(p)
        smaps[k], gts[k] = sm, gt
        print(f"[load] {k}: {len(sm)} scores, {len(gt)} GT nodes", flush=True)

    windows = []
    for d in TEST_DATES:
        dd = os.path.join(tf_root, "graph_" + d)
        if os.path.isdir(dd):
            windows += [os.path.join(dd, w) for w in sorted(os.listdir(dd))]
    print(f"[win] {len(windows)} windows", flush=True)

    agg = {k: {"n_win": 0, "n_win_gt": 0, "alerts": 0, "alerts_hit": 0,
               "gt_total": 0, "gt_cov": 0, "nodes": 0} for k in spaths}

    for wi, wp in enumerate(windows):
        G = torch.load(wp)
        nids = [int(x) for x in G.nodes()]
        idx = {nd: i for i, nd in enumerate(nids)}
        n = len(nids)
        edges = []
        for u, v in G.edges():
            iu, iv = idx.get(int(u)), idx.get(int(v))
            if iu is not None and iv is not None:
                edges.append((iu, iv))
        adj = build_adj(n, edges)
        for k in spaths:
            sc = np.array([smaps[k].get(nd, -1e9) for nd in nids], dtype=np.float64)
            y = np.array([1 if nd in gts[k] else 0 for nd in nids], dtype=np.int64)
            clusters = alert_pipeline(sc, adj, TOP_K, BFS_Q,
                                      MIN_CLUSTER, MAX_CLUSTER, MAX_ALERTS)
            hit, cov = 0, set()
            for c, _ in clusters:
                cc = np.array(c, dtype=np.int64)
                pos = cc[y[cc] == 1]
                if len(pos):
                    hit += 1
                    cov.update(pos.tolist())
            r = agg[k]
            r["n_win"] += 1
            r["nodes"] += n
            r["alerts"] += len(clusters)
            r["alerts_hit"] += hit
            r["gt_total"] += int(y.sum())
            r["gt_cov"] += len(cov)
            if int(y.sum()) > 0:
                r["n_win_gt"] += 1
        del G, adj
        if (wi + 1) % 20 == 0:
            print(f"  [{wi + 1}/{len(windows)}] done", flush=True)

    out = {}
    for k, r in agg.items():
        nal, nh = r["alerts"], r["alerts_hit"]
        P = nh / nal if nal else 0.0
        cov = r["gt_cov"] / r["gt_total"] if r["gt_total"] else 0.0
        F1 = 2 * P * cov / (P + cov) if (P + cov) else 0.0
        out[k] = {**r, "alert_precision": round(P, 4),
                  "gt_coverage": round(cov, 4),
                  "alert_f1": round(F1, 4),
                  "alerts_per_window": round(nal / r["n_win"], 3)}
        print(f"[{k}] windows={r['n_win']} (w/GT {r['n_win_gt']}) alerts={nal} "
              f"hit={nh} P={P:.4f} cov={cov:.4f} F1={F1:.4f} "
              f"alerts/win={nal / r['n_win']:.2f}", flush=True)
    json.dump({"epoch": a.epoch, "windows": len(windows), "result": out},
              open(a.out, "w"), indent=2)
    print("[saved]", a.out, flush=True)


if __name__ == "__main__":
    main()
