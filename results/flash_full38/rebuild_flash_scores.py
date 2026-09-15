#!/usr/bin/env python3
"""用 edge_losses CSV 的 loss 列重建 Flash 的 node scores（flash_score 全 0 是框架 bug）。
输出: ~/thr_flash_loss/pr_dir/scores_model_epoch_<ep>.pkl (pred_scores=node max loss)"""
import os, glob, shutil
import numpy as np
import pandas as pd
import torch
from collections import defaultdict

SRC = "/home/user/pids_artifacts_flash/evaluation/evaluation/9b252e2ec8a60deebce89bde07849dd0c667b81d59bf88118017d026046302fc/CADETS_E5/precision_recall_dir"
CSV_DIR = "/home/user/pids_artifacts_flash/training/training/0ade0b642ef6ffc282d4bed8c60b8d492fbca5a8b003562df7538c1c8a6fc899/CADETS_E5/edge_losses/test"
DST = os.path.expanduser("~/thr_flash_loss/pr_dir")
os.makedirs(DST, exist_ok=True)

# 1) 聚合每个 node 在所有 test TW 中的 max loss（一次算好，所有 epoch 共用）
node_to_maxloss = {}
for ep in [0, 1, 3, 5, 7, 9, 11]:
    d = os.path.join(CSV_DIR, f"model_epoch_{ep}")
    for f in sorted(glob.glob(os.path.join(d, "*.csv"))):
        df = pd.read_csv(f, usecols=["node", "loss"])
        m = df.groupby("node")["loss"].max()
        for nid, lv in m.items():
            if nid not in node_to_maxloss or lv > node_to_maxloss[nid]:
                node_to_maxloss[nid] = float(lv)
    print(f"epoch {ep} csv done, nodes so far: {len(node_to_maxloss)}", flush=True)
    break  # loss 各 epoch 不同！不能共用 — 改为按 epoch 分别聚合

# 上面的 break 逻辑不对，改为按 epoch 分别算
node_to_maxloss = {}
for ep in [0, 1, 3, 5, 7, 9, 11]:
    agg = {}
    d = os.path.join(CSV_DIR, f"model_epoch_{ep}")
    for f in sorted(glob.glob(os.path.join(d, "*.csv"))):
        df = pd.read_csv(f, usecols=["node", "loss"])
        m = df.groupby("node")["loss"].max()
        for nid, lv in m.items():
            if nid not in agg or lv > agg[nid]:
                agg[nid] = float(lv)
    src_pkl = os.path.join(SRC, f"scores_model_epoch_{ep}.pkl")
    o = torch.load(src_pkl, map_location="cpu")
    nodes = list(o["nodes"])
    scores = np.array([agg.get(nid, 0.0) for nid in nodes], dtype=np.float64)
    o["pred_scores"] = scores
    dst_pkl = os.path.join(DST, f"scores_model_epoch_{ep}.pkl")
    torch.save(o, dst_pkl)
    npos = int(np.sum(np.asarray(o["y_truth"]) == 1))
    print(f"epoch {ep}: N={len(nodes)} pos={npos} score[min={scores.min():.3f} max={scores.max():.3f} nonzero={np.count_nonzero(scores)}] -> {dst_pkl}", flush=True)
print("DONE")
