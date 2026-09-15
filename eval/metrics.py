# -*- coding: utf-8 -*-
"""eval/metrics.py — 检测指标（节点级/子图级通用）。

FPR/FNR 是论文卖点（指南 §7.2），必须一次写全：P/R/F1/FPR/FNR + PR-AUC/ROC-AUC。
纯 numpy 实现。
"""
from __future__ import annotations
import csv, os

import numpy as np


def binary_metrics(y_true, y_pred, score=None):
    """y: 1=攻击。score 越高越异常（用于 AUC）。返回 dict[float]。"""
    y = np.asarray(y_true, dtype=np.int64)
    p = np.asarray(y_pred, dtype=np.int64)
    tp = int(((p == 1) & (y == 1)).sum())
    fp = int(((p == 1) & (y == 0)).sum())
    tn = int(((p == 0) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    fnr = fn / (fn + tp) if fn + tp else 0.0
    m = {"TP": tp, "FP": fp, "TN": tn, "FN": fn, "Precision": prec, "Recall": rec,
         "F1": f1, "FPR": fpr, "FNR": fnr}
    if score is not None:
        s = np.asarray(score, dtype=np.float64)
        m["ROC_AUC"] = _auc(y == 1, s)
        m["PR_AUC"] = _pr_auc(y == 1, s)
        # 阈值无关排序质量口径（学术通行：provenance 论文多报 P@K 而非节点级 F1）
        for k in (10, 50, 100):
            m[f"P@{k}"] = _precision_at_k(y == 1, s, k)
        m["best_F1"] = _best_f1(y == 1, s)
    return m


def _precision_at_k(pos, score, k):
    """top-K 精确率：按分数取前 K 个节点中攻击节点占比。"""
    n = len(pos)
    k = min(k, n)
    if k == 0:
        return 0.0
    order = np.argsort(-score, kind="mergesort")[:k]
    return float(pos[order].sum() / k)


def _best_f1(pos, score):
    """最优阈值下的 F1（排序质量上限）：沿降序扫描所有切分点取最大。"""
    n_pos = int(pos.sum())
    if n_pos == 0:
        return 0.0
    order = np.argsort(-score, kind="mergesort")
    y = pos[order]
    tp = np.cumsum(y)
    ks = np.arange(1, len(y) + 1)
    prec = tp / ks
    rec = tp / n_pos
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    return float(f1.max())


def _auc(pos, score):
    """ROC-AUC = P(score(正) > score(负))；随机为 0.5，越高越好。

    修复 2026-09-12：原实现按**降序**秩（rank 1 = 最高分）套用升序 Mann-Whitney
    公式，实际返回的是 1-AUC（分数越高越好的检测器会被算成 <0.5）。此处改为
    升序秩，与 scipy/sklearn 的 roc_auc_score 一致。该函数此前未被论文任何表格
    使用（论文的 rank-AUC 来自 scripts_ablation/diag_mu_auc.py，实现正确）。
    """
    order = np.argsort(score, kind="mergesort")      # 升序：rank 1 = 最低分
    pos = pos[order]
    n_pos = int(pos.sum())
    n_neg = len(pos) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = np.arange(1, len(pos) + 1)
    return (ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def _pr_auc(pos, score):
    """PR-AUC：精确率-召回率曲线下面积（梯形，>=0 正类占比即随机基线）。"""
    order = np.argsort(-score, kind="mergesort")
    y = pos[order]
    n_pos = int(y.sum())
    if n_pos == 0:
        return float("nan")
    prec = np.cumsum(y) / np.arange(1, len(y) + 1)
    rec = np.cumsum(y) / n_pos
    # 梯形积分（按召回率单调递增）
    _trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(_trap(prec, rec)) if len(y) > 1 else float(prec[0])


def append_csv(path, row: dict):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)
