#!/usr/bin/env python3
"""阈值扫描评估（零重训成本）。

输入：pidsmaker 评估产物 precision_recall_dir/scores_model_epoch_*.pkl
      （含 pred_scores / y_truth / nodes / node2attacks，逐节点 max-loss 分数）

方法：对分数降序排序后做累积统计（O(N log N)），一次得到完整 P-R / MCC 曲线。

产出：把「固定阈值检出」与「排序能力」分开呈现
  - 官方 max_val_loss 阈值下的指标（对照框架结果）
  - best-F1 / best-MCC operating point
  - TPR @ 固定 FPR（1e-5 / 1e-4 / 1e-3 / 1e-2）
  - top-K 检出（K 扫描）
  - per-attack 覆盖（attack 0 / attack 1）
  - P-R 曲线图（各 epoch 叠加）
"""
import argparse
import json
import os

import numpy as np
import torch

OFFICIAL_THR = {0: 5.913, 1: 6.512, 3: 7.352, 5: 7.764, 7: 7.999, 9: 8.003, 11: 8.043}


def load_scores(p):
    o = torch.load(p, map_location="cpu")
    s = np.asarray(o["pred_scores"], dtype=np.float64)
    y = np.asarray(o["y_truth"], dtype=np.int64)
    nodes = list(o["nodes"])
    n2a = o["node2attacks"]
    return s, y, nodes, n2a


def sorted_stats(s, y):
    """按分数降序，返回累积 tp / fp / 分数序列。"""
    order = np.argsort(-s, kind="mergesort")
    ys = (y[order] == 1).astype(np.int64)
    ks = np.arange(1, len(s) + 1)
    cum_tp = np.cumsum(ys)
    cum_fp = ks - cum_tp
    return order, cum_tp, cum_fp, s[order]


def metrics_at_k(cum_tp, cum_fp, npos, nneg, k):
    k = int(min(max(k, 1), len(cum_tp)))
    tp = int(cum_tp[k - 1])
    fp = int(cum_fp[k - 1])
    fn = npos - tp
    tn = nneg - fp
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / npos if npos else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    d = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    m = (tp * tn - fp * fn) / d if d else 0.0
    return {"k": k, "thr": None, "tp": tp, "fp": fp, "fn": fn,
            "precision": round(prec, 5), "recall": round(rec, 5),
            "f1": round(f1, 5), "mcc": round(m, 5)}


def roc_auc_from_sorted(cum_tp, npos, nneg, n):
    # AUC = 平均 (tp/npos) 对 (fp/nneg) 的积分 ≈ 梯形法
    if npos == 0 or nneg == 0:
        return float("nan")
    tpr = cum_tp / npos
    fpr = np.arange(1, n + 1) / nneg
    integ = getattr(np, "trapezoid", None) or np.trapz
    area = integ(np.concatenate([[0.0], tpr]), np.concatenate([[0.0], fpr]))
    return float(area)


def attacks_detected(s, nodes, n2a, thr):
    hit = {}
    idx = {nid: i for i, nid in enumerate(nodes)}
    for nid, atts in n2a.items():
        i = idx.get(nid)
        if i is None:
            continue
        if s[i] > thr:
            for a in atts:
                hit[int(a)] = True
    return sorted(hit.keys())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="precision_recall_dir")
    ap.add_argument("--epochs", default="0,1,3,5,7,9,11")
    ap.add_argument("--out", default="thr_scan")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    epochs = [int(x) for x in a.epochs.split(",")]
    fprs = [1e-5, 1e-4, 1e-3, 1e-2]
    topks = [62, 100, 200, 500, 1000, 5000, 10000, 50000, 100000]

    summary = {}
    curves = {}
    for ep in epochs:
        p = os.path.join(a.dir, f"scores_model_epoch_{ep}.pkl")
        if not os.path.exists(p):
            print(f"[skip] {p} 不存在", flush=True)
            continue
        s, y, nodes, n2a = load_scores(p)
        npos = int((y == 1).sum())
        nneg = len(s) - npos
        order, cum_tp, cum_fp, s_sorted = sorted_stats(s, y)

        # 每个不同分数值对应一个"阈值档位"（用 > thr 判定 -> 取严格更小的位置）
        # 这里把位置 k 视作"取分数最高的 k 个"，是同一族的 operating point
        prec_arr = np.divide(cum_tp, np.arange(1, len(s) + 1),
                             out=np.zeros(len(s)), where=np.arange(1, len(s) + 1) > 0)
        rec_arr = cum_tp / npos if npos else np.zeros(len(s))
        with np.errstate(divide="ignore", invalid="ignore"):
            f1_arr = np.where((prec_arr + rec_arr) > 0,
                              2 * prec_arr * rec_arr / (prec_arr + rec_arr), 0.0)
        tp_a = cum_tp.astype(np.float64)
        fp_a = cum_fp.astype(np.float64)
        fn_a = npos - tp_a
        tn_a = nneg - fp_a
        den = np.sqrt((tp_a + fp_a) * (tp_a + fn_a) * (tn_a + fp_a) * (tn_a + fn_a))
        mcc_arr = np.divide(tp_a * tn_a - fp_a * fn_a, den,
                            out=np.zeros(len(s)), where=den > 0)

        print(f"\n===== epoch_{ep} | N={len(s)} | 正例={npos} =====", flush=True)
        res = {"n_nodes": len(s), "n_pos": npos,
               "auc": round(roc_auc_from_sorted(cum_tp, npos, nneg, len(s)), 5)}

        def record(name, k):
            d = metrics_at_k(cum_tp, cum_fp, npos, nneg, k)
            d["thr"] = float(s_sorted[k - 1])
            d["attacks"] = attacks_detected(s, nodes, n2a, d["thr"])
            res[name] = d
            print(f"  [{name:16s}] thr={d['thr']:.4f} k={d['k']} tp={d['tp']} fp={d['fp']} fn={d['fn']} "
                  f"P={d['precision']:.4f} R={d['recall']:.4f} F1={d['f1']:.4f} att={d['attacks']}", flush=True)
            return d

        # 官方阈值：位置 = 分数 > thr 的个数
        thr_off = OFFICIAL_THR.get(ep)
        if thr_off is not None:
            k_off = int(np.count_nonzero(s > thr_off))
            record("official", k_off)

        # best-F1 / best-MCC
        record("best_f1", int(np.argmax(f1_arr)) + 1)
        record("best_mcc", int(np.argmax(mcc_arr)) + 1)

        # TPR @ FPR（取满足 FPR<=f 的最大 k）
        res["tpr_at_fpr"] = {}
        fpr_arr = cum_fp / nneg if nneg else np.zeros(len(s))
        for f in fprs:
            k = int(np.searchsorted(fpr_arr, f, side="right"))
            d = record(f"tpr@fpr={f:g}", k) if k > 0 else None
            if d:
                res["tpr_at_fpr"][f] = {"thr": d["thr"], "tp": d["tp"], "fp": d["fp"], "recall": d["recall"]}

        # top-K
        res["topk"] = {}
        for k in topks:
            d = metrics_at_k(cum_tp, cum_fp, npos, nneg, k)
            res["topk"][k] = {"tp": d["tp"], "precision": d["precision"], "recall": d["recall"]}
            print(f"  [top-{k:<7d}] tp={d['tp']} P={d['precision']:.4f} R={d['recall']:.4f}", flush=True)

        # P-R 曲线（前 3000 名密集采样 + 其余降采样，保留小 K 区间的转折）
        step = max(1, len(s) // 800)
        idxs = list(range(0, min(3000, len(s)))) + list(range(3000, len(s), step))
        curves[ep] = [(float(rec_arr[i]), float(prec_arr[i])) for i in idxs]

        summary[ep] = res

    with open(os.path.join(a.out, "threshold_scan.json"), "w") as f:
        json.dump({"summary": summary, "official_thr": OFFICIAL_THR,
                   "curves": {str(k): v for k, v in curves.items()}}, f, indent=2)
    print(f"\n[saved] {a.out}/threshold_scan.json", flush=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # 面板 1：P-R 曲线 + 关键 operating point 标记
        plt.figure(figsize=(9, 6.5))
        for ep, pr in curves.items():
            xs = [q[0] for q in pr]
            ys = [q[1] for q in pr]
            plt.plot(xs, ys, label=f"epoch {ep}", lw=1.3, alpha=0.85)
        markers = [("official", "X", "k", 90), ("best_f1", "o", None, 60), ("best_mcc", "*", None, 160)]
        for ep, res in summary.items():
            for name, mk, col, size in markers:
                d = res.get(name)
                if not d:
                    continue
                plt.scatter([d["recall"]], [d["precision"]], marker=mk, s=size,
                            zorder=5, edgecolors="black", linewidths=0.5,
                            label=f"ep{ep} {name}" if ep in (0,) else None)
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Kairos on CADETS_E5 (38-bin, 62 GT nodes) - node-level P-R\n"
                  "X=official max_val_loss   o=best-F1   * =best-MCC")
        plt.grid(True, which="both", alpha=0.3)
        plt.legend(fontsize=8, loc="lower left", ncol=2)
        plt.tight_layout()
        plt.savefig(os.path.join(a.out, "pr_curve.png"), dpi=140)
        print(f"[saved] {a.out}/pr_curve.png", flush=True)

        # 面板 2：Recall vs 标记节点数（对数轴），更直观呈现"抓多少个节点能换多少检出"
        plt.figure(figsize=(9, 6.5))
        ks = [1, 10, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 500000, 2725340]
        for ep, res in summary.items():
            ys = [res["topk"].get(str(k), res["topk"].get(k, {}).get("recall", None)) for k in ks]
            xs = [k for k, v in zip(ks, ys) if v is not None]
            ys = [v for v in ys if v is not None]
            plt.plot(xs, ys, marker="o", ms=3.5, lw=1.3, label=f"epoch {ep}")
        plt.axhline(62 / 2725340, color="gray", ls="--", lw=1, label="random baseline (62/2.73M)")
        plt.xscale("log")
        plt.xlabel("Number of nodes flagged (top-K by anomaly score)")
        plt.ylabel("Recall (of 62 GT malicious nodes)")
        plt.title("Kairos on CADETS_E5 (38-bin) - Recall @ top-K")
        plt.grid(True, which="both", alpha=0.3)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(a.out, "recall_at_topk.png"), dpi=140)
        print(f"[saved] {a.out}/recall_at_topk.png", flush=True)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[warn] 画图失败: {e}", flush=True)


if __name__ == "__main__":
    main()
