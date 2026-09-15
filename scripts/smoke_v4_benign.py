# -*- coding: utf-8 -*-
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), *[".."] * 1))

"""v4 反向冒烟：v4 是否在良性测试集上引入假阳性？

不需要 ground_truth 和 templates（用 identity 归约，跳过 tered），
纯良性 → 良性 测试，看 v4 vs no-v4 的 FP 率（应该都 ≈0）。
如果 v4 引入大量 FP → 方案 A 失败，需调参或转 C。

数据集：bin.1 训练（前 1.5 万 records），bin.2 测试（前 2 万 records）。
归约：identity（μ≡1），naive 与 rate 退化为同一特征。
"""
import sys
import os
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = _R
sys.path.insert(0, ROOT)

from scripts.e5_f1_eval import parse_one, _edges_index
from reduction.base import IdentityOperator
from features.rate import node_feature_matrix, make_type_vocab, label_vector
from features.v4extras import extend_with_v4
from models.detector import BenignEnsemble
from eval.metrics import binary_metrics

DATA_DIR = _os.path.join(_os.path.dirname(_R), "dataset", "darpa e5", "cadets")
TRAIN = "bin.1"
TEST = "bin.2"
MAX_TR = 15000
MAX_TE = 20000


def run(v4: bool, tag: str, scale: str = "raw"):
    print(f"\n{'='*60}\n[{tag}] v4={v4} scale={scale}\n{'='*60}")
    t0 = time.time()
    Gt, _ = parse_one(DATA_DIR, TRAIN, MAX_TR)
    Gte, _ = parse_one(DATA_DIR, TEST, MAX_TE)
    print(f"[parse] ({time.time()-t0:.0f}s) "
          f"train={Gt.n_nodes()} 节点, test={Gte.n_nodes()} 节点")

    res_t = IdentityOperator().reduce(Gt)
    res_te = IdentityOperator().reduce(Gte)
    Gp_t, Gp_te = res_t.Gp, res_te.Gp

    vocab = make_type_vocab([Gp_t, Gp_te])
    Xt, nids_t, _ = node_feature_matrix(Gp_t, encoding="rate",
                                          tape_dim=8, tape_base=10000.0,
                                          semantic="onehot", vocab=vocab)
    Xte, nids_te, _ = node_feature_matrix(Gp_te, encoding="rate",
                                            tape_dim=8, tape_base=10000.0,
                                            semantic="onehot", vocab=vocab)
    if v4:
        e_t = _edges_index(Gp_t, nids_t)
        e_te = _edges_index(Gp_te, nids_te)
        Xt, _ = extend_with_v4(Xt, nids_t, Gp_t, e_t, scale=scale)
        Xte, _ = extend_with_v4(Xte, nids_te, Gp_te, e_te, scale=scale)

    det = BenignEnsemble(radius_q=0.05).fit([Xt], ["tr0"])

    # 训练集自身：应全判良性
    pred_t = det.predict(Xt)
    fp_rate_t = (~pred_t).sum() / len(pred_t)

    # 测试集（良性）：应几乎全判良性
    pred_te = det.predict(Xte)
    fp_rate_te = (~pred_te).sum() / len(pred_te)

    # 异常分分布
    s_t = det.anomaly_scores(Xt)
    s_te = det.anomaly_scores(Xte)

    print(f"  train: n={len(pred_t)} FP={fp_rate_t:.4f} "
          f"score=[{s_t.min():.4f}, {s_t.max():.4f}, mean={s_t.mean():.4f}]")
    print(f"  test : n={len(pred_te)} FP={fp_rate_te:.4f} "
          f"score=[{s_te.min():.4f}, {s_te.max():.4f}, mean={s_te.mean():.4f}]")
    return {"train_fp": float(fp_rate_t), "test_fp": float(fp_rate_te),
            "train_score_range": (float(s_t.min()), float(s_t.max())),
            "test_score_range": (float(s_te.min()), float(s_te.max()))}


def main():
    no_v4 = run(v4=False, tag="baseline")
    v4_raw = run(v4=True, tag="v4-raw", scale="raw")
    v4_rob = run(v4=True, tag="v4-robust", scale="robust")

    print(f"\n{'='*60}\n[对比]\n{'='*60}")
    print(f"  baseline  test_fp = {no_v4['test_fp']:.4f}")
    print(f"  v4-raw    test_fp = {v4_raw['test_fp']:.4f}  "
          f"Δ={v4_raw['test_fp']-no_v4['test_fp']:+.4f}")
    print(f"  v4-robust test_fp = {v4_rob['test_fp']:.4f}  "
          f"Δ={v4_rob['test_fp']-no_v4['test_fp']:+.4f}")
    best = min(v4_raw['test_fp'], v4_rob['test_fp'])
    if best - no_v4['test_fp'] < 0.02:
        print(f"  [OK] v4 在良性测试集上 FP 增量 < 2pp，方案 A 可用")
    else:
        print(f"  [WARN] v4 在良性测试集上 FP 增量 ≥ 2pp，需进一步调参或转 C")


if __name__ == "__main__":
    main()