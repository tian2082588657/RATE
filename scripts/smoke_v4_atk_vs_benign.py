# -*- coding: utf-8 -*-
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), *[".."] * 1))

"""C 阶段告警级口径原型：v4 特征下，攻击图的异常分分布尾部是否比良性图更重？

不需要 ground_truth。我们用以下代理指标：
  - 异常分 top-K% 节点数（攻击图应有更多高分节点）
  - 异常分 ≥ threshold 的节点比例（攻击图应有更大尾部）
  - 异常分 P95、P99（攻击图尾部应上翘）
  - score_max（攻击图应显著高于良性图）

如果 v4 让攻击图 vs 良性图的尾部差距扩大 → 告警级评估（C 阶段）能跑通。
不需要服务器，本地 3 文件（bin.1 train / bin.2 良性 test / bin.118 攻击 test）
即可验证。
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

DATA_DIR = _os.path.join(_os.path.dirname(_R), "dataset", "darpa e5", "cadets")
TRAIN = "bin.1"
BENIGN_TEST = "bin.2"
ATK_TEST = "bin.118"
MAX_TR = 15000
MAX_TE = 80000


def fit_and_score(Gt, G_b, G_atk, v4=False, scale="raw"):
    vocab = make_type_vocab([Gt, G_b, G_atk])
    Xt, nids_t, _ = node_feature_matrix(Gt, encoding="rate",
                                          tape_dim=8, tape_base=10000.0,
                                          semantic="onehot", vocab=vocab)
    Xb, nids_b, _ = node_feature_matrix(G_b, encoding="rate",
                                          tape_dim=8, tape_base=10000.0,
                                          semantic="onehot", vocab=vocab)
    Xa, nids_a, _ = node_feature_matrix(G_atk, encoding="rate",
                                          tape_dim=8, tape_base=10000.0,
                                          semantic="onehot", vocab=vocab)
    if v4:
        et = _edges_index(Gt, nids_t)
        eb = _edges_index(G_b, nids_b)
        ea = _edges_index(G_atk, nids_a)
        Xt, _ = extend_with_v4(Xt, nids_t, Gt, et, scale=scale)
        Xb, _ = extend_with_v4(Xb, nids_b, G_b, eb, scale=scale)
        Xa, _ = extend_with_v4(Xa, nids_a, G_atk, ea, scale=scale)

    det = BenignEnsemble(radius_q=0.05).fit([Xt], ["tr0"])
    return (det.anomaly_scores(Xb), det.anomaly_scores(Xa),
            det.anomaly_scores(Xt))


def stats(name, s):
    return {
        "name": name, "n": len(s),
        "max": float(s.max()), "p99": float(np.percentile(s, 99)),
        "p95": float(np.percentile(s, 95)),
        "mean": float(s.mean()),
        "ratio_above_0.5": float((s >= 0.5).sum() / len(s)),
        "ratio_above_0.8": float((s >= 0.8).sum() / len(s)),
    }


def compare(v4, scale):
    t0 = time.time()
    Gt, _ = parse_one(DATA_DIR, TRAIN, MAX_TR)
    Gb, _ = parse_one(DATA_DIR, BENIGN_TEST, MAX_TE)
    Ga, _ = parse_one(DATA_DIR, ATK_TEST, MAX_TE)
    print(f"[parse] ({time.time()-t0:.0f}s) "
          f"train={Gt.n_nodes()} 节点, "
          f"benign_test={Gb.n_nodes()} 节点, "
          f"atk_test={Ga.n_nodes()} 节点")
    s_b, s_a, s_train = fit_and_score(Gt, Gb, Ga, v4=v4, scale=scale)
    print(f"\n[v4={v4} scale={scale}] 分布对比")
    sb = stats("benign_test", s_b)
    sa = stats("atk_test", s_a)
    st = stats("train", s_train)
    for d in [st, sb, sa]:
        print(f"  [{d['name']:<11}] n={d['n']:>6} max={d['max']:.3f} "
              f"p99={d['p99']:.3f} p95={d['p95']:.3f} mean={d['mean']:.3f} "
              f">=0.5: {d['ratio_above_0.5']:.3f} >=0.8: {d['ratio_above_0.8']:.3f}")
    print(f"  尾部差距 Δmax(atk-benign) = {sa['max']-sb['max']:+.3f}")
    return sb, sa


def main():
    print("=" * 60)
    print("[baseline] v4=False")
    print("=" * 60)
    sb0, sa0 = compare(v4=False, scale="raw")
    print()
    print("=" * 60)
    print("[v4-raw]")
    print("=" * 60)
    sb1, sa1 = compare(v4=True, scale="raw")
    print()
    print("=" * 60)
    print("[C 阶段决策证据]")
    print("=" * 60)
    # 关键指标：攻击图 vs 良性图 max 差距
    print(f"  baseline Δmax = {sa0['max']-sb0['max']:+.3f}  "
          f"p99差={sa0['p99']-sb0['p99']:+.3f}")
    print(f"  v4      Δmax = {sa1['max']-sb1['max']:+.3f}  "
          f"p99差={sa1['p99']-sb1['p99']:+.3f}")
    gain = (sa1['max'] - sb1['max']) - (sa0['max'] - sb0['max'])
    print(f"  v4 vs baseline 尾部差距放大 = {gain:+.3f}")
    if gain > 0.1:
        print(f"  [OK] v4 让攻击图尾部更突出，C 阶段告警级口径可行")
    else:
        print(f"  [WARN] v4 尾部放大不显著，C 阶段需另寻指标（如子图聚类）")


if __name__ == "__main__":
    main()