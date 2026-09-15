# -*- coding: utf-8 -*-
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), *[".."] * 1))

"""本地真实数据 mini 冒烟：v4 vs no-v4 在 bin.118 攻击文件上的 best_F1。

直接 import e5_f1_eval 的真实流程（parse / reduce / 特征），但用 mini
数据（max-records-train=20000, max-records-attack=80000）避免长时间运行。
不依赖服务器，本机即可跑，输出可直接对比 v4 是否在真实数据上救回节点级 F1。

如果 v4 在真实数据上 best_F1 仍 < 0.04（与 v3 同量级）→ 转 C；
如果 ≥ 0.10 → 全量服务器跑；如果 0.04-0.10 → 调参后跑。
"""
import sys
import os
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = _R
sys.path.insert(0, ROOT)

from scripts.e5_f1_eval import (parse_one, load_gt, label_graph,
                                  reduce_all, attack_survival, find_templates)
from reduction.tered import TeRedOperator
from reduction.base import IdentityOperator
from features.rate import node_feature_matrix, make_type_vocab, label_vector
from features.v4extras import extend_with_v4
from models.detector import BenignEnsemble
from eval.metrics import binary_metrics
from rate_core import load_graphs

# ---- 配置：单文件 mini ----
DATA_DIR = _os.path.join(_os.path.dirname(_R), "dataset", "darpa e5", "cadets")
TRAIN = "bin.1"
ATK = "bin.118"
GT = _os.path.join(_R, "cache", "darpa", "ground_truth")
MAX_TR = 20000
MAX_ATK = 80000
CACHE = "cache"
TPL_GLOB = _os.path.join(_R, "cache", "darpa", "e5_templates", "*.jsonl")


def run_one(v4: bool):
    """跑一轮（v4 on/off），返回 (config_name, overall)。"""
    print(f"\n{'='*60}\n[v4={v4}] 开始")
    print(f"{'='*60}")

    # 1. ground truth
    if not os.path.exists(GT) or not os.listdir(GT):
        print(f"[skip] ground_truth 目录不存在或为空: {GT}")
        return None
    gt = load_gt(GT)
    print(f"[gt] {len(gt)} UUIDs")

    # 2. 解析
    t0 = time.time()
    Gt, _ = parse_one(DATA_DIR, TRAIN, MAX_TR)
    Ga, meta = parse_one(DATA_DIR, ATK, MAX_ATK)
    hit, atk = label_graph(Ga, gt)
    print(f"[parse] train={Gt.n_nodes()} 节点, "
          f"atk={Ga.n_nodes()} 节点 {Ga.n_edges()} 边, "
          f"GT 命中 {atk} ({time.time()-t0:.0f}s)")
    if atk == 0:
        print(f"[skip] 攻击文件 {ATK} GT 命中 0")
        return None

    # 3. 归约（简化版，不走 share_k）
    tpl_path = find_templates(TPL_GLOB)
    tpls = load_graphs(tpl_path)
    print(f"[tpl] {tpl_path} -> {len(tpls)}")

    t1 = time.time()
    id_op = IdentityOperator()
    res_t = id_op.reduce(Gt)
    res_a = id_op.reduce(Ga)
    print(f"[reduce-id] ({time.time()-t1:.0f}s)")

    t2 = time.time()
    tered = TeRedOperator(tpls, max_instances=300, max_total=5000, share_k=-1)
    res_t_te = tered.reduce(Gt)
    res_a_te = tered.reduce(Ga)
    print(f"[reduce-tered] train {Gt.n_nodes()}->{res_t_te.Gp.n_nodes()} "
          f"({time.time()-t2:.0f}s)")
    print(f"[reduce-tered] atk {Ga.n_nodes()}->{res_a_te.Gp.n_nodes()} "
          f"({time.time()-t2:.0f}s)")

    # 4. 特征（v4 开关）
    Gp_t, Gp_a = res_t_te.Gp, res_a_te.Gp
    Gp_t_id, Gp_a_id = res_t.Gp, res_a.Gp

    vocab = make_type_vocab([Gp_t_id, Gp_a_id, Gp_t, Gp_a])

    # 训练集：identity（μ=1 时 rate = naive），用 Gp_t_id 即可
    Xt, nids_t, _ = node_feature_matrix(Gp_t_id, encoding="rate",
                                          tape_dim=8, tape_base=10000.0,
                                          semantic="onehot", vocab=vocab)
    if v4:
        # edges_index for train
        from scripts.e5_f1_eval import _edges_index
        e_t = _edges_index(Gp_t_id, nids_t)
        Xt, _ = extend_with_v4(Xt, nids_t, Gp_t_id, e_t)

    # 测试集（攻击）：三配置：id+rate, tered+naive, tered+rate
    results = {}
    for cfg_name, Gp_test, enc in [
        ("id+rate", Gp_a_id, "rate"),
        ("tered+naive", Gp_a, "dual_naive"),
        ("tered+rate", Gp_a, "rate"),
    ]:
        Xa, nids_a, _ = node_feature_matrix(Gp_test, encoding=enc,
                                              tape_dim=8, tape_base=10000.0,
                                              semantic="onehot", vocab=vocab)
        if v4:
            from scripts.e5_f1_eval import _edges_index
            e_a = _edges_index(Gp_test, nids_a)
            Xa, _ = extend_with_v4(Xa, nids_a, Gp_test, e_a)
        ya = label_vector(Gp_test, nids_a)

        det = BenignEnsemble(radius_q=0.05).fit([Xt], ["tr0"])
        pred = (~det.predict(Xa)).astype(np.int64)
        score = det.anomaly_scores(Xa)
        m = binary_metrics(ya, pred, score)
        surv, _ = attack_survival(Ga, res_a_te)
        m["attack_survival"] = surv
        m["n_atk"] = int(ya.sum())
        results[cfg_name] = m
        print(f"  [{cfg_name}] TP={m['TP']} FP={m['FP']} FN={m['FN']} "
              f"P={m['Precision']:.3f} R={m['Recall']:.3f} "
              f"F1={m['F1']:.3f} P@50={m.get('P@50', float('nan')):.3f} "
              f"bestF1={m.get('best_F1', float('nan')):.3f} "
              f"surv={surv:.2f} atk={m['n_atk']}")

    return results


def main():
    if not os.path.exists(TPL_GLOB.split("*")[0]) or not os.path.exists(TPL_GLOB):
        print(f"[fatal] 模板目录不存在: {TPL_GLOB}")
        return
    # 先 no-v4 baseline，再 v4
    res_no = run_one(v4=False)
    res_v4 = run_one(v4=True)

    if res_no and res_v4:
        print(f"\n{'='*60}\n[对比表]\n{'='*60}")
        print(f"{'config':<14} {'baseline F1':<14} {'v4 F1':<14} "
              f"{'baseline bestF1':<18} {'v4 bestF1':<14}")
        for k in res_no:
            f1_b = res_no[k].get('F1', 0)
            f1_v = res_v4[k].get('F1', 0)
            bf_b = res_no[k].get('best_F1', 0)
            bf_v = res_v4[k].get('best_F1', 0)
            d = bf_v - bf_b
            print(f"{k:<14} {f1_b:<14.4f} {f1_v:<14.4f} "
                  f"{bf_b:<18.4f} {bf_v:<14.4f}  ΔbestF1={d:+.4f}")


if __name__ == "__main__":
    main()