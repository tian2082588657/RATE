# -*- coding: utf-8 -*-
"""diag_mu_auc.py — 在真实归约图上量化 μ 的判别力（免训练）。

回答：μ（Σμ/count 折叠比值）对攻击标签有没有信号？
     高 μ 节点是攻击节点还是良性聚合器？

输出每个测试文件：
  n_nodes, n_pos(攻击节点数)
  AUC(ratio_in), AUC(ratio_out)      # μ 比值
  AUC(count_in), AUC(count_out)      # 边条数（对照）
  P(攻击|ratio>1), 覆盖率(P(ratio>1|攻击))
"""
import sys, os
sys.path.insert(0, '/TeRed+RATE/code')
import numpy as np

from scripts.e6b_alert_replay import load_res
from scripts.e5_f1_eval import resolve_name, find_templates
from scripts.e6_alert_eval import parse_spec

DATA = '/TeRed+RATE/dataset/darpae5/cadets'
CACHE = 'cache'


def _rank(a):
    order = np.argsort(a, kind='mergesort')
    r = np.empty(len(a), dtype=np.float64)
    r[order] = np.arange(len(a), dtype=np.float64)
    return r


def auc(y, x):
    y = np.asarray(y)
    x = np.asarray(x, dtype=np.float64)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float('nan')
    r = _rank(x)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def main():
    tpl = find_templates(os.path.join(CACHE, 'darpa', 'e5_templates', '*.jsonl'))
    stem = os.path.splitext(os.path.basename(tpl))[0]
    rcache = os.path.join(CACHE, 'darpa', 'e5_f1_reduce')
    specs = ['bin.116', 'bin.117', 'bin.118', 'bin.120',
             'bin.119@500000', 'bin.6@400000', 'bin.7@400000']
    print(f"{'file':16s} {'n':>7s} {'pos':>5s} | {'AUC_ratio_in':>12s} "
          f"{'AUC_ratio_out':>13s} {'AUC_cnt_in':>10s} {'AUC_cnt_out':>11s} "
          f"| {'P(atk|r>1)':>10s} {'cov(r>1|atk)':>12s} {'rate_r>1':>8s} {'rate_r>1_atk':>12s}")
    agg = []
    for spec in specs:
        nm, mr = parse_spec(DATA, spec)
        rec_tag = f"at{mr}" if mr else "atfull"
        gid = resolve_name(DATA, nm)
        res = load_res(rcache, 'tered', stem, rec_tag, gid)
        G = res.Gp
        nids = list(G.nodes.keys())
        din_c, dout_c = G.degrees(use_mu=False)
        din_m, dout_m = G.degrees(use_mu=True)
        ic = np.array([din_c[n] for n in nids], dtype=np.float64)
        oc = np.array([dout_c[n] for n in nids], dtype=np.float64)
        im = np.array([din_m[n] for n in nids], dtype=np.float64)
        om = np.array([dout_m[n] for n in nids], dtype=np.float64)
        ri = im / np.maximum(ic, 1.0)
        ro = om / np.maximum(oc, 1.0)
        y = np.array([G.labels.get(n, 0) for n in nids], dtype=np.int64)
        pos = int(y.sum())
        gt1 = (ri > 1.0 + 1e-9) | (ro > 1.0 + 1e-9)
        p_atk_given = (float(y[gt1].sum()) / gt1.sum()) if gt1.sum() else float('nan')
        cov = (float(gt1[y == 1].sum()) / pos) if pos else float('nan')
        row = (auc(y, ri), auc(y, ro), auc(y, ic), auc(y, oc), p_atk_given, cov,
               float(gt1.mean()))
        agg.append((row, nids, gt1, y))
        print(f"{spec:16s} {len(nids):7d} {pos:5d} | {row[0]:12.4f} {row[1]:13.4f} "
              f"{row[2]:10.4f} {row[3]:11.4f} | {row[4]:10.4f} {row[5]:12.4f} "
              f"{row[6]:8.4f}")
    # 汇总
    import statistics as st
    print("\n[mean] ratio_in AUC=%.4f  ratio_out AUC=%.4f  cnt_in AUC=%.4f  cnt_out AUC=%.4f"
          % tuple(st.mean(a[0][i] for a in agg if a[0][i] == a[0][i]) for i in range(4)))
    tot = [a for a in agg]
    n_pos = sum(int(a[3].sum()) for a in tot)
    n_gt1 = sum(int(a[2].sum()) for a in tot)
    n_gt1_pos = sum(int((a[2] & (a[3] == 1)).sum()) for a in tot)
    print(f"[pooled] ratio>1 节点 {n_gt1} 个，其中攻击 {n_gt1_pos} 个 "
          f"→ P(攻击|ratio>1)={n_gt1_pos/max(n_gt1,1):.4f}；"
          f"攻击节点总数 {n_pos}，被 ratio>1 覆盖 {n_gt1_pos} → 覆盖%={n_gt1_pos/max(n_pos,1):.4f}")
    print("DIAG_DONE")


if __name__ == "__main__":
    main()
