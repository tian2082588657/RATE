# -*- coding: utf-8 -*-
"""analyze_a5b.py — 合并分析 A5（旧）与 A5b（μ 抢救）编码消融结果。"""
import csv, statistics as st
from collections import defaultdict

FILES = ['bin.116', 'bin.117', 'bin.118', 'bin.119@500000', 'bin.120',
         'bin.6@400000', 'bin.7@400000']


def load(path, tag):
    rows = []
    try:
        with open(path, encoding='utf-8') as f:
            for r in csv.DictReader(f):
                r['_src'] = tag
                rows.append(r)
    except FileNotFoundError:
        print(f'[warn] missing {path}')
    return rows


rows = load('/TeRed+RATE/code/results/e6d_ablation_a5.csv', 'A5')
rows += load('/TeRed+RATE/code/results/e6d_ablation_a5b.csv', 'A5b')
print('total rows:', len(rows))
print('columns:', list(rows[0].keys())[:24])
print()

# 只看 A5 group
a5 = [r for r in rows if r.get('group') == 'A5']
print('A5-group rows:', len(a5))
encs = sorted(set(r['encoding'] for r in a5))
print('encodings:', encs)
print()

MET = ['node_best_f1', 'best_F1_alert', 'node_cov', 'flat_topk_cov']


def num(r, k):
    try:
        return float(r[k])
    except Exception:
        return None


print('=' * 100)
print('【汇总】按 (编码, 来源) 聚合，7 文件均值')
print('=' * 100)
print(f"{'encoding':18s} {'src':4s} {'node_best_f1':>13s} {'best_F1_alert':>14s} "
      f"{'node_cov':>9s} {'flat_cov':>9s} {'n':>3s}")
byenc = defaultdict(lambda: defaultdict(list))
for r in a5:
    key = (r['encoding'], r['_src'])
    for k in MET:
        v = num(r, k)
        if v is not None:
            byenc[key][k].append(v)
for (enc, src), d in sorted(byenc.items(), key=lambda x: (x[0][1], x[0][0])):
    n = len(d['node_best_f1'])
    print(f"{enc:18s} {src:4s} "
          f"{st.mean(d['node_best_f1']):13.4f} {st.mean(d['best_F1_alert']):14.4f} "
          f"{st.mean(d['node_cov']):9.4f} {st.mean(d['flat_topk_cov']):9.4f} {n:3d}")

print()
print('=' * 100)
print('【逐文件 · node_best_f1】A5b 内部对比（同口径，最干净）')
print('=' * 100)
A5B = ['dual_naive', 'rate_ratio', 'rate_ratio_only', 'rate_rank']
per = defaultdict(dict)
for r in a5:
    if r['_src'] == 'A5b':
        per[r['file']][r['encoding']] = num(r, 'node_best_f1')
hdr = f"{'file':16s}" + ''.join(f"{e:>18s}" for e in A5B) + "   delta(rr-dn)"
print(hdr)
deltas = []
for f in FILES:
    if f not in per:
        continue
    d = per[f]
    dn, rr = d.get('dual_naive'), d.get('rate_ratio')
    delta = (rr - dn) if (dn is not None and rr is not None) else float('nan')
    if delta == delta:
        deltas.append(delta)
    print(f"{f:16s}" + ''.join(f"{d.get(e, float('nan')):18.4f}" for e in A5B)
          + f"   {delta:+.4f}")
print(f"{'MEAN':16s}" + ''.join(
    f"{st.mean([per[f][e] for f in per if e in per[f]]):18.4f}" for e in A5B)
    + f"   {st.mean(deltas):+.4f}")
print(f"\nrate_ratio vs dual_naive 节点级：{sum(1 for d in deltas if d > 0)}/"
      f"{len(deltas)} 文件提升，均值 {st.mean(deltas):+.4f} "
      f"（相对 {st.mean(deltas)/0.1694*100:+.1f}%）")

print()
print('=' * 100)
print('【逐文件 · best_F1_alert】A5b 内部对比')
print('=' * 100)
per2 = defaultdict(dict)
for r in a5:
    if r['_src'] == 'A5b':
        per2[r['file']][r['encoding']] = num(r, 'best_F1_alert')
print(f"{'file':16s}" + ''.join(f"{e:>18s}" for e in A5B))
d2 = []
for f in FILES:
    if f not in per2:
        continue
    d = per2[f]
    dn, rr = d.get('dual_naive'), d.get('rate_ratio')
    if dn is not None and rr is not None:
        d2.append(rr - dn)
    print(f"{f:16s}" + ''.join(f"{d.get(e, float('nan')):18.4f}" for e in A5B))
print(f"{'MEAN':16s}" + ''.join(
    f"{st.mean([per2[f][e] for f in per2 if e in per2[f]]):18.4f}" for e in A5B))
print(f"\n告警级 rate_ratio vs dual_naive：{sum(1 for x in d2 if x > 0)}/"
      f"{len(d2)} 提升，均值 {st.mean(d2):+.4f}")
print("ANALYZE_DONE")
