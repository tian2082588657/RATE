# -*- coding: utf-8 -*-
"""v4 特征工程全量验证：v3(no-v4) vs v4 汇总表生成器"""
import csv, glob, os

def norm(cfg):
    if cfg in ('identity', 'naive', 'RATE'):
        return cfg
    if cfg.startswith('TeRed'):
        return 'naive' if 'naive' in cfg else 'RATE'
    return 'identity'  # 无归约基线

rows = []
for p in sorted(glob.glob('e5_f1_v*_*.csv')):
    tag = 'v4' if '_v4_' in p else 'v3'
    with open(p, encoding='utf-8') as f:
        recs = list(csv.DictReader(f))
    # CSV 表头末列带 .gz 文件名，行数可能比表头多，取正常行
    for r in recs:
        rows.append({'file': p.split('bin.')[1].split('_')[0], 'tag': tag,
                     'config': norm(r['config']),
                     'best_F1': float(r['best_F1']), 'P@50': float(r['P@50']),
                     'P@100': float(r['P@100']), 'PR_AUC': float(r['PR_AUC']),
                     'runtime_s': float(r['runtime_s'])})

files = ['116', '117', '118', '119', '120']
confs = ['identity', 'naive', 'RATE']
atfull = {'116', '117', '118', '120'}   # 119 为 at500k 截断口径

def get(fn, tag, cfg):
    for x in rows:
        if x['file'] == fn and x['tag'] == tag and x['config'] == cfg:
            return x
    return None

lines = []
lines.append('# v3 vs v4 逐文件对比（best_F1 / P@50 / P@100 / PR_AUC / 提升倍数）\n')
for fn in files:
    note = ' (at500k 截断口径)' if fn not in atfull else ''
    lines.append(f'## bin.{fn}{note}\n')
    lines.append('| config | v3 best_F1 | v4 best_F1 | 提升x | v3 P@50 | v4 P@50 | v3 P@100 | v4 P@100 | v3 PR_AUC | v4 PR_AUC |')
    lines.append('|---|---|---|---|---|---|---|---|---|---|')
    for cfg in confs:
        a, b = get(fn, 'v3', cfg), get(fn, 'v4', cfg)
        if not a or not b:
            continue
        ratio = b['best_F1'] / a['best_F1'] if a['best_F1'] > 0 else float('inf')
        x = f'{ratio:.1f}' if ratio != float('inf') else 'inf'
        lines.append(f"| {cfg} | {a['best_F1']:.4f} | {b['best_F1']:.4f} | {x} | "
                     f"{a['P@50']:.2f} | {b['P@50']:.2f} | {a['P@100']:.2f} | {b['P@100']:.2f} | "
                     f"{a['PR_AUC']:.4f} | {b['PR_AUC']:.4f} |")
    lines.append('')

# 汇总：naive/RATE 跨文件均值（仅 atfull 4 文件）
lines.append('## atfull 4 文件（116/117/118/120）naive+RATE 汇总\n')
for cfg in ['naive', 'RATE']:
    v3s = [get(f, 'v3', cfg)['best_F1'] for f in files if f in atfull]
    v4s = [get(f, 'v4', cfg)['best_F1'] for f in files if f in atfull]
    m3, m4 = sum(v3s)/len(v3s), sum(v4s)/len(v4s)
    lines.append(f'- {cfg}: v3 mean best_F1={m3:.4f}  →  v4 mean={m4:.4f}  (x{m4/m3:.1f})')
    lines.append(f'  - 各文件 v3: ' + ', '.join(f'{x:.4f}' for x in v3s))
    lines.append(f'  - 各文件 v4: ' + ', '.join(f'{x:.4f}' for x in v4s))

# 提升格子统计（15 个格子 = 5 文件 × 3 配置）
up = down = 0
for fn in files:
    for cfg in confs:
        a, b = get(fn, 'v3', cfg), get(fn, 'v4', cfg)
        if a and b:
            if b['best_F1'] > a['best_F1']:
                up += 1
            else:
                down += 1
lines.append(f'\n## 提升方向统计\n- 15 个对比格（5 文件×3 配置）：提升 {up} 格，未提升 {down} 格')

with open('v3_vs_v4_summary.md', 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')
print('written v3_vs_v4_summary.md')
