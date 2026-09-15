# -*- coding: utf-8 -*-
"""patch_a5.py — 给 e6d_ablation_replay.py 的 A5 编码列表加 --a5-enc 参数。"""
p = '/TeRed+RATE/code/scripts/e6d_ablation_replay.py'
s = open(p, encoding='utf-8').read()

old_loop = 'for enc2 in ["dual_naive", "rate_single", "none"]:'
new_loop = 'for enc2 in [e.strip() for e in a.a5_enc.split(",") if e.strip()]:'
assert old_loop in s, 'A5 loop pattern not found'
s = s.replace(old_loop, new_loop)

old_arg = 'ap.add_argument("--results", required=True)'
new_arg = ('ap.add_argument("--results", required=True)\n'
           '    ap.add_argument("--a5-enc", default="dual_naive,rate_single,none")')
assert old_arg in s, 'results arg not found'
s = s.replace(old_arg, new_arg, 1)

open(p, 'w', encoding='utf-8').write(s)
print('PATCH_OK')
