# -*- coding: utf-8 -*-
"""同步 nbr_type_div 删列后的辅助文件（e5_f1_eval help、smoke_v4_extras 断言）。

2026-09-11：v4 从 4 列 -> 3 列，下列文本/断言需同步。
幂等：若所有锚点都已消失（说明已打过补丁）则跳过；否则要求每个锚点唯一命中。
"""
import ast
import os
import shutil

ROOT = "/TeRed+RATE/code"


def backup(path):
    bak = path + ".bak.pre_ablation"
    if not os.path.exists(bak):
        shutil.copy2(path, bak)
        print("BACKUP  ", bak)


def patch(path, pairs):
    src = open(path, encoding="utf-8").read()
    if all(old not in src for old, _ in pairs):
        print("SKIP (already applied)", path)
        return
    for i, (old, new) in enumerate(pairs):
        c = src.count(old)
        if c != 1:
            raise SystemExit(f"[FAIL] anchor {i} hits={c} in {path}")
        src = src.replace(old, new)
    ast.parse(src)
    open(path, "w", encoding="utf-8").write(src)
    print("PATCHED ", path)


ev = os.path.join(ROOT, "scripts/e5_f1_eval.py")
backup(ev)
patch(ev, [
    (
        '                    help="启用 v4 扩展特征(log_deg / out_in_ratio / "\n'
        '                         "nbr_type_div / self_loop)。默认开，--no-v4-extras 关闭")',
        '                    help="启用 v4 扩展特征(log_deg / out_in_ratio / "\n'
        '                         "self_loop)。默认开，--no-v4-extras 关闭")',
    ),
])

sm = os.path.join(ROOT, "scripts/smoke_v4_extras.py")
backup(sm)
patch(sm, [
    (
        '    assert X1.shape == (len(nids), d0 + 4), f"应为 ({len(nids)}, {d0+4}), 得 {X1.shape}"\n'
        '    assert names == ["log_deg", "out_in_ratio", "nbr_type_div", "self_loop"]',
        '    assert X1.shape == (len(nids), d0 + 3), f"应为 ({len(nids)}, {d0+3}), 得 {X1.shape}"\n'
        '    assert names == ["log_deg", "out_in_ratio", "self_loop"]',
    ),
    (
        "    self_loop_col = X1[:, d0 + 3]",
        "    self_loop_col = X1[:, d0 + 2]",
    ),
])

print("ALL DONE")
