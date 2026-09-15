# -*- coding: utf-8 -*-
"""主表口径升级 + v4 删列 的代码补丁（2026-09-11）。

1) features/v4extras.py      : 删除 nbr_type_div 列（4 列 -> 3 列）
2) scripts/e6_alert_eval.py  : alert_pipeline 默认改为 wmax=1.0 + sort_mode="cmax"
                               （纯 max 聚合 + 簇内 max 排序），并记录 agg_mode
3) scripts/e6b_alert_replay.py: 结果行同步记录 agg_mode

幂等：已应用则跳过；每个锚点都要求唯一命中（0 次/多次都报错）。
每个文件先备份为 <name>.bak.pre_ablation。
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
    else:
        print("BAK-EXIST", bak)


def patch(path, pairs, marker):
    src = open(path, encoding="utf-8").read()
    if marker in src:
        print("SKIP (already patched)", path)
        return
    for i, (old, new) in enumerate(pairs):
        c = src.count(old)
        if c != 1:
            raise SystemExit(f"[FAIL] anchor {i} hits={c} in {path}")
        src = src.replace(old, new)
    ast.parse(src)  # 语法校验
    open(path, "w", encoding="utf-8").write(src)
    print("PATCHED ", path)


# ---------------------------------------------------------------- 1. v4extras
v4 = os.path.join(ROOT, "features/v4extras.py")
backup(v4)
patch(v4, [
    (
        "v4 直接补这 4 列：log_deg / out_in_ratio / nbr_type_div / self_loop。",
        "v4 直接补这 3 列：log_deg / out_in_ratio / self_loop。\n\n"
        "(2026-09-11 消融后删除 nbr_type_div：归约图上 99.5% 节点该列恒为 1/9，\n"
        " 全图无多类型邻居，去掉后指标四位小数不变 -> 判定为死重列。)",
    ),
    (
        '    """在已有 X 上追加 4 列（log_deg / out_in_ratio / nbr_type_div /\n'
        "    self_loop），返回 (X_new, names_new)。",
        '    """在已有 X 上追加 3 列（log_deg / out_in_ratio / self_loop），\n'
        "    返回 (X_new, names_new)。",
    ),
    (
        '    names_new = ["log_deg", "out_in_ratio", "nbr_type_div", "self_loop"]',
        '    names_new = ["log_deg", "out_in_ratio", "self_loop"]',
    ),
    (
        "    # ---- 2. 邻居类型多样性 ----\n"
        "    type_set_per_node = [set() for _ in range(n)]\n"
        "    self_loop = np.zeros(n, dtype=np.float64)\n"
        "    if edges_idx is not None and len(edges_idx) > 0:\n"
        "        # 本图节点类型 vocab\n"
        "        all_types = set()\n"
        "        for nid in nids:\n"
        '            all_types.add(Gp.nodes[nid].get("type", "unknown"))\n'
        "        vocab_size = max(len(all_types), 1)\n"
        "\n"
        "        for s, d in edges_idx:\n"
        "            s, d = int(s), int(d)\n"
        "            if 0 <= s < n:\n"
        "                type_set_per_node[s].add(\n"
        '                    Gp.nodes[nids[s]].get("type", "unknown"))\n'
        "                if s == d:\n"
        "                    self_loop[s] += 1.0\n"
        "            if 0 <= d < n and s != d:\n"
        "                type_set_per_node[d].add(\n"
        '                    Gp.nodes[nids[d]].get("type", "unknown"))\n'
        "\n"
        "        # 归一化到 [0, 1]：多样性 / vocab_size\n"
        "        nbr_type_div = np.array([len(s) / vocab_size\n"
        "                                  for s in type_set_per_node])\n"
        "    else:\n"
        "        nbr_type_div = np.zeros(n, dtype=np.float64)\n"
        "\n"
        "    cols = [log_deg, out_in_ratio, nbr_type_div, self_loop]",
        "    # ---- 2. 自环计数（nbr_type_div 已于 2026-09-11 消融后删除）----\n"
        "    self_loop = np.zeros(n, dtype=np.float64)\n"
        "    if edges_idx is not None and len(edges_idx) > 0:\n"
        "        for s, d in edges_idx:\n"
        "            s, d = int(s), int(d)\n"
        "            if s == d and 0 <= s < n:\n"
        "                self_loop[s] += 1.0\n"
        "\n"
        "    cols = [log_deg, out_in_ratio, self_loop]",
    ),
], marker="nbr_type_div 已于 2026-09-11")

# ---------------------------------------------------------------- 2. e6_alert_eval
ev = os.path.join(ROOT, "scripts/e6_alert_eval.py")
backup(ev)
patch(ev, [
    (
        "def alert_pipeline(score, edges, n, top_k=100, bfs_q=75, min_cluster=3,\n"
        "                   max_cluster=200, max_alerts=20):\n"
        '    """top-K 异常种子 -> 受限 BFS 连通簇 -> 聚合分排序 -> top-B 告警。\n'
        "\n"
        "    返回 [(node_idx_list, agg_score), ...]（按 agg 降序，最多 max_alerts 个）。\n"
        '    """',
        "def alert_pipeline(score, edges, n, top_k=100, bfs_q=75, min_cluster=3,\n"
        '                   max_cluster=200, max_alerts=20, wmax=1.0,\n'
        '                   sort_mode="cmax"):\n'
        '    """top-K 异常种子 -> 受限 BFS 连通簇 -> 聚合分排序 -> top-B 告警。\n'
        "\n"
        "    返回 [(node_idx_list, agg_score), ...]（排序后，最多 max_alerts 个）。\n"
        "\n"
        "    默认口径（2026-09-11 消融升级）：wmax=1.0（纯 max 聚合）+\n"
        '    sort_mode="cmax"（按簇内最大异常分排序）。消融证据：cmax 排序 F1@1 是\n'
        "    旧 agg 排序的 3 倍（0.50 vs 0.17）、F1@20 +58%（0.30 vs 0.19）——旧\n"
        "    0.7max+0.3mean 里的 mean 项是稀释源。旧口径可用 wmax=0.7,\n"
        '    sort_mode="agg" 完整复现。\n'
        '    """',
    ),
    (
        "        if len(cluster) >= min_cluster:\n"
        "            cs = score[np.array(cluster, dtype=np.int64)]\n"
        "            agg = float(cs.max()) * 0.7 + float(cs.mean()) * 0.3\n"
        "            clusters.append((cluster, agg))\n"
        "    clusters.sort(key=lambda x: -x[1])\n"
        "    return clusters[:max_alerts]",
        "        if len(cluster) >= min_cluster:\n"
        "            cs = score[np.array(cluster, dtype=np.int64)]\n"
        "            cmax = float(cs.max())\n"
        "            agg = cmax * wmax + float(cs.mean()) * (1.0 - wmax)\n"
        '            key = cmax if sort_mode == "cmax" else agg\n'
        "            clusters.append((cluster, agg, key))\n"
        "    clusters.sort(key=lambda x: -x[2])\n"
        "    return [(c, a) for c, a, _ in clusters[:max_alerts]]",
    ),
    (
        '                   "max_alerts": a.max_alerts,\n'
        '                   "v4_scale": a.v4_scale,\n'
        '                   "runtime_s": round(time.time() - tc, 1)}',
        '                   "max_alerts": a.max_alerts,\n'
        '                   "v4_scale": a.v4_scale,\n'
        '                   "agg_mode": "w1.0_cmax",\n'
        '                   "runtime_s": round(time.time() - tc, 1)}',
    ),
], marker="wmax=1.0（纯 max 聚合）")

# ---------------------------------------------------------------- 3. e6b_alert_replay
rp = os.path.join(ROOT, "scripts/e6b_alert_replay.py")
backup(rp)
patch(rp, [
    (
        '                   "max_alerts": a.max_alerts, "v4_scale": a.v4_scale,\n'
        '                   "runtime_s": round(time.time() - tc, 1)}',
        '                   "max_alerts": a.max_alerts, "v4_scale": a.v4_scale,\n'
        '                   "agg_mode": "w1.0_cmax",\n'
        '                   "runtime_s": round(time.time() - tc, 1)}',
    ),
], marker='agg_mode": "w1.0_cmax"')

print("ALL DONE")
