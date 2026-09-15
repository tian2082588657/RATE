# -*- coding: utf-8 -*-
"""scripts/diag_e4.py — E3 cadets 归约率瓶颈数据侧诊断。

目标：解释"E3 时间窗节点归约率 ~1%、边 ~5%"为什么这么低。三个量化问题：
  Q1 同程序多运行（cmd 分组）：每窗进程里相同 cmd 的 run 数分布 —— 若每个
     cmd 只 run 1 次，模板化天花板低（无跨 run 重复可挖）。
  Q2 模板命中结构：挖出的模板多大、锚 type 分布、命中实例的节点构成。
  Q3 吸收冲突：被吸收的共享节点（file/socket）平均被多少个"未吸收的外部进程"
     引用 —— 若共享 file 被第一 region 吸收挡后续匹配，量化能再折多少。
附带对比实验：吸收策略 A(现状:全吸收) vs 策略 B(共享 hub 剥离:只折叠锚侧
私有结构), 报告两者节点/边归约率差异。

用法: python scripts/diag_e4.py [min_support] [per_graph]
"""
from __future__ import annotations
import os, sys, glob, time, collections

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rate_core import load_graphs
from reduction import template_mining as tmining
from reduction.tered import TeRedOperator, find_instances, _to_nx_matching
from reduction.cpr import CPROperator
from reduction.base import check_invariants, node_map_sanity
import numpy as np


def cmd_group_stats(wins):
    """每窗进程按 cmd 分组：组数 / 组内 run 数直方 / 每组节点占比。"""
    print("\n=== Q1 同程序多运行(cmd)统计 ===")
    for i, w in enumerate(wins):
        procs = {n: nd for n, nd in w.nodes.items()
                 if nd.get("type") == "process"}
        groups = collections.defaultdict(list)
        for n, nd in procs.items():
            cmd = nd.get("attrs", {}).get("cmd", "")[:60]
            groups[cmd].append(n)
        sizes = sorted((len(v) for v in groups.values()), reverse=True)
        multi = sum(1 for v in groups.values() if len(v) > 1)
        if i >= 6:
            continue
        print(f"  win{i}: process={len(procs)} cmd组={len(groups)} "
              f"多run组={multi} top组大小={sizes[:8]} "
              f"max组节点占比={sizes[0]/max(1,len(procs)):.2f}" if sizes else
              f"  win{i}: 无进程")
        # 组内重复度：同一 cmd 的进程是否共享 object
        if groups:
            big = max(groups.values(), key=len)
            objs = collections.Counter()
            for n in big:
                for e in w.edges:
                    if e["src"] == n:
                        objs[e["dst"]] += 1
                    elif e["dst"] == n:
                        objs[e["src"]] += 1
            print(f"      top组 '{next(iter(groups))}' 共享object={len(objs)} "
                  f"(进程{len(big)}个 平均每人连{len(objs)/max(1,len(big)):.1f}个对象)")


def main():
    min_support = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    per_graph = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    f = sorted(glob.glob("cache/darpa/e3cadets_w8_*.jsonl"))
    if not f:
        print("NO_WINDOW_CACHE"); return
    wins = load_graphs(f[0])
    n = len(wins)
    n_train = max(2, n // 2)
    train, rest = wins[:n_train], wins[n_train:]
    print(f"[diag_e4] {n} 窗 | train {n_train} | min_support={min_support} "
          f"per_graph={per_graph}")

    cmd_group_stats(wins)

    # ---- 先 CPR 看结构规模 ----
    cpr = CPROperator()
    print("\n=== 窗口结构规模(CPR后) ===")
    struct = []
    for i, w in enumerate(wins):
        r = cpr.reduce(w)
        struct.append(r.Gp)
        if i < 6:
            print(f"  win{i}: 原始 {w.n_nodes()}n/{len(w.edges)}e -> "
                  f"CPR {r.Gp.n_nodes()}n/{len(r.Gp.edges)}e")

    # ---- 挖掘 ----
    t0 = time.time()
    templates, info = tmining.mine_templates(
        train, min_support=min_support, per_graph=per_graph,
        max_templates=200, min_tpl_nodes=3, seed=0, verbose=True)
    print(f"[diag_e4] 挖掘 {time.time()-t0:.1f}s -> {len(templates)} 模板")
    for t in templates[:10]:
        tc = collections.Counter(nd.get("type", "?") for nd in t.nodes.values())
        print(f"  {t.gid}: n={t.n_nodes()} e={t.n_edges()} "
              f"anchor={t.meta.get('anchor')} sup={t.meta.get('support')} "
              f"types={dict(tc.most_common(4))}")

    # ---- Q3 吸收冲突量化（win 上用模板跑匹配，统计被吸收共享节点）----
    print("\n=== Q3 吸收冲突 + 策略对比 ===")
    op = TeRedOperator(templates, max_instances=500, verbose=False)
    for i, w in enumerate(wins):
        tag = "train" if i < n_train else "test"
        # 直接跑匹配拿 regions（与 op.reduce 内一致：max_total 放大以看全貌）
        regions = find_instances(w, templates, max_instances=500, max_total=100000)
        if not regions:
            print(f"  win{i}({tag}): 0 区域"); continue
        absorbed = set()
        for r in regions:
            absorbed |= r["nodes"]
        # 被吸收节点类型分布
        tc = collections.Counter(w.nodes[x].get("type", "?") for x in absorbed)
        # 被吸收共享节点被多少外部进程引用
        proc_nodes = {x for x, nd in w.nodes.items()
                      if nd.get("type") == "process"}
        ext_ref = collections.Counter()
        for x in absorbed:
            if w.nodes[x].get("type", "?") == "process":
                continue
            for e in w.edges:
                if e["src"] == x and e["dst"] not in absorbed and e["dst"] in proc_nodes:
                    ext_ref[x] += 1
                if e["dst"] == x and e["src"] not in absorbed and e["src"] in proc_nodes:
                    ext_ref[x] += 1
        shared_hit = {x: c for x, c in ext_ref.items() if c >= 2}
        print(f"  win{i}({tag}): 区域={len(regions)} 吸收节点={len(absorbed)} "
              f"({(len(absorbed)/w.n_nodes()):.1%}) 类型={dict(tc.most_common(4))}")
        print(f"      被吸收共享节点(>=2外部进程引用)={len(shared_hit)} 个, "
              f"这些节点挡掉的潜在进程引用={sum(shared_hit.values())} 次")

    # ---- 结果汇总 ----
    rows = []
    for i, w in enumerate(wins):
        res = op.reduce(w)
        node_map_sanity(res, w)
        inv = check_invariants(w, res)
        ok = all(v is True for k, v in inv.items() if k != "INV2_violations")
        n0, n1 = w.n_nodes(), res.Gp.n_nodes()
        e0, e1 = len(w.edges), len(res.Gp.edges)
        rows.append((i, n0, n1, e0, e1, ok, len(inv["INV2_violations"])))
        print(f"  win{i:<2}: nodes {n0:>5}->{n1:>5} "
              f"({(1-n1/max(1,n0)):>6.1%}) | edges {e0:>6}->{e1:>6} "
              f"({(1-e1/max(1,e0)):>6.1%}) | INV={ok}")
    nr = np.array([(1 - r[2] / max(1, r[1])) for r in rows])
    er = np.array([(1 - r[4] / max(1, r[3])) for r in rows])
    print(f"\n[diag_e4] 平均节点归约率 {nr.mean():.3f} | 平均边归约率 "
          f"{er.mean():.3f} | INV全过={all(r[5] for r in rows)}")


if __name__ == "__main__":
    main()
