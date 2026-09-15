# -*- coding: utf-8 -*-
"""scripts/diag_e5.py — E3 cadets 归约率上限量化（策略推演）。

结论先行（来自 diag_e4）：
  * 每窗 ~190 process / ~1700 unix_socket / ~190 file；cmd 全空，无法按程序身份聚类。
  * 匹配是 type-based：模板里 file/socket 是"可替换角色"而非固定节点 —— 折叠无需
    节点相同，只需 type 邻域模式相同。
  * 现瓶颈：模板只 3~5 节点(2 process+1 file / 4 process+1 file)，区域小 -> 折叠净赚少；
    且共享 hub 被首个区域吸收挡后续。

本脚本量化三件事：
  Q1 进程邻域模式重复度：每窗进程的 1-hop (邻居 type 多重集 + 边 etype) 指纹；
     若 190 进程只有 ~10 个不同指纹，则"一进程一折叠"模板化上限高（可折 ~180 区域）。
  Q2 对象共享度直方图：socket/file 被多少进程引用；判断"进程私有 vs 全局共享"。
  Q3 折叠上限模拟（两种模板粒度）：
     A) 模板=进程+全部 1-hop 邻居（进程全邻域，对象角色化）——贪心逐进程折叠，对象
        被占则跳过；报告最大节点归约率上限。
     B) 现有 v3 小模板实际表现（对照）。
     C) 若共享对象保留(不吸收、只折叠进程侧私有子结构)，归约率能到多少。

用法: python scripts/diag_e5.py
"""
from __future__ import annotations
import os, sys, glob, collections

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rate_core import load_graphs
from reduction.cpr import CPROperator
import numpy as np


def process_egonet_profile(g):
    """每个 process 的 1-hop 邻域指纹。返回 {proc: (dout 边etype计数字典, 邻居type计数, 对象集)}"""
    nbrs = collections.defaultdict(set)
    etype_out = collections.defaultdict(collections.Counter)
    etype_in = collections.defaultdict(collections.Counter)
    for e in g.edges:
        s, d = e["src"], e["dst"]
        nbrs[s].add(d); nbrs[d].add(s)
        etype_out[s][e["etype"]] += 1
        etype_in[d][e["etype"]] += 1
    procs = [n for n, nd in g.nodes.items() if nd.get("type") == "process"]
    prof = {}
    for p in procs:
        nb = [x for x in nbrs[p] if x in g.nodes]
        tcnt = collections.Counter(g.nodes[x].get("type", "?") for x in nb)
        prof[p] = {
            "n_nbr": len(nb),
            "type_cnt": tuple(sorted(tcnt.items())),
            "out": tuple(sorted(etype_out[p].items())),
            "inn": tuple(sorted(etype_in[p].items())),
            "nbrs": set(nb),
        }
    return prof


def main():
    f = sorted(glob.glob("cache/darpa/e3cadets_w8_*.jsonl"))[0]
    wins = load_graphs(f)
    cpr = CPROperator()
    print("=== Q1/Q2: 每窗进程邻域模式与对象共享度 ===")
    all_pat = collections.Counter()
    for i, w in enumerate(wins):
        r = cpr.reduce(w)
        g = r.Gp
        prof = process_egonet_profile(g)
        # 邻域 size 分布
        sizes = sorted((p["n_nbr"] for p in prof.values()), reverse=True)
        # 完整邻域指纹 (邻居type计数+出入etype) 去重
        full = collections.Counter((p["type_cnt"], p["out"], p["inn"])
                                   for p in prof.values())
        # 仅 type 指纹去重
        tonly = collections.Counter(p["type_cnt"] for p in prof.values())
        # 对象共享度
        obj_share = collections.Counter()
        for p, pf in prof.items():
            for o in pf["nbrs"]:
                if g.nodes[o].get("type") != "process":
                    obj_share[o] += 1
        share_hist = collections.Counter(obj_share.values())
        print(f"win{i}: proc={len(prof)} 邻域size 中位={np.median(sizes):.0f} "
              f"p90={np.percentile(sizes,90):.0f} max={max(sizes) if sizes else 0}")
        print(f"   完整邻域指纹去重={len(full)} (top={full.most_common(3)})")
        print(f"   仅邻居type指纹去重={len(tonly)} 重复度max={tonly.most_common(1)}")
        print(f"   共享对象: 被1进程用={share_hist.get(1,0)} "
              f"2-5进程={sum(v for k,v in share_hist.items() if 2<=k<=5)} "
              f">5进程={sum(v for k,v in share_hist.items() if k>5)}")
        all_pat.update(full)

    print("\n=== Q3 折叠上限模拟 ===")
    for i, w in enumerate(wins):
        r = cpr.reduce(w)
        g = r.Gp
        prof = process_egonet_profile(g)
        n0 = g.n_nodes()
        # 贪心按邻域大->小折叠：模板=proc+全部1-hop邻居，占用的对象已被折则跳过
        # 模拟三种策略下的"可折叠进程数 + 吸收节点数"
        # 策略A: 折叠进程+其全部邻居(进程侧对象全吸收)
        order = sorted(prof.items(), key=lambda kv: -kv[1]["n_nbr"])
        used = set()
        n_region = 0
        absorbed = set()
        for p, pf in order:
            if p in used:
                continue
            block = {p} | pf["nbrs"]
            if block & used:
                continue
            # 只统计非 process 对象数 >=1 的块
            objs = [x for x in pf["nbrs"] if g.nodes[x].get("type") != "process"]
            if not objs:
                continue
            used |= block
            n_region += 1
            absorbed |= block
        # 策略C: 折叠进程 + 其"私有"对象(被<=2进程引用的对象)；共享对象留外部
        obj_share = collections.Counter()
        for p, pf in prof.items():
            for o in pf["nbrs"]:
                if g.nodes[o].get("type") != "process":
                    obj_share[o] += 1
        order2 = sorted(prof.items(), key=lambda kv: -kv[1]["n_nbr"])
        used2 = set()
        n_region2 = 0
        absorbed2 = set()
        for p, pf in order2:
            if p in used2:
                continue
            priv = {x for x in pf["nbrs"]
                    if g.nodes[x].get("type") != "process" and obj_share.get(x, 9) <= 2}
            if not priv:
                continue
            block = {p} | priv
            if block & used2:
                continue
            used2 |= block
            n_region2 += 1
            absorbed2 |= block
        # 汇总：节点归约率（折叠后每区域生成1个摘要节点近似；保留图里实际还要 +1 出口节点）
        def _nr(na, nr_):
            # 每区域 2 个摘要节点(M/N) 替代 absorbed；外部共享对象保留时不产生额外节点
            return (n0 - (n0 - na + 2 * nr_)) / n0 if False else (na - 2 * nr_) / n0
        print(f"win{i}: 策略A(全邻域+对象全吸收): 区域={n_region} 吸收节点={len(absorbed)} "
              f"节点归约率~{_nr(len(absorbed), n_region):.1%}")
        print(f"       策略C(进程+私有对象, 共享保留): 区域={n_region2} 吸收节点={len(absorbed2)} "
              f"节点归约率~{_nr(len(absorbed2), n_region2):.1%}")


if __name__ == "__main__":
    main()
