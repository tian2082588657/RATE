# -*- coding: utf-8 -*-
"""scripts/diag_e7.py — win0 进程邻域结构精查（决定折叠层修法）。

问题：R3 显示剔除共享上下文后只剩 34/窗 进程锚 —— "进程+私有对象" 形态在
cadets 里到底存不存在、长什么样？逐进程统计：
  Q1: 每个 process 的 1-hop 邻居按 (type, share度) 分布
  Q2: 可折叠候选形态统计：
      a) 1 process + >=2 私有(k=1)对象
      b) 1 process + >=2 私有 + 共享文件(库)
      c) 纯共享(全是库文件/多进程共享)
  Q3: 若折叠只吸收 锚进程+私有对象（共享保留），每窗能折多少节点（结构上限）
"""
from __future__ import annotations
import os, sys, glob, collections

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rate_core import load_graphs
from reduction.cpr import CPROperator
from reduction.template_mining import _SUBJECT_WORDS

SUBJ = ("process", "subject")


def analyze(g, name=""):
    # CPR 结构图（模板匹配口径）
    cpr = CPROperator()
    gc = cpr.reduce(g).Gp
    procs = [n for n, nd in gc.nodes.items() if nd.get("type") in SUBJ]
    # 对象共享度：被多少个不同 process 直接引用
    ref = collections.Counter()
    for e in gc.edges:
        if gc.nodes.get(e["src"], {}).get("type") in SUBJ:
            ref[e["dst"]] += 1
    # 每 process 的 1-hop 对象邻居分组
    nbr = collections.defaultdict(list)      # process -> [(obj, share, type)]
    for e in gc.edges:
        if gc.nodes.get(e["src"], {}).get("type") in SUBJ:
            t = gc.nodes.get(e["dst"], {}).get("type", "?")
            nbr[e["src"]].append((e["dst"], ref[e["dst"]], t))
    n_proc = len(procs)
    n_priv = sum(1 for c in ref.values() if c == 1)
    n_shared = sum(1 for c in ref.values() if c >= 2)
    # 形态统计
    cnt_a = cnt_b = cnt_c = cnt_iso = 0
    max_priv = 0
    n_priv_total = 0
    proc_with_priv = 0
    for p in procs:
        objs = nbr.get(p, [])
        priv = [o for o in objs if o[1] == 1 and o[2] not in SUBJ]
        sh = [o for o in objs if o[1] >= 2 and o[2] not in SUBJ]
        pn = [o for o in objs if o[2] in SUBJ]   # process 邻居
        if len(priv) >= 2 and not sh and not pn:
            cnt_a += 1
        elif len(priv) >= 2 and (sh or pn):
            cnt_b += 1
        elif len(priv) < 2 and len(objs) >= 2:
            cnt_c += 1
        else:
            cnt_iso += 1
        if len(priv) >= 2:
            proc_with_priv += 1
            n_priv_total += len(priv)
        max_priv = max(max_priv, len(priv))
    print(f"[e7:{name}] CPR {gc.n_nodes()}n/{len(gc.edges)}e | "
          f"process={n_proc} 私有对象={n_priv} 共享对象={n_shared}")
    print(f"  形态a(纯私有>=2): {cnt_a} | "
          f"形态b(私有>=2+共享): {cnt_b} | "
          f"形态c(几乎无私有): {cnt_c} | 无邻居: {cnt_iso}")
    print(f"  有>=2私有对象的进程 {proc_with_priv}/{n_proc} | "
          f"其私有对象总数 {n_priv_total} | 单进程最多私有 {max_priv}")
    # 假设折叠：吸收 进程+私有对象，共享保留 -> 节点上限
    # 每进程折 1+len(priv) 节点为 M+N(2) ；共享/process邻居 保留
    foldable = [(p, len([o for o in nbr.get(p, []) if o[1] == 1 and o[2] not in SUBJ]))
                for p in procs]
    foldable = [x for x in foldable if x[1] >= 2]
    absorbed = sum(1 + k for _, k in foldable)
    new_nodes = len(foldable) * 2
    n0 = g.n_nodes()
    est = (absorbed - new_nodes) / n0 if n0 else 0
    print(f"  可折叠进程 {len(foldable)} | 吸收节点 {absorbed} -> 新节点 {new_nodes} | "
          f"预估节点归约 {est:.1%} (基于原始图 {n0} 节点)")
    return locals()


def main():
    f = sorted(glob.glob("cache/darpa/e3cadets_w8_*.jsonl"))[0]
    wins = load_graphs(f)
    print(f"[e7] win0 原始: {wins[0].n_nodes()}n/{len(wins[0].edges)}e")
    analyze(wins[0], "win0")
    analyze(wins[1], "win1")


if __name__ == "__main__":
    main()
