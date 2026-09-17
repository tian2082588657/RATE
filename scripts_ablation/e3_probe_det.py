# -*- coding: utf-8 -*-
"""e3_probe_det.py — 验证 TeRedOperator 的实例匹配是否跨进程可复现。

背景：E3 强度扫描里归约量对 max_total 非单调（节点删除量与边删除量都不单调），
与 "budget 越大匹配实例越多" 的直觉矛盾。怀疑 _seeded_match 的候选集是 set，
迭代顺序随 PYTHONHASHSEED 变化，同一输入在不同进程得到不同贪婪轨迹。

本脚本只做一次测试图的 find_instances（跳过 5 个训练图、跳过打分），
打印区域数、被吸收节点数与区域集合摘要，便于跨进程比对。

用法:
  PYTHONHASHSEED=0     python scripts_ablation/e3_probe_det.py ...
  PYTHONHASHSEED=12345 python scripts_ablation/e3_probe_det.py ...
"""
from __future__ import annotations
import os, sys, time, argparse, hashlib

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from rate_core import load_graphs
from reduction.tered import find_instances
from scripts.e6_alert_eval import parse_spec
from scripts_ablation.graphcache import parse_cached


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/TeRed+RATE/dataset/darpae5/cadets")
    ap.add_argument("--templates", required=True)
    ap.add_argument("--graph-cache", required=True)
    ap.add_argument("--spec", default="bin.7@400000")
    ap.add_argument("--max-total", type=int, default=4000)
    ap.add_argument("--max-instances", type=int, default=300)
    ap.add_argument("--share-k", type=int, default=-1)
    a = ap.parse_args()

    print("PYTHONHASHSEED=%r  max_total=%d" % (os.environ.get("PYTHONHASHSEED"),
                                               a.max_total), flush=True)
    tpls = load_graphs(a.templates)
    print("templates: %d" % len(tpls), flush=True)

    nm, mr = parse_spec(a.data_dir, a.spec)
    g, _meta = parse_cached(a.data_dir, nm, mr or None, a.graph_cache)
    print("graph %s: %d nodes, %d edges" % (a.spec, g.n_nodes(), g.n_edges()),
          flush=True)

    t = time.time()
    regions = find_instances(g, tpls, max_instances=a.max_instances,
                             max_total=a.max_total, share_k=a.share_k)
    absorbed = set()
    for r in regions:
        absorbed |= r["absorb"]
    region_ids = sorted(tuple(sorted(r["absorb"])) for r in regions)
    digest = hashlib.md5(repr(region_ids).encode()).hexdigest()

    print("n_regions=%d  absorbed=%d  digest=%s  (%.0fs)"
          % (len(regions), len(absorbed), digest, time.time() - t), flush=True)

    # 前 5 个区域的吸收集，便于肉眼比对
    for i, r in enumerate(regions[:5]):
        print("  region[%d] tpl=%s absorb=%d" % (i, r["tpl_id"], len(r["absorb"])),
              flush=True)


if __name__ == "__main__":
    main()
