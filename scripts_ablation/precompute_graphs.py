# -*- coding: utf-8 -*-
"""scripts_ablation/precompute_graphs.py — 串行预解析并缓存规范图。

背景：`parse_one` 无缓存，且 E1--E3 会反复用到同一批图。并发启动时多个进程会
同时 miss 并各自解析同一份 gz（I/O 竞争 + 重复劳动）。本脚本串行地把需要的图
解析一次写入 graphcache，之后所有实验都命中缓存。

用法:
  python scripts_ablation/precompute_graphs.py \
      --data-dir /TeRed+RATE/dataset/darpae5/cadets \
      --specs "bin.116,bin.117,bin.118,bin.119@500000,bin.120,bin.6@400000,bin.7@400000" \
      --train "bin.1,bin.2,bin.3,bin.4,bin.5" --train-records 400000 \
      --graph-cache /TeRed+RATE/code/cache/darpa/graphcache
"""
from __future__ import annotations
import os, sys, time, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from scripts_ablation.graphcache import parse_cached


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/TeRed+RATE/dataset/darpae5/cadets")
    ap.add_argument("--specs", default="bin.116,bin.117,bin.118,bin.119@500000,"
                                      "bin.120,bin.6@400000,bin.7@400000")
    ap.add_argument("--train", default="bin.1,bin.2,bin.3,bin.4,bin.5")
    ap.add_argument("--train-records", type=int, default=400000)
    ap.add_argument("--graph-cache", default="/TeRed+RATE/code/cache/darpa/graphcache")
    a = ap.parse_args()

    os.makedirs(a.graph_cache, exist_ok=True)
    jobs = []
    for s in a.specs.split(","):
        s = s.strip()
        if not s:
            continue
        nm, _, mr = s.partition("@")
        jobs.append((nm, int(mr) if mr else None))
    for nm in a.train.split(","):
        nm = nm.strip()
        if nm:
            jobs.append((nm, a.train_records))

    t0 = time.time()
    for i, (nm, mr) in enumerate(jobs, 1):
        t = time.time()
        g, _meta = parse_cached(a.data_dir, nm, mr, a.graph_cache)
        print("[pre] %2d/%d %-22s max_records=%-8s -> %7d 节点 / %8d 边 (%.0fs)"
              % (i, len(jobs), nm, mr, g.n_nodes(), g.n_edges(), time.time() - t),
              flush=True)
        del g
    print("[pre] DONE (%.0fs)" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
