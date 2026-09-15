# -*- coding: utf-8 -*-
"""scripts/e3_save_templates.py — 用最优配置挖 E3 cadets 模板并存缓存（服务器）。"""
from __future__ import annotations
import sys, os, glob, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rate_core import load_graphs, save_graphs
from reduction import template_mining as tmining


def main():
    ms = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    per = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    wins = load_graphs(glob.glob("cache/darpa/e3cadets_w8_*.jsonl")[0])
    train = wins[:4]
    t0 = time.time()
    templates, info = tmining.mine_templates(
        train, min_support=ms, per_graph=per, max_templates=300,
        min_tpl_nodes=3, seed=0, verbose=True)
    dst = "cache/darpa/e3cadets_templates_w8.jsonl"
    save_graphs(dst, templates)
    print(f"saved {len(templates)} templates -> {dst} "
          f"({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
