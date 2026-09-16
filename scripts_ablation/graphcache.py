# -*- coding: utf-8 -*-
"""scripts_ablation/graphcache.py — 规范图的磁盘解析缓存。

`parse_one` 每次都重新解 gz 并重建图（bin.116 全量含数百万事件，耗时数分钟）。
E1--E3 的批量实验里同一张图会被反复解析，因此加一层
(data_dir, file, max_records) -> (graph, meta) 的 pickle 缓存，
把解析成本从 O(运行次数) 降到 O(唯一图数)。

用法:
    from scripts_ablation.graphcache import parse_cached
    g, meta = parse_cached(a.data_dir, nm, a.max_records_train, ccache_dir)
"""
from __future__ import annotations
import os, hashlib

from rate_core import save_pickle, load_pickle
from scripts.e5_f1_eval import parse_one, resolve_name


def _key(data_dir, nm, max_records):
    real = resolve_name(data_dir, nm)
    raw = "%s|%s|%s" % (os.path.basename(os.path.normpath(data_dir)), real,
                        max_records)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]


def parse_cached(data_dir, nm, max_records, cache_dir, verbose=True):
    """带缓存的单图解析。max_records=None 表示全量。"""
    os.makedirs(cache_dir, exist_ok=True)
    cp = os.path.join(cache_dir, "%s.pkl" % _key(data_dir, nm, max_records))
    if os.path.exists(cp):
        if verbose:
            print("[gcache] 命中 %s (%s)" % (os.path.basename(cp), nm), flush=True)
        return load_pickle(cp)
    if verbose:
        print("[gcache] 解析 %s (max_records=%s)" % (nm, max_records), flush=True)
    g, meta = parse_one(data_dir, nm, max_records)
    save_pickle(cp, (g, meta))
    return g, meta
