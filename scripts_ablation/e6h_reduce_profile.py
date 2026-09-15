# -*- coding: utf-8 -*-
"""e6h_reduce_profile.py — 归约成本画像：模板级耗时 + 索引化加速估计。

回答审稿意见「reduction 端到端极贵，但没给任何优化后的测量或估计」。

方法（单次运行即得全部信息）：
  1. 复刻 reduction.tered.find_instances 的主循环，逐模板计时，记录
     候选种子数 / 实际匹配尝试数 / 命中实例数。
  2. 单独计时 collapse 阶段。
  3. 静态估计「按 (root type, 度) 索引模板」的候选规模：
       naive   = sum_t |{v : type(v)=root_type(t), deg(v) >= deg_t(root)}|
       indexed = sum_v |{t : root_type(t)=type(v), deg_t(root) <= deg(v)}|
     两者之比 = 索引化可去掉的冗余工作量上界。
输出：results/e6h_reduce_profile.csv（模板级）+ results/e6h_reduce_summary.json
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--spec", default="bin.119@500000")
    ap.add_argument("--templates", default="")
    ap.add_argument("--cache", default="cache/darpa")
    ap.add_argument("--results", default="results/e6h_reduce_profile.csv")
    ap.add_argument("--summary", default="results/e6h_reduce_summary.json")
    ap.add_argument("--share-k", type=int, default=-1)
    ap.add_argument("--max-instances", type=int, default=300)
    ap.add_argument("--max-total", type=int, default=5000)
    a = ap.parse_args()

    from rate_core import load_graphs
    from scripts.e5_f1_eval import parse_one, find_templates
    from scripts.e6_alert_eval import parse_spec
    from reduction.tered import (
        _to_nx_matching, _MatchIndex, _shared_objects, _pick_root, _seeded_match,
        TeRedOperator,
    )

    tpl_path = a.templates or find_templates(
        os.path.join(a.cache, "e5_templates", "*.jsonl"))
    tpls = load_graphs(tpl_path)
    print(f"[e6h] templates: {len(tpls)} from {tpl_path}", flush=True)

    nm, mr = parse_spec(a.data_dir, a.spec)
    t0 = time.time()
    g, meta = parse_one(a.data_dir, nm, mr or None)
    t_parse = time.time() - t0
    print(f"[e6h] parsed {a.spec}: {g.n_nodes()} nodes {g.n_edges()} edges "
          f"in {t_parse:.1f}s", flush=True)

    # ---------- 1. 复刻 find_instances 主循环并逐模板计时 ----------
    t0 = time.time()
    Gnx = _to_nx_matching(g)
    idx = _MatchIndex(Gnx)
    t_index = time.time() - t0
    shared = _shared_objects(g, a.share_k) if a.share_k >= 0 else set()

    removed = set()
    budget = a.max_total
    rows = []
    seeds_considered = 0
    match_attempts = 0
    t_match_total = 0.0
    for ti, tpl in enumerate(tpls):
        Tnx = _to_nx_matching(tpl)
        if Tnx.number_of_nodes() < 2:
            continue
        root_t = _pick_root(Tnx, tpl)
        if root_t is None:
            continue
        rtype = Tnx.nodes[root_t].get("type", "unknown")
        r_in, r_out = Tnx.in_degree(root_t), Tnx.out_degree(root_t)
        seeds = idx.by_type.get(rtype, [])
        seeds = [s for s in seeds
                 if Gnx.in_degree(s) >= r_in and Gnx.out_degree(s) >= r_out]
        n_seeds = len(seeds)
        seeds_considered += n_seeds
        n_inst, n_att = 0, 0
        tb = time.time()
        for s in seeds:
            if budget <= 0 or n_inst >= a.max_instances:
                break
            if s in removed:
                continue
            n_att += 1
            m = _seeded_match(idx, Tnx, root_t, s)
            if m is None:
                continue
            gids = set(m.values())
            if gids & removed:
                continue
            absorb = {x for x in gids if x not in shared}
            if len(absorb) < 2:
                continue
            removed |= absorb
            n_inst += 1
            budget -= 1
        tm = time.time() - tb
        t_match_total += tm
        match_attempts += n_att
        rows.append({
            "tpl_idx": ti, "tpl_id": getattr(tpl, "gid", str(ti)),
            "n_tpl_nodes": Tnx.number_of_nodes(), "root_type": rtype,
            "root_in": r_in, "root_out": r_out,
            "n_seeds": n_seeds, "n_attempts": n_att, "n_instances": n_inst,
            "match_s": round(tm, 4),
        })
        if (ti + 1) % 20 == 0:
            print(f"  [e6h] {ti+1}/{len(tpls)} templates, "
                  f"{t_match_total:.0f}s elapsed in matching", flush=True)

    # ---------- 2. collapse 阶段单独计时 ----------
    op = TeRedOperator(tpls, max_instances=a.max_instances,
                       max_total=a.max_total, share_k=a.share_k)
    t0 = time.time()
    res = op.reduce(g)
    t_reduce_total = time.time() - t0

    # ---------- 3. 静态索引化估计 ----------
    deg = {}
    for nid in Gnx.nodes:
        deg[nid] = (Gnx.in_degree(nid), Gnx.out_degree(nid))
    # naive: 每个模板扫它的 root type 节点
    tpl_roots = []
    for tpl in tpls:
        Tnx = _to_nx_matching(tpl)
        if Tnx.number_of_nodes() < 2:
            continue
        rt = _pick_root(Tnx, tpl)
        if rt is None:
            continue
        tpl_roots.append((Tnx.nodes[rt].get("type", "unknown"),
                          Tnx.in_degree(rt), Tnx.out_degree(rt)))
    naive = 0
    for (rt, ri, ro) in tpl_roots:
        naive += sum(1 for v in idx.by_type.get(rt, [])
                     if deg[v][0] >= ri and deg[v][1] >= ro)
    # indexed: 每个节点只查「root type 匹配且度不超过自身」的模板
    tpl_by_rtype = defaultdict(list)
    for (rt, ri, ro) in tpl_roots:
        tpl_by_rtype[rt].append((ri, ro))
    indexed = 0
    for v, (vi, vo) in deg.items():
        vt = Gnx.nodes[v].get("type", "unknown")
        for (ri, ro) in tpl_by_rtype.get(vt, ()):
            if ri <= vi and ro <= vo:
                indexed += 1

    summary = {
        "spec": a.spec,
        "n_nodes": g.n_nodes(), "n_edges": g.n_edges(),
        "n_templates": len(tpls),
        "parse_s": round(t_parse, 2),
        "index_build_s": round(t_index, 2),
        "match_total_s": round(t_match_total, 2),
        "reduce_total_s": round(t_reduce_total, 2),
        "collapse_s": round(t_reduce_total - t_match_total, 2),
        "seeds_considered": seeds_considered,
        "match_attempts": match_attempts,
        "regions": res.stats.get("n_regions"),
        "nodes_out": res.Gp.n_nodes(),
        "naive_seed_work": naive,
        "indexed_seed_work": indexed,
        "index_gain": round(naive / indexed, 2) if indexed else None,
    }
    with open(a.summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(a.results, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("[e6h] summary:", json.dumps(summary, ensure_ascii=False), flush=True)
    print(f"[e6h] done -> {a.results}", flush=True)


if __name__ == "__main__":
    main()
