# -*- coding: utf-8 -*-
"""scripts_ablation/e3_probe_strength.py — 标定"归约强度 <-> 归约率"的映射。

E3 主图需要一条可控的横轴（归约率）。可用的强度旋钮：
  max_total      全图区域总预算（越小 -> 归约越弱）
  max_instances  每模板实例数上限
  tpl_topk       只用前 k 个模板（模板越多 -> 命中越多 -> 归约越强）
本脚本只做归约、不做特征化与检测，用于快速找扫描点、看可达的归约率上限。

用法:
  python scripts_ablation/e3_probe_strength.py --test-files bin.116 \
      --max-total 200,500,1000,3000,5000,20000 --results results/e3_probe.csv
"""
from __future__ import annotations
import os, sys, time, argparse
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from rate_core import load_graphs
from reduction.tered import TeRedOperator
from eval.metrics import append_csv
from scripts.e5_f1_eval import parse_one, find_templates
from scripts.e6_alert_eval import parse_spec
from scripts_ablation.graphcache import parse_cached


def ints(s):
    return [int(x) for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/TeRed+RATE/dataset/darpae5/cadets")
    ap.add_argument("--test-files", default="bin.116")
    ap.add_argument("--templates", default=None)
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--share-k", type=int, default=-1)
    ap.add_argument("--max-instances", default="300")
    ap.add_argument("--max-total", default="200,500,1000,3000,5000,20000,100000")
    ap.add_argument("--tpl-topk", default="0", help="0 = 全部模板")
    ap.add_argument("--graph-cache", default="cache/darpa/graphcache")
    ap.add_argument("--results", default="results/e3_probe.csv")
    a = ap.parse_args()

    tpl_path = a.templates or find_templates(
        os.path.join(a.cache, "darpa", "e5_templates", "*.jsonl"))
    tpls_all = load_graphs(tpl_path)
    print("[probe] 模板 %s -> %d" % (tpl_path, len(tpls_all)), flush=True)

    mis = ints(a.max_instances)
    mts = ints(a.max_total)
    kks = ints(a.tpl_topk)

    specs = [s.strip() for s in a.test_files.split(",") if s.strip()]
    for spec in specs:
        nm, mr = parse_spec(a.data_dir, spec)
        g, _m = parse_cached(a.data_dir, nm, mr or None, a.graph_cache)
        n0, e0 = g.n_nodes(), g.n_edges()
        print("\n[probe] ===== %s: %d 节点 / %d 边 =====" % (spec, n0, e0),
              flush=True)
        for kk in kks:
            tpls = tpls_all if kk <= 0 else tpls_all[:kk]
            for mt in mts:
                for mi in mis:
                    t = time.time()
                    op = TeRedOperator(tpls, max_instances=mi, max_total=mt,
                                       share_k=a.share_k)
                    res = op.reduce(g)
                    st = res.stats
                    n1, e1 = res.Gp.n_nodes(), res.Gp.n_edges()
                    row = dict(
                        dataset="e5_cadets", file=spec, n_tpl=len(tpls),
                        tpl_topk=kk, max_instances=mi, max_total=mt,
                        share_k=a.share_k,
                        n_nodes_orig=n0, n_edges_orig=e0,
                        n_nodes_red=n1, n_edges_red=e1,
                        node_cut=round(1 - n1 / n0, 4) if n0 else 0.0,
                        edge_cut=round(1 - e1 / e0, 4) if e0 else 0.0,
                        n_regions=st.get("n_regions", 0),
                        runtime_s=round(time.time() - t, 1))
                    append_csv(a.results, row)
                    print("  tpl=%4d mi=%5d mt=%7d -> 节点 %6d(-%.1f%%) "
                          "边 %7d(-%.1f%%) 区域=%5d (%.0fs)"
                          % (len(tpls), mi, mt, n1, 100 * row["node_cut"],
                             e1, 100 * row["edge_cut"], row["n_regions"],
                             row["runtime_s"]), flush=True)
        del g
    print("\n[probe] 完成 -> %s" % a.results, flush=True)


if __name__ == "__main__":
    main()
