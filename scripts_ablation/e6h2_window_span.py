# -*- coding: utf-8 -*-
"""e6h2_window_span.py — 每个测试分区的窗口时间跨度与到达率。

配合 e6g 的归约耗时，回答审稿问题：
  「窗口大小是多少？reduction 时间与窗口到达速率的关系？
    如果 reduction 比窗口间隔还长，如何做 window-level detection？」
输出：results/e6h2_window_span.csv / .json
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--specs", default="bin.116,bin.117,bin.118,"
                                     "bin.119@500000,bin.120,bin.6@400000,bin.7@400000")
    ap.add_argument("--cache", default="cache/darpa")
    ap.add_argument("--results", default="results/e6h2_window_span.csv")
    ap.add_argument("--summary", default="results/e6h2_window_span.json")
    a = ap.parse_args()

    from scripts.e5_f1_eval import parse_one
    from scripts.e6_alert_eval import parse_spec

    rows = []
    for spec in [s.strip() for s in a.specs.split(",") if s.strip()]:
        nm, mr = parse_spec(a.data_dir, spec)
        g, meta = parse_one(a.data_dir, nm, mr or None)
        ts = [e.get("ts", 0) for e in g.edges if e.get("ts")]
        if not ts:
            print(f"[e6h2] {spec}: no timestamps", flush=True)
            continue
        lo, hi = min(ts), max(ts)
        span_s = (hi - lo) / 1e9
        n_ev = g.n_edges()
        rows.append({
            "spec": spec, "n_nodes": g.n_nodes(), "n_edges": n_ev,
            "t_lo": lo, "t_hi": hi,
            "span_s": round(span_s, 1),
            "span_h": round(span_s / 3600.0, 3),
            "events_per_s": round(n_ev / span_s, 1) if span_s > 0 else None,
        })
        print(f"[e6h2] {spec}: span {span_s/3600:.3f} h "
              f"({span_s:.0f}s), {n_ev} edges, "
              f"{n_ev/span_s:.1f} edges/s", flush=True)

    if rows:
        with open(a.results, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        spans = [r["span_h"] for r in rows]
        summ = {
            "n_partitions": len(rows),
            "span_h_min": min(spans), "span_h_max": max(spans),
            "span_h_mean": sum(spans) / len(spans),
            "rows": rows,
        }
        with open(a.summary, "w", encoding="utf-8") as f:
            json.dump(summ, f, ensure_ascii=False, indent=2)
        print(f"[e6h2] mean span {summ['span_h_mean']:.3f} h "
              f"({summ['span_h_mean']*3600:.0f}s) -> {a.summary}", flush=True)


if __name__ == "__main__":
    main()
