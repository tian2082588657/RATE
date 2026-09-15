# -*- coding: utf-8 -*-
"""scripts/scan_clean_v2.py — parse 口径纯良性候选预筛（与评测同口径）。

旧 gt_scan_e5.py 用 fastavro 直扫原始记录 uuid 字段，与评测
(darpa_tc.parse 建图 + 按节点 UUID 匹配) 口径不一致，会把攻击文件
误判为纯良性（bin.6 旧扫 0 命中、评测命中 6）。本脚本用评测同口径
重新验证候选文件：parse(全量) -> label_graph(当前 124-UUID GT)。

用法:
  python scripts/scan_clean_v2.py bin.20,bin.40,bin.52 --out results/scan_clean_v2.json
"""
from __future__ import annotations
import os, sys, json, time, argparse
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from adapters import darpa_tc
from scripts.e5_f1_eval import load_gt, label_graph, resolve_name

GT = {}
DATA_DIR = ""


def scan_one(nm):
    t0 = time.time()
    try:
        full = resolve_name(DATA_DIR, nm)
        p = os.path.join(DATA_DIR, full)
        gs, meta = darpa_tc.parse(p, max_records=None, verbose=False)
        if not gs:
            return {"file": nm, "error": "empty graph"}
        g = gs[0]
        hit, _ = label_graph(g, GT)
        return {"file": nm, "full": full, "records": meta.get("records"),
                "n_nodes": g.n_nodes(), "n_edges": g.n_edges(),
                "n_hit": hit, "runtime_s": round(time.time() - t0, 1)}
    except Exception as e:
        return {"file": nm, "error": str(e),
                "runtime_s": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/TeRed+RATE/dataset/darpae5/cadets")
    ap.add_argument("--gt", default="/TeRed+RATE/code/ground_truth")
    ap.add_argument("--files", required=True, help="逗号分隔候选短名")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    global GT, DATA_DIR
    GT = load_gt(a.gt)
    DATA_DIR = a.data_dir
    print(f"[scan2] GT: {len(GT)} UUIDs", flush=True)

    nms = [s.strip() for s in a.files.split(",") if s.strip()]
    t0 = time.time()
    with Pool(a.workers) as pool:
        rows = pool.map(scan_one, nms)
    for r in rows:
        if "error" in r:
            print(f"[scan2] {r['file']}: ERROR {r['error']} "
                  f"({r['runtime_s']}s)", flush=True)
        else:
            print(f"[scan2] {r['file']}: records={r['records']} "
                  f"nodes={r['n_nodes']} edges={r['n_edges']} "
                  f"GT命中={r['n_hit']} ({r['runtime_s']}s)", flush=True)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    clean = [r for r in rows if r.get("n_hit") == 0]
    print(f"\n[scan2] 完成 ({time.time()-t0:.0f}s): {len(clean)}/{len(rows)} "
          f"纯良性候选 (GT命中=0)", flush=True)


if __name__ == "__main__":
    main()
