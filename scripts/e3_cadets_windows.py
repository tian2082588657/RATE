# -*- coding: utf-8 -*-
"""scripts/e3_cadets_windows.py — DARPA E3 cadets 时间窗切分（M2 go/no-go 雏形）。

背景：E3 cadets 是长时间系统调用流（缓存单图 ~15.7 万边 / 6500s），TeRed 官方
模板来自程序单元测试，这里按 RATE 指南 §10 R2：把流按时间窗切成 N 张连续子图，
早期窗 = 良性行为（挖模板），晚期窗 = 验证 tered 归约率（攻击通常在尾部）。

用法（服务器）:
  python scripts/e3_cadets_windows.py 8   # 切成 8 窗，输出到 cache/darpa/e3cadets_w8.jsonl

输出：每窗一张 CanonicalGraph（节点带 type/attrs，边保留 ts/etype），meta 记录
窗口起止 ts、边数。节点按所在窗口重建（跨窗同一 uuid 会重复出现——这正模拟了
"同一程序在多个窗口重复运行"，是模板挖掘想要的结构）。
"""
from __future__ import annotations
import os, sys, glob, collections

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rate_core import CanonicalGraph, load_graphs, save_graphs


def split_windows(g, n_win=8, by="time"):
    """把一张大图按 ts 均匀切成 n_win 个窗口子图。"""
    es = sorted(g.edges, key=lambda e: e["ts"])
    if not es:
        return []
    out = []
    if by == "time":
        # 按时间均匀：每个窗口一个等宽 ts 桶
        lo, hi = es[0]["ts"], es[-1]["ts"]
        span = max(1, (hi - lo) / n_win)
        buckets = collections.defaultdict(list)
        for e in es:
            b = min(n_win - 1, int((e["ts"] - lo) / span))
            buckets[b].append(e)
        idxs = [buckets[i] for i in range(n_win)]
    else:  # 按边数均匀
        per = max(1, len(es) // n_win)
        idxs = [es[i * per:(i + 1) * per] for i in range(n_win)]
    for i, seg in enumerate(idxs):
        sg = CanonicalGraph(f"{g.gid}:win{i}")
        sg.meta = dict(g.meta)
        sg.meta["window"] = i
        if seg:
            sg.meta["ts_lo"], sg.meta["ts_hi"] = seg[0]["ts"], seg[-1]["ts"]
        seen = set()
        for e in seg:
            seen.add(e["src"]); seen.add(e["dst"])
            sg.edges.append(dict(e))
        for nid in seen:
            sg.nodes[nid] = dict(g.nodes.get(nid, {"type": "unknown", "attrs": {}}))
        sg.edges.sort(key=lambda e: e["ts"])
        out.append(sg)
    return out


def main():
    n_win = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    src = sys.argv[2] if len(sys.argv) > 2 else \
        glob.glob("cache/darpa/parsed/*.jsonl")[0]
    graphs = load_graphs(src)
    g = graphs[0]
    wins = split_windows(g, n_win=n_win, by="time")
    dst = os.path.join("cache", "darpa",
                       f"e3cadets_w{n_win}_{os.path.basename(src)}")
    save_graphs(dst, wins)
    print(f"[windows] {os.path.basename(src)} -> {len(wins)} 窗 -> {dst}")
    for w in wins:
        tc = collections.Counter(nd.get("type", "?") for nd in w.nodes.values())
        n0 = sum(1 for e in w.edges if e.get("mu", 1) == 1)
        print(f"  win{w.meta.get('window')}: nodes={w.n_nodes():>6} "
              f"edges={len(w.edges):>7} ts跨度={((w.meta.get('ts_hi',0)-w.meta.get('ts_lo',0))/1e9):>6.0f}s "
              f"proc={tc.get('process',0):>4} types={dict(tc.most_common(5))}")


if __name__ == "__main__":
    main()
