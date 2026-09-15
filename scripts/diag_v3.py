# -*- coding: utf-8 -*-
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), *[".."] * 1))

"""scripts/diag_v3.py — 新挖掘器在 StreamSpot 上的环节损耗分析。"""
import os, sys, glob, time
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from rate_core import load_graphs
from reduction import template_mining as tmining
from reduction.tered import _to_nx_matching, find_instances
from adapters.streamspot import parse
import run_pipeline as rp

def main():
    # 读解析缓存（无则解析）
    cdir = os.path.join(ROOT, "cache", "streamspot", "parsed")
    fs = sorted(glob.glob(os.path.join(cdir, "*.jsonl")))
    graphs = load_graphs(fs[0]) if fs else None
    if graphs is None:
        graphs, _ = parse(_os.path.join(_os.path.dirname(_R), "dataset", "streamspot", "all.tar.gz"),
                          gids=list(range(7)) + [500, 501])
    by = {int(g.gid): g for g in graphs}
    gids = [0, 1, 2, 3, 4, 5, 6]
    benign = [by[g] for g in gids]
    print("良性图:", gids, "规模:", [(g.gid, g.n_nodes(), len(g.edges)) for g in benign])

    t0 = time.time()
    sem = tmining.auto_semantic_types(benign, rare_frac=0.02, min_count=4)
    print("语义 type(自动探测):", sorted(sem))
    # 每图语义节点数与各 type 计数
    from collections import Counter
    for g in benign:
        c = Counter(nd.get("type", "?") for nd in g.nodes.values())
        print(f"  gid={g.gid} types={dict(c.most_common())}")
    print(f"[diag] auto_semantic 耗时 {time.time()-t0:.2f}s")

    # 候选统计
    t0 = time.time()
    cands = tmining.sample_candidates(benign, khop=1, per_graph=400,
                                      max_tpl_nodes=48, seed=0,
                                      khop_rare=2, verbose=True)
    print(f"[diag] 候选 {len(cands)}, 耗时 {time.time()-t0:.2f}s")
    from collections import Counter as C
    nn = C(min(c.n_nodes(), 20) for c in cands)
    tt = C(c.meta.get("anchor_type", "?") for c in cands)
    # 补 anchor_type
    tt = C()
    for c in cands:
        tt[c.nodes[c.meta.get("anchor")].get("type", "?")] += 1
    print("候选 n_nodes 分布(截20):", dict(sorted(nn.items())))
    print("候选锚 type 分布:", dict(tt))

    # WL 桶 + 去重叠支持度
    import collections
    buckets = collections.defaultdict(list)
    for c in cands:
        buckets[tmining.wl_signature(_to_nx_matching(c))].append(c)
    print(f"\nWL 桶数={len(buckets)}")
    freq = []
    for sig, subs in buckets.items():
        subs.sort(key=lambda s: (-s.n_nodes(), -s.n_edges()))
        occ, chosen = set(), []
        for s in subs:
            keys = {(s.meta.get("origin"), n) for n in s.nodes}
            if keys & occ:
                continue
            occ |= keys
            chosen.append(s)
        if len(chosen) >= 3:
            freq.append((len(chosen), chosen[0]))
    freq.sort(key=lambda x: -x[0])
    print(f"去重叠后 support>=3 的桶: {len(freq)}")
    for sup, rep in freq[:15]:
        print(f"  support={sup}  nodes={rep.n_nodes()}  edges={len(rep.edges)}  "
              f"anchor_type={rep.nodes.get(rep.meta.get('anchor')).get('type')}")

if __name__ == "__main__":
    main()
