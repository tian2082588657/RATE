# -*- coding: utf-8 -*-
"""临时 debug：ov 测试 INV4 失败原因。"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from rate_core import CanonicalGraph, star_graph
from reduction import template_mining as tmining
from reduction.tered import find_instances, TeRedOperator
from reduction.base import check_invariants

benign = [star_graph(f"b{i}", n_leaves=4, seed=i) for i in range(8)]
benign += [star_graph(f"b{i}", n_leaves=5, seed=i + 100) for i in range(6)]
tpls, info = tmining.mine_templates(benign, min_support=5, per_graph=10,
                                    max_tpl_nodes=12, seed=0, verbose=False)
print("templates:", len(tpls), "| sizes:", [(t.n_nodes(), t.n_edges(),
      t.meta.get("anchor"), t.nodes.get(t.meta.get("anchor", "0"))) for t in tpls])

G = CanonicalGraph("ov")
G.ensure_node("c1", ntype="t0")
G.ensure_node("c2", ntype="t0")
for i in range(4):
    G.ensure_node(f"l{i}", ntype="t1")
    G.add_edge("c1", f"l{i}", "e0")
    if i < 2:
        G.add_edge("c2", f"l{i}", "e0")
G.ensure_node("x", ntype="t1")
G.add_edge("c2", "x", "e0")
G.ensure_node("y", ntype="t1")
G.add_edge("c2", "y", "e0")
print("G:", G.n_nodes(), "nodes", len(G.edges), "edges")

regs = find_instances(G, tpls, max_instances=10)
for r in regs:
    print("region", r["tpl_id"], "n_nodes:", len(r["nodes"]), "nodes:", sorted(r["nodes"])[:12])

# 直接单步测 _seeded_match
from reduction.tered import _to_nx_matching, _MatchIndex, _pick_root, _seeded_match
idx = _MatchIndex(_to_nx_matching(G))
t = tpls[0]
Tnx = _to_nx_matching(t)
root = _pick_root(Tnx, t)
# 内联复制看中间态
succ_t = {n: set(Tnx.successors(n)) for n in Tnx.nodes}
pred_t = {n: set(Tnx.predecessors(n)) for n in Tnx.nodes}
order, seen, q = [], {root}, [root]
while q:
    n = q.pop(0)
    for nb in list(succ_t[n]) + list(pred_t[n]):
        if nb not in seen:
            seen.add(nb)
            q.append(nb)
            order.append(nb)
print("order:", order, "| succ_t:", dict(succ_t), "| pred_t:", dict(pred_t))
print("succ_g[c1]:", sorted(idx.succ["c1"]), "| pred_g[c1]:", sorted(idx.pred["c1"]))
removed = set()
for r in regs:
    removed |= r["nodes"]
print("removed:", len(removed), "kept:", G.n_nodes() - len(removed),
      "=> Gp nodes:", G.n_nodes() - len(removed) + 2 * len(regs))

op = TeRedOperator(tpls, max_instances=10)
res = op.reduce(G)
print("INV:", check_invariants(G, res))
