# -*- coding: utf-8 -*-
"""reduction/tered.py — TeRed 式区域塌缩归约（合并指南 §2.4 / §4.1）。

流程（对齐指南）：
  1. 在 G 上以模板库（良性频繁子图，见 template_mining.py）做 VF2 匹配（节点 type 约束）；
  2. 每个命中区域 R 塌缩为 入口 M + 出口 N + 汇总边 M→N：
       - μ(M→N) = |R 内被折叠的原始边数|（边质量数边）
       - 外部前驱边 (u→r∈R) 重连为 (u→M)，1:1 保边数
       - 外部后继边 (r∈R→v) 重连为 (N→v)，1:1 保边数
  3. 全程输出 node_map π 与 edge_map ρ（μ 的唯一来源）。
  4. 重叠实例：贪心——已被吸收的节点不再参与后续匹配。

v4（2026-09-03，diag_e6 修复）：**共享感知吸收**。病根：DARPA/E3 窗口里大量
"同程序多实例"是兄弟进程，共享系统库 file/父进程节点。原版把命中区域全部节点
标记 removed —— 第一个实例吸收共享对象后，后续实例无法再经它匹配 -> 区域数被
压到 ~20（理论上限 36~45+）。v4 规则：
  * find_instances 里共享对象（被 >= share_k 个进程引用的对象）**不 removed**，
    允许出现在多个区域的匹配中（不同区域经它连接是合法的"共享上下文"）；
  * 每个 region 记录 nodes(完整匹配) 与 absorb(进程锚+私有对象=真正塌缩集)；
  * TeRedOperator 折叠只用 absorb 塌缩；共享对象保留为外部节点，其边 1:1 重连。
"""
from __future__ import annotations
import copy
from collections import defaultdict

import networkx as nx
from networkx.algorithms import isomorphism as nxiso

from rate_core import CanonicalGraph
from reduction.base import ReductionOperator, ReductionResult


def _to_nx_matching(G: CanonicalGraph):
    """用于匹配的 DiGraph：节点带 type，边仅结构（与原版 TeRed 一致）。"""
    dg = nx.DiGraph()
    for nid, nd in G.nodes.items():
        dg.add_node(nid, type=nd.get("type", "unknown"))
    for e in G.edges:
        if not dg.has_edge(e["src"], e["dst"]):
            dg.add_edge(e["src"], e["dst"])
    return dg


def _node_match(a, b):
    return a.get("type") == b.get("type")


# ---------------- 匹配索引（每图一次） ----------------
class _MatchIndex:
    """预计算的匹配索引：type->节点表、出/入邻接 set。供 seed-and-grow 匹配用。"""

    def __init__(self, Gnx):
        self.Gnx = Gnx
        self.by_type = defaultdict(list)
        self.succ = {n: set(Gnx.successors(n)) for n in Gnx.nodes}
        self.pred = {n: set(Gnx.predecessors(n)) for n in Gnx.nodes}
        for n in Gnx.nodes:
            self.by_type[Gnx.nodes[n].get("type", "unknown")].append(n)


def _pick_root(Tnx, tpl):
    """模板根：meta.anchor 优先；否则取模板内 (入+出) 度最大的节点（种子判别力最强）。"""
    anchor = str(tpl.meta.get("anchor", "")) if tpl.meta else ""
    if anchor in Tnx.nodes:
        return anchor
    if Tnx.number_of_nodes() == 0:
        return None
    best = None
    for n in Tnx.nodes:
        d = Tnx.in_degree(n) + Tnx.out_degree(n)
        if best is None or d > best[1]:
            best = (n, d)
    return best[0]


def _seeded_match(idx: _MatchIndex, Tnx, root_t, seed_g, max_backtracks=4000):
    """在 idx 图上做一次种子生长匹配（root_t -> seed_g 固定为模板根）。返回 {t: g} 或 None。

    候选种子由外部给出；这里从 root 已映射出发，按 BFS 层序扩展模板节点，
    用"已映射邻居的出/入邻接交集"收紧候选，有向边逐条验证，DFS 回溯直到
    完整映射或穷尽。
    """
    Gnx, succ_g, pred_g = idx.Gnx, idx.succ, idx.pred
    succ_t = {n: set(Tnx.successors(n)) for n in Tnx.nodes}
    pred_t = {n: set(Tnx.predecessors(n)) for n in Tnx.nodes}
    # BFS 层序：从 root 出发按模板邻接分层
    order, seen, q = [], {root_t}, [root_t]
    while q:
        n = q.pop(0)
        # 必须 sorted：succ_t/pred_t 是集合，哈希随机化下迭代顺序随进程变化，
        # BFS 层序随之变化 -> _seeded_match 返回不同合法嵌入 -> 归约不可复现。
        # （与下方 cand 的 sorted 同为可复现性硬要求，勿改回 list()。）
        for nb in sorted(succ_t[n]) + sorted(pred_t[n]):
            if nb not in seen:
                seen.add(nb)
                order.append(nb)
                q.append(nb)
    mapping = {root_t: seed_g}
    used = {seed_g}
    attempt = [0]

    def bt(idx):
        if idx >= len(order):
            return True
        t = order[idx]
        cand = None
        for tnb in succ_t[t]:              # 模板 t -> tnb
            if tnb in mapping:
                c = pred_g.get(mapping[tnb], set())
                cand = c if cand is None else cand & c
        for tnb in pred_t[t]:              # 模板 tnb -> t
            if tnb in mapping:
                c = succ_g.get(mapping[tnb], set())
                cand = c if cand is None else cand & c
        if cand is None:
            return False
        typ = Tnx.nodes[t].get("type", "unknown")
        # 必须 sorted：cand 是 set，字符串哈希随机化后迭代顺序随进程变化，
        # DFS 会返回不同的合法嵌入 -> 归约结果不可复现（E3 扫描非单调的根因）。
        for g in sorted(cand):
            if g in used:
                continue
            if Gnx.nodes[g].get("type", "unknown") != typ:
                continue
            attempt[0] += 1
            if attempt[0] > max_backtracks:
                return False
            mapping[t] = g
            used.add(g)
            if bt(idx + 1):
                return True
            del mapping[t]
            used.discard(g)
        return False

    ok = bt(0)
    return dict(mapping) if ok else None


def _shared_objects(G, share_k=2):
    """被 >= share_k 个不同进程/主体引用的节点 -> 视为"共享上下文"。

    共享上下文(系统库 file、公共 socket、被大量子进程 fork 的父进程)不是
    "某一次运行"的专属产物，吸收它会挡掉后续所有经它的实例。v4：这些节点
    不吸收、保留为外部节点。注意：主体节点(process)若被多个主体引用(父进程
    hub)，同样不吸收 —— 否则第一个子实例吸收父节点后其余全部被挡。
    """
    act = {n for n, nd in G.nodes.items()
           if nd.get("type") in ("process", "subject")}
    ref = defaultdict(set)
    for e in G.edges:
        if e["src"] in act:
            ref[e["dst"]].add(e["src"])
        if e["dst"] in act:
            ref[e["src"]].add(e["dst"])
    return {n for n, s in ref.items() if len(s) >= share_k}


def find_instances(G: CanonicalGraph, templates, max_instances=100,
                   max_total=1000, verbose=False, max_backtracks=4000,
                   seed_degree_loosen=0, share_k=2, min_absorb=2):
    """贪心、非重叠地找模板实例（seed-and-grow，避免全图 VF2 穷举）。

    v4（share_k>=0）：实例的"吸收集"= 命中节点中 进程锚+私有对象（共享上下文
    不 removed，可被多实例复用）。region dict:
      {"tpl","tpl_id","nodes": 完整匹配节点, "absorb": 可塌缩节点}。
    min_absorb=2：吸收集 < 2 个节点的区域折叠无收益(还倒贴 M/N)，跳过。
    """
    Gnx = _to_nx_matching(G)
    idx = _MatchIndex(Gnx)
    shared = _shared_objects(G, share_k) if share_k >= 0 else set()
    removed = set()
    regions = []
    budget = max_total
    for tpl in templates:
        if budget <= 0:
            break
        Tnx = _to_nx_matching(tpl)
        if Tnx.number_of_nodes() < 2:
            continue
        root_t = _pick_root(Tnx, tpl)
        if root_t is None:
            continue
        rtype = Tnx.nodes[root_t].get("type", "unknown")
        r_in, r_out = Tnx.in_degree(root_t), Tnx.out_degree(root_t)
        seeds = idx.by_type.get(rtype, [])
        if seed_degree_loosen >= 0 and r_in + r_out > 0:
            seeds = [s for s in seeds
                     if Gnx.in_degree(s) >= r_in - seed_degree_loosen
                     and Gnx.out_degree(s) >= r_out - seed_degree_loosen]
        n_inst = 0
        for s in seeds:
            if budget <= 0 or n_inst >= max_instances:
                break
            if s in removed:
                continue
            m = _seeded_match(idx, Tnx, root_t, s, max_backtracks=max_backtracks)
            if m is None:
                continue
            gids = set(m.values())
            if gids & removed:
                continue
            # v4: 吸收集 = 命中节点中非共享者（共享上下文全保留，含父进程 hub）
            absorb = {x for x in gids if x not in shared}
            # 防退化：共享对象全部保留时 region 无塌缩意义 -> 至少吸收 >= min_absorb
            if len(absorb) < min_absorb:
                continue
            removed |= absorb
            regions.append({"tpl": tpl, "tpl_id": tpl.gid, "nodes": gids,
                            "absorb": absorb})
            n_inst += 1
            budget -= 1
        if verbose:
            print(f"[tered] 模板 {tpl.gid} 命中 {n_inst} 个实例")
    return regions


class TeRedOperator(ReductionOperator):
    """TeRed 主算子。templates: list[CanonicalGraph]（冻结的良性模板库）。"""
    name = "tered"

    def __init__(self, templates=None, max_instances=100, max_total=1000,
                 summary_etype="__summary__", **cfg):
        super().__init__(**cfg)
        self.templates = templates or []
        self.max_instances = int(max_instances)
        self.max_total = int(max_total)
        self.summary_etype = summary_etype

    def _reduce(self, G):
        regions = find_instances(G, self.templates,
                                 max_instances=self.max_instances,
                                 max_total=self.max_total,
                                 share_k=int(self.cfg.get("share_k", 2)),
                                 verbose=self.cfg.get("verbose", False))
        if not regions:
            # 无命中：返回恒等结果（保持算子链路可用）
            Gp = CanonicalGraph(G.gid + ":tered")
            for nid, nd in G.nodes.items():
                Gp.nodes[nid] = copy.deepcopy(nd)
            Gp.labels = dict(G.labels)
            edge_map = {}
            for i, e in enumerate(G.edges):
                Gp.edges.append(dict(e))
                edge_map[len(Gp.edges) - 1] = [i]
            nm = {n: n for n in G.nodes}
            return ReductionResult(Gp, nm, edge_map, {
                "nodes_before": G.n_nodes(), "edges_before": G.n_edges(),
                "n_regions": 0, "reduction_ratio": 0.0,
            }, self.name)

        # ---- 吸收集 = 真正塌缩的节点（共享对象保留为外部节点）----
        removed = set()
        for r in regions:
            removed |= r["absorb"]
        kept = set(G.nodes) - removed

        # ---- 每区域角色 ----
        region_of = {}
        for ridx, r in enumerate(regions):
            for nd in r["absorb"]:
                region_of[nd] = ridx
        out_external = defaultdict(set)     # absorb 节点 -> 外部目标
        for e in G.edges:
            if e["src"] in removed and e["dst"] not in removed:
                out_external[e["src"]].add(e["dst"])

        # ---- 构造归约图节点 ----
        Gp = CanonicalGraph(G.gid + ":tered")
        # 按原图顺序插入（而非遍历 kept 集合）：Gp 节点的插入顺序决定后续
        # 词典/序列化字节序，虽不影响语义，但会让两次同参运行产出不同 pkl。
        for nid in G.nodes:
            if nid not in kept:
                continue
            Gp.nodes[nid] = copy.deepcopy(G.nodes[nid])
            if nid in G.labels:
                Gp.labels[nid] = G.labels[nid]
        # ---- 按需创建 M/N（防孤儿）----
        # 先扫描每条边, 按 region 分类: 内部(internal) / 入边(src 不在任何 absorb) /
        # 出边(dst 不在任何 absorb)。need_M: 有入边或有内部边; need_N: 有出边或有内部边。
        internal_idxs = defaultdict(list)
        has_in = defaultdict(bool)
        has_out = defaultdict(bool)
        for i, e in enumerate(G.edges):
            s, t = e["src"], e["dst"]
            rs, rt = region_of.get(s), region_of.get(t)
            if rs is not None and rs == rt:
                internal_idxs[rs].append(i)          # 区域内部边 -> 汇总边 μ
            else:
                if rs is not None:
                    has_out[rs] = True
                if rt is not None:
                    has_in[rt] = True
        region_node_ids = {}
        for ridx, r in enumerate(regions):
            need_m = has_in[ridx] or bool(internal_idxs[ridx])
            need_n = has_out[ridx] or bool(internal_idxs[ridx])
            if not (need_m or need_n):       # 该区域 absorb 无边：跳过折叠
                continue
            m_id = f"teredM{ridx}"
            n_id = f"teredN{ridx}"
            region_node_ids[ridx] = (m_id if need_m else None,
                                     n_id if need_n else None)
            tpl = r["tpl"]
            members = sorted(r["absorb"])
            att = {"role": "entry", "template": r["tpl_id"], "members": members}
            if need_m:
                Gp.nodes[m_id] = {"type": tpl.nodes.get(tpl.meta.get("anchor", "0"),
                                                        tpl.nodes.get("0", {"type": "summary"}))
                                       .get("type", "summary"),
                                  "attrs": att}
                if any(G.labels.get(x, 0) == 1 for x in members):
                    Gp.labels[m_id] = 1
            if need_n:
                Gp.nodes[n_id] = {"type": "summary_node",
                                  "attrs": dict(att, role="exit")}
                if any(G.labels.get(x, 0) == 1 for x in members):
                    Gp.labels[n_id] = 1
        # ---- node_map（映射到实际创建的 M/N；无外部出边->M，有出边->N）----
        node_map = {}
        for ridx, r in enumerate(regions):
            m_id, n_id = region_node_ids.get(ridx, (None, None))
            for nd in sorted(r["absorb"]):
                if out_external.get(nd):
                    node_map[nd] = n_id if n_id else m_id
                else:
                    node_map[nd] = m_id if m_id else n_id
                if node_map[nd] is None:            # absorb 无边区域：保持原节点
                    node_map[nd] = nd
        for nid in G.nodes:
            node_map.setdefault(nid, nid)
        # ---- 重连边（internal 已在第一遍扫描收集进 internal_idxs）----
        edge_map = {}
        for i, e in enumerate(G.edges):
            s, t = e["src"], e["dst"]
            rs, rt = region_of.get(s), region_of.get(t)
            if rs is not None and rs == rt:
                continue                          # 内部边 -> 汇总边（勿双计）
            m_r, n_r = region_node_ids.get(rs, (None, None))
            m_t, n_t = region_node_ids.get(rt, (None, None))
            ns = n_r if rs is not None else s       # absorb 出边: 从 N 发出
            nt = m_t if rt is not None else t       # absorb 入边: 汇到 M
            pos = len(Gp.edges)
            Gp.edges.append({"src": ns, "dst": nt, "etype": e["etype"],
                             "ts": e["ts"], "mu": 1.0})
            edge_map[pos] = [i]
        # ---- 汇总边 M→N：μ = 被折叠内部边数 ----
        for ridx in range(len(regions)):
            idxs = internal_idxs[ridx]
            if not idxs:
                continue
            m_id, n_id = region_node_ids.get(ridx, (None, None))
            if m_id is None or n_id is None:
                continue
            pos = len(Gp.edges)
            Gp.edges.append({"src": m_id, "dst": n_id,
                             "etype": self.summary_etype, "ts": 0,
                             "mu": float(len(idxs))})
            edge_map[pos] = idxs

        return ReductionResult(Gp, node_map, edge_map, {
            "nodes_before": G.n_nodes(), "edges_before": G.n_edges(),
            "nodes_after": Gp.n_nodes(), "edges_after": Gp.n_edges(),
            "n_regions": len(regions),
            "n_removed_nodes": len(removed),
            "templates_used": sorted({r["tpl_id"] for r in regions}),
            "reduction_ratio": (G.n_nodes() - Gp.n_nodes()) / max(1, G.n_nodes()),
        }, self.name)
