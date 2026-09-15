# -*- coding: utf-8 -*-
"""reduction/template_mining.py — 良性模板挖掘 v5（指南 §4.3 的落地，2026-09-03）。

TeRed 官方模板来自目标程序的单元测试；DARPA/StreamSpot 是已发生数据，无法补跑，
因此改为：**从良性训练图集的"以语义节点为锚的 k-hop 邻域子图"上挖频繁子图**。

v3 起点（v2 症状：StreamSpot 只挖出 7 个 2~6 节点模板、归约率 0.1%）：
  1. 锚点语义化 + WL 色彩分桶 + 去重叠支持度 + 匹配验证（可折叠性保证）。
v4/v5 演进（2026-09-03 diag_e4~e7 证据驱动）：
  1. auto_semantic_types 主体词路优先（有实名 process/subject 就只锚主体，
     防 file_object_dir 等低频杂物污染锚 -> file-hub 大模板，边归约暴跌）；
  2. 模板默认大优先排序（同支持度先跑大模板，防小模板先吃共享节点）；
  3. cpr_first 结构图挖掘+验证（提速 ~6x）；
  4. dedup_key node/anchor 双口径（A/B 实测 node 端到端更优，见 mine_templates）；
  5. context_k 共享上下文剔除（可选项，cadets 上过激，保留给合成场景）。

与归约算子共用同一标签口径（节点 type 匹配、边标签忽略——与原版 TeRed 一致）。
缓存：模板一次挖掘、落盘 json，实验时按覆盖率抽样加载（归约强度扫描预留接口）。
"""
from __future__ import annotations
import json, os, random, collections
import networkx as nx
from networkx.algorithms import isomorphism as nxiso

from rate_core import CanonicalGraph, save_graphs, load_graphs


# ---------------- WL 色彩签名（分桶主键） ----------------
def wl_signature(dg, iters=3):
    """整图 WL 色彩多重集签名：对带 node type 的有向图，同构不变、O(V+E) 快。

    用于把候选子图分成"可能同构"的桶；桶内再用匹配器精验（假阳性会被剔除）。
    """
    col = {n: str(dg.nodes[n].get("type", "?")) for n in dg.nodes}
    for _ in range(iters):
        new = {}
        for n in dg.nodes:
            nbs = sorted(col[v] for v in dg.successors(n)) + \
                  sorted(col[v] for v in dg.predecessors(n))
            new[n] = col[n] + "/" + ",".join(nbs)
        col = new
    cnt = collections.Counter(col.values())
    return tuple(sorted(cnt.items()))


# ---------------- 语义锚点探测 ----------------
_SUBJECT_WORDS = ("process", "subject", "unit", "proc", "principal", "task")
"""溯源图里"行为主体" type 命名关键词（进程/任务）。主体数量少但占比未必低频：
StreamSpot 用 a/b/.. 抽象名（走稀有启发），DARPA 用 process/.. 实名（走主体词）。
"""


def auto_semantic_types(graphs, rare_frac=0.02, min_count=4):
    """返回应作为模板锚的"语义主体" type 集。

    两路判定（v5 修正）：
      1. **主体词路（优先）**：type 名含 process/subject/unit/proc/principal/task 的，
         直接视为行为主体。**只要存在任一实名主体 type，就只返回这些**——不再
         掺入稀有路径的杂物（E3 cadets 里 file_object_dir/object 等低频 type 会
         被当锚 -> 挖出 file-hub 大切片模板，折叠时互相挡、边归约率暴跌）。
      2. **稀有兜底**：无任何实名主体 type（StreamSpot 的 a/b 抽象名）时，才用
         "跨图低频 type" 启发挑主体。
    启发式：溯源图里行为主体(进程/服务)数量远少于被访问对象(文件/socket)；
    行为模板应以主体为锚（指南 §4.3）。
    """
    cnt = collections.Counter()
    named_subj = set()
    for g in graphs:
        for n, nd in g.nodes.items():
            t = nd.get("type", "?")
            cnt[t] += 1
            if any(w in str(t).lower() for w in _SUBJECT_WORDS):
                named_subj.add(t)
    subj = named_subj & set(cnt)
    if subj:                     # 有实名主体 -> 只用主体，防低频杂物污染锚
        return subj
    total = sum(cnt.values()) or 1
    rare = {t for t, c in cnt.items()
            if c / total <= rare_frac and c >= min_count}
    return rare


# ---------------- 候选抽取 ----------------
def _slice_hub_egonet(g, a, fanout=(2, 6), groups=24, max_tpl_nodes=48,
                      rng=None):
    """把 hub 锚（egonet 超 max_tpl_nodes）切成"锚 + 少量直接邻居"小组候选。

    动机：StreamSpot/DARPA 的进程锚通常连 10^3 量级对象，完整 1-hop egonet
    是 2000+ 节点的 hub，既超尺寸上限、又太特异无法跨图重复。真实可模板化
    的是进程的局部交互块（如 a 同时写 2~6 个对象）。这里把 a 的直接邻居
    随机分组，每组与 a 组成一个小候选 —— 组间节点不相交，天然支持非重叠折叠。
    """
    if rng is None:
        rng = random.Random(0)
    # 直接邻居（含入/出）
    nbrs = set()
    for e in g.edges:
        if e["src"] == a:
            nbrs.add(e["dst"])
        elif e["dst"] == a:
            nbrs.add(e["src"])
    nbrs = [n for n in nbrs if n in g.nodes]
    if len(nbrs) < 2:
        return []
    rng.shuffle(nbrs)
    kmin, kmax = max(2, fanout[0]), min(fanout[1], max_tpl_nodes - 1)
    out = []
    i, tries = 0, 0
    while len(out) < groups and tries < groups * 4 and i < len(nbrs):
        tries += 1
        k = rng.randint(kmin, kmax)
        sel = nbrs[i:i + k]
        i += k
        if len(sel) < 2:
            break
        seen = {a} | set(sel)
        sub = CanonicalGraph(f"{g.gid}:hub:{a}:{len(out)}")
        for nid in seen:
            sub.nodes[nid] = {"type": g.nodes[nid].get("type", "unknown"),
                              "attrs": dict(g.nodes[nid].get("attrs", {}))}
        for e in g.edges:                       # 组内全边（含邻居间边）
            if e["src"] in seen and e["dst"] in seen:
                sub.edges.append({"src": e["src"], "dst": e["dst"],
                                  "etype": e["etype"], "ts": 0, "mu": 1.0})
        if sub.n_edges() == 0:
            continue
        sub.meta = {"anchor": a, "k": 1, "origin": g.gid, "hub_slice": True}
        out.append(sub)
    return out


def sample_candidates(benign_graphs, khop=1, per_graph=400, max_tpl_nodes=48,
                      seed=0, anchor_types=None, rare_frac=0.02,
                      min_rare_count=4, min_tpl_nodes=3,
                      khop_rare=2, hub_fanout=(2, 6), hub_groups=24,
                      verbose=True):
    """抽 k-hop 邻域候选。锚点策略（v3 核心）：
      - 语义 type（显式 anchor_types 或自动低频探测）节点：全部抽取；
      - 其余节点：按每图配额均匀抽样（度 >= 2 优先，略过纯叶子云）。
    语义锚 khop=khop_rare（2 跳抓"主体→对象"行为），若超出 max_tpl_nodes 自动
    降为 1-hop；1-hop 仍是 hub（邻居数超限）则 _slice_hub_egonet 切片。
    每个候选携带 meta.anchor/origin 供去重叠计数。
    """
    rng = random.Random(seed)
    sem = set(anchor_types) if anchor_types else auto_semantic_types(
        benign_graphs, rare_frac=rare_frac, min_count=min_rare_count)
    cands = []
    n_sem, n_hub, n_hub_slices = 0, 0, 0
    for g in benign_graphs:
        types_here = collections.Counter(
            nd.get("type", "?") for nd in g.nodes.values())
        sem_here = sem & set(types_here)
        deg = collections.Counter()
        for e in g.edges:
            deg[e["src"]] += 1
            deg[e["dst"]] += 1
        sem_nodes = [n for n in g.nodes
                     if g.nodes[n].get("type", "?") in sem_here]
        rest_nodes = [n for n in g.nodes
                      if g.nodes[n].get("type", "?") not in sem_here
                      and deg.get(n, 0) >= 2]
        rng.shuffle(sem_nodes)
        rng.shuffle(rest_nodes)
        # 语义锚全抽；普通锚补足配额
        anchors = list(sem_nodes)
        n_sem += len(sem_nodes)
        room = max(0, per_graph - len(sem_nodes))
        anchors += rest_nodes[:room]
        for a in anchors:
            a_type = g.nodes[a].get("type", "?")
            k = khop_rare if a_type in sem_here else khop
            sub = g.khop_subgraph(a, k)
            if sub is None:
                continue
            if sub.n_nodes() > max_tpl_nodes and k > 1:
                sub = g.khop_subgraph(a, k - 1)      # 2-hop 蔓延过大 -> 降 1-hop
            if sub is None:
                continue
            if sub.n_nodes() > max_tpl_nodes:
                # hub：1-hop 仍超限 -> 邻居分组切片（不再整丢！）
                slices = _slice_hub_egonet(g, a, fanout=hub_fanout,
                                           groups=hub_groups,
                                           max_tpl_nodes=max_tpl_nodes, rng=rng)
                if slices:
                    n_hub += 1
                    n_hub_slices += len(slices)
                    cands.extend(slices)
                continue
            if sub.n_nodes() < min_tpl_nodes or sub.n_nodes() > max_tpl_nodes:
                continue
            if sub.n_edges() == 0:
                continue
            sub.meta["anchor"] = a
            sub.meta["origin"] = g.gid
            cands.append(sub)
    if verbose:
        print(f"[template_mining] 语义锚 type={sorted(sem)} (语义锚点 {n_sem}) | "
              f"hub 锚 {n_hub} 个 -> 切片 {n_hub_slices} | "
              f"抽得 {len(cands)} 个邻域候选")
    return cands


# ---------------- 共享上下文剔除（v5：让模板只含"私有交互"，避免区域重叠） ----------------
def _context_nodes(G, subject_types, k=2):
    """返回 G 中被 >= k 个不同主体(进程)引用的节点集 = 共享上下文。

    diag_e4/e5 证据：E3/E5 cadets 每窗有 ~190 进程，但进程 1-hop 邻域里的
    file/父进程几乎全是系统库级共享节点（libc 等被 56~167 进程引用）。若模板
    把这些共享节点编进去，折叠时第一实例一吸收、其余兄弟进程全被挡 -> 区域数
    被压到 ~1/窗。v5 挖矿前先把这类节点从候选图剔除：模板只编码"进程+私有对象"，
    标准 TeRed 算子即可折叠出大量互不重叠的区域。
    主体 = type 名命中 _SUBJECT_WORDS 的节点（process/subject/..）。
    """
    cnt = collections.Counter()
    for e in G.edges:
        if G.nodes.get(e["src"], {}).get("type", "?") in subject_types:
            cnt[e["dst"]] += 1
        if G.nodes.get(e["dst"], {}).get("type", "?") in subject_types:
            cnt[e["src"]] += 1
    return {n for n, c in cnt.items() if c >= k}


def _drop_nodes(g, kill):
    """返回去掉 kill 节点集后（含关联边）的图副本。"""
    gp = CanonicalGraph(g.gid + ":nctx")
    for nid, nd in g.nodes.items():
        if nid not in kill:
            gp.nodes[nid] = {"type": nd.get("type", "unknown"),
                             "attrs": dict(nd.get("attrs", {}))}
    for e in g.edges:
        if e["src"] not in kill and e["dst"] not in kill:
            gp.edges.append(dict(e))
    return gp


# ---------------- 主挖掘 ----------------
def _relabel_rep(rep: CanonicalGraph, idx):
    """桶代表 -> 模板（重排为规范 id 0..n-1，anchor 尽量为 0）。"""
    anchor = rep.meta.get("anchor")
    order = [anchor] if anchor in rep.nodes else []
    order += [n for n in sorted(rep.nodes) if n not in order]
    rid = {nid: str(i) for i, nid in enumerate(order)}
    tpl = CanonicalGraph(f"tpl_{idx}")
    for nid, nd in rep.nodes.items():
        tpl.nodes[rid[nid]] = {"type": nd.get("type", "unknown"),
                               "attrs": dict(nd.get("attrs", {}))}
    for e in rep.edges:
        tpl.edges.append({"src": rid[e["src"]], "dst": rid[e["dst"]],
                          "etype": e["etype"], "ts": 0, "mu": 1.0})
    tpl.meta = {"anchor": rid.get(anchor, "0"), "origin": rep.meta.get("origin")}
    return tpl


def mine_templates(benign_graphs, min_support=5, khop=1, per_graph=400,
                   max_tpl_nodes=48, max_templates=100, seed=0,
                   min_tpl_nodes=3, min_tpl_edges=2, anchor_types=None,
                   rare_frac=0.02, min_rare_count=4, khop_rare=2, verify=True,
                   cpr_first=True, hub_fanout=(2, 6), hub_groups=24,
                   dedup_key="node", context_k=0, verbose=True,
                   progress=None):
    """返回 (templates: list[CanonicalGraph], clusters_info)。

    模板 = 良性图集里能非重叠嵌入 >= min_support 次的连通结构块（经匹配器验证）。
    v5 流程：语义锚候选(可共享上下文剔除) -> WL 分桶 -> 桶内按 anchor 去重叠计数
    -> 代表成模板 -> seed-and-grow 匹配验证（可折叠实例数 >= min_support 才保留）。

    dedup_key: "node"(默认) | "anchor"。桶内去重叠计数口径：
      - "node": 按 (origin, node) 全节点集去重 —— v3 原口径。E3 8 窗 A/B 实测
        (2026-09-03 diag): node 端到端节点归约 9.7% > anchor 8.7%，边 8.7% >
        2.3%。原因：anchor 口径把"共享上下文"型大桶(如 sup=55 的 6-file+2-proc
        模板)放进来，但折叠时这些模板含共享节点、区域互相挡 -> 每窗只能折 1 次，
        反而挤掉可折叠的小模板。node 口径更严，保留的模板折叠互不冲突。
      - "anchor": 不同锚(进程)=不同实例（不把共享节点计入去重键）。能恢复被
        node 口径压掉的大桶（diag_e6 桶级验证 13 个 sup>=4 桶含 sup=55），
        适合"每窗同程序 64 次运行都想各折一次"的理想场景；但 cadets 里这些
        实例共享 libc/父进程，标准折叠仍互相挡，端到端反而更差。保留为可选项。

    context_k>0（默认 0=关）：从候选抽取用的图上剔除"被 >= context_k 个主体
    引用的共享上下文节点"（见 _context_nodes）。实验结论：cadets 上过激
    (E3 8 窗 node 9.7% -> 2.6%)——它的可折叠重复结构恰恰长在共享上下文里，
    剔除后进程锚大多无私有邻居可挖。仅适合对象共享度极高的合成场景，保留为
    可选项（tests/test_shared_context.py 覆盖）。

    cpr_first=True：挖掘前先对每张良性图做 CPR 结构压缩（并行边折叠成 μ=1 结构边，
    节点/结构不变）。诊断实测：StreamSpot 原始 8~32 万并行边，CPR 后 ~1.7 万结构边，
    挖掘提速一个量级且候选干净；匹配阶段 _to_nx_matching 本就把模板并行边压平，
    故从结构图挖模板与在原始图挖模板同构等价。
    """
    from reduction.tered import find_instances, _to_nx_matching
    from reduction.cpr import CPROperator

    if cpr_first:
        cpr = CPROperator()
        mine_graphs = []
        for g in benign_graphs:
            res = cpr.reduce(g)
            mine_graphs.append(res.Gp)
    else:
        mine_graphs = list(benign_graphs)
    sem_types = (set(anchor_types) if anchor_types else
                 auto_semantic_types(mine_graphs, rare_frac=rare_frac,
                                     min_count=min_rare_count))

    # ---- 候选抽取图：可选剔除共享上下文 ----
    cand_graphs = mine_graphs
    if context_k and context_k > 0:
        subj_types = {t for t in sem_types
                      if any(w in str(t).lower() for w in _SUBJECT_WORDS)}
        pruned, n_ctx = [], 0
        for g in mine_graphs:
            ctx = _context_nodes(g, subj_types, context_k)
            n_ctx += len(ctx)
            pruned.append(_drop_nodes(g, ctx))
        cand_graphs = pruned
        if verbose:
            print(f"[template_mining] context_k={context_k}: 剔除共享上下文节点 "
                  f"{n_ctx} 个 (跨 {len(mine_graphs)} 图)")

    cands = sample_candidates(cand_graphs, khop=khop, per_graph=per_graph,
                              max_tpl_nodes=max_tpl_nodes, seed=seed,
                              anchor_types=sem_types, rare_frac=rare_frac,
                              min_tpl_nodes=min_tpl_nodes, khop_rare=khop_rare,
                              hub_fanout=hub_fanout, hub_groups=hub_groups,
                              verbose=verbose)
    # ---- WL 分桶 ----
    buckets = collections.defaultdict(list)
    for c in cands:
        if progress:
            progress()
        buckets[wl_signature(_to_nx_matching(c))].append(c)
    # ---- 桶内去重叠计数（dedup_key: node 默认, anchor 可选 —— 见 docstring） ----
    freq = []            # (support, rep)
    for sig, subs in buckets.items():
        # 大候选优先（有结构意义），同结构取较大实例为代表
        subs.sort(key=lambda s: (-s.n_nodes(), -s.n_edges()))
        occ = set()
        chosen = []
        for s in subs:
            if dedup_key == "node":
                keys = {(s.meta.get("origin"), n) for n in s.nodes}
            else:                       # "anchor": 不同锚(进程)=不同实例
                keys = {(s.meta.get("origin"), s.meta.get("anchor"))}
            if keys & occ:
                continue
            occ |= keys
            chosen.append(s)
        if len(chosen) < min_support:
            continue
        freq.append((len(chosen), chosen[0]))
    freq.sort(key=lambda x: (-x[0], -x[1].n_nodes()))
    # ---- 匹配验证：模板必须真能在良性图上折叠 >= min_support 个非重叠区域 ----
    templates = []
    for sup, rep in freq:
        tpl = _relabel_rep(rep, len(templates))
        if tpl.n_nodes() < min_tpl_nodes or tpl.n_edges() < min_tpl_edges:
            continue
        ok = sup
        if verify:
            found = 0
            for g in mine_graphs:       # 在 CPR 结构图上验证：匹配去重后与原始图等价，快 ~6x
                regs = find_instances(g, [tpl], max_instances=min_support,
                                      max_total=min_support)
                found += len(regs)
                if found >= min_support:
                    break
            ok = min(sup, found)
        if ok < min_support:
            continue
        tpl.meta["support"] = ok
        templates.append(tpl)
        if len(templates) >= max_templates:
            break
    # 默认大优先：同支持度下先执行大模板（region 覆盖更多节点，且避免小模板
    # 先吃掉共享节点导致大模板无实例可折叠 —— 实测节点归约率 +4%）。
    templates.sort(key=lambda t: (-t.n_nodes(), -t.n_edges()))
    if verbose:
        print(f"[template_mining] {len(buckets)} 个 WL 桶, {len(freq)} 个支持度>="
              f"{min_support}, 匹配验证后采用 {len(templates)} 个模板")
    return templates, {"n_candidates": len(cands), "n_clusters": len(buckets),
                       "n_frequent": len(freq), "n_templates": len(templates),
                       "semantic_types": sorted(sem_types)}


# ---------------- 覆盖率抽样加载（预留归约强度扫描） ----------------
def load_subset(templates, coverage=1.0, seed=0, by="support"):
    """按覆盖率抽样模板。coverage=1 全量；0.1=覆盖总支持度 10% 的最小模板集。"""
    if coverage >= 1.0:
        return list(templates)
    total = sum(t.meta.get("support", 1) for t in templates)
    target = total * coverage
    acc, chosen = 0.0, []
    for t in sorted(templates, key=lambda t: -t.meta.get("support", 1)):
        chosen.append(t)
        acc += t.meta.get("support", 1)
        if acc >= target:
            break
    return chosen


# ---------------- 持久化 ----------------
def save_templates(templates, path):
    save_graphs(path, templates)


def load_templates(path):
    return load_graphs(path)
