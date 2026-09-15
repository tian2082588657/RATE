# -*- coding: utf-8 -*-
"""rate_core.py — 规范图格式 (Canonical Graph) 与通用工具。

合并指南 §6.1 的落地。所有数据集适配器输出该格式；所有归约算子、特征、
检测代码只认这个格式。原则：节点 id 在单图内唯一；边保留多重边（不预先去重，
去重是 CPR 算子的职责，且必须记账 μ）；μ 是"该条边代表几条原始边"。

节点:  nid -> {"type": str, "attrs": {原始元数据}}     (type ∈ {process,file,socket,unit,...} 或数据集的原始类型标签)
边:    {"src","dst","etype","ts": int(ns 或 0), "mu": float}
标签:  labels: nid -> 0(良性)/1(攻击)，评测用；可为空
"""
from __future__ import annotations
import json, hashlib, os, random, math, pickle, gzip, zlib


class CanonicalGraph:
    __slots__ = ("gid", "nodes", "edges", "labels", "meta")

    def __init__(self, gid: str, nodes=None, edges=None, labels=None, meta=None):
        self.gid = str(gid)
        self.nodes = nodes if nodes is not None else {}   # nid -> {"type","attrs"}
        self.edges = edges if edges is not None else []   # list of dict
        self.labels = labels if labels is not None else {}  # nid -> 0/1
        self.meta = meta if meta is not None else {}

    # ---------- 基础统计 ----------
    def n_nodes(self):
        return len(self.nodes)

    def n_edges(self):
        return len(self.edges)

    def ensure_node(self, nid, ntype="unknown", attrs=None):
        """取节点；不存在则创建占位节点（DARPA 事件先于对象定义时用）。"""
        nid = str(nid)
        if nid not in self.nodes:
            self.nodes[nid] = {"type": ntype, "attrs": dict(attrs) if attrs else {}}
        return self.nodes[nid]

    def add_edge(self, src, dst, etype, ts=0, mu=1.0):
        self.edges.append({"src": str(src), "dst": str(dst), "etype": str(etype),
                           "ts": int(ts or 0), "mu": float(mu)})

    # ---------- 度数 / 质量加权度 ----------
    def degrees(self, use_mu: bool = False):
        """返回 (din: dict, dout: dict)。use_mu=False 数边条数(=naive 口径)；
        use_mu=True 按 μ 加权(Σμ，=RATE 的质量口径)。全节点补 0。"""
        din, dout = {}, {}
        for nid in self.nodes:
            din[nid] = 0.0
            dout[nid] = 0.0
        for e in self.edges:
            w = float(e["mu"]) if use_mu else 1.0
            dout[e["src"]] = dout.get(e["src"], 0.0) + w
            din[e["dst"]] = din.get(e["dst"], 0.0) + w
        return din, dout

    def total_edge_mu(self) -> float:
        return sum(float(e["mu"]) for e in self.edges)

    # ---------- K-hop 邻域子图（指南 §2.2：在 G' 上做 K-hop 提取） ----------
    def khop_subgraph(self, anchor: str, k: int = 2):
        """anchor 的 k-hop 邻域，返回新 CanonicalGraph（仅拓扑，供模板挖掘/子图级检测）。"""
        if anchor not in self.nodes:
            return None
        frontier, seen = {anchor}, {anchor}
        for _ in range(k):
            nxt = set()
            for e in self.edges:
                if e["src"] in frontier and e["dst"] not in seen:
                    nxt.add(e["dst"])
                if e["dst"] in frontier and e["src"] not in seen:
                    nxt.add(e["src"])
            seen |= nxt
            frontier = nxt
            if not frontier:
                break
        sub = CanonicalGraph(f"{self.gid}:k{k}:{anchor}")
        for nid in seen:
            sub.nodes[nid] = {"type": self.nodes.get(nid, {}).get("type", "unknown"),
                              "attrs": dict(self.nodes.get(nid, {}).get("attrs", {}))}
            if nid in self.labels:
                sub.labels[nid] = self.labels[nid]
        for e in self.edges:
            if e["src"] in seen and e["dst"] in seen:
                sub.edges.append(dict(e))
        sub.meta = {"anchor": anchor, "k": k, "origin": self.gid}
        return sub

    # ---------- 序列化 ----------
    def to_dict(self):
        return {"gid": self.gid, "nodes": self.nodes, "edges": self.edges,
                "labels": self.labels, "meta": self.meta}

    @staticmethod
    def from_dict(d):
        return CanonicalGraph(d["gid"], d.get("nodes", {}), d.get("edges", []),
                              d.get("labels", {}), d.get("meta", {}))


def fingerprint(params: dict) -> str:
    """配置指纹：用于缓存目录命名，避免重复解析/归约。"""
    s = json.dumps(params, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:16]


# ---------------- 图缓存 IO ----------------
def save_graphs(path, graphs):
    """写 JSONL（gzip 可选）。每行一张图。"""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for g in graphs:
            f.write(json.dumps(g.to_dict(), ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def load_graphs(path):
    graphs = []
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                graphs.append(CanonicalGraph.from_dict(json.loads(line)))
    return graphs


def save_pickle(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=4)
    os.replace(tmp, path)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def cache_path(cache_root: str, dataset_key: str, stage: str, fp: str, ext="jsonl") -> str:
    d = os.path.join(cache_root, dataset_key, stage)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{fp}.{ext}")


# ---------------- 随机小图生成（不变量测试用） ----------------
def random_graph(gid="rand", n=30, m=None, ntypes=4, etypes=6, p_self=0.0,
                 mu=None, seed=0, labels_frac=0.1):
    """生成有向多重图。mu 为 None 时全部 1；否则从 mu 分布抽样(>0 整数)。"""
    rng = random.Random(seed)
    if m is None:
        m = int(n * rng.uniform(1.2, 2.5))
    nodes = {}
    for i in range(n):
        t = f"t{rng.randrange(ntypes)}"
        nodes[str(i)] = {"type": t, "attrs": {}}
    edges = []
    for _ in range(m):
        a, b = str(rng.randrange(n)), str(rng.randrange(n))
        if p_self < 1.0 and a == b:
            b = str((int(b) + 1) % n)
        mu_i = mu() if mu else 1.0
        edges.append({"src": a, "dst": b, "etype": f"e{rng.randrange(etypes)}",
                      "ts": rng.randrange(10 ** 9), "mu": float(mu_i)})
    labels = {}
    for i in range(n):
        if rng.random() < labels_frac:
            labels[str(i)] = 1
    g = CanonicalGraph(gid, nodes, edges, labels)
    g.meta["seed"] = seed
    return g


def star_graph(gid, center_type="t0", leaf_type="t1", n_leaves=4, etype="e0", seed=0):
    """手工用例：以 c 为中心的星型子图，供模板挖掘/匹配测试。"""
    rng = random.Random(seed)
    nodes = {str(-1): {"type": center_type, "attrs": {}}}
    edges = []
    for i in range(n_leaves):
        nodes[str(i)] = {"type": leaf_type, "attrs": {}}
        edges.append({"src": str(-1), "dst": str(i), "etype": etype,
                      "ts": 0, "mu": 1.0})
    return CanonicalGraph(gid, nodes, edges)


def graph_stats(g):
    din, dout = g.degrees(use_mu=True)
    din0, dout0 = g.degrees(use_mu=False)
    return {"gid": g.gid, "nodes": g.n_nodes(), "edges": g.n_edges(),
            "total_mu": g.total_edge_mu(),
            "max_mass_in": max(din.values()) if din else 0,
            "max_mass_out": max(dout.values()) if dout else 0}
