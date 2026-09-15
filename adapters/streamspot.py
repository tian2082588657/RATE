# -*- coding: utf-8 -*-
"""adapters/streamspot.py — StreamSpot 数据集适配器。

原始格式：all.tar.gz 内含一个 all.tsv，每行 6 列 tab 分隔：
    <src顶点id> <src标签> <dst顶点id> <dst标签> <事件标签> <图id>
行按图 id 有序排列，每张图约 1e5 条边。600 张图 = 6 个场景(scene = gid//100) × 100。

本适配器把每张图变成一个 CanonicalGraph：
    节点 = 顶点id；node.type = 顶点标签(a..h)；边 = src→dst；edge.etype = 事件标签。
图上标注 scene、graph_label(默认 scene<5 良性 / scene==5 攻击，可配置)。
"""
from __future__ import annotations
import csv, io, os, tarfile
from rate_core import CanonicalGraph


# 顶点/边标签枚举统计到 meta，便于后续语义特征统一字典
def parse(archive_path, gids=None, gid_range=None, split="scenes",
          max_lines=None, label_scene=5, verbose=True):
    """流式解析 all.tar.gz 中的 all.tsv。

    gids: 只保留这些图（list[int]）；文件按 gid 有序，读到超过目标即停。
    split: "scenes" -> gid//100 == label_scene 为攻击，其余良性；
           "threatrace" -> 复现 ThreaTrace 划分: gid in [0,125) 良性 / [125,150) 攻击。
    max_lines: 仅读取前 N 行（调试用）。
    返回 (graphs: list[CanonicalGraph], meta: dict)。
    """
    if not os.path.exists(archive_path):
        raise FileNotFoundError(archive_path)
    wanted = set()
    if gids:
        wanted = set(int(x) for x in gids)
    elif gid_range:
        wanted = set(range(int(gid_range[0]), int(gid_range[1])))
    else:
        wanted = None  # 全部

    label_fn = _label_fn(split, label_scene)

    graphs = []
    cur = None            # 正在累积的图
    n_lines = 0
    type_counts = {}      # 全量统计
    etype_counts = {}
    with tarfile.open(archive_path, "r:gz") as tf:
        member = None
        for m in tf.getmembers():
            if m.name.endswith(".tsv") or m.name == "all.tsv":
                member = m
                break
        if member is None:
            raise ValueError(f"{archive_path} 内未找到 .tsv 文件")
        f = tf.extractfile(member)
        text = io.TextIOWrapper(f, encoding="utf-8")
        reader = csv.reader(text, delimiter="\t")
        for row in reader:
            if len(row) < 6:
                continue
            n_lines += 1
            if max_lines and n_lines > max_lines:
                break
            g = int(row[5])
            if wanted is not None and g > max(wanted):
                break
            if wanted is not None and g not in wanted:
                if cur is not None and int(cur.gid) != g:
                    graphs.append(_finalize(cur, label_fn))
                    cur = None
                continue
            if cur is None or int(cur.gid) != g:
                if cur is not None:
                    graphs.append(_finalize(cur, label_fn))
                cur = CanonicalGraph(gid=g)
                cur.meta["scene"] = g // 100
                cur.meta["split"] = split
            _add_row(cur, row, type_counts, etype_counts)
        if cur is not None:
            graphs.append(_finalize(cur, label_fn))

    meta = {"name": "streamspot", "archive": os.path.basename(archive_path),
            "n_graphs": len(graphs), "n_lines_read": n_lines,
            "node_types": sorted(type_counts), "edge_types": sorted(etype_counts)}
    if verbose:
        print(f"[streamspot] 读取 {n_lines} 行 / {len(graphs)} 图 | "
              f"节点类型 {type_counts} | 事件类型 {etype_counts}")
    return graphs, meta


def _label_fn(split, label_scene):
    if split == "threatrace":
        def f(gid):
            g = int(gid)
            return 1 if 125 <= g < 150 else 0
        return f
    # scenes
    return lambda gid: 1 if int(gid) // 100 == int(label_scene) else 0


def _add_row(g, row, type_counts, etype_counts):
    src_id, src_lbl, dst_id, dst_lbl, ev, _ = row[0], row[1], row[2], row[3], row[4], row[5]
    type_counts.setdefault(src_lbl, 0)
    type_counts.setdefault(dst_lbl, 0)
    type_counts[src_lbl] += 1
    type_counts[dst_lbl] += 1
    etype_counts[ev] = etype_counts.get(ev, 0) + 1
    g.ensure_node(src_id, ntype=src_lbl)
    g.ensure_node(dst_id, ntype=dst_lbl)
    g.add_edge(src_id, dst_id, etype=ev, ts=len(g.edges))


def _finalize(g, label_fn):
    # 图级标注：整图节点的攻击标签（StreamSpot 无节点级真值，图级近似）
    gl = label_fn(g.gid)
    if gl == 1:
        for nid in g.nodes:
            g.labels[nid] = 1
    else:
        for nid in g.nodes:
            g.labels[nid] = 0
    g.meta["graph_label"] = gl
    return g
