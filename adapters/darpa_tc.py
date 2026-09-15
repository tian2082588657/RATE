# -*- coding: utf-8 -*-
"""adapters/darpa_tc.py — DARPA Transparent Computing 数据集适配器。

支持三种原始载体，输出同一规范图格式：
  * E3 (CDM18)：ta1-*.json.tar.gz，tar 内是行式 JSON（ndjson），cdm18。
  * E5 theia (CDM20)：目录内已附带 ta1-theia-*.json.*.gz（ndjson，cdm20）。
  * E5 cadets：*.bin.*.gz —— 实际是 **Avro 对象容器**(魔数 Obj\x01...)，
    需 fastavro 读取（pip install fastavro）。schema 已内嵌于容器头。

记录形态统一成：
  {'kind': 'Event'|'Subject'|'FileObject'|'NetFlowObject'|..., 'body': {...}, 'hostId': ...}
然后映射为溯源图：
  Subject(process) -> 节点 type=process
  FileObject/NetFlowObject/... -> 节点
  Event -> 有向边 subject -> predicateObject，etype=事件名(write/read/execve/...)

E5 cadets 的 bin 数据块记录与 ndjson 形态不同（见 iter_avro_records），已做兼容。
"""
from __future__ import annotations
import gzip, io, json, os, tarfile, uuid as _uuid

from rate_core import CanonicalGraph

# CDM 事件名 -> 短名（去掉 EVENT_ 前缀并小写）
def _evname(body):
    t = str(body.get("type", ""))
    if t.startswith("EVENT_"):
        return t[len("EVENT_"):].lower()
    names = body.get("names")
    arr = names.get("array", []) if isinstance(names, dict) else names
    if isinstance(arr, list) and arr:
        return str(arr[0]).lower()
    return t.lower() or "event"


def _scalar(v, default=None):
    """解开 Avro/JSON union 包装：{"string": x} / {"long": x} / {"int": x} 等。"""
    if isinstance(v, (bytes, bytearray)):
        # CDM20 Avro 的 UUID 字段是 bytes(16)，转标准 UUID 字符串作节点 id
        try:
            return str(_uuid.UUID(bytes=bytes(v))) if len(v) == 16 else v.hex()
        except Exception:
            return v.hex()
    if isinstance(v, dict):
        for k, val in v.items():
            if k in ("string", "long", "int", "double", "float", "boolean"):
                return val
        # 单键 UUID 包装 com.bbn.tc.schema.avro.cdm18.UUID -> 内部值
        if len(v) == 1:
            only = list(v.values())[0]
            if isinstance(only, (str, int)):
                return only
        return None
    return v


def _str(v):
    s = _scalar(v)
    return s if s is not None else ""


# ---------------- 记录迭代器 ----------------
def iter_ndjson_lines(path):
    """逐条 ndjson。path 可为 .gz 或 .tar.gz(取第一个成员) 或普通文件。"""
    if path.endswith(".tar.gz") or path.endswith(".tgz"):
        with tarfile.open(path, "r:gz") as tf:
            member = tf.getmembers()[0]
            f = tf.extractfile(member)
            for line in io.TextIOWrapper(f, encoding="utf-8"):
                line = line.strip()
                if line:
                    yield json.loads(line)
    elif path.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
    else:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


def _norm_ndjson(rec):
    """ndjson 记录 -> (kind, body, host_id)。"""
    datum = rec.get("datum") or rec
    body = None
    kind = ""
    if isinstance(datum, dict) and len(datum) == 1:
        k = list(datum.keys())[0]
        if "." in k:                       # com.bbn.tc.schema.avro.cdm18.Event
            kind = k.split(".")[-1]
            body = datum[k]
    if body is None:
        body = datum
    if not kind:
        kind = _str(rec.get("type", "")).replace("RECORD_", "") or \
               _str(body.get("type", "")).split("_")[0]
    host = _str(rec.get("hostId")) or _str(body.get("hostId"))
    return kind, body, host


def iter_avro_records(path):
    """E5 cadets .bin.*.gz：Avro 对象容器。需 fastavro。"""
    try:
        import fastavro
    except ImportError:
        raise RuntimeError("解析 DARPA E5 cadets 需要 fastavro："
                           "pip install fastavro -i https://pypi.org/simple "
                           "--trusted-host pypi.org --trusted-host files.pythonhosted.org")
    with gzip.open(path, "rb") as f:
        reader = fastavro.reader(f)
        for rec in reader:
            yield rec


# RECORD_* 大写枚举 -> 适配器内部分支名
_KIND_CANON = {
    "EVENT": "Event", "SUBJECT": "Subject", "FILE_OBJECT": "FileObject",
    "NET_FLOW_OBJECT": "NetFlowObject", "SRC_SINK_OBJECT": "SrcSinkObject",
    "IPC_OBJECT": "IpcObject", "UNIT": "Unit", "PRINCIPAL": "Principal",
    "PROVENANCE_TAG_NODE": "TagIdentifier", "TYPETEMPLATE": "TypeDeclaration",
    "TAGANNOTATION": "TagAnnotation",
}


def _norm_avro(rec):
    """Avro datum -> (kind, body, host_id)。E5 cadets 容器内为 TCCDMDatum 记录。"""
    body = rec
    kind = ""
    # 形态一：{"datum": {"com.bbn.tc.schema.avro.cdm20.Event": {...}}}
    d = rec.get("datum")
    if isinstance(d, dict):
        if len(d) == 1 and "." in list(d.keys())[0]:
            k = list(d.keys())[0]
            kind = k.split(".")[-1]
            body = d[k]
        else:
            body = d
    if not kind:
        # 形态二：外层 type 字段 "RECORD_EVENT"/"RECORD_SUBJECT"/... 是权威标记
        t = _str(rec.get("type"))
        if t.startswith("RECORD_"):
            kind = _KIND_CANON.get(t[len("RECORD_"):], t[len("RECORD_"):].title())
    if not kind:
        # 形态三：body 自带 record_type / type 前缀 / subject 字段的兜底
        kind = _str(body.get("record_type")) or ""
        if not kind:
            bt = _str(body.get("type"))
            for pref, k in (("EVENT_", "Event"), ("SUBJECT_", "Subject"),
                            ("FILE_OBJECT", "FileObject"), ("NET_FLOW", "NetFlowObject")):
                if bt.startswith(pref):
                    kind = k
                    break
        if not kind and "subject" in body:
            kind = "Event"
    host = _str(body.get("hostId")) or _str(rec.get("hostId"))
    return kind, body, host


# ---------------- 适配器主体 ----------------
def parse(path, name=None, kind=None, max_records=None, host_filter=None,
          object_types=None, ts_window=None, verbose=True):
    """解析单个原始文件为一张（或按 part 多张）CanonicalGraph。

    kind: 'e3_ndjson' | 'e5_ndjson' | 'e5_cadets_bin'。缺省按扩展名推断。
    max_records: 只解析前 N 条记录（冒烟/调试，取文件开头=良性时段）。
    ts_window: (t_lo, t_hi) 纳秒过滤事件。
    object_types: 感兴趣的 object 记录类型子集。
    返回 (graphs: list[CanonicalGraph], meta)。
    """
    if kind is None:
        base = os.path.basename(path).lower()
        if ".bin." in base or base.endswith(".bin.gz"):
            kind = "e5_cadets_bin"
        elif (name and "e5" in name) or "e5" in base:
            kind = "e5_ndjson"
        else:
            kind = "e3_ndjson"
    g = CanonicalGraph(gid=os.path.basename(path))
    g.meta["source"] = path
    g.meta["kind"] = kind

    n_rec = 0
    obj_def = {}   # uuid -> 定义过的对象类型（用于补全占位节点）
    def _obj_type(t):
        if not t:
            return "object"
        t = str(t)
        return {"FILE_OBJECT_FILE": "file", "FILE_OBJECT_DIRECTORY": "dir",
                "FILE_OBJECT_SOCKET": "socket", "FILE_OBJECT_FIFO": "fifo",
                "FILE_OBJECT_CHAR": "chardev", "FILE_OBJECT_LINK": "link",
                "FILE_OBJECT_REGULAR": "file", "NET_FLOW_OBJECT": "netflow",
                "SUBJECT_PROCESS": "process", "SUBJECT_UNIT": "unit",
                "SUBJECT_USER": "user", "UNPRINCIPLED": "object",
                "SRC_SINK_OBJECT": "sink", "IPC_OBJECT": "ipc",
                "PRINCIPAL_LOCAL": "principal"}.get(t, t.lower() or "object")

    if kind in ("e3_ndjson", "e5_ndjson"):
        it = iter_ndjson_lines(path)
        norm = _norm_ndjson
    else:
        it = iter_avro_records(path)
        norm = _norm_avro

    n_skip = 0
    for rec in it:
        n_rec += 1
        if max_records and n_rec > max_records:
            break
        kind_r, body, host = norm(rec)
        if host_filter and host and host not in host_filter:
            n_skip += 1
            continue
        body = body or {}
        try:
            if kind_r == "Subject":
                uid = _str(body.get("uuid"))
                if uid:
                    g.ensure_node(uid, ntype="process", attrs={
                        "cmd": _str(body.get("cmdLine")), "pid": _str(body.get("cid"))})
            elif kind_r in ("FileObject", "NetFlowObject", "Unit", "Principal",
                            "SrcSinkObject", "IpcObject"):
                uid = _str(body.get("uuid"))
                if uid:
                    t = _obj_type(body.get("type"))
                    if uid in g.nodes and g.nodes[uid]["type"] == "object":
                        g.nodes[uid]["type"] = t     # 占位节点补全
                    else:
                        g.ensure_node(uid, ntype=t)
                    obj_def[uid] = t
            elif kind_r == "Event":
                ts = _scalar(body.get("timestampNanos")) or 0
                if ts_window and not (ts_window[0] <= ts <= ts_window[1]):
                    n_skip += 1
                    continue
                subj = _str(body.get("subject"))
                obj = _str(body.get("predicateObject"))
                if not subj or not obj:
                    n_skip += 1
                    continue
                # CDM20 事件无 subjectType 字段 —— 默认 process，勿落入 object
                st_raw = body.get("subjectType")
                st = _obj_type(st_raw) if st_raw else ""
                g.ensure_node(subj, ntype=st or "process")
                ot = obj_def.get(obj, "object")
                g.ensure_node(obj, ntype=ot)
                etype = _evname(body)
                g.add_edge(subj, obj, etype=etype, ts=int(ts or 0))
        except Exception:
            n_skip += 1
            continue

    meta = {"name": name or os.path.basename(path), "kind": kind,
            "records": n_rec, "skipped": n_skip, "nodes": g.n_nodes(),
            "edges": g.n_edges(),
            "node_types": _type_hist(g), "edge_types": _etype_hist(g)}
    if verbose:
        print(f"[darpa:{kind}] {os.path.basename(path)} -> {g.n_nodes()} 节点 "
              f"{g.n_edges()} 边 (读取 {n_rec} 记录)")
    return [g] if g.n_edges() else [], meta


def parse_many(paths, **kw):
    """多个原始文件 -> 多张图。每文件一张。"""
    graphs, metas = [], []
    for p in paths:
        gs, m = parse(p, **kw)
        graphs.extend(gs)
        metas.append(m)
    return graphs, metas


def _type_hist(g):
    h = {}
    for n in g.nodes.values():
        h[n["type"]] = h.get(n["type"], 0) + 1
    return h


def _etype_hist(g):
    h = {}
    for e in g.edges:
        h[e["etype"]] = h.get(e["etype"], 0) + 1
    return h


# ---------------- 时间窗口切分（DARPA 无真值时的良/恶近似） ----------------
def split_by_time(graph, train_frac=0.5, seed=0):
    """按事件时间戳把一张 DARPA 图切成 训练(早)/测试(晚) 两张子图。
    返回 (train_g, test_g)。攻击通常发生在语料末尾 —— 此切分仅供搭建 pipeline
    冒烟；正式评测需各 TA 的 truth 标注文件(见 README 数据集一节)。"""
    es = sorted(graph.edges, key=lambda e: e["ts"])
    if not es:
        return graph, None
    cut = int(len(es) * train_frac)
    def _mk(seg, tag):
        sg = CanonicalGraph(f"{graph.gid}:{tag}")
        sg.meta = dict(graph.meta)
        seen = set()
        for e in seg:
            seen.add(e["src"]); seen.add(e["dst"])
            sg.edges.append(dict(e))
        for nid in seen:
            sg.nodes[nid] = dict(graph.nodes.get(nid, {"type": "unknown", "attrs": {}}))
        return sg
    return _mk(es[:cut], "train"), _mk(es[cut:], "test")
