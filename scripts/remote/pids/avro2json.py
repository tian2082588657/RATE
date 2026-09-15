# -*- coding: utf-8 -*-
"""avro2json.py — DARPA E5 avro OCF (.bin.*.gz) → ta3-java-consumer 兼容 JSON lines。

目标格式（从 theia consumer 输出实证逆向 + 真实 cadets bin 解码核对）：
  - 每行一个 TCCDMDatum record，标准 avro JSON 编码 + consumer 自定义：
      * record 字段按 schema 顺序输出，紧凑无空格（json.dumps separators=(",",":")）
      * union 非 null 值 → {"<branch_tag>": value}：
          primitive 分支 tag = "string"/"int"/"long"/"float"/"double"/"boolean"/"bytes"
          complex 分支 tag = "array"/"map"
          named 分支 tag = 全名（namespace.name），如 com.bbn.tc.schema.avro.cdm20.Event
      * union 的 null 分支 / 直接 null → 裸 null（与 consumer 一致，不包装 {"null":null}）
      * 非 union 的 fixed UUID(size=16) → 大写 hex 连字符 36 字符串（consumer 风格）
      * union 内 fixed / 普通 bytes → latin-1 解码（json.dumps ensure_ascii 负责转义）
  - 默认只输出 Event/Subject/FileObject/NetFlowObject 四类 datum（建库脚本所需），
    --all 可关闭过滤。

fastavro 解码注意：
  - union 的 record 分支读取时**不带 tag**（返回裸字段 dict）→ 需按字段名集合回匹配分支；
    parse_schema 展开命名引用后字段类型为 dict/primitive 字符串，无未知引用。

用法:
  python3 avro2json.py --in ta1-cadets-1-e5-official-2.bin.100.gz --out out.json
  python3 avro2json.py --in-dir ... --out-dir ...（按 .bin.*.gz 批量）
  python3 avro2json.py --in ... --out ... --limit 300000（只转前 N 条，供验证）
"""
from __future__ import annotations
import argparse, gzip, json, os, sys, time

try:
    import fastavro
except ImportError:
    sys.exit("需要 fastavro: pip install fastavro -i https://pypi.tuna.tsinghua.edu.cn/simple")

KEEP = {"Event", "Subject", "FileObject", "NetFlowObject"}
_PRIM = {"string", "bytes", "int", "long", "float", "double", "boolean", "null"}
_NAMED = {"record", "error", "enum", "fixed"}


def _collect_named(node, out, seen=None):
    """递归收集命名类型定义：fullname -> node。node 需已 inject_ns。"""
    if seen is None:
        seen = set()
    if isinstance(node, list):
        for x in node:
            _collect_named(x, out, seen)
        return
    if not isinstance(node, dict):
        return
    if id(node) in seen:
        return
    seen.add(id(node))
    t = node.get("type")
    nm = node.get("name")
    if nm and t in _NAMED:
        ns = node.get("namespace", "")
        out[(f"{ns}.{nm}" if ns else nm)] = node
    if t in ("record", "error"):
        for f in node.get("fields", []):
            _collect_named(f.get("type"), out, seen)
    elif t == "array":
        _collect_named(node.get("items"), out, seen)
    elif t == "map":
        _collect_named(node.get("values"), out, seen)
    elif isinstance(t, dict):
        _collect_named(t, out, seen)
    elif isinstance(t, list):
        _collect_named(t, out, seen)


def _inline_refs(node, index, seen=None):
    """把命名类型字符串引用替换为定义 dict（原地）。parse_schema 保留引用为字符串，
    fastavro 解码虽能用内部索引，但自研编码器需要完整类型信息。schema 可能成环
    （递归 record），须以 id 集合防重复遍历。"""
    if seen is None:
        seen = set()
    if isinstance(node, list):
        for i, x in enumerate(node):
            if isinstance(x, str) and x not in _PRIM and x != "null":
                node[i] = index.get(x, {"type": x})  # 找不到则以 type 兜底
            else:
                _inline_refs(x, index, seen)
        return
    if not isinstance(node, dict):
        return
    if id(node) in seen:
        return
    seen.add(id(node))
    t = node.get("type")
    for key in ("items", "values"):
        v = node.get(key)
        if isinstance(v, str) and v not in _PRIM and v != "null":
            node[key] = index.get(v, {"type": v})
        else:
            _inline_refs(v, index, seen)
    if t in ("record", "error"):
        for f in node.get("fields", []):
            ft = f["type"]
            if isinstance(ft, str) and ft not in _PRIM and ft != "null":
                f["type"] = index.get(ft, {"type": ft})
            else:
                _inline_refs(ft, index, seen)
    elif t == "array":
        _inline_refs(node.get("items"), index, seen)
    elif t == "map":
        _inline_refs(node.get("values"), index, seen)
    elif isinstance(t, dict):
        _inline_refs(t, index, seen)
    elif isinstance(t, list):
        _inline_refs(t, index, seen)


def _residual_refs(node, found=None, seen=None):
    """统计 schema 中仍为非 primitive 的字符串 type 引用（应已全部内联）。"""
    if found is None:
        found = set()
    if seen is None:
        seen = set()
    if isinstance(node, list):
        for x in node:
            if isinstance(x, str):
                if x not in _PRIM and x != "null":
                    found.add(x)
            else:
                _residual_refs(x, found, seen)
        return found
    if not isinstance(node, dict):
        return found
    if id(node) in seen:
        return found
    seen.add(id(node))
    t = node.get("type")
    for key in ("items", "values"):
        v = node.get(key)
        if isinstance(v, str):
            if v not in _PRIM and v != "null":
                found.add(v)
        else:
            _residual_refs(v, found, seen)
    if t in ("record", "error"):
        for f in node.get("fields", []):
            ft = f["type"]
            if isinstance(ft, str) and ft not in _PRIM and ft != "null":
                found.add(ft)
            else:
                _residual_refs(ft, found, seen)
    elif t == "array":
        _residual_refs(node.get("items"), found, seen)
    elif t == "map":
        _residual_refs(node.get("values"), found, seen)
    elif isinstance(t, dict):
        _residual_refs(t, found, seen)
    elif isinstance(t, list):
        _residual_refs(t, found, seen)
    return found


# ---- 编码器 -------------------------------------------------------------
def _fmt_uuid(b: bytes) -> str:
    """bytes(16) -> "XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX" 大写"""
    h = b.hex().upper()
    return f"{h[0:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"


def _fullname(ty: dict) -> str:
    ns = ty.get("namespace") or ""
    return f"{ns}.{ty['name']}" if ns else ty["name"]


def _branch_tag(ty) -> str:
    """union 分支 tag（named 用全名，primitive/complex 用 type 名）。"""
    if isinstance(ty, str):
        return ty
    t0 = ty.get("type")
    if t0 in ("array", "map", "string", "bytes", "int", "long", "float", "double", "boolean"):
        return t0
    nm = ty.get("name")
    if nm:
        ns = ty.get("namespace", "")
        return f"{ns}.{nm}" if ns else nm
    return t0 or ""


def _match(ty, val):
    """从 union 分支列表选出与值匹配的分支（不含 record 分支歧义场景）；无匹配返回 None。"""
    for b in ty:
        if isinstance(b, str):
            if b == "null":
                if val is None:
                    return b
            elif b == "boolean":
                if isinstance(val, bool):
                    return b
            elif b in ("int", "long"):
                if isinstance(val, int) and not isinstance(val, bool):
                    return b
            elif b in ("float", "double"):
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    return b
            elif b == "string":
                if isinstance(val, str):
                    return b
            elif b == "bytes":
                if isinstance(val, (bytes, bytearray)):
                    return b
        elif isinstance(b, dict):
            t0 = b.get("type")
            if t0 == "null":
                if val is None:
                    return b
            elif t0 == "boolean":
                if isinstance(val, bool):
                    return b
            elif t0 in ("int", "long"):
                if isinstance(val, int) and not isinstance(val, bool):
                    return b
            elif t0 in ("float", "double"):
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    return b
            elif t0 in ("string", "enum"):
                if isinstance(val, str):
                    return b
            elif t0 == "bytes":
                if isinstance(val, (bytes, bytearray)):
                    return b
            elif t0 == "fixed":
                if isinstance(val, (bytes, bytearray)):
                    return b
            elif t0 == "array":
                if isinstance(val, list):
                    return b
            elif t0 == "map":
                if isinstance(val, dict):
                    return b
    return None


def inject_ns(node, ns):
    """Avro 命名类型未显式 namespace 时继承 enclosing 定义（原地补全）。

    schema 顶层 name 可能为全名（如 com.bbn.tc.schema.avro.cdm20.TCCDMDatum），
    name 含点则拆出 namespace；record/enum/fixed 若无 namespace 字段则补为当前上下文 ns。
    """
    if isinstance(node, list):
        for x in node:
            inject_ns(x, ns)
        return
    if not isinstance(node, dict):
        return
    t = node.get("type")
    nm = node.get("name")
    cur = ns
    if nm and t in ("record", "error", "enum", "fixed"):
        if "." in nm:
            p, _, s = nm.rpartition(".")
            node["name"] = s
            node["namespace"] = p
            cur = p
        else:
            cur = node.get("namespace") or ns
            if not node.get("namespace"):
                node["namespace"] = ns
    if t in ("record", "error"):
        for f in node.get("fields", []):
            inject_ns(f.get("type"), cur)
    elif t == "array":
        inject_ns(node.get("items"), cur)
    elif t == "map":
        inject_ns(node.get("values"), cur)
    elif isinstance(t, dict):
        inject_ns(t, cur)
    elif isinstance(t, list):
        inject_ns(t, cur)


def make_encoder():
    """返回 (enc, resolve) 闭包；schema 需已 parse_schema + inject_ns。"""
    uuid_b = _fmt_uuid
    _cache = {}

    def _rec_index(types):
        """union 内 record/error 分支：字段名集合 -> 分支；分支按序保留。"""
        key = id(types)
        r = _cache.get(key)
        if r is None:
            by_fields = {}
            recs = []
            for b in types:
                if isinstance(b, dict) and b.get("type") in ("record", "error"):
                    fs = frozenset(f["name"] for f in b.get("fields", ()))
                    by_fields.setdefault(fs, b)
                    recs.append(b)
            r = (by_fields, recs)
            _cache[key] = r
        return r

    def _named_in(types, k):
        for b in types:
            if isinstance(b, dict) and b.get("type") in ("record", "error", "enum", "fixed"):
                if k == b.get("name") or k == _fullname(b):
                    return b
        return None

    def resolve(types, val):
        """union 值 → (branch, unwrapped_val)。branch=None 表示裸 null/无法解析。"""
        if val is None:
            return (None, None)
        if isinstance(val, dict):
            if len(val) == 1:  # 已带 tag 包装（{全名/短名: v} 或 {primitive: v}）
                k = next(iter(val))
                b = _named_in(types, k)
                if b is not None:
                    return (b, val[k])
                if k in _PRIM:
                    return (k, val[k])
                if k in ("array", "map"):
                    for bb in types:
                        if isinstance(bb, dict) and bb.get("type") == k:
                            return (bb, val[k])
                    return (k, val[k])
            # fastavro 裸 record dict → 按字段名集合匹配
            by_fields, recs = _rec_index(types)
            ks = frozenset(val.keys())
            b = by_fields.get(ks)
            if b is not None:
                return (b, val)
            for bb in types:  # map 分支（值 dict 非 record）
                if isinstance(bb, dict) and bb.get("type") == "map":
                    return (bb, val)
            best, best_n = None, -1  # 兜底：重叠字段最多的 record 分支
            for bb in recs:
                n = len(set(f["name"] for f in bb["fields"]) & ks)
                if n > best_n:
                    best_n, best = n, bb
            return (best, val) if best else (None, val)
        b = _match(types, val)  # 非 dict：primitive / fixed / bytes
        if b is None:
            return (None, val)
        return (b, val) if not isinstance(b, str) else (b, val)

    def enc(ty, val):
        if isinstance(ty, str):
            if ty == "string":
                return val
            if ty == "bytes":
                return bytes(val).decode("latin-1")
            if ty in ("int", "long", "float", "double", "boolean", "null"):
                return val
            return val  # 未知（parse_schema 后不应出现）
        if isinstance(ty, list):
            if val is None:
                return None
            b, v = resolve(ty, val)
            if b is None:
                return None
            if isinstance(b, str):
                return {b: enc(b, v)}
            t0 = b.get("type")
            if t0 == "null":
                return None
            return {_branch_tag(b): enc(b, v)}
        # dict：record/enum/fixed/array/map
        t0 = ty.get("type")
        if t0 in ("record", "error"):
            out = {}
            for f in ty["fields"]:
                fn = f["name"]
                out[fn] = enc(f["type"], val.get(fn))
            return out
        if t0 == "enum":
            return val
        if t0 == "fixed":
            bv = bytes(val)
            nm = ty.get("name")
            if nm == "UUID" and len(bv) == 16:
                return uuid_b(bv)
            return bv.decode("latin-1")
        if t0 == "array":
            items = ty["items"]
            return [enc(items, x) for x in val]
        if t0 == "map":
            vs = ty["values"]
            return {k: enc(vs, v) for k, v in val.items()}
        if t0 == "string":
            return val
        if t0 == "bytes":
            return bytes(val).decode("latin-1")
        if t0 in ("int", "long", "float", "double", "boolean"):
            return val
        return val

    return enc, resolve


def convert_file(in_path, out_path, keep=KEEP, verbose=True, limit=None):
    t0 = time.time()
    n_in = n_out = 0
    enc, resolve = make_encoder()
    with gzip.open(in_path, "rb") as fi:
        reader = fastavro.reader(fi)
        schema = fastavro.parse_schema(reader.writer_schema)
        inject_ns(schema, "")
        named_idx = {}
        _collect_named(schema, named_idx)
        _inline_refs(schema, named_idx)
        res = _residual_refs(schema)
        if res:
            sys.exit(f"仍有未解析命名引用: {sorted(res)[:10]}")
        fields = schema["fields"]
        datum_field = next((f["type"] for f in fields if f["name"] == "datum"), None)
        if datum_field is None:
            sys.exit("schema 无 datum 字段，非 TCCDMDatum？")
        with open(out_path, "w", encoding="utf-8") as fo:
            for rec in reader:
                n_in += 1
                if limit and n_in > limit:
                    break
                d = rec.get("datum")
                kind = ""
                if d is not None:
                    b, _v = resolve(datum_field, d)
                    if isinstance(b, dict):
                        kind = b.get("name", "")
                if keep is not None and kind not in keep:
                    continue
                line = enc(schema, rec)
                fo.write(json.dumps(line, ensure_ascii=True, separators=(",", ":")))
                fo.write("\n")
                n_out += 1
    dt = time.time() - t0
    if verbose:
        print(f"[avro2json] {os.path.basename(in_path)}: {n_in} in -> {n_out} out "
              f"({dt:.1f}s, {n_out/dt:.0f} rec/s)", flush=True)
    return n_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path")
    ap.add_argument("--out", dest="out_path")
    ap.add_argument("--in-dir", dest="in_dir")
    ap.add_argument("--out-dir", dest="out_dir")
    ap.add_argument("--all", action="store_true", help="输出全部类型（默认只 4 类）")
    ap.add_argument("--limit", type=int, default=None, help="每文件最多转 N 条（验证用）")
    a = ap.parse_args()
    keep = None if a.all else KEEP

    if a.in_dir:
        os.makedirs(a.out_dir, exist_ok=True)
        files = sorted(f for f in os.listdir(a.in_dir)
                       if ".bin." in f and f.endswith(".gz"))
        total = 0
        for f in files:
            outp = os.path.join(a.out_dir, f[:-3] + ".json")  # .gz 剥掉
            total += convert_file(os.path.join(a.in_dir, f), outp, keep=keep, limit=a.limit)
        print(f"[avro2json] 批量完成: {len(files)} 文件 {total} 行")
    else:
        if not a.in_path or not a.out_path:
            sys.exit("需 --in/--out 或 --in-dir/--out-dir")
        convert_file(a.in_path, a.out_path, keep=keep, limit=a.limit)


if __name__ == "__main__":
    main()
