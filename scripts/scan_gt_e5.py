# -*- coding: utf-8 -*-
"""scripts/scan_gt_e5.py — E5 cadets 全量文件扫描 ground truth UUID 命中。

用法: python scripts/scan_gt_e5.py <gt_dir> <data_dir> <out_json> [workers]
gt_dir 含 node_Nginx_Drakon_APT*.csv (UUID,attributes,index_id 每行一条)。
输出: {file: {"hits": [...], "ts_min": .., "ts_max": .., "n_records": ..}}
"""
from __future__ import annotations
import sys, os, glob, json, gzip, uuid as _uuid
from multiprocessing import Pool

GT = set()
DATA_DIR = ""
CTX = {"gt": None}


def load_gt(gt_dir):
    gt = set()
    for f in sorted(glob.glob(os.path.join(gt_dir, "node_*.csv"))):
        for line in open(f, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            u = line.split(",")[0].strip()
            if u:
                gt.add(u.lower())
    return gt


def _scalar_uuid(v):
    """Avro CDM20 uuid 字段(bytes16/str/dict union) -> 小写字符串或 None。"""
    if isinstance(v, (bytes, bytearray)):
        if len(v) == 16:
            try:
                return str(_uuid.UUID(bytes=bytes(v))).lower()
            except Exception:
                return None
        return None
    if isinstance(v, str):
        return v.lower()
    if isinstance(v, dict):
        for k in ("string",):
            if k in v:
                return str(v[k]).lower()
    return None


def scan_file(path):
    import fastavro
    hits = []
    ts_min, ts_max = None, None
    n = 0
    try:
        with gzip.open(path, "rb") as f:
            for rec in fastavro.reader(f):
                n += 1
                d = rec.get("datum") if isinstance(rec, dict) else None
                body = rec if not isinstance(d, dict) else None
                # 形态一 {"datum": {com.bbn...Event: {...}}}
                if isinstance(d, dict) and len(d) == 1 and "." in list(d.keys())[0]:
                    body = list(d.values())[0]
                elif isinstance(d, dict):
                    body = d
                if not isinstance(body, dict):
                    continue
                u = _scalar_uuid(body.get("uuid"))
                if u is not None and u in CTX["gt"]:
                    hits.append(u)
                # 时间范围取 Event.timestampNanos
                if ts_min is None or (rec_kind(body) == "ev"):
                    pass
                t = body.get("timestampNanos")
                if t is not None:
                    if isinstance(t, dict):
                        t = t.get("long") or t.get("int") or t.get("double")
                    try:
                        t = int(t)
                        if ts_min is None or t < ts_min:
                            ts_min = t
                        if ts_max is None or t > ts_max:
                            ts_max = t
                    except Exception:
                        pass
    except Exception as e:
        return {"file": os.path.basename(path), "error": repr(e)}
    return {"file": os.path.basename(path), "hits": sorted(set(hits)),
            "n_hit": len(set(hits)), "ts_min": ts_min, "ts_max": ts_max,
            "n_records": n}


def rec_kind(body):
    return "ev" if "subject" in body else "obj"


def main():
    gt_dir, data_dir, out_json = sys.argv[1], sys.argv[2], sys.argv[3]
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else 16
    global GT, DATA_DIR
    GT = load_gt(gt_dir)
    CTX["gt"] = GT
    files = sorted(glob.glob(os.path.join(data_dir, "*.gz")))
    # 按记录号自然顺序: bin.gz, bin.1.gz, bin.10.gz... 排序后按数字重排
    import re
    def key(p):
        b = os.path.basename(p)
        m = re.search(r"\.bin(?:\.(\d+))?\.gz$", b)
        return int(m.group(1)) if m and m.group(1) else -1
    files.sort(key=key)
    print(f"[scan] {len(files)} files, GT={len(GT)} uuids, workers={workers}",
          flush=True)
    res = []
    with Pool(workers, initializer=_init, initargs=(GT,)) as p:
        for i, r in enumerate(p.imap_unordered(_scan, files, chunksize=1)):
            n_hit = r.get("n_hit", 0)
            if n_hit or "error" in r:
                print(f"[scan] {r['file']}: hits={n_hit} "
                      f"{'ERR '+r['error'] if 'error' in r else ''}", flush=True)
            res.append(r)
            if (i + 1) % 10 == 0:
                print(f"[scan] ... {i+1}/{len(files)} done", flush=True)
    order = {f: i for i, f in enumerate(files)}
    res.sort(key=lambda r: order.get(os.path.join(data_dir, r["file"]), 9999))
    json.dump(res, open(out_json, "w"), indent=1)
    tot = sum(r.get("n_hit", 0) for r in res)
    hitf = [r["file"] for r in res if r.get("n_hit")]
    print(f"[scan] DONE total_hits={tot} files_with_hits={len(hitf)}: {hitf}",
          flush=True)


def _init(gt):
    CTX["gt"] = gt


def _scan(path):
    return scan_file(path)


if __name__ == "__main__":
    main()
