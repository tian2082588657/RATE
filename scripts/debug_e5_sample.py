# -*- coding: utf-8 -*-
"""scripts/debug_e5_sample.py — 抽样打印 E5 theia(ndjson) 与 E5 cadets(Avro) 的原始记录形态。"""
from __future__ import annotations
import gzip, json, os, sys

import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), *[".."] * 1))

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def sample_ndjson(path, n=6):
    print(f"===== {path} 前 {n} 行 ndjson =====")
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            line = line.strip()
            if line:
                try:
                    rec = json.loads(line)
                    print(f"[{i}] keys={list(rec.keys())}")
                    print(json.dumps(rec, ensure_ascii=False)[:600])
                except Exception as e:
                    print(f"[{i}] JSON error: {e}: {line[:200]}")

def sample_avro(path, n=6):
    print(f"===== {path} 前 {n} 条 Avro =====")
    import fastavro
    with gzip.open(path, "rb") as f:
        reader = fastavro.reader(f)
        print("schema name:", reader.writer_schema.get("name"))
        print("schema fields:", [fl.get("name") for fl in reader.writer_schema.get("fields", [])])
        for i, rec in enumerate(reader):
            if i >= n:
                break
            print(f"[{i}] keys={list(rec.keys())}")
            print(json.dumps(rec, ensure_ascii=False, default=str)[:800])

if __name__ == "__main__":
    sample_ndjson(_os.path.join(_os.path.dirname(_R), "dataset", "darpa e5", "theia", "ta1-theia-1-e5-official-1.json.1.gz"))
    print()
    sample_avro(_os.path.join(_os.path.dirname(_R), "dataset", "darpa e5", "cadets", "ta1-cadets-1-e5-official-2.bin.1.gz"))
