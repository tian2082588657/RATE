# -*- coding: utf-8 -*-
"""probe_gt_def.py — 探查 GT uuid 在 bin 中的记录形态。

对指定 bin，定位指定 uuid 的出现上下文：
  - DEF：某 record 的顶层 "uuid" 字段 == 目标（节点定义记录）→ 打印该 record 顶层键集合
  - REF：某 record 的 subject / predicateObject 字段 == 目标（事件引用）
输出每类首个样本的键集合，用于判断 build_db 为何漏定义。
用法: ~/pids-lite/bin/python probe_gt_def.py --bin <path> --uuid <U1[,U2...]>
"""
from __future__ import annotations
import argparse, gzip, sys
import fastavro


def norm_uuid(v):
    if isinstance(v, (bytes, bytearray)) and len(v) == 16:
        h = v.hex().upper()
        return f"{h[0:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"
    if isinstance(v, str):
        s = v.strip().upper()
        if len(s) == 36 and s.count("-") == 4:
            return s
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--uuid", required=True)
    a = ap.parse_args()
    targets = {u.strip().upper() for u in a.uuid.split(",")}
    print(f"[probe] 目标 {len(targets)}: {sorted(targets)[:3]}...", flush=True)

    found = {u: {"def": 0, "ref": 0, "def_keys": None} for u in targets}
    n = 0
    with gzip.open(a.bin, "rb") as fi:
        reader = fastavro.reader(fi)
        for datum in reader:
            n += 1
            rec = datum
            if isinstance(rec, dict):
                d = rec.get("datum")
                if isinstance(d, dict):
                    rec = d
            if not isinstance(rec, dict):
                continue
            u = norm_uuid(rec.get("uuid"))
            if u in found:
                st = found[u]
                st["def"] += 1
                if st["def_keys"] is None:
                    st["def_keys"] = sorted(rec.keys())[:20]
                    print(f"[DEF] {u} keys={st['def_keys']}", flush=True)
            for k in ("subject", "predicateObject"):
                v = rec.get(k)
                if isinstance(v, (bytes, bytearray, str)):
                    ru = norm_uuid(v)
                    if ru in found:
                        st = found[ru]
                        if st["def"] == 0 and st["ref"] < 1:
                            print(f"[REF] {ru} via Event.{k} (rec keys={sorted(rec.keys())[:12]})", flush=True)
                        st["ref"] += 1
            if n % 2000000 == 0:
                print(f"  ...{n} datum", flush=True)
    print(f"[done] 总 {n} datum")
    for u in sorted(found):
        st = found[u]
        print(f"[sum] {u}: def={st['def']} ref={st['ref']}")


if __name__ == "__main__":
    main()
