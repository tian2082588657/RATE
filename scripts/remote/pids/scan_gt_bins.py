# -*- coding: utf-8 -*-
"""scan_gt_bins.py — 在全部 122 个原始 cadets avro bin 中定位 GT 恶意节点 UUID 的落点。

对每个 .bin.*.gz（fastavro 流式解码）统计：该文件中出现过的 GT uuid 集合。
命中来源覆盖三类（保守全收，决策端"命中即需补灌"）：
  1. 节点定义记录：Subject/FileObject/NetFlowObject 记录自带的 "uuid" 字段（FileObject 在 baseObject.uuid）
  2. Event 引用：Event 记录的 subject / predicateObject 字段（union[null,UUID] → bytes16）
  3. 嵌套兜底：任一 dict 的 "uuid" 键（避免 schema 变体漏判）

输出（--out）：每行  bin文件 basename \t 命中数 \t uuid1,uuid2,...
用法:
  ~/pids-lite/bin/python scan_gt_bins.py \
      --gt ~/gt_missing.txt --data /TeRed+RATE/dataset/darpae5/cadets --out ~/gt_bin_hits.tsv
"""
from __future__ import annotations
import argparse, gzip, os, sys

import fastavro


def norm_uuid(v):
    """bytes16 / 大写hex字符串 -> 大写连字符 uuid；其余返回 None。"""
    if isinstance(v, (bytes, bytearray)) and len(v) == 16:
        h = v.hex().upper()
        return f"{h[0:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"
    if isinstance(v, str):
        s = v.strip().upper()
        if len(s) == 36 and s.count("-") == 4:
            return s
    return None


def scan_file(path, gt):
    hits = set()
    with gzip.open(path, "rb") as fi:
        reader = fastavro.reader(fi)
        for datum in reader:
            # datum 顶层 record（TCCDMDatum）可能包一层 "datum" 或直接是分支 record
            rec = datum
            if isinstance(rec, dict):
                d = rec.get("datum")
                if isinstance(d, dict):
                    rec = d
            if not isinstance(rec, dict):
                continue
            # 1) node/event 自带的 uuid 键
            u = norm_uuid(rec.get("uuid"))
            if u in gt:
                hits.add(u)
            # 2) FileObject -> baseObject.uuid
            bo = rec.get("baseObject")
            if isinstance(bo, dict):
                u = norm_uuid(bo.get("uuid"))
                if u in gt:
                    hits.add(u)
            # 3) Event 引用 subject / predicateObject（UUID union 解码为 bytes16 或 None）
            for k in ("subject", "predicateObject"):
                v = rec.get(k)
                if isinstance(v, (bytes, bytearray, str)):
                    u = norm_uuid(v)
                    if u in gt:
                        hits.add(u)
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True, help="GT uuid 列表（每行 attack1/attack2\\tUUID）")
    ap.add_argument("--data", default="", help="cadets bin 目录（--file 模式可空）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--file", default="", help="只扫这一个文件（并行分片用）")
    ap.add_argument("--limit", type=int, default=0, help="只扫前 N 个文件（调试用）")
    a = ap.parse_args()

    gt = set()
    with open(a.gt) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            gt.add(parts[-1].strip().upper())
    print(f"[gt] {len(gt)} 个目标 uuid", flush=True)

    if a.file:
        files = [a.file]  # 全路径，直接扫
    elif a.data:
        files = sorted(f for f in os.listdir(a.data) if ".bin." in f and f.endswith(".gz"))
    else:
        sys.exit("需提供 --data 或 --file")
    print(f"[data] {len(files)} 个 bin 文件", flush=True)
    if a.limit:
        files = files[:a.limit]

    out = open(a.out, "w")
    t_all = {}
    for i, f in enumerate(files, 1):
        path = os.path.join(a.data, f) if a.data else f
        hits = scan_file(path, gt)
        if hits:
            line = f"{f}\t{len(hits)}\t{','.join(sorted(hits))}"
        else:
            line = f"{f}\t0\t-"
        out.write(line + "\n")
        out.flush()
        for u in hits:
            t_all[u] = t_all.get(u, []) + [f]
        print(f"[{i}/{len(files)}] {f}: {len(hits)} 命中", flush=True)
    out.close()
    print("[done] 明细 ->", a.out, flush=True)


if __name__ == "__main__":
    main()
