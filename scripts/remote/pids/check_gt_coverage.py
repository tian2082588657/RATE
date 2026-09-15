# -*- coding: utf-8 -*-
"""check_gt_coverage.py — 检查 cadets_e5 库内已覆盖的 GT 恶意节点。

读 PIDSMaker Ground_Truth orthrus 版两个 GT CSV（attack1: node_Nginx_Drakon_APT.csv,
attack2: node_Nginx_Drakon_APT_17.csv），对三张节点表统计 uuid 命中，输出：
  - 每组 GT 的节点总数 / 已入库数 / 缺失 uuid 列表（存 --out 文件）
用法:
  python3 check_gt_coverage.py [--out ~/gt_missing.txt]
"""
from __future__ import annotations
import argparse, os

GT_FILES = {
    "attack1": "/home/user/pidsmaker/Ground_Truth/orthrus/E5-CADETS/node_Nginx_Drakon_APT.csv",
    "attack2": "/home/user/pidsmaker/Ground_Truth/orthrus/E5-CADETS/node_Nginx_Drakon_APT_17.csv",
}
TABLES = ["subject_node_table", "file_node_table", "netflow_node_table"]


def read_gt(path):
    uuids = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            u = line.split(",")[0].strip().strip('"')
            if u and u.lower() != "uuid":
                uuids.append(u.upper())
    return uuids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.expanduser("~/gt_missing.txt"))
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=5432)
    ap.add_argument("--user", default="postgres")
    ap.add_argument("--password", default="postgres")
    ap.add_argument("--db", default="cadets_e5")
    a = ap.parse_args()

    import psycopg2
    conn = psycopg2.connect(database=a.db, host=a.host, user=a.user,
                            password=a.password, port=a.port)
    cur = conn.cursor()
    missing = {}  # group -> [uuid,...]
    for grp, path in GT_FILES.items():
        uuids = read_gt(path)
        found = set()
        for tbl in TABLES:
            for i in range(0, len(uuids), 500):
                chunk = uuids[i:i + 500]
                ph = ",".join(["%s"] * len(chunk))
                cur.execute(f"select node_uuid from {tbl} where node_uuid in ({ph})", chunk)
                found.update(r[0].upper() for r in cur.fetchall())
        miss = [u for u in uuids if u not in found]
        missing[grp] = miss
        print(f"[{grp}] GT 节点 {len(uuids)} 个 | 已覆盖 {len(found)} | 缺失 {len(miss)}", flush=True)
    with open(a.out, "w") as f:
        for grp in missing:
            for u in missing[grp]:
                f.write(f"{grp}\t{u}\n")
    print(f"[done] 缺失 uuid 列表 -> {a.out}", flush=True)
    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
