# -*- coding: utf-8 -*-
"""build_db_cadets.py — DARPA TC E5 CADETS 子集建库（自包含 fork）。

语义与 PIDSMaker `create_database_e5.py` 完全一致（正则抽取 + 建 4 表 + sha256 hash），
但去除对 pidsmaker.utils 的 import（其顶层 import torch），使本脚本只需轻量依赖
（psycopg2-binary / pyyaml / tqdm），可在系统 python3 直跑。

用法:
  python3 build_db_cadets.py \
      --raw-dir /TeRed+RATE/dataset/darpae5/cadets_json \
      --db cadets_e5 --host localhost --port 5432 \
      --user postgres --password postgres \
      [--fl-dir ~/pidsmaker/dataset_preprocessing/darpa_tc] \
      [--dataset CADETS_E5] [--dry-run]

说明:
  - 文件清单默认取 PIDSMaker filelist.py 的 CADETS_E5 list，并过滤 os.path.exists
    （即只处理已转换出的 .json 文件，跳过缺失分片 .json.1/.json.2）。
  - 四遍扫描与 create_database_e5.py 相同：netflow → subject → file → event。
  - cadets 数据特性（实测 bin.1 前 30 万条）：Subject.cmdLine 全为 null、properties 全空、
    FileObject 无 filename 字段（该 CDM 版 FileObject 只有 baseObject{permission,epoch,properties}）。
    官方正则（要求 cmdLine+path / filename 字面量）只适配 theia/trace 风格数据，对 cadets 会
    导致 subject/file 表为空、event 边全部无法解析。仓库下游（训练/构图）只读
    uuid→hash_id→index_id 映射与 event_table 边，不读 path/cmd/src_addr 等文本列，且
    init DDL 中这些列均可空 → 默认在官方正则未命中时**兜底插入 uuid-only 行**（hash 规则不变），
    保证 cadets 图完整；--strict 可复刻官方原版行为。
"""
from __future__ import annotations
import argparse, hashlib, os, re as _re, sys, time

import psycopg2
from psycopg2 import extras as ex
from tqdm import tqdm

# ---- 从 pidsmaker.utils 内联的工具函数（语义一致） --------------------------
def stringtomd5(originstr: str) -> str:
    """PIDSMaker 的 'stringtomd5'：实际为 SHA-256 hexdigest。"""
    return hashlib.sha256(originstr.encode("utf-8")).hexdigest()


edge_reversed = [
    "EVENT_EXECUTE", "EVENT_LSEEK", "EVENT_MMAP", "EVENT_OPEN", "EVENT_ACCEPT",
    "EVENT_READ", "EVENT_RECVFROM", "EVENT_RECVMSG", "EVENT_READ_SOCKET_PARAMS",
    "EVENT_CHECK_FILE_ATTRIBUTES", "READ",
]

exclude_edge_type = {
    "EVENT_FCNTL", "EVENT_OTHER", "EVENT_ADD_OBJECT_ATTRIBUTE", "EVENT_FLOWS_TO",
}


# ---- 四个入库阶段（与 create_database_e5.py 逐行同构） ----------------------
def store_netflow(file_path, cur, connect, index_id, flist, strict=False):
    netobjset = set()
    netobj2hash = {}
    n_full = n_fb = 0
    for file in tqdm(flist):
        with open(file_path + file, "r") as f:
            for line in f:
                if "NetFlowObject" in line:
                    try:
                        res = re_findall_netflow(line)
                        nodeid, srcaddr, srcport, dstaddr, dstport = res
                        nodeproperty = srcaddr + "," + srcport + "," + dstaddr + "," + dstport
                        hashstr = stringtomd5(nodeid)
                        netobj2hash[nodeid] = [hashstr, nodeproperty]
                        netobj2hash[hashstr] = nodeid
                        netobjset.add(hashstr)
                        n_full += 1
                    except Exception:
                        if not strict:
                            u = re_netflow_uuid(line)
                            if u and u not in netobj2hash:
                                netobj2hash[u] = [stringtomd5(u), None]
                                n_fb += 1
    datalist = []
    net_uuid2hash = {}
    for i in netobj2hash.keys():
        if len(i) != 64:
            prop = netobj2hash[i][1]
            if prop:  # 完整行: uuid, hash, addr,port,addr,port, index
                datalist.append([i] + [netobj2hash[i][0]] + prop.split(",") + [index_id])
            else:  # 兜底行: uuid, hash, 空地址, index
                datalist.append([i, netobj2hash[i][0], None, None, None, None, index_id])
            net_uuid2hash[i] = netobj2hash[i][0]
            index_id += 1
    sql = "insert into netflow_node_table values %s on conflict do nothing"
    ex.execute_values(cur, sql, datalist, page_size=10000)
    connect.commit()
    print(f"[netflow] {len(datalist)} 行入库 (full={n_full}, fallback={n_fb})", flush=True)
    return index_id, net_uuid2hash


def store_subject(file_path, cur, connect, index_id, flist, strict=False):
    subject_obj2hash = {}
    n_full = n_fb = 0
    fail_count = 0
    for file in tqdm(flist):
        with open(file_path + file, "r") as f:
            for line in f:
                if "schema.avro.cdm20.Subject" in line:
                    m = re_findall_subject(line)
                    if m:
                        try:
                            subject_obj2hash[m[0]] = [m[-1], m[-3]]  # {uuid:[path, cmd]}
                            n_full += 1
                        except Exception:
                            fail_count += 1
                    elif not strict:
                        u = re_subject_obj_uuid(line)
                        if u and u not in subject_obj2hash:
                            subject_obj2hash[u] = [None, None]  # uuid-only 兜底行
                            n_fb += 1
    datalist = []
    subject_uuid2hash = {}
    for i in subject_obj2hash.keys():
        if len(i) != 64:
            v = subject_obj2hash[i]
            if isinstance(v, list):
                datalist.append([i, stringtomd5(i), v[0], v[1], index_id])
            else:
                datalist.append([i, stringtomd5(i), None, None, index_id])
            subject_uuid2hash[i] = stringtomd5(i)
            index_id += 1
    sql = "insert into subject_node_table values %s on conflict do nothing"
    ex.execute_values(cur, sql, datalist, page_size=10000)
    connect.commit()
    print(f"[subject] {len(datalist)} 行入库 (full={n_full}, fallback={n_fb}, fail={fail_count})", flush=True)
    return index_id, subject_uuid2hash


def store_file(file_path, cur, connect, index_id, flist, strict=False):
    file_obj2hash = {}
    n_full = n_fb = 0
    fail_count = 0
    for file in tqdm(flist):
        with open(file_path + file, "r") as f:
            for line in f:
                if "avro.cdm20.FileObject" in line:
                    try:
                        m = re_findall_file(line)
                        file_obj2hash[m[0]] = m[-1]
                        n_full += 1
                    except Exception:
                        if not strict:
                            u = re_file_uuid(line)
                            if u and u not in file_obj2hash:
                                file_obj2hash[u] = None  # uuid-only 兜底行
                                n_fb += 1
                        fail_count += 1
    datalist = []
    file_uuid2hash = {}
    for i in file_obj2hash.keys():
        if len(i) != 64:
            datalist.append([i, stringtomd5(i), file_obj2hash[i], index_id])
            file_uuid2hash[i] = stringtomd5(i)
            index_id += 1
    sql = "insert into file_node_table values %s on conflict do nothing"
    ex.execute_values(cur, sql, datalist, page_size=10000)
    connect.commit()
    print(f"[file] {len(datalist)} 行入库 (full={n_full}, fallback={n_fb}, fail={fail_count})", flush=True)
    return index_id, file_uuid2hash


def create_node_list(cur):
    nodeid2msg = {}
    for tbl in ("netflow_node_table", "subject_node_table", "file_node_table"):
        cur.execute(f"select * from {tbl}")
        for i in cur.fetchall():
            nodeid2msg[i[1]] = i[-1]
    return nodeid2msg


def store_event(file_path, cur, connect, reverse, nodeid2msg,
                subject_uuid2hash, file_uuid2hash, net_uuid2hash, flist):
    datalist = []
    n_hit = n_skip = 0
    for file in tqdm(flist):
        with open(file_path + file, "r") as f:
            for line in f:
                if '{"datum":{"com.bbn.tc.schema.avro.cdm20.Event"' in line:
                    rel = re_etype(line)
                    if not rel or rel in exclude_edge_type:
                        n_skip += 1
                        continue
                    su = re_subject_uuid(line)
                    pu = re_object_uuid(line)
                    if su and pu and su[0] in subject_uuid2hash and (
                            pu[0] in subject_uuid2hash or pu[0] in file_uuid2hash
                            or pu[0] in net_uuid2hash):
                        ev = re_event_uuid(line)
                        ts = re_ts(line)
                        if ev is None or ts is None:
                            continue
                        subjectId = subject_uuid2hash[su[0]]
                        if pu[0] in file_uuid2hash:
                            objectId = file_uuid2hash[pu[0]]
                        elif pu[0] in net_uuid2hash:
                            objectId = net_uuid2hash[pu[0]]
                        else:
                            objectId = subject_uuid2hash[pu[0]]
                        if rel in reverse:
                            datalist.append([objectId, nodeid2msg.get(objectId),
                                             rel, subjectId, nodeid2msg.get(subjectId), ev, ts])
                        else:
                            datalist.append([subjectId, nodeid2msg.get(subjectId),
                                             rel, objectId, nodeid2msg.get(objectId), ev, ts])
                        n_hit += 1
                        if len(datalist) >= 50000:
                            write_event_in_DB(cur, connect, datalist)
                            datalist = []
    if datalist:
        write_event_in_DB(cur, connect, datalist)
    print(f"[event] {n_hit} 边入库 (skip={n_skip})", flush=True)


def write_event_in_DB(cur, connect, datalist):
    sql = "insert into event_table values %s on conflict do nothing"
    ex.execute_values(cur, sql, datalist, page_size=50000)
    connect.commit()


# ---- 正则（与 create_database_e5.py 完全一致） ------------------------------
def re_findall_netflow(line):
    return _re.findall(
        'NetFlowObject":{"uuid":"(.*?)"(.*?)"localAddress":{"string":"(.*?)"},'
        '"localPort":{"int":(.*?)},"remoteAddress":{"string":"(.*?)"},'
        '"remotePort":{"int":(.*?)}', line)[0]


def re_findall_subject(line):
    m = _re.findall(
        'avro.cdm20.Subject":{"uuid":"(.*?)"(.*?)"cmdLine":{"string":"(.*?)"}(.*?)"path":"(.*?)"',
        line)
    return m[0] if m else None


def re_findall_file(line):
    m = _re.findall('avro.cdm20.FileObject":{"uuid":"(.*?)",(.*?)"filename":"(.*?)"', line)
    return m[0] if m else None


def re_etype(line):
    m = _re.findall('"type":"(.*?)"', line)
    return m[0] if m else None


def re_subject_uuid(line):
    return _re.findall('"subject":{"com.bbn.tc.schema.avro.cdm20.UUID":"(.*?)"', line)


def re_object_uuid(line):
    return _re.findall('"predicateObject":{"com.bbn.tc.schema.avro.cdm20.UUID":"(.*?)"', line)


def re_event_uuid(line):
    m = _re.findall('{"datum":{"com.bbn.tc.schema.avro.cdm20.Event":{"uuid":"(.*?)",', line)
    return m[0] if m else None


# ---- cadets 数据兜底用 uuid 正则（官方 create_database_e5.py 无，见模块 docstring）----
def re_subject_obj_uuid(line):
    m = _re.findall('avro.cdm20.Subject":{"uuid":"(.*?)",', line)
    return m[0] if m else None


def re_file_uuid(line):
    m = _re.findall('avro.cdm20.FileObject":{"uuid":"(.*?)",', line)
    return m[0] if m else None


def re_netflow_uuid(line):
    m = _re.findall('NetFlowObject":{"uuid":"(.*?)",', line)
    return m[0] if m else None


def re_ts(line):
    m = _re.findall('"timestampNanos":(.*?),', line)
    if not m:
        return None
    try:
        return int(m[0])
    except Exception:
        return None


# ---- 主流程 ----------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True, help="转换后 json 所在目录（尾斜杠可选）")
    ap.add_argument("--db", default="cadets_e5")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=5432)
    ap.add_argument("--user", default="postgres")
    ap.add_argument("--password", default="postgres")
    ap.add_argument("--fl-dir", default=os.path.expanduser("~/pidsmaker/dataset_preprocessing/darpa_tc"),
                    help="filelist.py 所在目录")
    ap.add_argument("--dataset", default="CADETS_E5")
    ap.add_argument("--strict", action="store_true",
                    help="严格复刻官方 create_database_e5.py：Subject/File/NetFlow 行仅当官方正则命中才入库")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    raw_dir = a.raw_dir if a.raw_dir.endswith("/") else a.raw_dir + "/"

    # 载入 PIDSMaker filelist（纯数据模块）并过滤存在的文件
    sys.path.insert(0, a.fl_dir)
    import filelist as _fl
    fl_all = _fl.get_filelist(a.dataset)
    flist = [f for f in fl_all if os.path.exists(raw_dir + f)]
    print(f"[filelist] {a.dataset}: 全 {len(fl_all)} 条 -> 现存 {len(flist)} 条", flush=True)
    if not flist:
        sys.exit(f"raw_dir 下无匹配文件: {raw_dir}")
    if a.dry_run:
        for f in flist[:10]:
            print("  ", f)
        print("  ... (dry-run 结束)")
        return

    connect = psycopg2.connect(database=a.db, host=a.host, user=a.user,
                               password=a.password, port=a.port)
    cur = connect.cursor()

    index_id = 0
    t0 = time.time()
    index_id, net_uuid2hash = store_netflow(raw_dir, cur, connect, index_id, flist, strict=a.strict)
    index_id, subject_uuid2hash = store_subject(raw_dir, cur, connect, index_id, flist, strict=a.strict)
    index_id, file_uuid2hash = store_file(raw_dir, cur, connect, index_id, flist, strict=a.strict)
    nodeid2msg = create_node_list(cur)
    store_event(raw_dir, cur, connect, edge_reversed, nodeid2msg,
                subject_uuid2hash, file_uuid2hash, net_uuid2hash, flist)
    cur.close()
    connect.close()
    print(f"[build_db] 完成，总耗时 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
