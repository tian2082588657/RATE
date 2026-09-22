# -*- coding: utf-8 -*-
"""图表.xlsx 就地更新（保留内嵌图表；openpyxl 重存会丢 chart，故直接改 XML）。

用法: python update_xlsx_v4.py  (在 figures/ 目录下运行)
"""
import csv, glob, re, statistics as st, zipfile, shutil, os, sys

SRC = "图表.xlsx"
DST = "图表.xlsx.new"

# ---------- 1. 从 v4 CSV 计算精确均值 ----------
rows = []
for f in glob.glob("../results/v4_final/e1v4_bin*.csv"):
    rows += list(csv.DictReader(open(f, encoding="utf-8")))


def m(fig, enc, fld="node_best_f1"):
    v = [float(r[fld]) for r in rows
         if r["fig"] == fig and r["encoding"] == enc and r.get(fld) not in ("", None)]
    return st.mean(v)


# E1_encoding_matrix 列序: dual_naive, none, rate, rate_single, semantic_only
enc_cols = ["dual_naive", "none", "rate", "rate_single", "semantic_only"]
e1_node = {c: m("tered", c) for c in enc_cols}
e1_alert = {c: m("tered", c, "best_F1_alert") for c in enc_cols}

# E3_strength_sweep
import collections
d3 = collections.defaultdict(lambda: collections.defaultdict(list))
for f in glob.glob("../results/e3redo/*.csv"):
    stn = int(re.match(r"st(\d+)_", os.path.basename(f)).group(1))
    for r in csv.DictReader(open(f, encoding="utf-8")):
        d3[stn][r["encoding"]].append(r)

e3 = {}
for stn, per in d3.items():
    allr = sum(per.values(), [])
    no = sum(int(r["n_nodes_orig"]) for r in allr)
    ng = sum(int(r["n_nodes_Gp"]) for r in allr)
    cuts = [1.0 - int(r["n_edges_Gp"]) / int(r["n_edges_orig"]) for r in allr]
    e3[stn] = {
        "none": st.mean(float(r["node_best_f1"]) for r in per.get("none", [])),
        "rate": st.mean(float(r["node_best_f1"]) for r in per.get("rate", [])),
        "nodes_kept": ng / no,
        "edge_cut": st.mean(cuts),
    }

# ---------- 2. 定位 sheet ----------
z = zipfile.ZipFile(SRC)
names = z.namelist()
sheet_xml = {}
for n in names:
    if n.startswith("xl/worksheets/sheet"):
        sheet_xml[n] = z.read(n).decode("utf-8")


def first_text(xml):
    mm = re.search(r"<t>([^<]*)</t>", xml)
    return mm.group(1) if mm else ""


target = {}
for n, xml in sheet_xml.items():
    target[first_text(xml)] = n
print("sheets:", target)


def set_cell(xml, ref, val):
    """把 r=ref 的数值单元格替换为新值；val=None 则清空。"""
    pat = re.compile(r'(<c r="%s"[^>]*?)(?:><v>[^<]*</v></c>|></c>)' % ref)
    if val is None:
        out, k = pat.subn(lambda mo: mo.group(1) + "/>", xml, count=1)
    else:
        pat2 = re.compile(r'(<c r="%s"[^>]*?)><v>[^<]*</v>' % ref)
        out, k = pat2.subn(lambda mo: mo.group(1) + "><v>%.10g</v>" % val, xml, count=1)
    if k != 1:
        raise SystemExit("cell %s not replaced (k=%d)" % (ref, k))
    return out


# sheet1 = E1_encoding_matrix (tered 行 = r3)
s = sheet_xml["xl/worksheets/sheet1.xml"]
for i, c in enumerate(enc_cols):
    s = set_cell(s, chr(ord("B") + i) + "3", e1_node[c])
s = set_cell(s, "G3", st.mean(list(e1_node.values())))
sheet_xml["xl/worksheets/sheet1.xml"] = s

# sheet2 = E1_alert (tered 行 = r3)
s = sheet_xml["xl/worksheets/sheet2.xml"]
for i, c in enumerate(enc_cols):
    s = set_cell(s, chr(ord("B") + i) + "3", e1_alert[c])
s = set_cell(s, "G3", st.mean(list(e1_alert.values())))
sheet_xml["xl/worksheets/sheet2.xml"] = s

# sheet3 = E3_strength_sweep
s = sheet_xml["xl/worksheets/sheet3.xml"]
order = [250, 1000, 4000, 16000]
for i, stn in enumerate(order):
    r = str(i + 2)
    s = set_cell(s, "B" + r, None)              # dual_naive：本轮扫描未跑
    s = set_cell(s, "C" + r, e3[stn]["none"])
    s = set_cell(s, "D" + r, e3[stn]["rate"])
    s = set_cell(s, "E" + r, e3[stn]["nodes_kept"])
    s = set_cell(s, "F" + r, e3[stn]["edge_cut"])
sheet_xml["xl/worksheets/sheet3.xml"] = s

# ---------- 3. 重打包 ----------
zo = zipfile.ZipFile(DST, "w", zipfile.ZIP_DEFLATED)
for item in z.infolist():
    data = z.read(item.filename)
    if item.filename in sheet_xml:
        data = sheet_xml[item.filename].encode("utf-8")
    zo.writestr(item, data)
zo.close()
z.close()
shutil.move(DST, SRC)
print("updated", SRC)
