# -*- coding: utf-8 -*-
"""figures/make_excel.py — 把实验 CSV 汇总成带原生图表的 Excel 工作簿。

用法:
  python figures/make_excel.py --results-dir <目录> --out 图表.xlsx

生成 sheet:
  E1_encoding_matrix   图来源 x 编码 -> 节点级检测质量（含柱状图）
  E1_alert             同上，告警级指标
  E2_slot_equivalence  槽位级等价性（配对）
  E3_strength_sweep    归约强度 vs 归约率 vs 检测质量（含折线图 = 主图）
  StreamSpot           第二数据集
每个 sheet 先落原始聚合表，再放原生图表对象。
"""
from __future__ import annotations
import os, sys, re, csv, glob, argparse
from collections import defaultdict

try:
    from openpyxl import Workbook
    from openpyxl.chart import BarChart, LineChart, Reference, ScatterChart, Series
    from openpyxl.styles import Font, Alignment, PatternFill
    from openpyxl.utils import get_column_letter
except ImportError:
    print("需要 openpyxl: pip install openpyxl", file=sys.stderr)
    raise

HDR_FILL = PatternFill("solid", fgColor="D9E1F2")
HDR_FONT = Font(bold=True, size=11)


def read_csvs(pattern):
    rows = []
    for p in sorted(glob.glob(pattern)):
        try:
            with open(p, "r", encoding="utf-8", newline="") as fh:
                for r in csv.DictReader(fh):
                    r["_src"] = os.path.basename(p)
                    rows.append(r)
        except Exception as e:
            print("skip %s: %s" % (p, e))
    return rows


def fnum(r, k):
    try:
        v = r.get(k, "")
        return float(v) if v not in ("", None, "NA", "nan") else None
    except (TypeError, ValueError):
        return None


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def write_table(ws, header, rows, num_fmt="0.000"):
    ws.append(header)
    for c in range(1, len(header) + 1):
        cell = ws.cell(row=1, column=c)
        cell.fill = HDR_FILL
        cell.font = HDR_FONT
        cell.alignment = Alignment(horizontal="center", wrap_text=True)
    for r in rows:
        ws.append(r)
    for r in range(2, ws.max_row + 1):
        for c in range(1, len(header) + 1):
            cell = ws.cell(row=r, column=c)
            if isinstance(cell.value, float):
                cell.number_format = num_fmt
    for i, h in enumerate(header, 1):
        w = max(11, min(24, len(str(h)) + 4))
        ws.column_dimensions[get_column_letter(i)].width = w
    ws.freeze_panes = "A2"


def sheet_e1(wb, rows, fig_col="fig", title="E1_encoding_matrix",
             metric="node_best_f1", extra=("node_cov", "node_auc")):
    if not rows:
        return
    encs = sorted({r.get("encoding", "?") for r in rows})
    figs = sorted({r.get(fig_col, "?") for r in rows})
    ws = wb.create_sheet(title)
    header = row_head = ["figure \\ encoding"] + encs + ["MEAN"]
    table = []
    for fg in figs:
        line = [fg]
        vals = []
        for en in encs:
            m = mean([fnum(r, metric) for r in rows
                      if r.get(fig_col) == fg and r.get("encoding") == en])
            line.append(m if m is not None else "")
            vals.append(m)
        line.append(mean(vals))
        table.append(line)
    write_table(ws, header, table)
    ch = BarChart()
    ch.type = "col"
    ch.title = "%s (%s)" % (metric, title)
    ch.y_axis.title = metric
    ch.x_axis.title = "figure"
    data = Reference(ws, min_col=2, max_col=1 + len(encs), min_row=1,
                     max_row=1 + len(figs))
    cats = Reference(ws, min_col=1, min_row=2, max_row=1 + len(figs))
    ch.add_data(data, titles_from_data=True)
    ch.set_categories(cats)
    ch.height, ch.width = 9, 16
    ws.add_chart(ch, "A%d" % (ws.max_row + 3))
    return ws


def _edge_cut(r):
    """边归约率 = 1 - n_edges_Gp / n_edges_orig；缺列时回退到显式 edge_cut。"""
    ec = fnum(r, "edge_cut")
    if ec is not None:
        return ec
    gp, orig = fnum(r, "n_edges_Gp"), fnum(r, "n_edges_orig")
    if gp is None or not orig:
        return None
    return 1.0 - gp / orig


def _max_total(r):
    """实验强度（预算）。优先用 max_total 列，否则从文件名 mt<N> 解析。"""
    v = fnum(r, "max_total")
    if v is not None:
        return int(round(v))
    m = re.search(r'mt(\d+)', r.get("_src", ""))
    return int(m.group(1)) if m else None


def sheet_e3(wb, rows):
    """主图：横轴为 max_total 归约预算，纵轴节点级检测质量。"""
    if not rows:
        return
    encs = sorted({r.get("encoding", "?") for r in rows})
    strengths = sorted({st for st in (_max_total(r) for r in rows) if st is not None})
    ws = wb.create_sheet("E3_strength_sweep")
    header = ["max_total"] + encs + ["nodes_kept"] + ["edge_cut"]
    table = []
    for st in strengths:
        band = [r for r in rows if _max_total(r) == st]
        nk = mean([fnum(r, "n_nodes_Gp") / fnum(r, "n_nodes_orig")
                   if fnum(r, "n_nodes_orig") else None for r in band])
        ec = mean([_edge_cut(r) for r in band])
        line = [st] + [mean([fnum(r, "node_best_f1") for r in band
                              if r.get("encoding") == en]) for en in encs]
        line.append(nk)
        line.append(ec)
        table.append([v if v is not None else "" for v in line])
    write_table(ws, header, table)
    ch = LineChart()
    ch.title = "Detection quality vs reduction budget"
    ch.x_axis.title = "max_total (template instance budget)"
    ch.y_axis.title = "node best-F1"
    data = Reference(ws, min_col=2, max_col=1 + len(encs), min_row=1,
                     max_row=1 + len(table))
    cats = Reference(ws, min_col=1, min_row=2, max_row=1 + len(table))
    ch.add_data(data, titles_from_data=True)
    ch.set_categories(cats)
    ch.height, ch.width = 10, 18
    ws.add_chart(ch, "A%d" % (ws.max_row + 3))
    return ws


def sheet_simple(wb, rows, title, keys, metrics):
    if not rows:
        return
    ws = wb.create_sheet(title)
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r.get(k, "?") for k in keys)].append(r)
    header = list(keys) + list(metrics) + ["n"]
    table = []
    for g in sorted(groups):
        rs = groups[g]
        table.append(list(g)
                     + [(mean([fnum(r, m) for r in rs]) if mean([fnum(r, m) for r in rs]) is not None else "")
                        for m in metrics]
                     + [len(rs)])
    write_table(ws, header, table)
    return ws


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = a.results_dir

    wb = Workbook()
    wb.remove(wb.active)

    e1 = read_csvs(os.path.join(d, "e1", "*.csv")) or read_csvs(os.path.join(d, "e1_*.csv"))
    e2 = read_csvs(os.path.join(d, "e2", "sum_*.csv")) or read_csvs(os.path.join(d, "e2_*.csv"))
    e3 = read_csvs(os.path.join(d, "e3", "*.csv")) or read_csvs(os.path.join(d, "e3_*.csv"))
    ss = read_csvs(os.path.join(d, "e4s", "*.csv")) or read_csvs(os.path.join(d, "e4s_*.csv"))

    sheet_e1(wb, e1, "fig", "E1_encoding_matrix", "node_best_f1")
    sheet_e1(wb, e1, "fig", "E1_alert", "best_F1_alert",
             extra=("F1_alert@5", "F1_alert@20"))
    sheet_e3(wb, e3)
    sheet_simple(wb, e2, "E2_slot_equivalence", ["config"],
                 ["n_kept", "covered_kept", "recall_kept", "alert_P_kept", "F1_kept"])
    sheet_simple(wb, ss, "StreamSpot", ["operator", "encoding"],
                 ["Precision", "Recall", "F1", "ROC_AUC"])

    if not wb.sheetnames:
        wb.create_sheet("empty").append(["no results found in " + d])
    wb.save(a.out)
    print("written", a.out, "sheets:", wb.sheetnames)


if __name__ == "__main__":
    main()
