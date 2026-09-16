# -*- coding: utf-8 -*-
"""figures/make_drawio.py — 生成论文用 draw.io 框图（可直接 import 到 app.diagrams.net）。

产出：
  fig1_pipeline.drawio      端到端流水线（数据 -> 归约 -> 编码 -> 检测 -> 告警 -> 评估）
  fig2_distortion.drawio    归约失真机制（度语义：入口膨胀 / 出口膨胀 / 内部消失 / 未触碰不变）
  fig3_matrix.drawio        实验设计矩阵（图来源 x 编码 的对照格）
"""
from __future__ import annotations
import os, io

OUT = os.path.dirname(os.path.abspath(__file__))

# ---------------- draw.io style 常量 ----------------
S_BOX = ("rounded=1;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#37474F;"
         "fontColor=#1A237E;fontSize=12;arcSize=8;")
S_BOX_FILL = ("rounded=1;whiteSpace=wrap;html=1;fillColor=#E3F2FD;strokeColor=#1565C0;"
              "fontColor=#0D47A1;fontSize=12;arcSize=8;")
S_BOX_ACCENT = ("rounded=1;whiteSpace=wrap;html=1;fillColor=#FFF3E0;strokeColor=#E65100;"
                "fontColor=#BF360C;fontSize=12;arcSize=8;fontWeight=1;")
S_BOX_OK = ("rounded=1;whiteSpace=wrap;html=1;fillColor=#E8F5E9;strokeColor=#2E7D32;"
            "fontColor=#1B5E20;fontSize=12;arcSize=8;")
S_GROUP = ("rounded=0;whiteSpace=wrap;html=1;fillColor=none;strokeColor=#90A4AE;"
           "dashed=1;fontColor=#546E7A;fontSize=11;verticalAlign=top;"
           "align=left;spacingLeft=6;spacingTop=2;")
S_NOTE = ("text;html=1;strokeColor=none;fillColor=none;align=left;"
          "verticalAlign=middle;fontColor=#546E7A;fontSize=11;")
S_EDGE = ("edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;strokeColor=#455A64;"
          "strokeWidth=1;endArrow=block;endFill=1;fontSize=11;fontColor=#455A64;")
S_EDGE_DASH = ("edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;strokeColor=#90A4AE;"
               "strokeWidth=1;dashed=1;endArrow=block;endFill=1;fontSize=11;"
               "fontColor=#78909C;")


class Diagram:
    def __init__(self, name, w=1180, h=800):
        self.name = name
        self.w, self.h = w, h
        self.cells = []
        self._n = 0

    def box(self, value, x, y, w, h, style=S_BOX, cid=None):
        self._n += 1
        cid = cid or "n%d" % self._n
        self.cells.append(
            '<mxCell id="%s" value="%s" style="%s" vertex="1" parent="1">'
            '<mxGeometry x="%d" y="%d" width="%d" height="%d" as="geometry"/>'
            '</mxCell>' % (cid, value, style, x, y, w, h))
        return cid

    def edge(self, src, dst, label="", style=S_EDGE, cid=None,
             exit_x=None, exit_y=None, entry_x=None, entry_y=None):
        self._n += 1
        cid = cid or "e%d" % self._n
        extra = ""
        if exit_x is not None:
            extra += '<mxPoint x="%d" y="%d" as="exitPoint"/>' % (exit_x, exit_y)
        if entry_x is not None:
            extra += '<mxPoint x="%d" y="%d" as="entryPoint"/>' % (entry_x, entry_y)
        self.cells.append(
            '<mxCell id="%s" value="%s" style="%s" edge="1" parent="1" '
            'source="%s" target="%s">'
            '<mxGeometry relative="1" as="geometry">%s</mxGeometry></mxCell>'
            % (cid, label, style, src, dst, extra))
        return cid

    def xml(self):
        return (
            '<mxfile host="app.diagrams.net" type="device">\n'
            '  <diagram name="%s" id="%s">\n'
            '    <mxGraphModel dx="1400" dy="900" grid="1" gridSize="10" '
            'page="1" pageWidth="%d" pageHeight="%d" math="0" shadow="0">\n'
            '      <root>\n'
            '        <mxCell id="0"/>\n'
            '        <mxCell id="1" parent="0"/>\n'
            '%s\n'
            '      </root>\n'
            '    </mxGraphModel>\n'
            '  </diagram>\n'
            '</mxfile>\n'
            % (self.name, self.name, self.w, self.h,
               "\n".join("        " + c for c in self.cells)))

    def save(self, path):
        with io.open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(self.xml())
        print("written", path, len(self.xml()), "bytes")


# ================= FIG 1: 端到端流水线 =================
def fig1():
    d = Diagram("fig1_pipeline", 1180, 830)
    W = 1180

    # --- Stage 1 数据源 ---
    d.box("Datasets (three captures)", 30, 24, 1120, 96, S_GROUP, "g1")
    ds = [
        ("DARPA TC E5 CADETS<br/>(381 GB, 38 bins)", 60),
        ("DARPA TC E3 CADETS<br/>(time-windowed)", 430),
        ("StreamSpot<br/>(scenes split)", 800),
    ]
    for t, x in ds:
        d.box(t, x, 52, 320, 52, S_BOX, "ds_%d" % x)

    # --- Stage 2 规范图 ---
    g = d.box("Canonical provenance graph  G = (V, E, &mu;)<br/>"
              "events &rarr; typed nodes + timestamped edges",
              340, 160, 500, 58, S_BOX_FILL, "g_graph")
    for t, x in ds:
        d.edge("ds_%d" % x, "g_graph")

    # --- Stage 3 归约算子 ---
    d.box("Reduction operator R  (both branches share one encoder)", 30, 252,
          1120, 116, S_GROUP, "g3")
    idn = d.box("Identity  (no reduction)<br/>G' = G, &pi; = id",
                90, 286, 380, 62, S_BOX, "op_id")
    ter = d.box("TeRed-style collapse<br/>template regions &rarr; summary M/N",
                690, 286, 380, 62, S_BOX, "op_te")
    d.edge("g_graph", "op_id")
    d.edge("g_graph", "op_te")

    gp = d.box("Reduced graph  G' + node map &pi;, edge mass &mu;'",
               340, 402, 500, 52, S_BOX_FILL, "g_gp")
    d.edge("op_id", "g_gp")
    d.edge("op_te", "g_gp")

    # --- Stage 4 编码 ---
    d.box("Feature encoding (shared by both branches)", 30, 480, 1120, 116,
          S_GROUP, "g4")
    e1 = d.box("semantic channel<br/>type one-hot", 60, 514, 320, 62, S_BOX,
               "enc_sem")
    e2 = d.box("topology channel (RATE)<br/>in: &Sigma;&mu;   out: &Sigma;&mu;",
               430, 514, 320, 62, S_BOX_ACCENT, "enc_rate")
    e3 = d.box("v4 structural columns<br/>(log-deg, ratio, self-loop)",
               800, 514, 320, 62, S_BOX, "enc_v4")
    d.edge("g_gp", "enc_sem", style=S_EDGE_DASH)
    d.edge("g_gp", "enc_rate")
    d.edge("g_gp", "enc_v4", style=S_EDGE_DASH)

    # --- Stage 5 检测 ---
    det = d.box("Benign single-class ensemble<br/>"
                "per-group centroid + 5th-percentile cosine radius &rarr; "
                "anomaly score = 1 &minus; max sim",
                300, 636, 580, 56, S_BOX_FILL, "det")
    for c in ("enc_sem", "enc_rate", "enc_v4"):
        d.edge(c, "det", style=S_EDGE_DASH)

    # --- Stage 6 告警聚合 ---
    agg = d.box("BFS alert aggregation &nbsp;|&nbsp; top-k = 100 seeds, "
                "P75 threshold, cluster &ge; 3, &le; 200, budget B",
                250, 730, 680, 50, S_BOX_OK, "agg")
    d.edge("det", "agg")

    # --- Stage 7 输出 ---
    ev = d.box("Evaluation<br/>node F1 &middot; alert F1@B &middot; "
               "common-unit recall", 700, 812, 420, 46, S_NOTE, "ev")
    out = d.box("Alert clusters", 60, 812, 300, 46, S_NOTE, "out")
    d.edge("agg", "out", style=S_EDGE_DASH)
    d.edge("agg", "ev", style=S_EDGE_DASH)

    d.save(os.path.join(OUT, "fig1_pipeline.drawio"))


# ================= FIG 2: 归约失真机制 =================
def fig2():
    d = Diagram("fig2_distortion", 1180, 720)

    d.box("Region S collapsed into summary node s", 40, 20, 1100, 30, S_NOTE,
          "cap1")

    # 左：原图
    d.box("Original graph  G", 40, 60, 520, 400, S_GROUP, "gL")
    d.box("external<br/>node u", 80, 110, 110, 46, S_BOX, "u_in")
    d.box("external<br/>node w", 80, 380, 110, 46, S_BOX, "w_out")
    d.box("v", 240, 240, 90, 46, S_BOX_FILL, "v")
    d.box("S  (benign template region)", 390, 100, 140, 330, S_GROUP, "gS")
    for i, yy in enumerate((130, 200, 270, 340)):
        d.box("x%d" % (i + 1), 405, yy, 110, 44, S_BOX, "x%d" % i)
    # edges
    d.edge("u_in", "v", "1 edge")
    d.edge("u_in", "x0", "1 edge")
    d.edge("v", "w_out", "1 edge")
    d.edge("x3", "w_out", "1 edge")
    d.box("deg&#8314;(u) = 2 &nbsp;&nbsp; deg&#8315;(w) = 2",
          80, 450, 400, 40, S_NOTE + "fontColor=#0D47A1;", "noteL")

    # 右：归约图
    d.box("Reduced graph  G'", 620, 60, 520, 400, S_GROUP, "gR")
    d.box("external<br/>node u", 660, 110, 110, 46, S_BOX, "u_in2")
    d.box("external<br/>node w", 660, 380, 110, 46, S_BOX, "w_out2")
    d.box("v", 820, 240, 90, 46, S_BOX_FILL, "v2")
    d.box("s", 960, 170, 90, 46, S_BOX_ACCENT, "s")
    d.box("summary node s<br/>carries &mu; = mass sum", 940, 300, 180, 60,
          S_NOTE + "fontColor=#BF360C;", "noteS")
    # edges：入边汇总
    d.edge("u_in2", "v2", "1")
    d.edge("u_in2", "s", "&mu;=1")
    d.edge("v2", "w_out2", "1")
    d.edge("s", "w_out2", "&mu;=3")
    d.box("deg&#8314;(u) = 2 &nbsp;(entry inflated)<br/>"
          "deg&#8315;(w) = 2 &nbsp;(exit inflated)<br/>"
          "x1..x4 interior: gone &nbsp;&nbsp; v: untouched",
          660, 450, 460, 70, S_NOTE + "fontColor=#BF360C;", "noteR")

    # 下：结论条
    d.box("A degree in a reduced graph is the <b>mass sum</b> of incident edges, "
          "not their count &nbsp;&rarr;&nbsp; counting edges inflates boundary "
          "degrees and erases interior ones;<br/>counting <b>mass</b> (&Sigma;&mu;) "
          "keeps deg&#8314;/deg&#8315; unchanged for untouched nodes and "
          "mass-conserving for boundary nodes.",
          40, 560, 1100, 70, S_BOX_OK + "align=center;fontSize=12;", "conc")

    d.save(os.path.join(OUT, "fig2_distortion.drawio"))


# ================= FIG 3: 实验设计矩阵 =================
def fig3():
    d = Diagram("fig3_matrix", 1180, 640)
    d.box("Figure/encoding matrix  —  every cell is a full "
          "train&rarr;score&rarr;alert run", 40, 16, 1100, 30, S_NOTE, "cap")

    cols = ["no topology<br/>(none)", "single-channel<br/>(&Sigma;&mu;)",
            "dual count<br/>(naive)", "dual mass<br/>(RATE)"]
    rows = ["Identity<br/>(no reduction)", "TeRed collapse<br/>(reduced)"]
    x0, y0, cw, ch = 300, 70, 215, 130
    for j, c in enumerate(cols):
        d.box(c, x0 + j * cw, y0, cw - 14, 46, S_BOX_FILL, "c%d" % j)
    for i, r in enumerate(rows):
        d.box(r, 40, y0 + 60 + i * ch, 240, ch - 30, S_BOX, "r%d" % i)
    labels = [
        ["0.0068 (collapsed)", "?", "0.186", "0.169 / 0.173"],
        ["E1 control", "E1 control", "E1 control", "0.220 (matched width)"],
    ]
    for i in range(2):
        for j in range(4):
            d.box(labels[i][j], x0 + j * cw, y0 + 60 + i * ch, cw - 14,
                  ch - 30, S_BOX_OK + "align=center;", "m%d%d" % (i, j))

    d.box("Rows = graph provenance (reduced vs. not); columns = the topological "
          "channel the detector consumes.<br/>"
          "The bottom-left / top-right contrast isolates the effect of reduction "
          "from the effect of encoding.",
          40, 420, 1100, 56, S_NOTE, "note")

    d.box("Equivalence test (common unit, slots not touched by reduction)<br/>"
          "identity+dual-mass  vs.  reduced+dual-mass:  "
          "recall, F1@20, precision  &rarr;  no detectable loss",
          40, 500, 1100, 56, S_BOX_OK + "align=center;", "eq")

    d.save(os.path.join(OUT, "fig3_matrix.drawio"))


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
