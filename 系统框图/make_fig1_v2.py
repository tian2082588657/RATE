# -*- coding: utf-8 -*-
"""make_fig1_v2.py -- Fig.1 assembled from real graphic primitives.

Follows 绘图提示词-改进.md: every data-transformation stage is drawn with a
graphic primitive (graph glyph / sin wave / segmented band / concentric rings /
bar chart / convergence wedge / check-cross symbols); text is only a short
label beside the graphic.

Layout: two serpentine rows of panels on a 992-unit (= 17.5 cm printed) canvas.
  row1 (L->R): (1) Provenance Graph (2) Collapse (3) Reduced Graph (4) RATE Encoding
  row2 (L->R): (5) Detector (6) Alert Aggregation (7) Evaluation
  band (full width): take-aways as check / bang / cross symbols

Glyph policy (verified against Arial): mu pi rho Sigma beta +- != == . ' ->
U+2212  all render; U+2218 U+2208 U+2282 U+22C6 U+2717 U+207B are tofu -> avoid.
"""
import math
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import figlib as F                                                    # noqa: E402
from figlib import (box, ellipse, circle, poly, text, sinwave,        # noqa: E402
                    graph_glyph, merge_glyph, band, ruler, bars,
                    check, cross, bang, strike, panel,
                    to_drawio, render_png,
                    INK, MUTED, EDGE_C, GREEN, ORANGE, RED, BLUE)

MU = "\u03bc"          # mu
MUB = "\u03bc\u0304"   # mu with combining macron (averaged edge mass)
PI = "\u03c0"
RHO = "\u03c1"
SIG = "\u03a3"
BET = "\u03b2"
PM = "\u00b1"
MINUS = "\u2212"
PRIME = "\u2032"
DOT = "\u00b7"

# ------------------------------------------------------------------ palette
SPEC = {
    "p1": dict(fill="#DAE8FC", stroke="#6C8EBF", title="#6C8EBF"),
    "p2": dict(fill="#F5F5F5", stroke="#9E9E9E", title="#757575"),
    "p3": dict(fill="#D5E8D4", stroke="#82B366", title="#82B366"),
    "p4": dict(fill="#FFE6CC", stroke="#D79B00", title="#D79B00"),
    "p5": dict(fill="#D5E8D4", stroke="#82B366", title="#82B366"),
    "p6": dict(fill="#FFF2CC", stroke="#D6B656", title="#D6B656"),
    "p7": dict(fill="#E1D5E7", stroke="#9673A6", title="#9673A6"),
}

# ---------------------------------------------------------------- geometry
M = 8.0
IW = 976.0
GAPX = 16.0
ROWGAP = 34.0
BANDGAP = 28.0
BAND_H = 72.0
ARROW_SW = 2.6

W1 = [210.0, 196.0, 210.0, 312.0]
X1 = []
_c = M
for _wv in W1:
    X1.append(_c)
    _c += _wv + GAPX

_w2 = (IW - 2 * GAPX) / 3.0
X2 = [M, M + _w2 + GAPX, M + 2 * (_w2 + GAPX)]

H1 = 260.0
Y1 = M
Y2 = Y1 + H1 + ROWGAP


def inner(px, pw):
    return px + F.PAD, pw - 2 * F.PAD


def halo(cid, x, y, w, h):
    """Opaque chip that keeps a label readable when it sits on top of artwork."""
    return box(cid, x, y, w, h, "", fill="#FFFFFF", stroke="#FFFFFF", sw=1.0)


# ============================================== (1) Provenance Graph
def p1_rows(x, w):
    def graph(px, py, pw, cid):
        box(cid, x + 27, py + 57, 152, 52, "", fill="#FFE0B2", stroke=ORANGE,
            sw=1.3, dashed=True)
        nod = [(22, 26, "process", "p"), (82, 18, "file", "f"),
               (152, 30, "process", "p"), (46, 76, "file", "f"),
               (104, 82, "socket", "k"), (160, 90, "file", "f"),
               (32, 128, "socket", "k"), (112, 136, "process", "p")]
        edg = [(0, 1), (1, 2), (0, 3), (3, 4), (1, 4), (4, 5), (2, 5),
               (3, 6), (4, 7), (5, 7)]
        graph_glyph(cid, [(x + nx, py + ny, t, l) for nx, ny, t, l in nod],
                    edg, r=11, font=10.0)
        text(cid, x + 26, py + 33, 138, 13, "matched region S", font=11.5,
             align="left", color=ORANGE, bold=True)

    def caption(px, py, pw, cid):
        text(cid, x, py, w, 18, "G = (V, E)", font=13.5, bold=True)

    def legend(px, py, pw, cid):
        yy = py + 13
        for cx_, lab, col in [(13, "process", "#FFFFFF"),
                              (86, "file", "#E3F2FD"),
                              (128, "socket", "#FFF3E0")]:
            circle(cid, x + cx_, yy, 8, lab[0], fill=col, stroke=EDGE_C,
                   font=9.5)
            text(cid, x + cx_ + 11, py + 2, 56, 22, lab, font=11.5,
                 align="left")

    return [(152.0, graph), (18.0, caption), (26.0, legend)]


# ===================================================== (2) Collapse
def p2_rows(x, w):
    def stage(px, py, pw, cid):
        nod = [(20, 26, "file", "f"), (20, 68, "socket", "k"),
               (66, 48, "process", "p")]
        graph_glyph(cid, [(x + nx, py + ny, t, l) for nx, ny, t, l in nod],
                    [(0, 1), (0, 2), (1, 2)], r=9, font=9.5)
        box(cid, x + 12, py + 14, 62, 66, "", fill="none", stroke=ORANGE,
            sw=1.3, dashed=True)
        text(cid, x + 12, py + 1, 62, 12, "region S", font=11, color=ORANGE,
             bold=True)
        merge_glyph(cid, x + 78, py + 24, 46, 52, n=3, sw=1.3, fan=0.6)
        circle(cid, x + 132, py + 50, 25, "", fill="#FFE0B2", stroke=ORANGE,
               sw=1.7)
        box(cid, x + 118, py + 40, 28, 20, MU, fill="#FFFFFF",
            stroke=ORANGE, font=11.5, sw=1.0)
        text(cid, x + 74, py + 78, 116, 14, "summary node s", font=11,
             color=ORANGE, bold=True)

    def caption(px, py, pw, cid):
        text(cid, x, py, w, 18, "TeRed collapse", font=13.5, bold=True)

    def badge(px, py, pw, cid):
        poly(cid, [(x + 6, py + 18), (x + 74, py + 18)], color=EDGE_C, sw=1.4,
             arrow=True)
        box(cid, x + 28, py + 8, 22, 20, MU, fill="#FFF3E0", stroke=ORANGE,
            font=11.5, sw=1.0)
        text(cid, x + 6, py + 32, w - 12, 18,
             "%s(e%s) = merged edge count" % (MU, PRIME), font=11)

    return [(104.0, stage), (18.0, caption), (52.0, badge)]


# ================================================= (3) Reduced Graph
def p3_rows(x, w):
    def graph(px, py, pw, cid):
        nod = [(22, 26, "process", "p"), (82, 18, "file", "f"),
               (152, 30, "process", "p"), (100, 84, "summary", "s"),
               (162, 96, "file", "f"), (30, 130, "socket", "k"),
               (116, 138, "process", "p")]
        edg = [(0, 1), (1, 2), (0, 3), (1, 3), (3, 2), (3, 4), (3, 5),
               (3, 6)]
        graph_glyph(cid, [(x + nx, py + ny, t, l) for nx, ny, t, l in nod],
                    edg, r=11, font=10.0)
        # mu badge sits on each edge incident to the summary node
        for bx, by in [(x + 84, py + 40), (x + 136, py + 88),
                       (x + 56, py + 106)]:
            box(cid, bx, by, 20, 15, MU, fill="#FFF3E0", stroke=ORANGE,
                font=9.5, sw=1.0)

    def caption(px, py, pw, cid):
        text(cid, x, py, w, 18,
             "G%s    %s    %s    back-map" % (PRIME, PI, RHO), font=13,
             bold=True)

    def note_(px, py, pw, cid):
        text(cid, x, py, w, 26,
             "s absorbs region S\n%s badge = edge mass" % MU, font=11.5,
             color=MUTED)

    return [(152.0, graph), (18.0, caption), (26.0, note_)]


# ================================================ (4) RATE Encoding
def p4_rows(x, w):
    TAPE = ["TAPE(deg%s)" % MINUS, "TAPE(deg+)",
            "TAPE(%s%s)" % (MUB, MINUS), "TAPE(%s+)" % MUB]
    SEGS = ["deg%s" % MINUS, "deg+", "%s%s" % (MUB, MINUS), "%s+" % MUB,
            "1-hot", "struct"]
    COLS = ["#BBDEFB", "#FFF3E0", "#F8BBD0", "#C8E6C9", "#FFF9C4", "#D1C4E9"]

    def stage(px, py, pw, cid):
        bw, bh, pitch = 106.0, 27.0, 29.0
        bx = x + 40.0
        bus = x + 28.0
        cys = [py + 2 + i * pitch + bh / 2 for i in range(4)]
        circle(cid, x + 13, py + 20, 11, "v" + PRIME, fill="#FFFFFF",
               stroke=EDGE_C, font=10.5, sw=1.3)
        # one descriptor node feeds the four sinusoidal channels through a bus
        poly(cid, [(x + 24, py + 20), (bus, py + 20)], color=EDGE_C, sw=1.4)
        poly(cid, [(bus, cys[0]), (bus, cys[-1])], color=EDGE_C, sw=1.4)
        for i, lab in enumerate(TAPE):
            by = py + 2 + i * pitch
            poly(cid, [(bus, by + bh / 2), (bx - 3, by + bh / 2)],
                 color=EDGE_C, sw=1.4, arrow=True)
            box(cid, bx, by, bw, bh, "", fill="#FFFFFF", stroke=BLUE, sw=1.1)
            sinwave(cid, bx + 4, by + 6.5, 28, 14, cycles=1.5, color=BLUE)
            text(cid, bx + 34, by, bw - 36, bh, lab, font=11, align="left")
        # node-type indicator and derived structural features join the concat
        for j, (lab, col) in enumerate([("1-hot type", "#FFF9C4"),
                                        ("structural", "#D1C4E9")]):
            box(cid, bx, py + 120 + j * 18, bw, 16, lab, fill=col,
                stroke=EDGE_C, font=11, sw=1.0)
        # six-way fan-in: source i lands on descriptor slot i, so the
        # concatenation order is readable straight off the drawing
        ys = cys + [py + 128, py + 146]
        ty = [py + 4 + i * 22.5 + 10 for i in range(6)]
        for sy, t in zip(ys, ty):
            poly(cid, [(bx + bw, sy), (x + 168, (sy + t) / 2.0), (x + 184, t)],
                 color=EDGE_C, sw=1.1, arrow=True)
        for i, s in enumerate(SEGS):             # concatenated descriptor
            box(cid, x + 186, py + 4 + i * 22.5, 112, 20, s, fill=COLS[i],
                stroke=EDGE_C, font=11.5, sw=1.0, rounded=False)
        text(cid, x + 186, py + 140, 112, 14, "descriptor", font=12.5,
             bold=True, color=ORANGE)

    def def_(px, py, pw, cid):
        merge_glyph(cid, x + 6, py + 2, 46, 16, n=3, sw=1.2, fan=0.5)
        text(cid, x + 58, py, w - 58, 20,
             "m%s = %s %s(e%s),    %s = m / deg" % (PM, SIG, MU, PRIME, MUB),
             font=12.5)

    def star(px, py, pw, cid):
        text(cid, x, py, w, 20, "RATE* = count when %s %s 1" % (MU, "\u2261"),
             font=12.5, bold=True, color=ORANGE)

    return [(158.0, stage), (20.0, def_), (20.0, star)]


# ====================================================== (5) Detector
def p5_rows(x, w):
    def members(px, py, pw, cid):
        text(cid, x, py, w, 14, "one-class ensemble, 5 members", font=12)
        cx0 = x + w / 2.0 - 2 * 46.0
        for i in range(5):
            cx = cx0 + i * 46.0
            circle(cid, cx, py + 30, 11, str(i + 1), fill="#FFFFFF",
                   stroke=GREEN, font=10.5, sw=1.3)
            poly(cid, [(cx, py + 42), (cx, py + 49)], color=EDGE_C, sw=1.2,
                 arrow=True)
        box(cid, x + 18, py + 50, w - 36, 17,
            "cosine sim. to benign (P5 radius)", fill="#FFFFFF",
            stroke=GREEN, font=11.5, sw=1.1)

    def score(px, py, pw, cid):
        poly(cid, [(x + w / 2.0, py), (x + w / 2.0, py + 7)], color=EDGE_C,
             sw=1.7, arrow=True)
        box(cid, x + 72, py + 8, w - 144, 26,
            "anomaly = 1 %s max sim" % MINUS, fill="#FFFFFF", stroke=GREEN,
            font=13.5, bold=True, sw=1.4)
        poly(cid, [(x + w / 2.0, py + 35), (x + w / 2.0, py + 42)],
             color=EDGE_C, sw=1.7, arrow=True)

    def scores(px, py, pw, cid):
        text(cid, x, py + 3, 74, 24, "node\nscores", font=11.5, bold=True,
             align="left")
        band(cid, x + 78, py + 5, w - 78, 20,
             ["v1", "v2", "v3", "v4", "...", "vn"], stroke=EDGE_C)

    def callout(px, py, pw, cid):
        y = py + 2
        box(cid, x, y, w, 58, "", fill="#FFFFFF", stroke=MUTED, dashed=True,
            sw=1.1)
        text(cid, x + 8, y + 4, 128, 50,
             "99.93%% of surviving\nedges: %s = 1, so\nRATE* = count" % MU,
             font=11, align="left", color=MUTED)
        poly(cid, [(x + 138, y + 30), (x + 212, y + 30)], color=MUTED, sw=1.5,
             arrow=True)
        strike(cid, x + 134, y + 22, x + 216, y + 38, color=RED, sw=1.7)
        halo(cid, x + 136, y + 9, 82, 14)
        text(cid, x + 136, y + 9, 82, 14, "naive mass", font=11, color=RED,
             align="left")
        poly(cid, [(x + 254, y + 46), (x + 254, y + 14)], color=GREEN, sw=2.0,
             arrow=True)
        halo(cid, x + 262, y + 18, 34, 16)
        text(cid, x + 262, y + 18, 34, 16, MUB, font=12, color=GREEN,
             align="left")

    return [(70.0, members), (48.0, score), (30.0, scores), (64.0, callout)]


# ============================================= (6) Alert Aggregation
def p6_rows(x, w):
    def rings(px, py, pw, cid):
        cx, cy = x + 82.0, py + 76.0
        for r in (34.0, 74.0):                  # BFS depth from the seed
            circle(cid, cx, cy, r, "", fill="none", stroke="#90A4AE",
                   dashed=True, sw=1.1)
        random.seed(11)
        pts = []
        for _ in range(13):
            a = random.uniform(0, 2 * math.pi)
            rr = 72.0 * (0.22 + 0.78 * random.random() ** 0.55)
            pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
        dist = [math.hypot(a - cx, b - cy) for a, b in pts]
        order = sorted(range(len(pts)), key=lambda i: -dist[i])
        r75 = dist[order[4]]                     # P75 score threshold radius
        killed = set(i for i in range(len(pts)) if dist[i] > r75)
        keep = [pts[i] for i in range(len(pts)) if i not in killed]
        circle(cid, cx, cy, r75, "", fill="none", stroke=RED, dashed=True,
               sw=1.6)
        halo(cid, cx + 24, cy - r75 - 8, 40, 15)
        text(cid, cx + 24, cy - r75 - 8, 40, 15, "P75", font=11, color=RED,
             bold=True)
        if keep:
            xs = [p[0] for p in keep]
            ys = [p[1] for p in keep]
            ellipse(cid, min(xs) - 8, min(ys) - 8,
                    max(xs) - min(xs) + 16, max(ys) - min(ys) + 16, "",
                    fill="#C8E6C9", stroke=GREEN, dashed=True, sw=1.5)
        for i, (a, b) in enumerate(pts):
            if i in killed:
                circle(cid, a, b, 4, "", fill="#ECEFF1", stroke="#90A4AE",
                       sw=1.0)
                strike(cid, a - 6, b - 6, a + 6, b + 6, color=RED, sw=1.6)
            else:
                circle(cid, a, b, 4.5, "", fill="#2E7D32", stroke="#1B5E20",
                       sw=1.0)
        circle(cid, cx, cy, 9, "", fill="#FFB74D", stroke=ORANGE, sw=1.6)
        halo(cid, cx - 20, cy + 11, 42, 15)
        text(cid, cx - 20, cy + 11, 42, 15, "seed", font=11, color=ORANGE,
             bold=True)
        text(cid, x + 4, py + 6, 60, 13, "BFS", font=11, color=MUTED,
             align="left")
        halo(cid, x + 96, py + 118, 68, 16)
        text(cid, x + 96, py + 118, 68, 16, "cluster", font=11.5, color=GREEN,
             bold=True)
        poly(cid, [(cx + 62, cy), (cx + 96, cy)], color=EDGE_C, sw=1.4,
             arrow=True)
        bars(cid, x + 190, py + 24, 104, 88, [0.95, 0.62, 0.78, 0.44, 0.30],
             color="#F9A825", stroke=EDGE_C)
        poly(cid, [(x + 184, py + 116), (x + 294, py + 116)], color=EDGE_C,
             sw=1.5)
        text(cid, x + 186, py + 117, 110, 14, "budget " + BET, font=11.5,
             bold=True)

    def caption(px, py, pw, cid):
        text(cid, x, py, w, 18,
             "K = 100 seeds  %s  P75  %s  cluster 3%s200" % (DOT, DOT, MINUS),
             font=12)

    def out(px, py, pw, cid):
        text(cid, x, py + 2, 60, 18, "alerts", font=13, bold=True,
             align="left", color=ORANGE)
        band(cid, x + 64, py + 2, w - 64, 18,
             ["A1", "A2", "A3", "A4", "..."], stroke=EDGE_C)

    return [(152.0, rings), (18.0, caption), (22.0, out)]


# ==================================================== (7) Evaluation
def p7_rows(x, w):
    ARMS = ["identity", "TeRed+count", "TeRed+RATE*", "identity+ratio"]

    def arms(px, py, pw, cid):
        for i, lab in enumerate(ARMS):
            y = py + 7 + i * 22
            text(cid, x + 2, y - 9, 96, 18, lab, font=11.5, align="left",
                 bold=(i == 2))
            poly(cid, [(x + 104, y), (x + 246, y)], color=EDGE_C,
                 sw=2.4 if i == 2 else 1.8, arrow=True)
        poly(cid, [(x + 250, py + 7), (x + 250, py + 77)], color=EDGE_C,
             sw=2.0)
        poly(cid, [(x + 250, py + 77), (x + 250, py + 90)], color=EDGE_C,
             sw=2.0, arrow=True)
        ruler(cid, x, py + 90, w, 15, ticks=17)
        text(cid, x, py + 106, w, 14, "common original unit", font=12,
             color=MUTED)

    def metrics(px, py, pw, cid):
        labs = ["node best-F1", "node ROC-AUC", "coverage",
                "alert F1@" + BET]
        bw = (w - 3 * 6.0) / 4.0
        for i, s in enumerate(labs):
            box(cid, x + i * (bw + 6.0), py, bw, 34, s, fill="#FFFFFF",
                stroke="#9673A6", font=11.5, sw=1.1)

    def extra(px, py, pw, cid):
        nod = [(20, 14, "process", "p"), (58, 8, "file", "f"),
               (18, 46, "socket", "k"), (58, 40, "process", "p")]
        graph_glyph(cid, [(x + nx, py + ny, t, l) for nx, ny, t, l in nod],
                    [(0, 1), (0, 2), (1, 3), (2, 3)], r=8, font=8.5)
        text(cid, x + 2, py + 56, 84, 15, "StreamSpot corpus", font=11.5,
             bold=True)
        poly(cid, [(x + 96, py + 2), (x + 96, py + 74)], color="#B0BEC5",
             sw=1.0, dashed=True)
        cx, cy = x + 218.0, py + 30.0
        circle(cid, cx, cy, 13, "v", fill="#FFFFFF", stroke=EDGE_C, font=10,
               sw=1.4)
        for nx, ny in [(cx, cy - 26), (cx + 34, cy), (cx, cy + 26),
                       (cx - 34, cy)]:
            poly(cid, [(nx, ny),
                       (cx + (nx - cx) * 0.36, cy + (ny - cy) * 0.36)],
                 color=EDGE_C, sw=1.2, arrow=True)
            circle(cid, nx, ny, 7, "", fill="#E8EAF6", stroke=EDGE_C, sw=1.1)
        text(cid, x + 148, py + 62, 152, 15, "message-passing detector",
             font=11.5, bold=True)

    return [(124.0, arms), (36.0, metrics), (78.0, extra)]


# =================================================================== build
def stage_arrow(x_from, x_to, y, label, sw=ARROW_SW):
    """Bold flow arrow in the inter-panel gutter, with its one-word label."""
    poly("1", [(x_from, y), (x_to - 2, y)], color=EDGE_C, sw=sw, arrow=True)
    mid = (x_from + x_to) / 2.0
    text("1", mid - 30, y - 19, 60, 16, label, font=12.5, color=EDGE_C,
         bold=True)


def band_takeaways(yb):
    cid = "panband"
    el = box("1", M, yb, IW, BAND_H, "", fill="#FAFAFA", stroke="#78909C",
             dashed=True, sw=1.3, rounded=True)
    el["id"] = cid
    el["is_panel"] = True
    F.RECTS[cid] = (M, yb, IW, BAND_H)
    text(cid, M + 10, yb + 5, 200, 15, "Take-aways", font=12.5, bold=True,
         align="left", color=MUTED)
    items = [("check", "encoding governs detectability", GREEN),
             ("check", "21% fewer edges", GREEN),
             ("check", "24% less scoring time", GREEN),
             ("bang", "reduction 28-60x offline", ORANGE),
             ("cross", "no false-alarm rate", RED)]
    n = len(items)
    cw = IW / float(n)
    for i, (kind, lab, col) in enumerate(items):
        bx = M + i * cw + 12
        cy = yb + 45
        if i:
            poly(cid, [(M + i * cw, yb + 22), (M + i * cw, yb + BAND_H - 6)],
                 color="#CFD8DC", sw=1.0, dashed=True)
        if kind == "check":
            check(cid, bx + 9, cy, s=9.0, color=col, sw=2.8)
        elif kind == "bang":
            bang(cid, bx + 9, cy, s=10.0, color=col, sw=2.8)
        else:
            cross(cid, bx + 9, cy, s=8.5, color=col, sw=2.8)
        text(cid, bx + 26, yb + 25, cw - 42, 40, lab, font=12, align="left",
             color=col, bold=True)


def build():
    row1 = [(X1[0], W1[0], "pan1", "1. Provenance Graph", SPEC["p1"], p1_rows),
            (X1[1], W1[1], "pan2", "2. Collapse", SPEC["p2"], p2_rows),
            (X1[2], W1[2], "pan3", "3. Reduced Graph", SPEC["p3"], p3_rows),
            (X1[3], W1[3], "pan4", "4. RATE Encoding", SPEC["p4"], p4_rows)]
    for px, pw, cid, title, spec, mk in row1:
        ix, iw = inner(px, pw)
        panel(cid, px, Y1, pw, title, mk(ix, iw), spec, target_h=H1)

    row2 = [(X2[0], "pan5", "5. Detector", SPEC["p5"], p5_rows),
            (X2[1], "pan6", "6. Alert Aggregation", SPEC["p6"], p6_rows),
            (X2[2], "pan7", "7. Evaluation", SPEC["p7"], p7_rows)]
    H2 = 0.0
    for px, cid, title, spec, mk in row2:
        ix, iw = inner(px, _w2)
        rs = mk(ix, iw)
        H2 = max(H2, F.TITLE_H + 2 * F.PAD +
                 F.natural([dict(h=h) for h, _ in rs]))
    for px, cid, title, spec, mk in row2:
        ix, iw = inner(px, _w2)
        panel(cid, px, Y2, _w2, title, mk(ix, iw), spec, target_h=H2)

    yb = Y2 + H2 + BANDGAP
    band_takeaways(yb)

    # --- stage-to-stage flow -------------------------------------------------
    ymid = Y1 + 132.0
    stage_arrow(X1[0] + W1[0], X1[1], ymid, "G")
    stage_arrow(X1[1] + W1[1], X1[2], ymid, "G" + PRIME)
    stage_arrow(X1[2] + W1[2], X1[3], ymid, "v" + PRIME)

    # (4) -> (5): leave the row, continue along the gutter, re-enter below.
    # The continuation is dashed so it cannot be mistaken for a data path.
    xa = X1[3] + W1[3] / 2.0
    xc = X2[0] + _w2 / 2.0
    y0 = Y1 + H1
    ym = y0 + 14.0
    poly("1", [(xa, y0), (xa, ym)], color=EDGE_C, sw=ARROW_SW, arrow=True)
    poly("1", [(xa, ym), (xc, ym)], color=EDGE_C, sw=1.4, dashed=True)
    poly("1", [(xc, ym), (xc, Y2)], color=EDGE_C, sw=ARROW_SW, arrow=True)
    text("1", (xa + xc) / 2.0 - 26, ym + 4, 52, 14, "desc", font=12.5,
         bold=True, color=EDGE_C)

    stage_arrow(X2[0] + _w2, X2[1], Y2 + 132.0, "scores")
    stage_arrow(X2[1] + _w2, X2[2], Y2 + 132.0, "alerts")

    yb2 = Y2 + H2
    poly("1", [(X2[2] + _w2 / 2.0, yb2), (X2[2] + _w2 / 2.0, yb)],
         color=EDGE_C, sw=ARROW_SW, arrow=True)
    text("1", X2[2] + _w2 / 2.0 + 10, yb2 + 8, 76, 16, "verdict", font=12.5,
         bold=True, align="left", color=EDGE_C)

    return yb + BAND_H + M


if __name__ == "__main__":
    H = build()
    out_dio = os.path.join(HERE, "fig1_system_architecture_v2.drawio")
    out_png = os.path.join(HERE, "fig1_preview_v2.png")
    to_drawio(out_dio, H)
    render_png(out_png, H, scale=4.0)
    print("H = %.1f units  ->  %.2f x %.2f cm"
          % (H, IW * 0.5 / 72.0 * 2.54, H * 0.5 / 72.0 * 2.54))
    print("drawio : %s (%d bytes)" % (out_dio, os.path.getsize(out_dio)))
    print("png    : %s (%d bytes)" % (out_png, os.path.getsize(out_png)))
    print("cells  : %d" % len(F.E))
