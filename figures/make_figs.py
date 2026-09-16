# -*- coding: utf-8 -*-
"""make_figs.py — 生成论文用的图（PNG，300+ dpi 等效）。

fig1_pipeline.png  系统框图（与 figures/fig1_pipeline.drawio 内容一致）
fig2_sweep.png     归约强度扫描主图（读 results/e3/ 的真实数据）

用法:
  python figures/make_figs.py           # 全部
  python figures/make_figs.py fig1      # 只做框图
"""
from __future__ import annotations
import os, sys, glob, csv

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

BG = (255, 255, 255)
INK = (26, 35, 126)
INK2 = (38, 50, 56)
BOX = (227, 242, 253)
BOX2 = (255, 255, 255)
EDGE = (55, 71, 79)
ACC = (13, 71, 161)
LINE = (69, 90, 100)
MUTED = (84, 110, 122)

FONTS = ["C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/segoeui.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]


def font(size):
    for p in FONTS:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def box(d, xy, text, f, fill=BOX, outline=ACC, ink=INK, radius=12, width=2):
    d.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)
    x0, y0, x1, y1 = xy
    lines = text.split("\n")
    lh = f.size + 6
    ty = (y0 + y1) / 2 - len(lines) * lh / 2 + 3
    for i, ln in enumerate(lines):
        w = d.textlength(ln, font=f)
        d.text(((x0 + x1) / 2 - w / 2, ty + i * lh), ln, font=f, fill=ink)


def arrow(d, p0, p1, width=2, colour=LINE):
    d.line([p0, p1], fill=colour, width=width)
    (x0, y0), (x1, y1) = p0, p1
    import math
    ang = math.atan2(y1 - y0, x1 - x0)
    L, W = 14, 7
    for s in (-1, 1):
        d.line([(x1, y1),
                (x1 - L * math.cos(ang) + s * W * math.sin(ang),
                 y1 - L * math.sin(ang) - s * W * math.cos(ang))],
               fill=colour, width=width)


def fig1(path):
    W, H = 2400, 1500
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    f_t = font(40)
    f_b = font(30)
    f_s = font(26)

    # title band
    d.text((60, 34), "Reduction-aware topology encoding (RATE): evaluation pipeline",
           font=f_t, fill=INK)
    d.line([(60, 92), (W - 60, 92)], fill=(200, 210, 220), width=2)

    # row 1: datasets
    y = 130
    ds = [("DARPA TC E5\nCADETS (381 GB)", 60),
          ("StreamSpot\n(scenes split)", 1180)]
    d.text((60, y - 4), "(a) Captures", font=f_s, fill=MUTED)
    for t, x in ds:
        box(d, (x, y + 30, x + 460, y + 140), t, f_b)
    gx = 700
    box(d, (gx, y + 30, gx + 900, y + 140),
        "Canonical provenance graph  G = (V, E, \u03bc)\nevents \u2192 typed nodes + timestamped edges",
        f_b, fill=BOX2)
    arrow(d, (520, y + 85), (gx - 6, y + 85))
    arrow(d, (1180, y + 85), (gx + 906, y + 85))

    # row 2: reduction
    y2 = 340
    d.text((60, y2 - 34), "(b) Graph reduction  (the only difference between the two arms)",
           font=f_s, fill=MUTED)
    box(d, (60, y2, 900, y2 + 150),
        "Identity operator  (no reduction)\nG' = G,  all edges carry \u03bc = 1", f_b)
    box(d, (1080, y2, 1920, y2 + 150),
        "Collapse reduction  (TeRed-style)\ntemplate regions \u2192 summary nodes\n\u03bc = 1 for most nodes, with node map \u03c0",
        f_b)
    arrow(d, (1150, y + 142), (480, y2 - 4))
    arrow(d, (1200, y + 142), (1500, y2 - 4))

    # row 3: encoder
    y3 = 580
    d.text((60, y3 - 34), "(c) Topology encoder  (shared by both arms; the paper's object of study)",
           font=f_s, fill=MUTED)
    box(d, (360, y3, 2040, y3 + 170),
        "Dual-channel TAPE descriptor of node degree\n"
        "count arm:   degree = number of incident edges      \u2190 blind to reduction\n"
        "RATE arm:    degree = MASS of incident edges (\u03bc) + collapse ratio \u03bc\u0304 when \u03bc \u2260 1\n"
        "with \u03bc = 1 the two coincide: RATE degenerates to the count encoder",
        f_b, fill=BOX)
    arrow(d, (480, y2 + 154), (700, y3 - 4))
    arrow(d, (1500, y2 + 154), (1300, y3 - 4))

    # row 4: features + detector
    y4 = 850
    d.text((60, y4 - 34), "(d) Detection", font=f_s, fill=MUTED)
    box(d, (60, y4, 800, y4 + 150),
        "Structural features (3 columns)\nlog-degree, out/in ratio, self-loop flag\n"
        "+ semantic node-type one-hot", f_b, fill=BOX2)
    box(d, (900, y4, 1560, y4 + 150),
        "Label-free one-class ensemble\n5 members, cosine similarity\n5th-percentile radius",
        f_b, fill=BOX2)
    box(d, (1660, y4, 2340, y4 + 150),
        "Budgeted alert aggregation\ntop-100 seeds, BFS at P75\ncluster 3..200, <= 20 alerts",
        f_b, fill=BOX2)
    arrow(d, (840, y4 + 75), (900, y4 + 75))
    arrow(d, (1560, y4 + 75), (1660, y4 + 75))
    arrow(d, (1200, y3 + 174), (1200, y4 - 4))

    # row 5: metrics
    y5 = 1090
    d.text((60, y5 - 34), "(e) What is measured", font=f_s, fill=MUTED)
    box(d, (60, y5, 700, y5 + 150),
        "Necessity\nno topology channel ->\nnode F1 collapses on BOTH graphs", f_b,
        fill=(255, 243, 224), outline=(230, 145, 56), ink=(120, 60, 0))
    box(d, (760, y5, 1500, y5 + 150),
        "Sufficiency\nRATE: reduced vs unreduced\nindistinguishable on a common unit", f_b,
        fill=(232, 245, 233), outline=(46, 125, 50), ink=(20, 70, 24))
    box(d, (1560, y5, 2340, y5 + 150),
        "Cost\n-15% edges, -23% scoring\ndominated end to end by the reduction", f_b,
        fill=(237, 231, 246), outline=(94, 53, 177), ink=(45, 20, 100))
    arrow(d, (400, y4 + 154), (380, y5 - 4))
    arrow(d, (1150, y4 + 154), (1130, y5 - 4))
    arrow(d, (1900, y4 + 154), (1950, y5 - 4))

    d.text((60, H - 60),
           "Solid arrows: data flow.  Both arms use the identical encoder, detector and alert pipeline; "
           "only the reduction and the encoder's definition of degree differ.",
           font=f_s, fill=MUTED)
    im.save(path, dpi=(300, 300))
    print("wrote", path, im.size)


# --------------------------------------------------------------------------
def load_e3():
    """读 E3 扫描结果，返回 {max_total: (mean_reduction_ratio, mean_nodeF1)}。"""
    out = {}
    for d in sorted(glob.glob(os.path.join(ROOT, "results", "e3*"))):
        for p in sorted(glob.glob(os.path.join(d, "*.csv"))):
            try:
                rows = list(csv.DictReader(open(p, encoding="utf-8-sig")))
            except Exception:
                continue
            if not rows or "node_best_f1" not in rows[0]:
                continue
            mt = os.path.basename(p).split("_")[0]
            for r in rows:
                pass
            out.setdefault(mt, []).extend(rows)
    return out


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    out = os.path.join(HERE)
    if which in ("all", "fig1"):
        fig1(os.path.join(out, "fig1_pipeline.png"))
    if which in ("all", "fig2"):
        print("fig2 needs E3 results; use make_fig2.py once results/e3 exists")


if __name__ == "__main__":
    main()
