# -*- coding: utf-8 -*-
"""figures/make_fig_png.py — 用 PIL 直接绘制论文数据图（无需 matplotlib）。

产出:
  figures/fig2_encoding_strength.png
    面板 (a): 编码谱系（未归约 / 归约 x 五种编码）节点级 best-F1，7 分区均值
    面板 (b): 归约强度扫描（真实边归约率 -> 节点级 best-F1），逐分区曲线

数据来源:
  E1 : results/e1p/all_e1.csv                （none / rate_single / dual_naive / rate）
  E1b: results/e1b_semantic_only_all.csv     （semantic_only 对照）
  E3 : results/work/e3/*.csv                 （小窗口强度扫描）
       results/e3full/*.csv                  （全尺寸分区强度扫描）

用法:
  python figures/make_fig_png.py [--e3dirs "results/work/e3,results/e3full"] [--out PATH]
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import re
from collections import defaultdict

from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------- 样式
W, H = 1680, 720
BG = (255, 255, 255)
FG = (26, 26, 26)
GRID = (214, 219, 227)
AXIS = (60, 66, 76)

# 面板 (a)：完整编码谱系（由弱到强）
ENC_ORDER = ["semantic_only", "none", "rate_single", "dual_naive", "rate"]
ENC_SHORT = {
    "semantic_only": "no degree",
    "none": "scalar deg.",
    "rate_single": "\u03a3\u03bc",
    "dual_naive": "count",
    "rate": "RATE",
}
ENC_LABEL = {
    "semantic_only": "semantic only (no degree)",
    "none": "scalar-degree-only",
    "rate_single": "\u03a3\u03bc (mass scalar)",
    "dual_naive": "count (dual)",
    "rate": "RATE (dual)",
}
# 色盲友好 + 灰度可辨
ENC_COLOR = {
    "semantic_only": (120, 128, 142),
    "none": (150, 158, 170),
    "rate_single": (108, 172, 228),
    "dual_naive": (240, 160, 60),
    "rate": (196, 62, 74),
}
FIG_COLOR = {"identity": (108, 172, 228), "tered": (169, 49, 65)}
MARKER = {"semantic_only": "o", "none": "s", "rate_single": "s",
          "dual_naive": "^", "rate": "D"}

# 面板 (b)：只画两条族（全编码 / 无度列编码），每条族内逐分区一条线
ENC_B = ["none", "rate"]
ENC_B_LABEL = {"none": "no-degree encoder", "rate": "full encoder (RATE)"}

PROD_CUT = 21.4          # 生产设置（7 分区聚合）的边归约率，仅作参考线
                         # 2026-09-22 更新：可复现归约下 38,673,370 -> 30,412,456 边 (−21.4%)


def _font(size, bold=False):
    cands = [
        r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\segoeuib.ttf" if bold else r"C:\Windows\Fonts\segoeui.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for c in cands:
        if os.path.exists(c):
            try:
                return ImageFont.truetype(c, size)
            except Exception:
                pass
    return ImageFont.load_default()


F_TICK = _font(22)
F_LABEL = _font(25)
F_TITLE = _font(28, bold=True)
F_LEG = _font(21)
F_ANNO = _font(21)
F_NOTE = _font(19)


def read_csvs(patterns):
    """patterns: 逗号分隔的 glob 列表；只保留 dataset 以 e5 开头的行。"""
    rows = []
    for pat in patterns.split(","):
        pat = pat.strip()
        if not pat:
            continue
        if os.path.isdir(pat):
            pat = os.path.join(pat, "*.csv")
        for p in sorted(glob.glob(pat)):
            with open(p, encoding="utf-8", newline="") as fh:
                for r in csv.DictReader(fh):
                    ds = str(r.get("dataset", ""))
                    if not ds or ds == "dataset":
                        continue          # 跳过拼接文件里的重复表头
                    r["_src"] = os.path.basename(p)
                    rows.append(r)
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


def _edge_cut(r):
    """边归约率 = 1 - n_edges_Gp / n_edges_orig。"""
    gp, orig = fnum(r, "n_edges_Gp"), fnum(r, "n_edges_orig")
    if gp is None or not orig:
        ec = fnum(r, "edge_cut")
        return ec
    return 1.0 - gp / orig


class Panel:
    """一个坐标区：负责把数据坐标映射到像素并画轴/网格/刻度。"""

    def __init__(self, draw, box, title, xlabel, ylabel):
        self.d = draw
        self.x0, self.y0, self.x1, self.y1 = box
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.xlim = (0.0, 1.0)
        self.ylim = (0.0, 1.0)
        self.xticks = []
        self.yticks = []

    def _text_size(self, text, font):
        if hasattr(font, "getbbox"):
            l, t, r, b = font.getbbox(text)
            return (r - l, b - t)
        return (len(text) * 10, 20)

    def set_x(self, lo, hi, ticks):
        self.xlim = (lo, hi)
        self.xticks = ticks

    def set_y(self, lo, hi, ticks):
        self.ylim = (lo, hi)
        self.yticks = ticks

    def px(self, v):
        lo, hi = self.xlim
        return self.x0 + (v - lo) / (hi - lo) * (self.x1 - self.x0)

    def py(self, v):
        lo, hi = self.ylim
        return self.y1 - (v - lo) / (hi - lo) * (self.y1 - self.y0)

    def draw_frame(self, xticklabels=None, yticklabels=None, yfmt="%.2f"):
        d = self.d
        for t in self.yticks:
            y = self.py(t)
            d.line([(self.x0, y), (self.x1, y)], fill=GRID, width=1)
        for t in self.xticks:
            x = self.px(t)
            d.line([(x, self.y0), (x, self.y1)], fill=GRID, width=1)
        d.line([(self.x0, self.y0), (self.x0, self.y1)], fill=AXIS, width=2)
        d.line([(self.x0, self.y1), (self.x1, self.y1)], fill=AXIS, width=2)
        for i, t in enumerate(self.yticks):
            lab = yticklabels[i] if yticklabels else (yfmt % t)
            d.text((self.x0 - 12, self.py(t)), lab, font=F_TICK, fill=FG,
                   anchor="rm")
        if xticklabels:
            for t, lab in zip(self.xticks, xticklabels):
                d.text((self.px(t), self.y1 + 12), lab, font=F_TICK, fill=FG,
                       anchor="ma")
        else:
            for t in self.xticks:
                d.text((self.px(t), self.y1 + 12), ("%.2f" % t), font=F_TICK,
                       fill=FG, anchor="ma")
        d.text(((self.x0 + self.x1) / 2, self.y1 + 58), self.xlabel,
               font=F_LABEL, fill=FG, anchor="ma")
        if self.ylabel:
            lw, lh = self._text_size(self.ylabel, F_LABEL)
            d.text((self.x0 - 18 - lw, self.y0 - 10 - lh),
                   self.ylabel, font=F_LABEL, fill=FG)
        d.text(((self.x0 + self.x1) / 2, self.y0 - 40), self.title,
               font=F_TITLE, fill=FG, anchor="ma")

    # --- 图元
    def bar(self, xc, val, halfw, color, hatch=False):
        d = self.d
        x_l, x_r = self.px(xc - halfw), self.px(xc + halfw)
        y_v, y_0 = self.py(val), self.py(0.0)
        d.rectangle([x_l, y_v, x_r, y_0], fill=color, outline=(40, 44, 52),
                    width=1)
        if hatch:
            step = 9                      # 细密斜纹，灰度打印仍可辨
            for k in range(-int((y_0 - y_v) / step) - 2,
                           int((x_r - x_l) / step) + 2):
                x1 = x_l + k * step
                y1 = y_v
                x2 = x1 + (y_0 - y_v)
                y2 = y_0
                x1c, y1c = max(x1, x_l), max(y1, y_v)
                x2c, y2c = min(x2, x_r), min(y2, y_0)
                if x1c < x2c and y1c < y2c:
                    d.line([(x1c, y1c), (x2c, y2c)], fill=(255, 255, 255),
                           width=1)

    def line(self, pts, color, marker="o", msize=7, lw=3):
        d = self.d
        if len(pts) > 1:
            d.line(pts, fill=color, width=lw, joint="curve")
        for (x, y) in pts:
            if marker == "o":
                d.ellipse([x - msize, y - msize, x + msize, y + msize],
                          fill=color, outline=(255, 255, 255), width=2)
            elif marker == "s":
                d.rectangle([x - msize, y - msize, x + msize, y + msize],
                            fill=color, outline=(255, 255, 255), width=2)
            elif marker == "^":
                d.polygon([(x, y - msize - 1), (x - msize, y + msize),
                           (x + msize, y + msize)], fill=color,
                          outline=(255, 255, 255))
            else:
                d.polygon([(x, y - msize), (x + msize, y), (x, y + msize),
                           (x - msize, y)], fill=color,
                          outline=(255, 255, 255))


def legend(d, items, x, y, dy=34):
    for i, (lab, color, mk) in enumerate(items):
        yy = y + i * dy
        if mk == "bar":
            d.rectangle([x, yy - 11, x + 26, yy + 9], fill=color,
                        outline=(40, 44, 52), width=1)
        else:
            d.line([(x, yy), (x + 26, yy)], fill=color, width=3)
            d.ellipse([x + 13 - 6, yy - 6, x + 13 + 6, yy + 6], fill=color)
        d.text((x + 36, yy), lab, font=F_LEG, fill=FG, anchor="lm")


def panel_a(d, box, e1):
    """编码谱系：未归约 / 归约 x 五种编码 的节点级 best-F1（7 分区均值）。"""
    p = Panel(d, box, "(a) Encoding spectrum, node best-F1",
              "topology encoding", "")
    vals = {}
    for fig in ("identity", "tered"):
        for enc in ENC_ORDER:
            vals[(fig, enc)] = mean([fnum(r, "node_best_f1") for r in e1
                                     if r.get("fig") == fig
                                     and r.get("encoding") == enc])
    ymax = 0.20
    p.set_x(-0.5, len(ENC_ORDER) - 0.5, list(range(len(ENC_ORDER))))
    p.set_y(0.0, ymax, [0.0, 0.05, 0.10, 0.15, 0.20])
    p.draw_frame(xticklabels=[ENC_SHORT[e] for e in ENC_ORDER])

    nfig = 2
    total = 0.56
    half = total / (2 * nfig) - 0.010
    gap = 0.065
    offs = {"identity": -gap, "tered": gap}
    for gi, enc in enumerate(ENC_ORDER):
        pair = {fig: vals.get((fig, enc)) for fig in ("identity", "tered")}
        top = max([v for v in pair.values() if v is not None] or [0.0])
        for fig in ("identity", "tered"):
            v = pair[fig]
            if v is None:
                continue
            p.bar(gi + offs[fig], v, half, FIG_COLOR[fig],
                  hatch=(fig == "tered"))
        # 两条臂的数值叠放在较高柱上方（左=未归约，右=归约）
        ytxt = p.py(top) - 12
        for fig in ("tered", "identity"):
            v = pair[fig]
            if v is None:
                continue
            d.text((p.px(gi), ytxt), "%.3f" % v, font=F_ANNO,
                   fill=FIG_COLOR[fig], anchor="mb")
            ytxt -= 24
    legend(d, [("unreduced graph", FIG_COLOR["identity"], "bar"),
               ("reduced graph", FIG_COLOR["tered"], "bar")],
           box[0] + 18, box[1] - 8)


def panel_b(d, box, e3):
    """归约强度扫描：真实边归约率 -> 节点级 best-F1，逐分区。
    只画归约图上的点：未归约图是另一个正集（reduced-unit 不可比），
    不能把 x=0 的 identity 行接进同一曲线。"""
    p = Panel(d, box, "(b) Detection under increasing reduction",
              "edge-reduction ratio (%)", "node best-F1")
    per = defaultdict(list)                       # (partition, enc) -> [(cut%, f1)]
    for r in e3:
        enc, part = r.get("encoding"), r.get("file")
        cut, f1 = _edge_cut(r), fnum(r, "node_best_f1")
        if enc is None or cut is None or f1 is None:
            continue
        per[(part, enc)].append((cut * 100.0, f1))
    if not per:
        p.set_x(0, 50, [0, 10, 20, 30, 40, 50])
        p.set_y(0, 1, [0, 0.5, 1.0])
        p.draw_frame()
        d.text(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2),
               "E3 pending", font=F_TITLE, fill=(180, 80, 80), anchor="mm")
        return

    xmax = 50.0
    allv = [v for (part, e), pts in per.items() if e in ENC_B
            for _, v in pts]
    ymax = max(allv + [0.01]) * 1.35
    p.set_x(0.0, xmax, [0, 10, 20, 30, 40, 50])
    step = 0.05
    ntick = max(2, int(ymax / step) + 1)
    p.set_y(0.0, ymax, [round(i * step, 2) for i in range(ntick)])
    p.draw_frame(xticklabels=["0", "10", "20", "30", "40", "50"])

    # 参考线：生产设置的归约率（标签放在线右侧中部，避开顶部图例）
    xp = p.px(PROD_CUT)
    d.line([(xp, p.y0), (xp, p.y1)], fill=(190, 120, 120), width=2)
    d.text((xp + 6, p.py(ymax * 0.52)), "production setting, 21.4%",
           font=F_NOTE, fill=(150, 80, 80))

    # 逐分区曲线：先画无度列族，再画全编码族（后者更重要，压在上层）
    for enc in ENC_B:
        for (part, e), pts in sorted(per.items()):
            if e != enc:
                continue
            pts = sorted(pts)
            pix = [(p.px(x), p.py(y)) for x, y in pts]
            p.line(pix, ENC_COLOR[enc], MARKER[enc],
                   msize=6 if enc == "rate" else 5,
                   lw=3 if enc == "rate" else 2)
    legend(d, [(ENC_B_LABEL[e], ENC_COLOR[e], "line") for e in ENC_B],
           box[0] + 22, box[1] + 24)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e1", default="results/e1p/all_e1.csv")
    ap.add_argument("--e1b", default="results/e1b_semantic_only_all.csv")
    ap.add_argument("--e3dirs", default="results/work/e3,results/e3full")
    ap.add_argument("--out", default="figures/fig2_encoding_strength.png")
    a = ap.parse_args()

    e1 = read_csvs(a.e1) + [r for r in read_csvs(a.e1b)]
    e3 = read_csvs(a.e3dirs)
    print("E1 rows %d | E3 rows %d" % (len(e1), len(e3)))

    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    panel_a(d, (170, 110, 790, 570), e1)
    panel_b(d, (1010, 110, 1630, 570), e3)
    d.text((W / 2, 34), "Encoding, reduction strength and detection quality",
           font=_font(30, bold=True), fill=FG, anchor="ma")
    img.save(a.out)
    print("written", a.out)


if __name__ == "__main__":
    main()
