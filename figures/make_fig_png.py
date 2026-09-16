# -*- coding: utf-8 -*-
"""figures/make_fig_png.py — 用 PIL 直接绘制论文数据图（无需 matplotlib）。

产出:
  figures/fig2_encoding_strength.png
    面板 (a): 编码谱系（未归约 / 归约 × 四种编码）节点级 best-F1
    面板 (b): 归约强度扫描（边归约率 → 节点级 best-F1），四种编码

数据来源:
  E1: results/e1p/all_e1.csv
  E3: results/work/e3/*.csv  （可选；缺失时只画面板 (a)）

用法:
  python figures/make_fig_png.py [--e1 PATH] [--e3dir DIR] [--out PATH]
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
from collections import defaultdict

from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------- 样式
W, H = 1680, 720
BG = (255, 255, 255)
FG = (26, 26, 26)
GRID = (214, 219, 227)
AXIS = (60, 66, 76)

ENC_ORDER = ["none", "rate_single", "dual_naive", "rate"]
ENC_LABEL = {
    "none": "no topology",
    "rate_single": "Σμ (mass scalar)",
    "dual_naive": "count (dual)",
    "rate": "RATE (dual, μ-weighted)",
}
# 色盲友好 + 灰度可辨；未归约=浅蓝实体，归约=深红带斜纹
ENC_COLOR = {
    "none": (150, 158, 170),
    "rate_single": (108, 172, 228),
    "dual_naive": (240, 160, 60),
    "rate": (196, 62, 74),
}
FIG_COLOR = {"identity": (108, 172, 228), "tered": (169, 49, 65)}
MARKER = {"none": "o", "rate_single": "s", "dual_naive": "^", "rate": "D"}


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


def read_csvs(pattern):
    rows = []
    for p in sorted(glob.glob(pattern)):
        with open(p, encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh):
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
        # 返回 (width, height)
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
        # 网格
        for t in self.yticks:
            y = self.py(t)
            d.line([(self.x0, y), (self.x1, y)], fill=GRID, width=1)
        for t in self.xticks:
            x = self.px(t)
            d.line([(x, self.y0), (x, self.y1)], fill=GRID, width=1)
        # 轴
        d.line([(self.x0, self.y0), (self.x0, self.y1)], fill=AXIS, width=2)
        d.line([(self.x0, self.y1), (self.x1, self.y1)], fill=AXIS, width=2)
        # 刻度文字
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
        # 轴标题
        d.text(((self.x0 + self.x1) / 2, self.y1 + 58), self.xlabel,
               font=F_LABEL, fill=FG, anchor="ma")
        if self.ylabel:
            # PIL anchor 在某些字体下失效，改用 bbox 手动定位
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
            # 白斜纹（稀疏），灰度打印仍可区分
            step = max(14, int((y_0 - y_v) / 8))
            for k in range(-int((y_0 - y_v) / step) - 2,
                           int((x_r - x_l) / step) + 2):
                x1 = x_l + k * step
                y1 = y_v
                x2 = x1 + (y_0 - y_v)
                y2 = y_0
                # 裁剪到矩形
                x1c, y1c = max(x1, x_l), max(y1, y_v)
                x2c, y2c = min(x2, x_r), min(y2, y_0)
                if x1c < x2c and y1c < y2c:
                    d.line([(x1c, y1c), (x2c, y2c)], fill=(255, 255, 255),
                           width=2)

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
    """编码谱系：未归约 / 归约 × 四种编码 的节点级 best-F1（均值）。"""
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
    p.draw_frame(xticklabels=["no topo.", "Σμ", "count", "RATE"])

    nfig = 2
    total = 0.56            # 每组占用宽度
    half = total / (2 * nfig) - 0.010
    gap = 0.065
    offs = {"identity": -gap, "tered": gap}
    # 只标注归约柱，避免未归约 count/RATE 重叠
    for gi, enc in enumerate(ENC_ORDER):
        for fig in ("identity", "tered"):
            v = vals.get((fig, enc))
            if v is None:
                continue
            p.bar(gi + offs[fig], v, half, FIG_COLOR[fig],
                  hatch=(fig == "tered"))
            if fig == "tered":
                d.text((p.px(gi + offs[fig]), p.py(v) - 14), "%.3f" % v,
                       font=F_ANNO, fill=FG, anchor="mb")
    legend(d, [("unreduced graph", FIG_COLOR["identity"], "bar"),
               ("reduced graph", FIG_COLOR["tered"], "bar")],
           box[0] + 18, box[1] - 8)


def panel_b(d, box, e3):
    """归约强度扫描：边归约率 → 节点级 best-F1。"""
    p = Panel(d, box, "(b) Detection under increasing reduction, node best-F1",
              "edge reduction ratio", "")
    if not e3:
        p.set_x(0, 1, [0, 0.5, 1.0])
        p.set_y(0, 1, [0, 0.5, 1.0])
        p.draw_frame()
        d.text(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2),
               "E3 pending", font=F_TITLE, fill=(180, 80, 80), anchor="mm")
        return
    # 每个 (strength, encoding) → edge_cut / node_best_f1
    per = defaultdict(list)
    for r in e3:
        gp, og = fnum(r, "n_edges_Gp"), fnum(r, "n_edges_orig")
        cut = fnum(r, "edge_cut")
        if cut is None and gp is not None and og:
            cut = 1.0 - gp / og
        if cut is None:
            continue
        per[(round(cut, 4), r.get("encoding"))].append(fnum(r, "node_best_f1"))

    series = {}
    for enc in ENC_ORDER:
        pts = sorted((c, mean(v)) for (c, e), v in per.items() if e == enc)
        pts = [(c, v) for c, v in pts if v is not None]
        if pts:
            series[enc] = pts

    allv = [v for pts in series.values() for _, v in pts]
    ymax = max(allv + [0.01]) * 1.28
    allx = [c for pts in series.values() for c, _ in pts]
    xmax = max(allx) if allx else 1.0
    p.set_x(-0.02, max(xmax * 1.12, 0.05),
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    p.set_y(0.0, ymax, [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30][
        : max(2, int(ymax / 0.05) + 1)])
    p.draw_frame()

    for enc, pts in series.items():
        pix = [(p.px(c), p.py(v)) for c, v in pts]
        p.line(pix, ENC_COLOR[enc], MARKER[enc])
    legend(d, [(ENC_LABEL[e], ENC_COLOR[e], "line")
               for e in ENC_ORDER if e in series],
           box[2] - 330, box[1] + 26)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e1", default="results/e1p/all_e1.csv")
    ap.add_argument("--e3dir", default="results/work/e3")
    ap.add_argument("--out", default="figures/fig2_encoding_strength.png")
    a = ap.parse_args()

    e1 = read_csvs(a.e1)
    e3 = read_csvs(os.path.join(a.e3dir, "*.csv"))
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
