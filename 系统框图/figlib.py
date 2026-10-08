# -*- coding: utf-8 -*-
"""figlib -- tiny dual-emitter drawing library.

One geometry model is emitted twice:
  * native mxGraph XML  (.drawio)   -> editable in draw.io
  * PIL raster          (.png)      -> instant preview, no draw.io install

Element kinds: box, ellipse, poly (polyline, optional arrow head), text.
Composite helpers (band, ruler, check, cross, bang, sinwave, graph, merge)
are built from those primitives so both emitters stay trivial.

Authoring scale: 1 unit = 1 pt.  A canvas of 992 units prints at 17.5 cm, so
authored font 14 -> printed 7 pt, authored 15 -> 7.5 pt.
"""
import math
import os

W = 992.0
MARGIN = 8.0
F_T = 15.0          # panel titles (bold)
F_L = 14.0          # every other label (>= 7 pt printed)
LH = 17.0
GAP = 5.0
TITLE_H = 38.0
PAD = 6.0
ROW_GAP = 5.0

INK = "#1B1B1B"
MUTED = "#546E7A"
EDGE_C = "#37474F"
GREEN = "#2E7D32"
ORANGE = "#E65100"
RED = "#C62828"
BLUE = "#1565C0"

E = []
RECTS = {}
_ids = [0]


def nid(p="e"):
    _ids[0] += 1
    return "%s%d" % (p, _ids[0])


def _mk(kind, parent, **kw):
    el = dict(kind=kind, id=nid(kind[0]), parent=parent)
    el.update(kw)
    E.append(el)
    return el


# ------------------------------------------------------------------ shape API
def box(parent, x, y, w, h, text="", fill="#FFFFFF", stroke=EDGE_C, font=F_L,
        bold=False, dashed=False, rounded=True, sw=1.0, align="center",
        valign="middle", color=INK):
    return _mk("box", parent, x=x, y=y, w=w, h=h, text=text, fill=fill,
               stroke=stroke, font=font, bold=bold, dashed=dashed,
               rounded=rounded, sw=sw, align=align, valign=valign, color=color)


def ellipse(parent, x, y, w, h, text="", fill="#FFFFFF", stroke=EDGE_C,
            font=F_L, bold=False, dashed=False, sw=1.0, color=INK):
    return _mk("ellipse", parent, x=x, y=y, w=w, h=h, text=text, fill=fill,
               stroke=stroke, font=font, bold=bold, dashed=dashed, sw=sw,
               color=color)


def circle(parent, cx, cy, r, text="", **kw):
    return ellipse(parent, cx - r, cy - r, 2 * r, 2 * r, text, **kw)


def poly(parent, pts, color=EDGE_C, sw=1.0, dashed=False, arrow=False,
         label="", font=F_L):
    el = _mk("poly", parent, pts=list(pts), color=color, sw=sw, dashed=dashed,
             arrow=arrow, label=label, font=font)
    return el


def text(parent, x, y, w, h, s, font=F_L, align="center", color=INK,
         bold=False):
    return _mk("text", parent, x=x, y=y, w=w, h=h, text=s, font=font,
               align=align, color=color, bold=bold)


def note(parent, x, y, w, h, s, font=F_L, align="left", color=MUTED):
    return _mk("text", parent, x=x, y=y, w=w, h=h, text=s, font=font,
               align=align, color=color, bold=False)


# ------------------------------------------------------------ composite parts
def sinwave(parent, x, y, w, h, cycles=1.5, color=BLUE, sw=1.4):
    pts = []
    n = 24
    for i in range(n + 1):
        t = i / float(n)
        pts.append((x + w * t, y + h / 2.0 -
                    h / 2.0 * math.sin(2 * math.pi * cycles * t)))
    return poly(parent, pts, color=color, sw=sw)


def graph_glyph(parent, nodes, edges, r=9.0, node_fill="#FFFFFF",
                node_stroke=EDGE_C, sw=1.0, hi=None, hi_fill="#FFE0B2",
                hi_alpha_rect=True, labels=True, font=9.0):
    """nodes = [(x, y, type, label)], edges = [(i, j)]."""
    out = []
    if hi is not None:
        xs = [nodes[i][0] for i in hi]
        ys = [nodes[i][1] for i in hi]
        pad = r + 7
        out.append(box(parent, min(xs) - pad, min(ys) - pad,
                       max(xs) - min(xs) + 2 * pad,
                       max(ys) - min(ys) + 2 * pad, "",
                       fill="#FFE0B2", stroke=ORANGE, rounded=True,
                       sw=1.0, dashed=False))
    for i, j in edges:
        x1, y1 = nodes[i][0], nodes[i][1]
        x2, y2 = nodes[j][0], nodes[j][1]
        dx, dy = x2 - x1, y2 - y1
        d = math.hypot(dx, dy) or 1.0
        out.append(poly(parent, [(x1 + dx / d * r, y1 + dy / d * r),
                                 (x2 - dx / d * (r + 3), y2 - dy / d * (r + 3))],
                        sw=sw, arrow=True))
    for i, (x, y, tp, lab) in enumerate(nodes):
        f = node_fill
        if tp == "file":
            f = "#E3F2FD"
        elif tp == "socket":
            f = "#FFF3E0"
        elif tp == "summary":
            f = "#FFE0B2"
        out.append(circle(parent, x, y, r, lab if labels else "", fill=f,
                          stroke=node_stroke, font=font, bold=lab == "s",
                          sw=1.4 if tp == "summary" else sw))
    return out


def merge_glyph(parent, x, y, w, h, n=4, color=EDGE_C, sw=1.2, arrow=True,
                fan=0.55):
    """Several lines from the left converging into one node at the right."""
    pts_all = []
    for i in range(n):
        yy = y + h * (i + 0.5) / n
        pts_all.append(poly(parent, [(x, yy), (x + w * (1 - fan), yy),
                                     (x + w, y + h / 2.0)], color=color,
                            sw=sw, arrow=(arrow and i == 0)))
    return pts_all


def band(parent, x, y, w, h, segs, colors=None, stroke=EDGE_C):
    """Segmented colour band (tensor / descriptor)."""
    colors = colors or ["#BBDEFB", "#FFF3E0", "#F8BBD0", "#C8E6C9",
                        "#FFF9C4", "#D1C4E9", "#FFE0B2"]
    n = len(segs)
    out = []
    sw_ = w / float(n)
    for i, s in enumerate(segs):
        out.append(box(parent, x + i * sw_, y, sw_, h, s,
                       fill=colors[i % len(colors)], stroke=stroke,
                       font=F_L, sw=0.8))
    return out


def ruler(parent, x, y, w, h, ticks=11, label="", color=EDGE_C):
    out = [box(parent, x, y, w, h, "", fill="#FFFFFF", stroke=color, sw=1.2,
               rounded=False)]
    for i in range(ticks):
        tx = x + w * (i + 0.5) / ticks
        out.append(box(parent, tx - 0.5, y + h - 5, 1.0, 5, "", fill=color,
                       stroke=color, sw=0))
    if label:
        out.append(text(parent, x, y - 20, w, 18, label, font=F_L))
    return out


def bars(parent, x, y, w, h, heights, color="#7CB342", stroke=EDGE_C):
    out = []
    n = len(heights)
    bw = w / (2.0 * n - 1)
    for i, hh in enumerate(heights):
        bx = x + i * bw * 2
        out.append(box(parent, bx, y + h - h * hh, bw, h * hh, "",
                       fill=color, stroke=stroke, sw=0.8, rounded=False))
    return out


def check(parent, cx, cy, s=9.0, color=GREEN, sw=2.6):
    return poly(parent, [(cx - s, cy), (cx - s * 0.25, cy + s * 0.7),
                         (cx + s, cy - s * 0.8)], color=color, sw=sw)


def cross(parent, cx, cy, s=8.0, color=RED, sw=2.6):
    return [poly(parent, [(cx - s, cy - s), (cx + s, cy + s)], color=color,
                 sw=sw),
            poly(parent, [(cx - s, cy + s), (cx + s, cy - s)], color=color,
                 sw=sw)]


def bang(parent, cx, cy, s=9.0, color=ORANGE, sw=2.4):
    return [poly(parent, [(cx, cy - s), (cx, cy + s * 0.25)], color=color,
                 sw=sw),
            circle(parent, cx, cy + s * 0.75, 1.5, fill=color, stroke=color,
                   sw=0)]


def strike(parent, x1, y1, x2, y2, color=RED, sw=1.6):
    return poly(parent, [(x1, y1), (x2, y2)], color=color, sw=sw)


# ------------------------------------------------------------------- geometry
def auto_h(s, w, font=F_L):
    cpl = max(5.0, (w - 8.0) / (font * 0.55))
    lines = sum(max(1, int(math.ceil(len(seg) / cpl)))
                for seg in s.split("\n"))
    return max(20.0, lines * LH + 7.0)


def natural(rows, gap=GAP):
    return sum(r["h"] for r in rows) + gap * (len(rows) - 1)


def panel(cid, px, py, pw, title, rows, spec, target_h=None):
    """rows = list of (height, draw_fn(x, y, iw, cid)) tuples."""
    x0, iw = px + PAD, pw - 2 * PAD
    nat = natural([dict(h=h) for h, _ in rows], GAP)
    gap = GAP
    if target_h:
        gap = min(15.0, GAP + max(0.0, target_h - TITLE_H - 2 * PAD - nat) /
                  max(1, len(rows) - 1))
    block = natural([dict(h=h) for h, _ in rows], gap)
    total = target_h or (TITLE_H + 2 * PAD + nat)
    box("1", px, py, pw, total, "", fill=spec["fill"], stroke=spec["stroke"],
        rounded=True, sw=1.0)
    E[-1]["id"] = cid
    E[-1]["is_panel"] = True
    RECTS[cid] = (px, py, pw, total)
    tb = box("1", px, py, pw, TITLE_H, title, fill=spec["title"],
             stroke=spec["title"], font=F_T, color="#FFFFFF", bold=True,
             align="left", rounded=True, valign="top")
    RECTS[tb["id"]] = (px, py, pw, TITLE_H)
    cy = py + TITLE_H + PAD + max(0.0, (total - TITLE_H - 2 * PAD - block) / 2.0)
    for h, fn in rows:
        fn(x0, cy, iw, cid)
        cy += h + gap
    return total


# =============================================================== drawio output
def esc(t):
    return (t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
             .replace('"', "&quot;"))


def _style(el, container=False):
    if el["kind"] == "text":
        return ("text;html=1;fontSize=%d;fontColor=%s;align=%s;"
                "verticalAlign=middle;fontFamily=Helvetica;%s"
                % (el["font"], el["color"], el.get("align", "left"),
                   "fontStyle=1;" if el.get("bold") else ""))
    if el["kind"] == "poly":
        s = ("endArrow=%s;endFill=1;html=1;rounded=0;strokeWidth=%s;"
             "strokeColor=%s;%s"
             % ("block" if el["arrow"] else "none", el["sw"], el["color"],
                "dashed=1;dashPattern=2 3;" if el["dashed"] else ""))
        return s
    p = []
    if container:
        p.append("container=1;collapsible=0;")
    if el["kind"] == "ellipse":
        p.append("ellipse;")
    else:
        p.append("rounded=1;" if el["rounded"] else "rounded=0;")
    p.append("whiteSpace=wrap;html=1;")
    p.append("fillColor=%s;strokeColor=%s;" % (el["fill"], el["stroke"]))
    if el.get("dashed"):
        p.append("dashed=1;dashPattern=2 3;")
    p.append("strokeWidth=%s;" % el["sw"])
    p.append("fontSize=%d;fontFamily=Helvetica;fontColor=%s;"
             % (el["font"], el["color"]))
    if el.get("bold"):
        p.append("fontStyle=1;")
    p.append("align=%s;verticalAlign=%s;"
             % (el.get("align", "center"), el.get("valign", "middle")))
    p.append("arcSize=10;")
    return "".join(p)


def to_drawio(path, H):
    o = ['<mxfile host="app.diagrams.net" type="device">',
         '  <diagram name="fig1_system_architecture" id="fig1">',
         '    <mxGraphModel dx="1600" dy="900" grid="0" gridSize="10" '
         'page="1" pageWidth="%d" pageHeight="%d" math="0" shadow="0">'
         % (int(W), int(H) + 4),
         '      <root>', '        <mxCell id="0"/>',
         '        <mxCell id="1" parent="0"/>']
    panel_ids = [k for k, v in RECTS.items() if k.startswith("p")]
    for el in E:
        par = el["parent"]
        ox = oy = 0.0
        if par in panel_ids:
            ox, oy = RECTS[par][0], RECTS[par][1]
        if el["kind"] == "poly":
            pts = el["pts"]
            o.append('        <mxCell id="%s" style="%s" edge="1" parent="%s">'
                     % (el["id"], _style(el), par))
            o.append('          <mxGeometry relative="1" as="geometry">')
            o.append('            <mxPoint x="%s" y="%s" as="sourcePoint"/>'
                     % (round(pts[0][0] - ox, 1), round(pts[0][1] - oy, 1)))
            o.append('            <mxPoint x="%s" y="%s" as="targetPoint"/>'
                     % (round(pts[-1][0] - ox, 1), round(pts[-1][1] - oy, 1)))
            if len(pts) > 2:
                o.append('            <Array as="points">')
                for px_, py_ in pts[1:-1]:
                    o.append('              <mxPoint x="%s" y="%s"/>'
                             % (round(px_ - ox, 1), round(py_ - oy, 1)))
                o.append('            </Array>')
            o.append('          </mxGeometry>')
            o.append('        </mxCell>')
            continue
        gx, gy = round(el["x"] - ox, 1), round(el["y"] - oy, 1)
        o.append('        <mxCell id="%s" value="%s" style="%s" vertex="1" '
                 'parent="%s">' % (el["id"], esc(el["text"]),
                                   _style(el, container=bool(el.get("is_panel"))),
                                   par))
        o.append('          <mxGeometry x="%s" y="%s" width="%s" height="%s" '
                 'as="geometry"/>' % (gx, gy, round(el["w"], 1),
                                      round(el["h"], 1)))
        o.append('        </mxCell>')
    o += ['      </root>', '    </mxGraphModel>', '  </diagram>', '</mxfile>']
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(o) + "\n")
    return path


# ================================================================== png output
def render_png(path, H, scale=2.6):
    from PIL import Image, ImageDraw, ImageFont
    reg = "C:/Windows/Fonts/arial.ttf"
    bold = "C:/Windows/Fonts/arialbd.ttf"
    cache = {}

    def fnt(size, b=False):
        k = (int(size * scale), b)
        if k not in cache:
            cache[k] = ImageFont.truetype(bold if b else reg, k[0])
        return cache[k]

    img = Image.new("RGB", (int(W * scale) + 8, int(H * scale) + 8), "white")
    d = ImageDraw.Draw(img)

    def wrap(s, f, maxw):
        out = []
        for seg in s.split("\n"):
            cur = ""
            for wd in seg.split(" "):
                t = (cur + " " + wd).strip()
                if d.textlength(t, font=f) <= maxw or not cur:
                    cur = t
                else:
                    out.append(cur)
                    cur = wd
            out.append(cur)
        return out

    def dash_line(x1, y1, x2, y2, col, wd, on=6.0, off=4.0):
        dx, dy = x2 - x1, y2 - y1
        L = math.hypot(dx, dy) or 1.0
        n = max(1, int(L / (on + off)))
        for i in range(n + 1):
            a = (i * (on + off)) / L
            b = min(1.0, a + on / L)
            d.line([x1 + dx * a, y1 + dy * a, x1 + dx * b, y1 + dy * b],
                   fill=col, width=wd)

    def arrow_head(p, q, col, sz):
        dx, dy = q[0] - p[0], q[1] - p[1]
        if abs(dx) >= abs(dy):
            ax, ay = (1 if dx >= 0 else -1), 0
        else:
            ax, ay = 0, (1 if dy >= 0 else -1)
        d.polygon([(q[0], q[1]),
                   (q[0] - ax * sz - ay * sz * .55, q[1] - ay * sz - ax * sz * .55),
                   (q[0] - ax * sz + ay * sz * .55, q[1] - ay * sz + ax * sz * .55)],
                  fill=col)

    for el in E:
        k = el["kind"]
        if k == "poly":
            pts = [(x * scale, y * scale) for x, y in el["pts"]]
            col = el["color"]
            wd = max(1, int(round(el["sw"] * scale)))
            for i in range(len(pts) - 1):
                if el["dashed"]:
                    dash_line(pts[i][0], pts[i][1], pts[i + 1][0],
                              pts[i + 1][1], col, wd)
                else:
                    d.line([pts[i], pts[i + 1]], fill=col, width=wd)
            if el["arrow"]:
                arrow_head(pts[-2], pts[-1], col, 6.5 * scale)
            continue
        if k == "text":
            f = fnt(el["font"], el.get("bold", False))
            lines = wrap(el["text"], f, max(20.0, el["w"] * scale - 7 * scale))
            lh = el["font"] * 1.22 * scale
            ty = (el["y"] + el["h"] / 2.0) * scale - len(lines) * lh / 2.0
            if el.get("align") == "left":
                tx = (el["x"] + 5.0) * scale
                anc = "lm"
            else:
                tx = (el["x"] + el["w"] / 2.0) * scale
                anc = "mm"
            for i, t in enumerate(lines):
                d.text((tx, ty + i * lh + lh / 2.0), t, font=f, fill=el["color"],
                       anchor=anc)
            continue
        x, y = el["x"] * scale, el["y"] * scale
        w_, h_ = el["w"] * scale, el["h"] * scale
        fill = None if el["fill"] in ("none", "") else el["fill"]
        stroke = None if el["stroke"] in ("none", "") else el["stroke"]
        wd = max(1, int(round(el["sw"] * scale)))
        if k == "ellipse":
            if el["dashed"]:
                cx, cy = x + w_ / 2.0, y + h_ / 2.0
                rx, ry = w_ / 2.0, h_ / 2.0
                n = 48
                for i in range(0, n, 2):
                    a0 = 2 * math.pi * i / n
                    a1 = 2 * math.pi * (i + 1) / n
                    d.line([cx + rx * math.cos(a0), cy + ry * math.sin(a0),
                            cx + rx * math.cos(a1), cy + ry * math.sin(a1)],
                           fill=stroke or EDGE_C, width=wd)
                if fill:
                    d.ellipse([x, y, x + w_, y + h_], fill=fill, outline=None)
            else:
                d.ellipse([x, y, x + w_, y + h_], fill=fill, outline=stroke,
                          width=wd)
        else:
            r = 10 * scale if el.get("rounded") else 0
            if el["dashed"]:
                if fill:
                    d.rounded_rectangle([x, y, x + w_, y + h_], radius=r,
                                        fill=fill)
                dash_line(x, y, x + w_, y, stroke or EDGE_C, wd)
                dash_line(x + w_, y, x + w_, y + h_, stroke or EDGE_C, wd)
                dash_line(x + w_, y + h_, x, y + h_, stroke or EDGE_C, wd)
                dash_line(x, y + h_, x, y, stroke or EDGE_C, wd)
            elif fill or stroke:
                if el["sw"] > 0:
                    d.rounded_rectangle([x, y, x + w_, y + h_], radius=r,
                                        fill=fill, outline=stroke, width=wd)
                elif fill:
                    d.rounded_rectangle([x, y, x + w_, y + h_], radius=r,
                                        fill=fill)
        if not el.get("text"):
            continue
        f = fnt(el["font"], el.get("bold", False))
        lines = wrap(el["text"], f, w_ - 7 * scale)
        lh = el["font"] * 1.22 * scale
        if el.get("valign") == "top":
            ty = y + 3 * scale
        else:
            ty = y + h_ / 2.0 - len(lines) * lh / 2.0
        tx = x + 5 * scale if el.get("align") == "left" else x + w_ / 2.0
        anc = "lm" if el.get("align") == "left" else "mm"
        for i, t in enumerate(lines):
            d.text((tx, ty + i * lh + lh / 2.0), t, font=f, fill=el["color"],
                   anchor=anc)
    img.save(path)
    return path

