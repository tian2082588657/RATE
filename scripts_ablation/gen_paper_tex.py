# -*- coding: utf-8 -*-
"""从修正后的 CSV 生成论文各表的 LaTeX 表体（v3 口径：coverage bug 已修）。

用法: python gen_paper_tex.py <results_dir> [--tex]

读取:
  e6_alert_full_v3.csv  三配置全流程（identity / TeRed+naive / TeRed+RATE）
  e6d_ablation_v3.csv   消融：A1 管线 / A3 排序键 / A4 结构特征 / A5 编码（含固定预算列）
  e7_origunit.csv       统一评估单元（归约图告警映射回原图 GT）

对齐关系（重要，勿改）:
  - e6d A1 variant 'base'        = bfs q75 + wmax=1.0 + sort=cmax  → 论文「reported」
  - e6d A1 variant 'legacy_agg'  = bfs q75 + wmax=0.7 + sort=agg   → 论文「cluster mean mixture」
  - e6d A1 variant 'sort_p95' 在 A3 组                            → 论文「within-cluster P95」
  - legacy_w1 (wmax=1.0,sort=agg) 与 base 逐位相同（已验证）——正文里那条「同一处改动」的论据
  - config→encoding: identity+rate→'rate'; TeRed+naive→'dual_naive';
    TeRed+RATE→'rate'; TeRed+RATE*→'rate_ratio'
    （原图上 μ≡1，rate≡dual_naive，故 identity 行编码不影响结论；
     归约图上 'rate'=Σμ 质量加权双通道；'rate_ratio'=count+折叠比值=§3.4 的 RATE*，
     即论文报告的 TeRed+RATE。'TeRed+RATE'（Σμ）保留作 tab:abl-enc 的对照行。
     A5 另给 rate_single / none / rate_ratio / rate_ratio_only / rate_rank。）
"""
import csv, sys, os, statistics as st

BASE = sys.argv[1] if len(sys.argv) > 1 else "."
# 论文报告的 TeRed+RATE == 脚本里的 TeRed+RATE*（rate_ratio 编码，即 §3.4 的 RATE*）。
# 脚本里的 "TeRed+RATE"（rate 编码，Σμ 质量加权）保留作 tab:abl-enc 的编码对照行。
CFG2PAPER = {"identity+rate": "identity", "TeRed+naive": "TeRed+count",
             "TeRed+RATE": "TeRed+RATE", "TeRed+RATE*": "TeRed+RATE"}
CONFIGS = ["identity+rate", "TeRed+naive", "TeRed+RATE*"]


def load(fn):
    p = os.path.join(BASE, fn)
    if not os.path.exists(p):
        print(f"% [warn] missing {fn}")
        return None
    with open(p, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def vals(rows, fld):
    out = []
    for r in rows:
        try:
            out.append(float(r[fld]))
        except (KeyError, ValueError, TypeError):
            pass
    return out


def m(rows, fld, nd=3):
    v = vals(rows, fld)
    return f"{st.mean(v):.{nd}f}" if v else "---"


def ms(rows, fld, nd=3):
    v = vals(rows, fld)
    if not v:
        return "---"
    if len(v) == 1:
        return f"{v[0]:.{nd}f}"
    return f"{st.mean(v):.{nd}f}$\\pm${st.pstdev(v):.{nd}f}"


def sel(rows, **kw):
    return [r for r in rows
            if all(r.get(k, "") == v for k, v in kw.items())]


a3f = load("e6_alert_full_v3.csv")
abf = load("e6d_ablation_v3.csv") + load("e6d_ratestar_v3.csv")
e7f = load("e7_origunit.csv")

NF = max([len(set(r["file"] for r in x)) for x in (a3f, abf, e7f) if x] or [0])
print(f"% ===== gen_paper_tex: files per group = {NF} =====")

# ---------------------------------------------------------------- tab:main
if abf:
    print("\n% ==== tab:main (cols: node best-F1 | node cov | F1@5 | F1@10 | orig recall | load) ====")
    for cfg in CONFIGS:
        rows = sel(abf, group="A1", variant="base", config=cfg)
        o = sel(e7f, config=cfg) if e7f else []
        print(f"  {CFG2PAPER[cfg]:12s} & {m(rows,'node_best_f1',4)} & {m(rows,'node_cov',3)} "
              f"& {m(rows,'F1_alert@5')} & {m(rows,'F1_alert@10')} "
              f"& {m(o,'orig_recall',3)} & {m(rows,'n_alerted_nodes',0)} \\\\   % n={len(rows)}")

# ---------------------------------------------------------------- tab:abl-a1
if abf:
    print("\n% ==== tab:abl-a1 (pipeline; best alert-F1 | F1@5) ====")
    for var in ["base", "legacy_agg", "no_bfs", "bfs_q50", "bfs_q90"]:
        cells = []
        for cfg in CONFIGS:
            rows = sel(abf, group="A1", variant=var, config=cfg)
            cells.append(m(rows, "best_F1_alert"))
        print(f"  {var:12s} & " + " & ".join(cells) + " \\\\")
    print("  % F1@5 variant:")
    for var in ["base", "legacy_agg", "no_bfs", "bfs_q50", "bfs_q90"]:
        cells = []
        for cfg in CONFIGS:
            rows = sel(abf, group="A1", variant=var, config=cfg)
            cells.append(m(rows, "F1_alert@5"))
        print(f"  {var:12s} & " + " & ".join(cells) + " \\\\")

# ---------------------------------------------------------------- tab:abl-a3
if abf:
    print("\n% ==== tab:abl-a3 (ranking key, TeRed+RATE, fixed budgets) ====")
    for label, grp, var in [("within-cluster max (reported)", "A1", "base"),
                            ("cluster mean mixture", "A1", "legacy_agg"),
                            ("within-cluster P95", "A3", "sort_p95")]:
        rows = sel(abf, group=grp, variant=var, config="TeRed+RATE")
        cells = [m(rows, f"F1_alert@{b}") for b in (1, 2, 3, 5, 10, 20)]
        print(f"  {label:30s} & " + " & ".join(cells) + " \\\\")

# ---------------------------------------------------------------- tab:abl-v4
if abf:
    print("\n% ==== tab:abl-v4 (features, TeRed+count; node best-F1 | F1@5 | cov | delta node) ====")
    base = sel(abf, group="A1", variant="base", config="TeRed+naive")
    base_n = st.mean(vals(base, "node_best_f1")) if vals(base, "node_best_f1") else None
    if base_n:
        print(f"  {'full (three features)':30s} & {m(base,'node_best_f1',4)} & {m(base,'F1_alert@5')} "
              f"& {m(base,'node_cov')} & --- \\\\")
    for var, lab in [("v4_drop_log_deg", "$-$ log-degree"),
                     ("v4_drop_out_in_ratio", "$-$ out/in ratio"),
                     ("v4_drop_self_loop", "$-$ self-loop flag")]:
        rows = sel(abf, group="A4", variant=var, config="TeRed+naive")
        vn = vals(rows, "node_best_f1")
        dl = f"{(st.mean(vn)/base_n-1)*100:+.0f}\\%" if (vn and base_n) else "---"
        print(f"  {lab:30s} & {m(rows,'node_best_f1',4)} & {m(rows,'F1_alert@5')} "
              f"& {m(rows,'node_cov')} & {dl} \\\\")

# ---------------------------------------------------------------- tab:abl-enc
if abf:
    print("\n% ==== tab:abl-enc (identical reduced graph; node best-F1 | F1@5 | F1@10 | cov) ====")
    ENC = [("dual_naive", "A5", "dual-channel edge count (reduction-blind)"),
           ("rate", "A1", "dual-channel $\\Sigma\\mu$ mass-weighted"),
           ("rate_ratio", "A5", "count $+$ collapse-ratio (= reported TeRed+RATE)"),
           ("rate_ratio_only", "A5", "collapse-ratio channel only"),
           ("rate_rank", "A5", "rank-normalised mass-weighted"),
           ("rate_single", "A5", "single-channel mass-weighted"),
           ("none", "A5", "no topology encoding")]
    for enc, grp, lab in ENC:
        if grp == "A5":
            rows = [r for r in abf if r["group"] == "A5" and r["encoding"] == enc]
        else:
            rows = sel(abf, group="A1", variant="base",
                       config="TeRed+RATE", encoding=enc)
        if not rows:
            print(f"  % {lab}: no rows (enc={enc})")
            continue
        print(f"  {lab:46s} & {m(rows,'node_best_f1',4)} & {m(rows,'F1_alert@5')} "
              f"& {m(rows,'F1_alert@10')} & {m(rows,'node_cov')} \\\\   % n={len(rows)} enc={enc} grp={grp}")

# ---------------------------------------------------------------- tab:abl-budget
if abf:
    print("\n% ==== tab:abl-budget (fixed-budget F1@b) ====")
    for cfg in CONFIGS:
        rows = sel(abf, group="A1", variant="base", config=cfg)
        cells = [m(rows, f"F1_alert@{b}") for b in (1, 2, 3, 5, 10, 20)]
        print(f"  {CFG2PAPER[cfg]:12s} & " + " & ".join(cells) + " \\\\")

# ---------------------------------------------------------------- tab:runtime
if a3f:
    print("\n% ==== tab:runtime (scoring stage) ====")
    files = []
    for r in a3f:
        if r["file"] not in files:
            files.append(r["file"])
    print("  % file & identity nodes & tered nodes & identity s & tered s")
    for f in files:
        ri = sel(a3f, config="identity+rate", file=f)
        rt = sel(a3f, config="TeRed+RATE", file=f)
        print(f"  {f:10s} & {m(ri,'n_nodes_Gp',0)} & {m(rt,'n_nodes_Gp',0)} "
              f"& {m(ri,'runtime_s',1)} & {m(rt,'runtime_s',1)} \\\\")
    ti = sum(vals(sel(a3f, config="identity+rate"), "runtime_s"))
    tt = sum(vals(sel(a3f, config="TeRed+RATE"), "runtime_s"))
    print(f"  % 合计 identity={ti:.1f}s tered={tt:.1f}s  ({(tt/ti-1)*100:+.1f}%)")

# ---------------------------------------------------------------- tab:origunit
if e7f:
    print("\n% ==== tab:origunit (common unit: alerts mapped back to original graph) ====")
    for cfg in CONFIGS:
        rows = sel(e7f, config=cfg)
        if not rows:
            continue
        print(f"  {CFG2PAPER[cfg]:12s} & {ms(rows,'orig_recall',3)} & {ms(rows,'F1_orig',3)} "
              f"& {m(rows,'alert_P_orig')} & {m(rows,'node_cov_red')} \\\\   % n={len(rows)}")
    print("\n% ---- GT 存活/塌缩统计 (TeRed+RATE 行) ----")
    rows = sel(e7f, config="TeRed+RATE")
    for r in rows:
        print(f"  {r['file']:10s} orig_gt={r['n_orig_gt']:>3s} kept={r['n_gt_kept']:>3s} "
              f"absorbed={r['n_gt_absorbed']:>3s} boundary={r['n_boundary_gt']:>2s} "
              f"worstDdeg={r['worst_deg_delta']:>5s} regions={r['n_regions']} removed={r['n_removed']}")
    if rows:
        tot = sum(int(r["n_orig_gt"]) for r in rows)
        kept = sum(int(r["n_gt_kept"]) for r in rows)
        bnd = sum(int(r["n_boundary_gt"]) for r in rows)
        wdd = max(float(r["worst_deg_delta"]) for r in rows)
        print(f"  % 合计 orig_gt={tot} kept={kept} absorbed={tot-kept} "
              f"({kept/tot*100:.1f}% 存活) boundary={bnd} worst|Δdeg|={wdd}")

# ---------------------------------------------------------------- 节点级配对检验
if abf:
    print("\n% ==== 节点级配对（rate_ratio vs dual_naive，A5 组同图同管线） ====")
    a = {r["file"]: float(r["node_best_f1"])
         for r in sel(abf, group="A5", encoding="rate_ratio")}
    b = {r["file"]: float(r["node_best_f1"])
         for r in sel(abf, group="A5", encoding="dual_naive")}
    common = sorted(set(a) & set(b))
    if common:
        w = sum(1 for f in common if a[f] > b[f])
        l = sum(1 for f in common if a[f] < b[f])
        t = len(common) - w - l
        print(f"  n={len(common)} 胜{w}/负{l}/平{t}  均值 {st.mean([a[f] for f in common]):.4f} "
              f"vs {st.mean([b[f] for f in common]):.4f}")
