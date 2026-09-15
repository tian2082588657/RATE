# -*- coding: utf-8 -*-
"""汇总 e5_f1_per_file.csv / e5_f1_smoke.csv -> 三配置对比总表（M2 裁决）。

用法: python scripts/e5_f1_summarize.py results/e5_f1_per_file.csv [更多csv...]
输出: 每配置 micro-P/R/F1（按 TP/FP/FN 累计） + 每文件明细。
"""
import sys, csv, collections

FILES = sys.argv[1:] or ["results/e5_f1_per_file.csv"]

rows = []
for f in FILES:
    try:
        with open(f, newline="", encoding="utf-8") as fh:
            rows += list(csv.DictReader(fh))
    except FileNotFoundError:
        print(f"skip {f}")

by_cfg = collections.defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0, "files": []})
per_cfg_file = collections.defaultdict(list)
for r in rows:
    cfg = r.get("config", "?")
    try:
        tp, fp, fn = int(r["TP"]), int(r["FP"]), int(r["FN"])
    except (KeyError, ValueError):
        continue
    c = by_cfg[cfg]
    c["TP"] += tp; c["FP"] += fp; c["FN"] += fn
    c["files"].append(r)
    per_cfg_file[cfg].append(r)

print("=" * 96)
print("E5 CADETS 三配置节点级 F1（M2 裁决） — ground truth: Nginx Drakon APT (ubc-provenance)")
print("=" * 96)
print("%-14s %6s %6s %6s %8s %8s %8s %10s" %
      ("配置", "TP", "FP", "FN", "P", "R", "F1", "文件数"))
order = ["无归约基线", "TeRed+naive", "TeRed+RATE"]
for cfg in order + [k for k in by_cfg if k not in order]:
    c = by_cfg.get(cfg)
    if not c:
        continue
    tp, fp, fn = c["TP"], c["FP"], c["FN"]
    p = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * rc / (p + rc) if p + rc else 0.0
    print("%-14s %6d %6d %6d %8.4f %8.4f %8.4f %10d" %
          (cfg, tp, fp, fn, p, rc, f1, len(c["files"])))

print("\n— 每文件明细 —")
for cfg in order:
    print(f"\n[{cfg}]")
    for r in per_cfg_file.get(cfg, []):
        try:
            print("  %-10s TP=%-4s FP=%-6s FN=%-4s F1=%-8s surv=%-6s atk_in_Gp=%s" % (
                (r.get("attack_files", "?")[-10:]), r["TP"], r["FP"], r["FN"],
                r.get("F1", "?"), r.get("surv_" + r.get("attack_files", "x"), "?"),
                r.get("n_atk", "?")))
        except Exception:
            print("  row:", {k: r[k] for k in list(r)[:8]})
