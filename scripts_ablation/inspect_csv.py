import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), *[".."] * 1))

import csv, os, io

BASE = _os.path.join(_R, "results", "v3_final")
OUT = _os.path.join(_R, "scripts_ablation", "_out.txt")
files = ["e6_alert_full_v3.csv", "e6d_ablation_v3.csv", "e6d_ratestar_v3.csv", "e7_origunit.csv"]

buf = io.StringIO()
for fn in files:
    p = os.path.join(BASE, fn)
    if not os.path.exists(p):
        print(f"!! missing {fn}", file=buf)
        continue
    with open(p, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"\n===== {fn}: {len(rows)} rows =====", file=buf)
    print("cols:", list(rows[0].keys()), file=buf)
    # distinct config/variant combos
    def uniq(k):
        return sorted({r.get(k, "") for r in rows})
    for k in ("config", "encoding", "group", "variant", "file"):
        if k in rows[0]:
            print(f"  {k}: {uniq(k)}", file=buf)

with open(OUT, "w", encoding="utf-8") as f:
    f.write(buf.getvalue())
print("done")
