# -*- coding: utf-8 -*-
"""从服务器下载 v3 最终结果并核验覆盖完整性。"""
import csv
import os
import subprocess
from collections import Counter

SSH = r"C:\WINDOWS\System32\OpenSSH\ssh.exe"
D = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "results", "v3_final")
TARGET = os.environ.get("TERED_SSH_TARGET", "user@your-server-host")
FILES = ["e6_alert_full_v3.csv", "e6d_ablation_v3.csv", "e7_origunit.csv"]

os.makedirs(D, exist_ok=True)
for f in FILES:
    r = subprocess.run([SSH, TARGET,
                        f"cat /TeRed+RATE/code/results/{f}"],
                       capture_output=True)
    if r.returncode != 0:
        print("DOWNLOAD FAIL", f, r.stderr.decode("utf-8", "replace")[:200])
        continue
    data = r.stdout.decode("utf-8", "replace")
    with open(os.path.join(D, f), "w", encoding="utf-8", newline="") as fh:
        fh.write(data)
    print(f, len(data.splitlines()), "lines")

print("\n=== coverage check ===")
for f in FILES:
    p = os.path.join(D, f)
    if not os.path.exists(p):
        continue
    rows = list(csv.DictReader(open(p, encoding="utf-8-sig")))
    files = sorted(set(r["file"] for r in rows))
    cfgs = Counter(r["config"] for r in rows)
    print(f"{f}: {len(rows)} rows, {len(files)} files {files}")
    print("   configs:", dict(cfgs))
