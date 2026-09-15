#!/bin/bash
cd /TeRed+RATE/code || exit 1
echo "=== procs ==="
ps -ef | grep -E "e6e_gnn_baselin[e]|e6g_cost_profil[e]|followu[p]" | head -6
echo "=== logs ==="
ls -la logs/ | grep -E "e6e|e6g"
echo "=== tail main ==="
tail -3 logs/e6e_gnn_baseline.log
echo "=== rows ==="
wc -l results/e6e_gnn_baseline.csv
