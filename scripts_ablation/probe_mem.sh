#!/bin/bash
cd /TeRed+RATE/code || exit 1
PY=${HOME}/miniconda3/envs/pids/bin/python
export PYTHONUNBUFFERED=1
$PY - <<'EOF'
import sys, time
sys.path.insert(0, '/TeRed+RATE/code')
import numpy as np, torch
try:
    import torch_sparse
    print("torch_sparse:", torch_sparse.__version__)
except Exception as e:
    print("torch_sparse: MISSING", e)
try:
    import torch_scatter
    print("torch_scatter:", torch_scatter.__version__)
except Exception as e:
    print("torch_scatter: MISSING", e)
from rate_core import load_pickle
from scripts.e5_f1_eval import resolve_name, find_templates
from scripts.e6b_alert_replay import load_res
tpl = find_templates('cache/darpa/e5_templates/*.jsonl')
stem = tpl.split('/')[-1].replace('.jsonl','')
rc = 'cache/darpa/e5_f1_reduce'
gid = resolve_name('/TeRed+RATE/dataset/darpae5/cadets', 'bin.1')
res = load_res(rc, 'tered', stem, 'tr400000', gid)
G = res.Gp
print("tered bin.1: nodes", G.n_nodes(), "edges", len(G.edges))
res2 = load_res(rc, 'id', stem, 'tr400000', gid)
print("id    bin.1: nodes", res2.Gp.n_nodes(), "edges", len(res2.Gp.edges))
EOF
