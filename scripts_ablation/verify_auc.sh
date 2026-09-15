#!/bin/bash
cd /TeRed+RATE/code || exit 1
PY=${HOME}/miniconda3/envs/pids/bin/python
export PYTHONUNBUFFERED=1
$PY - <<'EOF'
import numpy as np, sys
sys.path.insert(0, '/TeRed+RATE/code')
from sklearn.metrics import roc_auc_score
from eval.metrics import _auc, _pr_auc
rng = np.random.default_rng(0)
for trial in range(5):
    n = 2000; npos = 30
    y = np.zeros(n, dtype=np.int64); y[:npos] = 1
    s = rng.normal(0, 1, n) + y * rng.normal(0.8, 0.5, n)
    mine = _auc(y == 1, s); ref = roc_auc_score(y, s)
    print(f"trial{trial}: mine={mine:.6f} sklearn={ref:.6f} diff={abs(mine-ref):.2e}")
print("PR_AUC sample:", round(_pr_auc(y == 1, s), 6))
print("AUC_OK" if abs(mine-ref) < 1e-9 else "AUC_MISMATCH")
EOF
