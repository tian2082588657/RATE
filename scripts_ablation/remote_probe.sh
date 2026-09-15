#!/bin/bash
PY=${HOME}/miniconda3/envs/pids/bin/python
$PY - <<'EOF'
import torch, torch_geometric
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("pyg", torch_geometric.__version__)
try:
    import numpy, sklearn
    print("numpy", numpy.__version__, "sklearn", sklearn.__version__)
except Exception as e:
    print("extra import fail:", e)
EOF
echo "--- gpu ---"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
echo "--- code tree ---"
ls /TeRed+RATE/code/scripts_ablation 2>/dev/null
echo "--- templates ---"
ls -la /TeRed+RATE/code/cache/darpa/e5_templates/
