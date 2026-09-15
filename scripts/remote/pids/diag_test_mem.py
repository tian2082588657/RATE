#!/usr/bin/env python
"""诊断 Kairos 测试阶段显存增长：逐 batch 打印 alloc/reserved。

用法：
    ~/pids-lite/bin/python diag_test_mem.py [--max-batches N] [--window-limit W]
"""
import argparse
import os
import sys
import time

os.environ.setdefault("WANDB_MODE", "offline")

# 先解析本脚本自己的参数，再把 sys.argv 还原为流水线参数（避免 argparse 冲突）
_ap = argparse.ArgumentParser()
_ap.add_argument("--max-batches", type=int, default=0, help="0=全部")
_ap.add_argument("--window-limit", type=int, default=0, help="只看前 W 个窗口")
_ap.add_argument("--log-every", type=int, default=50)
_ap.add_argument("--gc-at", type=int, default=0, help="在第 N 与 2N 个 batch 时做 GC 张量统计")
_my_args, _ = _ap.parse_known_args()

sys.argv = [
    "main", "kairos", "CADETS_E5",
    "--database_host", "localhost",
    "--database_user", "postgres",
    "--database_password", "postgres",
    "--database_port", "5432",
    "--artifact_dir", os.path.expanduser("~/pids_artifacts_full38"),
]

import torch  # noqa: E402

from pidsmaker.config import get_runtime_required_args, get_yml_cfg  # noqa: E402
from pidsmaker.detection.training_methods.inference_loop import (  # noqa: E402
    test_edge_level,
    test_node_level,
)
from pidsmaker.factory import build_model  # noqa: E402
from pidsmaker.tasks.batching import get_preprocessed_graphs  # noqa: E402
from pidsmaker.utils.utils import get_device  # noqa: E402


def gb(x):
    return x / (1024 ** 3)


def gpu_tensor_stats(top=8):
    """扫描 GC 中存活的 CUDA 张量，按形状汇总占用。"""
    import gc

    gc.collect()
    total = 0
    n = 0
    by_shape = {}
    for o in gc.get_objects():
        try:
            if torch.is_tensor(o) and o.is_cuda:
                b = o.numel() * o.element_size()
                total += b
                n += 1
                k = (tuple(o.shape), str(o.dtype))
                by_shape[k] = by_shape.get(k, 0) + b
        except Exception:
            continue
    print(f"[gc] 存活CUDA张量 {n} 个 / 合计 {gb(total):.3f}GB", flush=True)
    for (shape, dtype), b in sorted(by_shape.items(), key=lambda kv: -kv[1])[:top]:
        print(f"[gc]   shape={shape} dtype={dtype} -> {gb(b):.3f}GB", flush=True)

    # 找出小张量的引用者（定位泄漏容器）
    small = []
    for o in gc.get_objects():
        try:
            if torch.is_tensor(o) and o.is_cuda and o.numel() <= 8:
                small.append(o)
        except Exception:
            continue
    print(f"[gc] 微小张量(numel<=8) 共 {len(small)} 个", flush=True)
    import random

    for t in random.sample(small, min(3, len(small))):
        print(f"[gc-ref] tensor shape={tuple(t.shape)} dtype={t.dtype} "
              f"refcount={sys.getrefcount(t)}", flush=True)
        for r in gc.get_referrers(t)[:6]:
            rn = type(r).__name__
            rs = repr(r)[:160].replace("\n", " ")
            print(f"[gc-ref]    <- {rn}: {rs}", flush=True)
    return total


def main():
    a = _my_args
    print(f"[diag] max_batches={a.max_batches} window_limit={a.window_limit}", flush=True)

    args, unknown = get_runtime_required_args(return_unknown_args=True)
    if unknown:
        print(f"[warn] unknown args: {unknown}", flush=True)
    cfg = get_yml_cfg(args)
    device = get_device(cfg)
    print(f"[cfg] device={device} is_node_level={cfg._is_node_level}", flush=True)

    t0 = time.time()
    train_data, val_data, test_data, max_node_num = get_preprocessed_graphs(cfg)
    print(f"[data] loaded in {time.time() - t0:.1f}s | max_node={max_node_num} "
          f"| test windows={len(test_data)}", flush=True)

    # 固定随机种子，保证补丁前后 loss 可比
    import numpy as _np

    torch.manual_seed(0)
    _np.random.seed(0)
    torch.cuda.manual_seed_all(0)

    model = build_model(data_sample=train_data[0][0], device=device, cfg=cfg, max_node_num=max_node_num)
    model.eval()
    torch.cuda.reset_peak_memory_stats(device=device)
    print(f"[model] built | alloc={gb(torch.cuda.memory_allocated()):.2f}GB "
          f"reserved={gb(torch.cuda.memory_reserved()):.2f}GB", flush=True)

    test_fn = test_node_level if cfg._is_node_level else test_edge_level

    n_batches = 0
    t_start = time.time()
    for wi, graphs in enumerate(test_data):
        if a.window_limit and wi >= a.window_limit:
            break
        for bi, g in enumerate(graphs):
            if a.max_batches and n_batches >= a.max_batches:
                break
            try:
                g.to(device=device)
                out_losses = test_fn(
                    data=g,
                    model=model,
                    split="test",
                    model_epoch_file="model_epoch_0",
                    cfg=cfg,
                    device=device,
                )
                if n_batches < 3 or n_batches == 199:
                    import numpy as _np

                    print(f"[loss] batch={n_batches} mean={_np.mean(out_losses):.6f} n={len(out_losses)}",
                          flush=True)
            except torch.cuda.OutOfMemoryError as e:
                print(f"[OOM] window={wi} batch={bi} n_batches={n_batches} "
                      f"edges={g.num_edges if hasattr(g, 'num_edges') else '?'} "
                      f"alloc={gb(torch.cuda.memory_allocated()):.2f}GB "
                      f"reserved={gb(torch.cuda.memory_reserved()):.2f}GB", flush=True)
                print(f"[OOM-DETAIL] {str(e)[:300]}", flush=True)
                return 1
            finally:
                g.to("cpu")
                torch.cuda.empty_cache()
            n_batches += 1
            if n_batches % a.log_every == 0:
                print(f"[batch {n_batches}] win={wi} alloc={gb(torch.cuda.memory_allocated()):.2f}GB "
                      f"reserved={gb(torch.cuda.memory_reserved()):.2f}GB "
                      f"peak={gb(torch.cuda.max_memory_allocated()):.2f}GB "
                      f"elapsed={time.time() - t_start:.0f}s", flush=True)
            if n_batches in (a.gc_at, a.gc_at * 2) and a.gc_at > 0:
                gpu_tensor_stats()
        if a.max_batches and n_batches >= a.max_batches:
            break

    print(f"[done] {n_batches} batches in {time.time() - t_start:.0f}s | "
          f"peak={gb(torch.cuda.max_memory_allocated()):.2f}GB", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
