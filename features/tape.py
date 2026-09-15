# -*- coding: utf-8 -*-
"""features/tape.py — TAPE 正弦/余弦位置编码（度数/质量度 投影）。

公式（对齐 DC-TAPE / RATE 定义）：
    x = log1p(值)                         （ThreaTrace PositionEncoder 同款平滑）
    angle_i = x / base ** (2i/D)          i = 0..D/2-1
    out[..,2i] = sin(angle_i); out[..,2i+1] = cos(angle_i)
base 可配（RATE 数学定义默认 10000；ThreaTrace 实现用 100/1000 —— 用 --tape-base 切换）。

纯 numpy 实现，不依赖 torch，保证本机(无 GPU)可直接跑。
"""
from __future__ import annotations
import numpy as np


def tape_embedding(values, dim: int = 8, base: float = 10000.0):
    """values: (n,) 非负数组（度数/质量度）。返回 (n, dim) 浮点矩阵。"""
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    dim = int(dim)
    if dim % 2 == 1:
        dim += 1
    half = dim // 2
    x = np.log1p(np.maximum(v, 0.0))
    freqs = 1.0 / (np.power(float(base), 2.0 * np.arange(half) / dim))
    ang = x[:, None] * freqs[None, :]          # (n, half)
    out = np.empty((len(v), dim), dtype=np.float64)
    out[:, 0::2] = np.sin(ang)
    out[:, 1::2] = np.cos(ang)
    return out
