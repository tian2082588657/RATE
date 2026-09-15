# -*- coding: utf-8 -*-
"""models/torch_net.py —（可选）GraphSAGE / N-GAT 语义特征网络。

从 threatrace 项目移植（SAGENet + PositionEncoder），仅供接入 GPU 后的
"语义特征" 环节使用。本机无 torch 也能跑其余全部流程；import 本模块前需：
    pip install torch --index-url https://download.pytorch.org/whl/cpu   # 本机 CPU
    # 服务器(your-server-host, RTX3060) 装对应 CUDA 版
"""
from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import SAGEConv
    _HAS_TORCH = True
except Exception as _e:                                   # pragma: no cover
    _HAS_TORCH = False
    _IMPORT_ERR = _e


def require_torch():
    if not _HAS_TORCH:                                    # pragma: no cover
        raise RuntimeError("需要 torch(+torch_geometric)。提示：" + str(_IMPORT_ERR))


class PositionEncoder(nn.Module):
    """DC-TAPE 度数位置编码（threatrace PositionEncoder 移植；base 可配 10000）。"""
    def __init__(self, pos_dim=64, base=10000.0):
        super().__init__()
        self.pos_dim = pos_dim
        half = pos_dim // 2
        freqs = 1.0 / (base ** (torch.arange(0, half).float() * 2.0 / pos_dim))
        self.register_buffer("freqs", freqs)

    def forward(self, pos):
        p = torch.log1p(pos.float())
        v = p.unsqueeze(-1) * self.freqs
        return torch.cat([torch.sin(v), torch.cos(v)], dim=-1)


class SAGENet(nn.Module):
    """2 层 GraphSAGE 节点分类（threatrace sage_focal 移植）。"""
    def __init__(self, feature_num, hidden=128, label_num=2, dropout=0.5):
        super().__init__()
        require_torch()
        self.conv1 = SAGEConv(feature_num, hidden)
        self.conv2 = SAGEConv(hidden, label_num)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
