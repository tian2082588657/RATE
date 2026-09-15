# -*- coding: utf-8 -*-
"""adapters/base.py — 数据集适配器接口与公共约定。

每个适配器实现 parse(...) -> list[CanonicalGraph]，并把"良/恶"依据写进
图的 labels（节点级 0/1）。无法给出节点级 ground truth 的数据集，labels 留空，
由上层按窗口/场景打标签。
"""
from __future__ import annotations
import abc


class DatasetAdapter(abc.ABC):
    name: str = "base"

    @abc.abstractmethod
    def parse(self, **kw):
        """返回 (graphs: list[CanonicalGraph], meta: dict)。"""
        raise NotImplementedError


def default_benign_split(graphs, benign_pred, attack_pred, train_frac=0.8, seed=0):
    """按图粒度划分 良性训练 / 良性验证 / 攻击测试。返回三个 list。
    benign_pred/attack_pred: 接受 gid 返回 bool 的谓词。
    """
    import random
    benign = [g for g in graphs if benign_pred(g.gid)]
    attack = [g for g in graphs if attack_pred(g.gid)]
    rng = random.Random(seed)
    rng.shuffle(benign)
    n_train = max(1, int(len(benign) * train_frac))
    train, val = benign[:n_train], benign[n_train:]
    return train, val, attack
