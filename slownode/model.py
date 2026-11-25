"""缩小版文本分类模型 + 变长合成数据集，纯 NumPy 实现。

模型：Embedding → 按有效 token 均值池化 → ReLU MLP → 10 分类。
每个 step 每张卡对自己的本地批做完整前向/反向，梯度按样本数加权
all-reduce（模拟数据并行），再统一更新参数 —— 剔除卡后本地批变大时，
加权平均在数学上仍然等价于全量批的梯度。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np

# 裁剪前的中间量可能触发溢出告警，裁剪后结果安全，统一忽略
np.seterr(over="ignore", divide="ignore", invalid="ignore")

from .config import SimConfig


@dataclass
class Dataset:
    tokens: np.ndarray      # (N, Lmax)，不足长度补 0
    lengths: np.ndarray     # (N,) 真实长度
    labels: np.ndarray      # (N,)
    long_pool: np.ndarray   # 长样本池的样本下标

    @property
    def size(self) -> int:
        return len(self.labels)


def build_dataset(cfg: SimConfig, rng: np.random.Generator) -> Dataset:
    n = cfg.dataset_size
    log_lens = rng.normal(cfg.seq_len_mu, cfg.seq_len_sigma, size=n)
    lengths = np.clip(np.exp(log_lens), cfg.seq_len_clip[0],
                      cfg.seq_len_clip[1]).astype(int)
    lmax = int(lengths.max())
    tokens = np.zeros((n, lmax), dtype=np.int64)
    for i in range(n):
        L = int(lengths[i])
        tokens[i, :L] = rng.integers(1, cfg.vocab_size, size=L)  # 0 保留给 padding
    # 用一个固定「教师」网络生成标签，让标签与输入真有可学习的关系
    w_teacher = rng.normal(0, 0.1, size=(cfg.embed_dim, cfg.num_classes))
    emb_teacher = rng.normal(0, 0.1, size=(cfg.vocab_size, cfg.embed_dim))
    mean_emb = np.stack([emb_teacher[tokens[i, :lengths[i]]].mean(axis=0)
                         for i in range(n)])
    logits = mean_emb @ w_teacher
    logits = np.clip(logits, -30, 30)
    labels = np.argmax(logits, axis=1)
    # 混入 8% 标签噪声，让损失有不可消除的底噪，更像真实训练
    flip = rng.random(n) < 0.08
    labels[flip] = rng.integers(0, cfg.num_classes, size=flip.sum())
    order = np.argsort(lengths)
    long_pool = order[int(n * (1 - cfg.long_pool_ratio)):]
    return Dataset(tokens=tokens, lengths=lengths, labels=labels, long_pool=long_pool)


class TinyModel:
    def __init__(self, cfg: SimConfig, rng: np.random.Generator):
        self.cfg = cfg
        k = 1.0 / np.sqrt(cfg.embed_dim)
        self.E = rng.normal(0, k, size=(cfg.vocab_size, cfg.embed_dim))
        kh = 1.0 / np.sqrt(cfg.embed_dim)
        self.W1 = rng.normal(0, kh, size=(cfg.embed_dim, cfg.hidden_dim))
        self.b1 = np.zeros(cfg.hidden_dim)
        self.W2 = rng.normal(0, 1.0 / np.sqrt(cfg.hidden_dim),
                             size=(cfg.hidden_dim, cfg.num_classes))
        self.b2 = np.zeros(cfg.num_classes)
        # SGD momentum 状态
        self._mom = {name: np.zeros_like(p) for name, p in self._params()}

    def _params(self):
        return [("E", self.E), ("W1", self.W1), ("b1", self.b1),
                ("W2", self.W2), ("b2", self.b2)]

    def forward_backward(self, tokens: np.ndarray, lengths: np.ndarray,
                         labels: np.ndarray):
        """对一个本地批做前向+反向，返回 (平均损失, 梯度字典, 样本数)。"""
        B = len(labels)
        # 均值池化（只统计有效 token）
        emb = self.E[tokens]                                  # (B,L,K)
        mask = np.arange(tokens.shape[1])[None, :] < lengths[:, None]
        denom = lengths[:, None].astype(float)
        pooled = (emb * mask[:, :, None]).sum(axis=1) / denom  # (B,K)
        h_pre = pooled @ self.W1 + self.b1
        h = np.maximum(h_pre, 0.0)
        logits = h @ self.W2 + self.b2
        logits = np.clip(logits, -30, 30)
        logits -= logits.max(axis=1, keepdims=True)
        e = np.exp(logits)
        probs = np.clip(e / e.sum(axis=1, keepdims=True), 1e-12, 1.0)
        loss = float(-np.log(probs[np.arange(B), labels]).mean())

        # ---- 反向 ----
        dlogits = probs.copy()
        dlogits[np.arange(B), labels] -= 1.0
        dlogits /= B
        gW2 = h.T @ dlogits
        gb2 = dlogits.sum(axis=0)
        dh = dlogits @ self.W2.T
        dh_pre = dh * (h_pre > 0)
        gW1 = pooled.T @ dh_pre
        gb1 = dh_pre.sum(axis=0)
        dpooled = dh_pre @ self.W1.T                            # (B,K)
        gE = np.zeros_like(self.E)
        # 每个 token 的梯度 = dpooled / length，散射累加到 embedding 表
        for i in range(B):
            L = int(lengths[i])
            contrib = np.broadcast_to(dpooled[i] / L, (L, dpooled.shape[1]))
            np.add.at(gE, tokens[i, :L], contrib)
        return loss, {"E": gE, "W1": gW1, "b1": gb1,
                      "W2": gW2, "b2": gb2}, B

    def allreduce_and_step(self, local_grads: List[dict], local_bs: List[int],
                           lr: float, momentum: float):
        """按样本数加权聚合各卡梯度并做动量 SGD 更新。"""
        total = max(sum(local_bs), 1)
        agg = {name: np.zeros_like(p) for name, p in self._params()}
        for grads, bs in zip(local_grads, local_bs):
            w = bs / total
            for name in agg:
                agg[name] += w * grads[name]
        # 全局梯度范数裁剪
        grad_norm = np.sqrt(sum((g ** 2).sum() for g in agg.values()))
        if grad_norm > self.cfg.grad_clip:
            scale = self.cfg.grad_clip / max(grad_norm, 1e-12)
            for name in agg:
                agg[name] *= scale
        for name, p in self._params():
            self._mom[name] = momentum * self._mom[name] + agg[name]
            p -= lr * self._mom[name]
            np.clip(p, -self.cfg.weight_clip, self.cfg.weight_clip, out=p)

    def eval_loss(self, ds: Dataset, rng: np.random.Generator,
                  batch: int = 512) -> float:
        """全量评估集上的平均损失（评估不改变参数）。"""
        idx = rng.permutation(ds.size)
        losses, weights = [], []
        for s in range(0, ds.size, batch):
            chunk = idx[s:s + batch]
            loss, _, b = self.forward_backward(
                ds.tokens[chunk], ds.lengths[chunk], ds.labels[chunk])
            losses.append(loss * b)
            weights.append(b)
        return float(np.sum(losses) / np.sum(weights))
