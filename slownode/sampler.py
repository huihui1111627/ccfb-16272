"""每步给每张卡分配数据。

正常模式模拟真实训练的数据均衡分片：在给定本地批样本数下，用「长短互补」
的贪心分配让每张卡拿到的总 token 数尽量一致（差异只剩采样随机性）。
skew 模式下被指定的卡只从「长样本池」采样，模拟数据分片不均导致的慢；
rebalance 通过修改 Topology 的分配权重改变各卡的样本预算。
"""
from __future__ import annotations

from typing import List, Dict, Optional, Set

import numpy as np

from .config import SimConfig
from .model import Dataset
from .topology import Topology


class Sampler:
    def __init__(self, cfg: SimConfig, ds: Dataset, rng: np.random.Generator):
        self.cfg = cfg
        self.ds = ds
        self.rng = rng
        self.skew_gpus: Set[int] = set()

    def set_skew(self, gpus: Set[int]):
        self.skew_gpus = set(gpus)

    def draw_step(self, topo: Topology, local_batches: np.ndarray,
                  active_ids: List[int]):
        ds = self.ds
        out = {}
        # 先算出本步需要的总样本数，一次性抽出候选池
        total_bs = int(sum(local_batches))
        normal_ids = [g for g in active_ids if g not in self.skew_gpus]
        skew_ids = [g for g in active_ids if g in self.skew_gpus]
        normal_bs = sum(int(b) for g, b in zip(active_ids, local_batches)
                        if g not in self.skew_gpus)

        # 倾斜卡：只从长样本池取
        for g in skew_ids:
            bs = int(dict(zip(active_ids, local_batches))[g])
            if bs <= 0:
                continue
            pick = self.rng.choice(ds.long_pool, size=bs, replace=True)
            toks, lens, labs = ds.tokens[pick], ds.lengths[pick], ds.labels[pick]
            out[g] = (toks, lens, labs, int(lens.sum()))

        if normal_bs > 0:
            cand = self.rng.integers(0, ds.size, size=normal_bs)
            cand_lens = ds.lengths[cand]
            # 长短互补贪心：按长度排序后，从两端交替取，轮询分给各卡，
            # 使各卡 token 总量尽量接近
            order = np.argsort(cand_lens)
            sorted_idx = np.concatenate([order[::2], order[::-2]])
            buckets = {g: [] for g in normal_ids}
            bs_map = dict(zip(active_ids, local_batches))
            cursors = {g: 0 for g in normal_ids}
            gi = 0
            for cand_pos in sorted_idx:
                # 找下一个还有配额的卡
                tries = 0
                while cursors[normal_ids[gi]] >= bs_map[normal_ids[gi]] \
                        and tries < len(normal_ids):
                    gi = (gi + 1) % len(normal_ids)
                    tries += 1
                g = normal_ids[gi]
                if cursors[g] >= bs_map[g]:
                    break
                buckets[g].append(cand[cand_pos])
                cursors[g] += 1
                gi = (gi + 1) % len(normal_ids)
            for g, picks in buckets.items():
                if not picks:
                    continue
                pick = np.array(picks)
                toks, lens, labs = ds.tokens[pick], ds.lengths[pick], ds.labels[pick]
                out[g] = (toks, lens, labs, int(lens.sum()))
        return out

    def mean_token_per_sample(self) -> float:
        return float(self.ds.lengths.mean())
