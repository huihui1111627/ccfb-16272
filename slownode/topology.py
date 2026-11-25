"""集群拓扑与卡状态。

一张卡的生命周期：active（正常/观察中）→ ejected（剔除）→ 重新 join。
数据分配通过 alloc_weight 表达：采样器按权重把 token 预算分到各卡，
rebalance 只改权重、不动拓扑。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set

import numpy as np

from .config import SimConfig

STATE_ACTIVE = "active"
STATE_WATCH = "watch"          # 已判定持续慢、等待置信度/冷却
STATE_EJECTED = "ejected"


@dataclass
class GPUState:
    gpu_id: int
    host_id: int
    capability: float                       # 固有效能（1 附近）
    state: str = STATE_ACTIVE
    alloc_weight: float = 1.0               # 数据分配权重
    ejected_step: Optional[int] = None
    rejoin_step: Optional[int] = None
    eject_count: int = 0
    label: str = ""                         # 控制器当前给的原因标签


class Topology:
    def __init__(self, cfg: SimConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.gpus: List[GPUState] = []
        for g in range(cfg.num_gpus):
            cap = float(1.0 + rng.uniform(-cfg.gpu_cap_spread, cfg.gpu_cap_spread))
            self.gpus.append(GPUState(gpu_id=g, host_id=g // cfg.gpus_per_host,
                                      capability=cap))
        # 初始等权分配
        self._equalize()
        self.change_log: List[dict] = []

    # ---- 基本查询 ----
    @property
    def active_ids(self) -> List[int]:
        return [g.gpu_id for g in self.gpus if g.state != STATE_EJECTED]

    @property
    def ejected_ids(self) -> List[int]:
        return [g.gpu_id for g in self.gpus if g.state == STATE_EJECTED]

    def host_of(self, gpu: int) -> int:
        return self.gpus[gpu].host_id

    def host_active_ids(self, host: int) -> List[int]:
        return [g.gpu_id for g in self.gpus
                if g.host_id == host and g.state != STATE_EJECTED]

    def host_all_ids(self, host: int) -> List[int]:
        return [g.gpu_id for g in self.gpus if g.host_id == host]

    def num_active(self) -> int:
        return len(self.active_ids)

    def alloc_weights(self, ids: Optional[List[int]] = None) -> np.ndarray:
        ids = ids if ids is not None else self.active_ids
        w = np.array([self.gpus[g].alloc_weight for g in ids], dtype=float)
        return w / w.sum()

    # ---- 拓扑变更 ----
    def eject(self, gpu: int, step: int, reason: str):
        gs = self.gpus[gpu]
        assert gs.state != STATE_EJECTED
        gs.state = STATE_EJECTED
        gs.ejected_step = step
        gs.eject_count += 1
        gs.label = reason
        self._equalize()
        self.change_log.append({"step": step, "action": "eject",
                                "gpu": gpu, "host": gs.host_id, "reason": reason})

    def rejoin(self, gpu: int, step: int, reason: str):
        gs = self.gpus[gpu]
        assert gs.state == STATE_EJECTED
        gs.state = STATE_ACTIVE
        gs.rejoin_step = step
        gs.label = ""
        self._equalize()
        self.change_log.append({"step": step, "action": "rejoin",
                                "gpu": gpu, "host": gs.host_id, "reason": reason})

    def set_alloc_weights(self, weights: Dict[int, float], step: int, reason: str):
        for g, w in weights.items():
            self.gpus[g].alloc_weight = max(w, 0.05)
        # 被剔除卡权重清零（保持不变即可，采样时只看 active）
        self.change_log.append({"step": step, "action": "rebalance",
                                "weights": dict(weights), "reason": reason})

    def snapshot_weights(self) -> Dict[int, float]:
        return {g.gpu_id: g.alloc_weight for g in self.gpus}

    def restore_weights(self, snapshot: Dict[int, float], step: int, reason: str):
        for g, w in snapshot.items():
            self.gpus[g].alloc_weight = w
        self.change_log.append({"step": step, "action": "rebalance_rollback",
                                "weights": dict(snapshot), "reason": reason})

    def _equalize(self):
        """拓扑变化后，活动卡恢复等权（保证全局批大小不变）。"""
        ids = self.active_ids
        for g in self.gpus:
            g.alloc_weight = 1.0 if g.gpu_id in ids else 0.0

    # ---- 批大小 / 学习率 ----
    def local_batches(self, ids: Optional[List[int]] = None) -> np.ndarray:
        """全局批大小固定模式：按分配权重拆分 global_batch。"""
        ids = ids if ids is not None else self.active_ids
        frac = self.alloc_weights(ids)
        raw = frac * self.cfg.global_batch
        # 每张活动卡至少 1 个样本，先保底再分配剩余预算
        floors = np.maximum(np.floor(raw).astype(int), 1)
        over = floors.sum() - self.cfg.global_batch
        if over > 0:
            # 极端情况下全局批被保底撑爆，按最小批次优先扣减
            order = np.argsort(floors)
            k = 0
            while over > 0 and k < len(floors):
                g = order[k % len(floors)]
                if floors[g] > 1:
                    floors[g] -= 1
                    over -= 1
                k += 1
            return floors
        remainder = self.cfg.global_batch - floors.sum()
        order = np.argsort(-(raw - floors))
        for i in range(remainder):
            floors[order[i % len(order)]] += 1
        return floors

    def local_batch_equal(self) -> int:
        """每卡批大小固定模式（剔除后全局批缩小）。"""
        return max(1, self.cfg.global_batch // self.cfg.num_gpus)

    def lr_for_topology(self, step: int, last_change_step: int) -> float:
        """拓扑变化后的 LR 策略。

        global 模式：有效批大小不变，LR 理论上不变，仅做一个短暂的线性再热身，
        防止剩余卡突然承担更大本地批带来的梯度噪声突变。
        local 模式：有效批大小随活动卡数缩小，按线性缩放规则（Goyal 等）缩小 LR，
        再热身结束后保持该比例。
        """
        cfg = self.cfg
        if cfg.batch_policy == "local":
            # 线性缩放规则：有效批大小缩到 scale 倍（scale≤1），LR 同比缩放。
            # 只在 scale 异常大（理论上不会，因活动卡数≤总卡数）时用 clip 兜底
            scale = self.num_active() / cfg.num_gpus
            target_lr = cfg.base_lr * min(scale, cfg.lr_scale_clip)
        else:
            target_lr = cfg.base_lr
        # 训练开头的统一初始化热身
        if last_change_step == 0 and step < cfg.init_warmup_steps:
            return target_lr * (step + 1) / cfg.init_warmup_steps
        dt = step - last_change_step
        if dt < cfg.topology_warmup_steps:
            frac = (dt + 1) / cfg.topology_warmup_steps
            return target_lr * (0.25 + 0.75 * frac)
        return target_lr
