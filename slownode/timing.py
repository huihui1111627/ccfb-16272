"""计时仿真：同步数据并行的单屏障模型。

真实的同步数据并行每个 step 只有一个 all-reduce 同步点：每张卡各自完成
数据加载→前向→反向→通信，最后在屏障处等最慢的卡。因此单步墙钟取
各卡「三阶段总耗时」的最大值，而不是三个阶段分别取 max 再相加——
后者等于人为设三道屏障，会把健康集群的等待占比虚高一倍以上。

等待占比 = 全卡(墙钟-本地耗时)之和 / 全卡本地耗时之和。

噪声分两层，贴近真实集群：
  - 卡级慢漂移 d_g：AR(1) 过程，描述同一张卡在一段时间内的系统性摆动；
  - 步间白噪声：小幅独立抖动。
固定能力差 capability 是卡间差异的主因，健康集群等待占比稳定在 ~15%。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import SimConfig
from .topology import Topology


class TimingNoise:
    """每张卡一个 AR(1) 慢漂移状态。"""

    def __init__(self, cfg: SimConfig, num_gpus: int,
                 rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.drift = rng.normal(0, cfg.ar_sd, size=num_gpus)

    def advance(self, active_ids: list) -> np.ndarray:
        cfg = self.cfg
        ids = np.array(active_ids)
        innov_sd = cfg.ar_sd * np.sqrt(1 - cfg.ar_coef ** 2)
        self.drift[ids] = cfg.ar_coef * self.drift[ids] \
            + self.rng.normal(0, innov_sd, size=len(ids))
        white = self.rng.normal(0, cfg.noise_cv, size=len(ids))
        return 1.0 + self.drift[ids] + white


@dataclass
class StepTiming:
    data_t: np.ndarray
    compute_t: np.ndarray
    comm_t: np.ndarray
    token_count: np.ndarray
    local_total: np.ndarray
    phase_wall: np.ndarray
    step_wall: float
    wait_ms: float
    busy_ms: float
    wait_ratio: float          # 原始等待占比（随卡数变化）
    wait_ratio_normalized: float  # 按满配 64 卡归一化，跨拓扑可比
    active_ids: list


def simulate_timing(cfg: SimConfig, topo: Topology, step: int,
                    token_count: np.ndarray, active_ids: list,
                    data_f: np.ndarray, compute_f: np.ndarray,
                    comm_f: np.ndarray, noise: TimingNoise) -> StepTiming:
    n = len(active_ids)
    base_tokens = cfg.global_batch / cfg.num_gpus * 32.0
    workload = token_count / base_tokens
    mult = noise.advance(active_ids)
    caps = np.array([topo.gpus[g].capability for g in active_ids])

    def phase(base_ms: float, factor: np.ndarray, scales: bool):
        wl = workload if scales else np.ones(n)
        # 同一张卡的三个阶段共用一个噪声乘子（卡的快慢作用在整条流水线上）
        return np.maximum(base_ms * wl / caps * factor * mult, 0.1)

    data_t = phase(cfg.base_data_ms, data_f, True)
    compute_t = phase(cfg.base_compute_ms, compute_f, True)
    comm_t = phase(cfg.base_comm_ms, comm_f, False)

    # 单屏障：每卡先算完自己的完整流水，再等最慢的卡
    local_total = data_t + compute_t + comm_t
    step_wall = float(local_total.max())
    phase_wall = np.array([data_t.max(), compute_t.max(), comm_t.max()])

    busy_ms = float(local_total.sum())
    wait_ms = float((step_wall - local_total).sum())
    wait_ratio = wait_ms / max(busy_ms, 1e-9)
    # 归一化：把当前活动卡数折算回满配规模，剔除卡后该指标不会因分母变小而虚高
    scale = n / cfg.num_gpus
    normalized = wait_ratio * scale
    return StepTiming(
        data_t=data_t, compute_t=compute_t, comm_t=comm_t,
        token_count=token_count, local_total=local_total,
        phase_wall=phase_wall, step_wall=step_wall,
        wait_ms=wait_ms, busy_ms=busy_ms,
        wait_ratio=wait_ratio, wait_ratio_normalized=normalized,
        active_ids=list(active_ids))
