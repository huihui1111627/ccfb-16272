"""慢因事件注入。

每类事件对应题目里的一种真实慢因，ground-truth 标签同时用于：
  1. 事件结束后给检测器打分（误剔除、发现时延）；
  2. 页面上把「真实原因」和「系统判定原因」并排展示。

事件对计时的作用统一用乘性因子（factor）表达，作用在
data / compute / comm 三个分量上；数据倾斜事件单独作用在工作量上。
多个事件同时命中时因子相乘，并做统一截断。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict

import numpy as np

from .config import SimConfig

# ground-truth 标签
TRUTH_TRANSIENT = "transient"            # 偶发一次
TRUTH_DATA_SKEW = "data_skew"            # 跟着数据走
TRUTH_PERSISTENT = "persistent"          # 机器问题导致的持续慢
TRUTH_PROGRESSIVE = "progressive"        # 硬件快坏了，越来越慢
TRUTH_THERMAL = "thermal_periodic"       # 温度降频，跟时间段相关（卡级周期）
TRUTH_HOST_NIC = "host_nic_periodic"     # 网卡被抢，机器级周期
TRUTH_CLUSTER_STORAGE = "cluster_storage"  # 共享存储抖动，集群级时间段
TRUTH_RECOVERY = "recovery"              # 维修/换卡：慢因消失

COMPONENT_DATA = "data"
COMPONENT_COMPUTE = "compute"
COMPONENT_COMM = "comm"
COMPONENT_ALL = "all"


@dataclass
class SlowEvent:
    id: str
    kind: str                       # 取 TRUTH_* 常量
    start: int
    end: int                        # 半开区间 [start, end)；-1 表示持续到结束
    gpus: Optional[List[int]] = None    # 作用的卡；None 表示全集群
    host: Optional[int] = None          # 机器级事件：机器编号
    component: str = COMPONENT_COMPUTE
    factor: float = 1.5                   # 降速因子（1.5 = 慢 50%）
    prob: float = 1.0                     # transient：每步触发概率
    period: int = 0                       # periodic：周期长度（步）
    duty: float = 0.5                     # periodic：每周期内处于慢态的占比
    ramp: int = 0                         # periodic：慢态边缘的渐变步数
    slope: float = 0.0                    # progressive：每步新增的因子增量
    factor_max: float = 3.0
    workload_mult: float = 1.0            # data_skew：工作量（token 数）倍数
    rng: Optional[np.random.Generator] = field(default=None, repr=False)

    def active(self, step: int) -> bool:
        end = self.end if self.end >= 0 else 10 ** 12
        return self.start <= step < end

    def strength(self, step: int) -> float:
        """返回该步的降速因子（1 表示无影响）。"""
        if not self.active(step):
            return 1.0
        if self.kind == TRUTH_TRANSIENT:
            return self.factor if self.rng.random() < self.prob else 1.0
        if self.kind == TRUTH_PROGRESSIVE:
            f = 1.0 + self.slope * (step - self.start)
            return float(min(f, self.factor_max))
        if self.kind in (TRUTH_THERMAL, TRUTH_HOST_NIC):
            phase = (step - self.start) % self.period
            on_len = max(1, int(round(self.period * self.duty)))
            if phase >= on_len:
                return 1.0
            # 边缘渐变，模拟温度爬升/网卡占用爬升
            if self.ramp and phase < self.ramp:
                return 1.0 + (self.factor - 1.0) * (phase / self.ramp)
            if self.ramp and phase > on_len - self.ramp:
                return 1.0 + (self.factor - 1.0) * ((on_len - phase) / self.ramp)
            return self.factor
        # persistent / cluster_storage 等：窗口内恒定
        return self.factor


class EventManager:
    """汇总当前步所有生效事件，给出每张卡每个分量的总因子与工作量倍数。"""

    def __init__(self, cfg: SimConfig, events: List[SlowEvent], rng: np.random.Generator):
        self.cfg = cfg
        self.events = events
        self.rng = rng
        self.num_gpus = cfg.num_gpus
        # 每张卡当前步生效的事件 id（页面展示「真实原因」用）
        self.active_event_ids: List[List[str]] = [[] for _ in range(self.num_gpus)]
        # 每张卡当前步生效事件的 ground-truth 类别
        self.active_kinds: List[List[str]] = [[] for _ in range(self.num_gpus)]

    def step_factors(self, step: int):
        """返回 (data_f, compute_f, comm_f, workload_mult) 四个长度 num_gpus 的数组。"""
        n = self.num_gpus
        data_f = np.ones(n)
        compute_f = np.ones(n)
        comm_f = np.ones(n)
        workload = np.ones(n)
        self.active_event_ids = [[] for _ in range(n)]
        self.active_kinds = [[] for _ in range(n)]

        for ev in self.events:
            if not ev.active(step):
                continue
            targets = ev.gpus if ev.gpus is not None else list(range(n))
            s = ev.strength(step)
            for g in targets:
                self.active_event_ids[g].append(ev.id)
                self.active_kinds[g].append(ev.kind)
                if ev.workload_mult != 1.0:
                    workload[g] *= ev.workload_mult
                if s == 1.0:
                    continue
                s = min(s, self.cfg.factor_clip)
                if ev.component in (COMPONENT_DATA, COMPONENT_ALL):
                    data_f[g] = min(data_f[g] * s, self.cfg.factor_clip)
                if ev.component in (COMPONENT_COMPUTE, COMPONENT_ALL):
                    compute_f[g] = min(compute_f[g] * s, self.cfg.factor_clip)
                if ev.component in (COMPONENT_COMM, COMPONENT_ALL):
                    comm_f[g] = min(comm_f[g] * s, self.cfg.factor_clip)
        return data_f, compute_f, comm_f, workload

    def active_truth_label(self, gpu: int) -> str:
        """该卡当前步的真实慢因（优先级：硬件类 > 环境类 > 数据类）。"""
        kinds = self.active_kinds[gpu]
        for prio in (TRUTH_PROGRESSIVE, TRUTH_PERSISTENT, TRUTH_THERMAL,
                     TRUTH_HOST_NIC, TRUTH_CLUSTER_STORAGE,
                     TRUTH_TRANSIENT, TRUTH_DATA_SKEW):
            if prio in kinds:
                return prio
        return ""

    def persistent_bad_gpus(self, step: int) -> set:
        """当前处于「真坏卡」状态的卡（持续慢/退化），用于误剔除统计。"""
        bad = set()
        for ev in self.events:
            if ev.kind in (TRUTH_PERSISTENT, TRUTH_PROGRESSIVE) and ev.active(step):
                if ev.gpus:
                    bad.update(ev.gpus)
        return bad
