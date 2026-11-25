"""训练仿真主引擎。

每个 step 的流程严格对应真实同步数据并行：
  事件因子 → 数据分配/采样 → 每卡前向反向（真算）→ 计时仿真（屏障等待）
  → 检测器更新 → 控制器决策（剔除/重分配/撤回/换卡）→ 梯度加权聚合与参数更新。

run_mode:
  full    —— 检测+处理全开；
  observe —— 只检测不处理（对照组，量化干预收益）；
  clean   —— 不注入任何慢因（健康基线，用于损失曲线对比）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict

import numpy as np

from .config import SimConfig
from .events import EventManager, SlowEvent, TRUTH_DATA_SKEW, TRUTH_RECOVERY
from .topology import Topology
from .sampler import Sampler
from .model import build_dataset, TinyModel
from .timing import simulate_timing, TimingNoise
from .detector import Detector, CAUSE_NONE
from .controller import Controller


@dataclass
class StepRecord:
    step: int
    wall_ms: float
    wait_ratio: float
    wait_ratio_normalized: float
    loss: float
    lr: float
    active_count: int
    data_t: list
    compute_t: list
    comm_t: list
    total_t: list
    tokens: list
    truth: list
    detected_cause: list
    confidence: list
    gpu_state: list
    host_id: list


class Simulator:
    def __init__(self, cfg: SimConfig, events: List[SlowEvent],
                 run_mode: str = "full", seed: Optional[int] = None):
        assert run_mode in ("full", "observe", "clean")
        self.cfg = cfg
        self.run_mode = run_mode
        seed = cfg.seed if seed is None else seed
        # 三个独立 RNG 流，事件/采样/初始化互不干扰，且三种 run_mode 完全对齐
        self.rng_model = np.random.default_rng(seed)
        self.rng_sample = np.random.default_rng(seed + 1)
        self.rng_noise = np.random.default_rng(seed + 2)
        self.rng_event = np.random.default_rng(seed + 3)

        self.dataset = build_dataset(cfg, np.random.default_rng(seed + 4))
        self.model = TinyModel(cfg, self.rng_model)
        self.topo = Topology(cfg, self.rng_model)
        self.sampler = Sampler(cfg, self.dataset, self.rng_sample)
        eff_events = [] if run_mode == "clean" else self._seed_event_rngs(events)
        self.events = EventManager(cfg, eff_events, self.rng_event)
        self.detector = Detector(cfg, cfg.num_gpus, cfg.gpus_per_host)
        self.controller = Controller(cfg, self.topo, self.detector)
        self.timing_noise = TimingNoise(cfg, cfg.num_gpus, self.rng_noise)
        self.records: List[StepRecord] = []
        self.wait_history: List[float] = []
        self.wait_norm_history: List[float] = []
        self.wall_history: List[float] = []
        self.loss_history: List[float] = []
        self._skew_gpus: set = set()
        self._setup_recovery_queues(eff_events)

    def _seed_event_rngs(self, events: List[SlowEvent]) -> List[SlowEvent]:
        for i, ev in enumerate(events):
            ev.rng = np.random.default_rng(self.cfg.seed + 100 + i)
        return events

    def _setup_recovery_queues(self, events: List[SlowEvent]):
        """持续慢事件若有结束时间，结束即视为维修/换卡完成，排队重新加回。"""
        for ev in events:
            if ev.kind in ("persistent", "progressive") and ev.end > 0 and ev.gpus:
                for g in ev.gpus:
                    self.controller.queue_rejoin(g, ev.end, "维修完成，新卡上线重新加入")

    # ------------------------------------------------------------------
    def run(self, log_every: int = 250) -> dict:
        cfg = self.cfg
        for step in range(cfg.total_steps):
            rec = self._step(step)
            self.records.append(rec)
            if step % log_every == 0 or step == cfg.total_steps - 1:
                print(f"[{self.run_mode}] step {step:5d}  wall={rec.wall_ms:7.1f}ms "
                      f"wait={rec.wait_ratio:6.1%}  loss={rec.loss:.4f}  "
                      f"active={rec.active_count}")
        return self.summary()

    def _step(self, step: int) -> StepRecord:
        cfg = self.cfg
        # 1) 事件因子
        data_f, compute_f, comm_f, workload_mult = self.events.step_factors(step)
        # 数据倾斜事件 → 通知采样器
        skew = set()
        for ev in self.events.events:
            if ev.kind == TRUTH_DATA_SKEW and ev.active(step) and ev.gpus:
                skew.update(ev.gpus)
        self.sampler.set_skew(skew)
        self._skew_gpus = skew

        # 2) 拓扑与数据分配
        active_ids = self.topo.active_ids
        if cfg.batch_policy == "global":
            local_batches = self.topo.local_batches(active_ids)
        else:
            b = self.topo.local_batch_equal()
            local_batches = np.full(len(active_ids), b)
        batches = dict(zip(active_ids, local_batches))
        drawn = self.sampler.draw_step(self.topo, local_batches, active_ids)

        # 工作量由真实采样的 token 数决定（倾斜卡从长样本池取数，天然 token 更多）。
        # 不再额外乘 workload_mult，避免与采样器的实际分配重复计账
        token_count = np.zeros(len(active_ids))
        for i, g in enumerate(active_ids):
            token_count[i] = drawn[g][3]

        # 3) 每卡真正的前向/反向
        local_grads, local_bs, losses = [], [], []
        for g in active_ids:
            toks, lens, labs, _ = drawn[g]
            loss, grads, bs = self.model.forward_backward(toks, lens, labs)
            local_grads.append(grads)
            local_bs.append(bs)
            losses.append(loss * bs)
        step_loss = float(np.sum(losses) / max(sum(local_bs), 1))

        # 4) 计时仿真
        tm = simulate_timing(cfg, self.topo, step, token_count, active_ids,
                             data_f[active_ids], compute_f[active_ids],
                             comm_f[active_ids], self.timing_noise)

        # 5) 检测器更新（对照组也更新，只是不触发动作）
        self.detector.update(step, active_ids, tm.data_t, tm.compute_t,
                             tm.comm_t, token_count)
        self.wait_history.append(tm.wait_ratio)
        self.wait_norm_history.append(tm.wait_ratio_normalized)
        self.wall_history.append(tm.step_wall)
        self.loss_history.append(step_loss)
        self.controller.attach_history(self.wait_history, self.wall_history)
        if self.run_mode == "full":
            self.controller.maybe_act(step, tm.wait_ratio, tm.step_wall)

        # 6) LR（拓扑感知）与参数更新
        lr = self.topo.lr_for_topology(step, self.controller.last_change_step)
        self.model.allreduce_and_step(local_grads, local_bs, lr, cfg.momentum)

        # 7) 记录（固定 64 长度，被剔卡填 0，方便画热力图）
        return self._record(step, tm, step_loss, lr, active_ids)

    def _record(self, step, tm, step_loss, lr, active_ids) -> StepRecord:
        n = self.cfg.num_gpus
        data_t = [0.0] * n
        compute_t = [0.0] * n
        comm_t = [0.0] * n
        total_t = [0.0] * n
        tokens = [0] * n
        truth = [""] * n
        cause = [CAUSE_NONE] * n
        conf = [0.0] * n
        state = [self.topo.gpus[g].state for g in range(n)]
        hosts = [self.topo.gpus[g].host_id for g in range(n)]
        for i, g in enumerate(active_ids):
            data_t[g] = round(float(tm.data_t[i]), 2)
            compute_t[g] = round(float(tm.compute_t[i]), 2)
            comm_t[g] = round(float(tm.comm_t[i]), 2)
            total_t[g] = round(float(tm.local_total[i]), 2)
            tokens[g] = int(tm.token_count[i])
            truth[g] = self.events.active_truth_label(g)
            v = self.detector.latest.get(g)
            if v is not None:
                cause[g] = v.cause
                conf[g] = round(float(v.confidence), 3)
        return StepRecord(
            step=step, wall_ms=round(tm.step_wall, 2),
            wait_ratio=round(tm.wait_ratio, 5),
            wait_ratio_normalized=round(tm.wait_ratio_normalized, 5),
            loss=round(step_loss, 5),
            lr=round(lr, 5), active_count=len(active_ids),
            data_t=data_t, compute_t=compute_t, comm_t=comm_t,
            total_t=total_t, tokens=tokens, truth=truth,
            detected_cause=cause, confidence=conf,
            gpu_state=state, host_id=hosts)

    # ------------------------------------------------------------------
    def summary(self) -> dict:
        return {
            "run_mode": self.run_mode,
            "steps": self.cfg.total_steps,
            "num_gpus": self.cfg.num_gpus,
            "wall_total_ms": round(float(np.sum(self.wall_history)), 1),
            "mean_wait_ratio": round(float(np.mean(self.wait_history)), 5),
            "final_loss": round(float(np.mean(self.loss_history[-20:])), 5),
            "ejected": self.topo.ejected_ids,
            "actions": [a.__dict__ for a in self.controller.actions],
            "topology_log": self.topo.change_log,
        }
