"""慢节点处理控制器。

把检测器的判决翻译成动作，并为每次动作负责：
  - 偶发 / 温度周期 / 集群时间窗：不动作，只记录；
  - 数据驱动：按效率的倒数重分配数据权重，观察窗口内无改善则撤回；
  - 持续慢 / 退化：置信度达标、经过冷却与数量保护后剔除，
    观察窗口内若发现慢因周期性消失（误判为持续慢）则撤回并加回；
  - 机器级：同机坏卡比例达标才整台处理，否则只观察；
  - 剔除后按策略调整每卡批大小与 LR（见 topology.lr_for_topology）；
  - 换卡/维修完成的卡重新 join，清空检测器历史重新观察。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .config import SimConfig
from .detector import (Detector, GPUVerdict, CAUSE_DATA, CAUSE_PERSISTENT,
                       CAUSE_PROGRESSIVE, CAUSE_HOST, CAUSE_CLUSTER,
                       CAUSE_PERIODIC_GPU, CAUSE_TRANSIENT, CAUSE_NONE)
from .topology import Topology, STATE_EJECTED, STATE_WATCH, STATE_ACTIVE
from .events import TRUTH_RECOVERY

EJECT_CAUSES = {CAUSE_PERSISTENT, CAUSE_PROGRESSIVE}
NO_ACTION_CAUSES = {CAUSE_CLUSTER, CAUSE_PERIODIC_GPU, CAUSE_TRANSIENT, CAUSE_NONE}


@dataclass
class PendingEval:
    action_id: int
    kind: str                  # rebalance / eject / host_eject
    step: int
    gpus: List[int]
    baseline_wait: float       # 动作前窗口的平均等待占比
    baseline_wall: float       # 动作前窗口的平均步墙钟
    weight_snapshot: Optional[dict] = None
    cached_hit_seq: Optional[List[int]] = None  # 剔除前的异常命中序列（撤回复核用）
    detail: dict = field(default_factory=dict)


@dataclass
class ActionRecord:
    action_id: int
    step: int
    kind: str
    gpus: List[int]
    cause: str
    confidence: float
    rationale: str
    status: str = "evaluating"     # evaluating / kept / rolled_back / rejected
    result: dict = field(default_factory=dict)


class Controller:
    def __init__(self, cfg: SimConfig, topo: Topology, detector: Detector):
        self.cfg = cfg
        self.topo = topo
        self.detector = detector
        self.actions: List[ActionRecord] = []
        self.pending: List[PendingEval] = []
        self._next_id = 1
        self.cooldown: Dict[int, int] = {}       # gpu -> 最早可再次动作的步
        self.rebalance_blocked: Dict[int, int] = {}  # 撤回过的卡 -> 屏蔽到何时
        self.data_streak: Dict[int, int] = {}    # 数据原因连续命中的判决次数
        self.data_streak_need = 4                # 连续 4 个判决点才动手
        self.rebalance_maintaining: set = set() # 已成功重分配、进入维护态的卡
        self.last_change_step = 0
        self.rejoin_queue: List[dict] = []       # 外部注入的换卡/维修事件

    def queue_rejoin(self, gpu: int, ready_step: int, reason: str):
        self.rejoin_queue.append({"gpu": gpu, "ready": ready_step, "reason": reason})

    # ------------------------------------------------------------------
    def maybe_act(self, step: int, wait_ratio: float, step_wall: float):
        # 1) 换卡/维修到期的卡先加回
        self._process_rejoin_queue(step)
        # 2) 评估窗口到期的历史动作
        self._evaluate_pending(step)
        # 3) 预热期 / 拓扑稳定期 / 非决策点都不做新判决。
        #    剔除或加回会改变所有卡的工作量，稳定期内的统计不能作为判据
        if step < self.cfg.warmup_steps or step % self.cfg.decide_every != 0:
            return
        if step - self.last_change_step < self.cfg.topology_settle_steps:
            return
        verdicts = self.detector.classify(step)
        self._mark_watch_states(verdicts)

        # 3a) 数据驱动 → 重分配。要求连续多个判决点稳定判为数据原因，
        #     且该卡不在冷却/屏蔽期，防止偶发波动触发动作。
        #     硬件类原因（持续慢/退化）不走重分配，直接走剔除流程
        data_candidates = []
        for g, v in verdicts.items():
            if v.cause == CAUSE_DATA:
                self.data_streak[g] = self.data_streak.get(g, 0) + 1
            else:
                self.data_streak[g] = 0
            if (v.cause == CAUSE_DATA
                    and g not in self.rebalance_maintaining
                    and self.data_streak[g] >= self.data_streak_need
                    and self._off_cooldown(g, step)
                    and step >= self.rebalance_blocked.get(g, 0)):
                data_candidates.append((g, v.workload_ratio))
        # 只处理工作量最高的前 15% 张卡，避免健康卡因采样噪声被卷入
        if data_candidates:
            top_k = max(1, int(self.topo.num_active() * self.cfg.data_top_fraction))
            data_candidates.sort(key=lambda x: -x[1])
            data_gpus = [g for g, _ in data_candidates[:top_k]]
            self._do_rebalance(step, data_gpus, verdicts)

        # 3b) 机器级故障 → 按比例决定是否整台处理
        self._maybe_host_eject(step, verdicts)

        # 3c) 卡级持续慢/退化 → 单卡剔除（置信度门槛）
        pending_gpus = {g for p in self.pending for g in p.gpus}
        for g, v in verdicts.items():
            if v.cause not in EJECT_CAUSES:
                continue
            if g in pending_gpus:
                continue  # 已有剔除动作在观察窗内，等复核结论，不重复提案
            if not self._eligible_to_eject(g, v, step):
                continue
            self._do_eject(step, [g], v.cause, v.confidence,
                           "；".join(v.evidence))

    # ------------------------------------------------------------------
    def _do_rebalance(self, step: int, gpus: List[int],
                      verdicts: Dict[int, GPUVerdict]):
        cfg = self.cfg
        # 目标权重 ∝ 1/效率比（效率低=单位 token 慢的卡少拿数据），
        # 用近窗效率中位数；工作量高但效率正常的卡按同侪效率归一
        weights = {}
        for g in self.topo.active_ids:
            h = self.detector.hist[g]
            if len(h.efficiency) == 0:
                weights[g] = 1.0
                continue
            recent = list(h.efficiency)[-cfg.persistent_steps:]
            rel = float(np.median(recent)) / max(
                float(np.median([np.median(self.detector.hist[x].efficiency)
                                 for x in self.topo.active_ids
                                 if len(self.detector.hist[x].efficiency)])), 1e-9)
            weights[g] = 1.0 / max(rel, 0.5)
        # 平滑：新旧权重各占一半，避免一步切太猛
        old = self.topo.snapshot_weights()
        smoothed = {g: 0.5 * old.get(g, 1.0) + 0.5 * w for g, w in weights.items()}
        snapshot = old
        self.topo.set_alloc_weights(smoothed, step,
                                    f"数据驱动重分配，覆盖卡 {gpus}")
        # 发起即进入维护态，防止评估窗内重复发起；评估失败时在撤回逻辑里解除
        for g in gpus:
            self.rebalance_maintaining.add(g)
            self.data_streak[g] = 0
        rec = self._record(step, "rebalance", gpus, CAUSE_DATA,
                           max(verdicts[g].confidence for g in gpus),
                           "效率归一化的 token 权重（与单位耗时成反比），新旧权重各半平滑")
        self.pending.append(PendingEval(
            rec.action_id, "rebalance", step, gpus,
            baseline_wait=self._recent_wait(step, cfg.rebalance_evals),
            baseline_wall=self._recent_wall(step, cfg.rebalance_evals),
            weight_snapshot=snapshot))
        for g in gpus:
            self.cooldown[g] = step + cfg.cooldown_steps

    def _maybe_host_eject(self, step: int, verdicts: Dict[int, GPUVerdict]):
        cfg = self.cfg
        hosts = {}
        for g, v in verdicts.items():
            if v.cause != CAUSE_HOST:
                continue
            hosts.setdefault(self.topo.host_of(g), []).append((g, v))
        for host, items in hosts.items():
            host_ids = self.topo.host_active_ids(host)
            bad = [g for g, _ in items]
            if len(bad) / max(len(host_ids), 1) < cfg.host_eject_fraction:
                continue
            conf = max(v.confidence for _, v in items)
            if conf < cfg.confidence_eject:
                continue
            # 整台机器共用一个冷却键，避免重复提案刷屏
            host_key = host * cfg.gpus_per_host
            if not self._off_cooldown(host_key, step):
                continue
            if not self._within_eject_budget(len(bad)):
                # 只记录一次拒绝并进入长冷却，而不是每个判决点重复记录
                self._record(step, "host_eject_rejected", bad, CAUSE_HOST, conf,
                             f"机器{host}疑似故障但超过最大剔除比例保护，持续观察")
                self.cooldown[host_key] = step + 10 * cfg.cooldown_steps
                continue
            if any(not self._off_cooldown(g, step) for g in bad):
                continue
            self._do_eject(step, bad, CAUSE_HOST, conf,
                           f"机器{host}上{len(bad)}/{len(host_ids)}卡持续异常")
            self.cooldown[host_key] = step + cfg.cooldown_steps

    def _do_eject(self, step: int, gpus: List[int], cause: str,
                  confidence: float, rationale: str):
        cfg = self.cfg
        if not self._within_eject_budget(len(gpus)):
            self._record(step, "eject_rejected", gpus, cause, confidence,
                         "超过最大剔除比例保护，拒绝剔除")
            return
        before = self.topo.num_active()
        cached = {}
        for g in gpus:
            # 剔除前缓存近窗命中序列，供观察窗结束时复核是否误判
            h = self.detector.hist[g]
            cached[g] = list(h.weak_hits)[-self.cfg.history_len:]
            self.topo.eject(g, step, cause)
            self.detector.reset_gpu(g)
            self.cooldown[g] = step + cfg.cooldown_steps
        self.last_change_step = step
        rec = self._record(step, "eject", gpus, cause, confidence,
                           f"{rationale}；活动卡 {before}→{self.topo.num_active()}，"
                           f"全局批大小保持{cfg.global_batch}，LR 再热身{cfg.topology_warmup_steps}步")
        self.pending.append(PendingEval(
            rec.action_id, "eject", step, gpus,
            baseline_wait=self._recent_wait(step, cfg.eject_evals),
            baseline_wall=self._recent_wall(step, cfg.eject_evals),
            cached_hit_seq=cached))

    # ------------------------------------------------------------------
    def _evaluate_pending(self, step: int):
        cfg = self.cfg
        still = []
        for p in self.pending:
            window = cfg.rebalance_evals if p.kind == "rebalance" else cfg.eject_evals
            if step - p.step < window:
                still.append(p)
                continue
            cur_wait = self._recent_wait(step, window)
            cur_wall = self._recent_wall(step, window)
            rec = self._find(p.action_id)
            improved_wait = cur_wait <= p.baseline_wait * (1 - cfg.rebalance_gain) \
                if p.kind == "rebalance" else \
                cur_wait <= p.baseline_wait * (1 - cfg.eject_gain)
            improved_wall = cur_wall < p.baseline_wall
            result = {"baseline_wait": p.baseline_wait, "after_wait": cur_wait,
                      "baseline_wall": p.baseline_wall, "after_wall": cur_wall}

            if p.kind == "rebalance":
                # 重分配效果看等待占比；没改善就撤回旧权重
                if improved_wait or improved_wall:
                    rec.status, rec.result = "kept", result
                    # 进入维护态：该卡的数据问题已被处理，不再重复触发重分配
                    for g in p.gpus:
                        self.data_streak[g] = 0
                        self.rebalance_maintaining.add(g)
                else:
                    self.topo.restore_weights(
                        p.weight_snapshot, step,
                        f"动作#{p.action_id}观察{window}步无改善，撤回数据重分配")
                    rec.status, rec.result = "rolled_back", result
                    # 屏蔽期 = 3 个评估窗口；期间该卡的数据问题不再触发重分配，
                    # 避免「重分配→撤回→再重分配」的震荡
                    for g in p.gpus:
                        self.cooldown[g] = step + cfg.cooldown_steps
                        self.rebalance_blocked[g] = step + 3 * cfg.rebalance_evals
                        self.data_streak[g] = 0
                        self.rebalance_maintaining.discard(g)
            else:
                # 剔除复核：只有「命中序列像周期」且「客观指标没改善」同时成立
                # 才撤回。剔除后等待/墙钟确实下降说明这张卡就是瓶颈，即使序列
                # 形态像周期也保留，避免把有用的剔除动作错误撤回
                misjudged = self._ejection_misjudged(p)
                objectively_helped = improved_wall or (
                    cur_wait < p.baseline_wait)
                if misjudged and not objectively_helped:
                    for g in p.gpus:
                        if self.topo.gpus[g].state == STATE_EJECTED:
                            self.topo.rejoin(g, step,
                                             f"动作#{p.action_id}证据复核：慢因周期性消失且无改善，撤回剔除")
                            self.detector.reset_gpu(g)
                            self.last_change_step = step
                    rec.status, rec.result = "rolled_back", {
                        **result, "reason": "慢因在观察窗内恢复且无客观改善，判定为误剔除并撤回"}
                else:
                    rec.status, rec.result = "kept", result
        self.pending = still

    def _ejection_misjudged(self, p: PendingEval) -> bool:
        """复核剔除是否误判。

        周期性慢因（温度/网卡争抢）本该在分类阶段被拦截，但事件若在观察窗口
        边界处相位特殊，可能漏过周期检测而被判成持续慢。剔除时缓存了该卡
        近窗的逐步异常命中序列，这里用严格标准复核（避免把带噪声的真坏卡
        误撤回）：
          - 异常命中率必须低于 40%（真持续慢通常 >70%）；
          - 且存在长度 ≥15 的连续恢复段；
          - 且至少有一个长度 ≥8 的成块异常段（周期形态而非零散噪声）。
        机器级动作不自动撤回（等维修事件）。
        """
        rec = self._find(p.action_id)
        if rec.cause == CAUSE_HOST:
            return False
        for g in p.gpus:
            seq = (p.cached_hit_seq or {}).get(g, [])
            if len(seq) < 60:
                continue
            hit_rate = sum(seq) / len(seq)
            text = "".join(str(x) for x in seq)
            if hit_rate < 0.4 and "000000000000000" in text and "11111111" in text:
                return True
        return False

    # ------------------------------------------------------------------
    def _process_rejoin_queue(self, step: int):
        ready = [q for q in self.rejoin_queue if q["ready"] <= step]
        self.rejoin_queue = [q for q in self.rejoin_queue if q["ready"] > step]
        for q in ready:
            g = q["gpu"]
            if self.topo.gpus[g].state == STATE_EJECTED:
                self.topo.rejoin(g, step, q["reason"])
                self.detector.reset_gpu(g)
                self.last_change_step = step
                self._record(step, "rejoin", [g], TRUTH_RECOVERY, 1.0, q["reason"])

    def _eligible_to_eject(self, g: int, v: GPUVerdict, step: int) -> bool:
        cfg = self.cfg
        if self.topo.gpus[g].state == STATE_EJECTED:
            return False
        if not self._off_cooldown(g, step):
            return False
        if v.confidence < cfg.confidence_eject:
            return False
        # 效率比不够突出不剔（剔除最小严重度保护）
        if v.efficiency_ratio < cfg.eject_min_slowdown:
            return False
        # 历史必须覆盖至少一个完整的退化/周期观察窗口，否则刚进慢相位的
        # 周期事件（或剔除重分配后的短暂扰动）会被当成永久故障误剔
        if len(self.detector.hist[g].efficiency) < cfg.progressive_window:
            return False
        if not self._within_eject_budget(1):
            return False
        return True

    def _within_eject_budget(self, extra: int) -> bool:
        ejected = len(self.topo.ejected_ids)
        return (ejected + extra) / self.cfg.num_gpus <= self.cfg.max_eject_fraction

    def _off_cooldown(self, g: int, step: int) -> bool:
        return step >= self.cooldown.get(g, 0)

    def _mark_watch_states(self, verdicts: Dict[int, GPUVerdict]):
        for g, v in verdicts.items():
            gs = self.topo.gpus[g]
            if gs.state == STATE_EJECTED:
                continue
            if v.cause in EJECT_CAUSES or v.cause == CAUSE_HOST:
                gs.state = STATE_WATCH
                gs.label = v.cause
            else:
                gs.state = STATE_ACTIVE
                gs.label = v.cause if v.cause != CAUSE_NONE else ""

    def _record(self, step: int, kind: str, gpus: List[int], cause: str,
                conf: float, rationale: str) -> ActionRecord:
        rec = ActionRecord(self._next_id, step, kind, list(gpus), cause,
                           conf, rationale)
        self._next_id += 1
        self.actions.append(rec)
        return rec

    def _find(self, action_id: int) -> ActionRecord:
        return next(a for a in self.actions if a.action_id == action_id)

    # 模拟器注入近窗统计（避免控制器自己存一份时序）
    def attach_history(self, wait_history: List[float], wall_history: List[float]):
        self._wait_hist = wait_history
        self._wall_hist = wall_history

    def _recent_wait(self, step: int, window: int) -> float:
        h = getattr(self, "_wait_hist", [])
        return float(np.mean(h[-window:])) if h else 0.0

    def _recent_wall(self, step: int, window: int) -> float:
        h = getattr(self, "_wall_hist", [])
        return float(np.mean(h[-window:])) if h else 1.0
