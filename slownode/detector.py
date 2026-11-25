"""慢节点检测器。

设计原则（对应题目「误判代价很大，有把握才动手」）：
  1. 同侪比较全部用中位数 + MAD（中位数绝对偏差），单张离群卡不会拉偏基线；
  2. 每个异常先做分量归因（data / compute / comm），再做工作量归因，
     区分「卡本身慢」和「分到的活多」；
  3. 原因分六类，每类的判决都要求多个独立证据同时成立并累计观察窗口；
  4. 输出 0~1 的置信度，剔除动作要求 ≥0.9 且经过多个决策周期确认。
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Optional

import numpy as np

from .config import SimConfig
from .events import (TRUTH_DATA_SKEW, TRUTH_TRANSIENT, TRUTH_PERSISTENT,
                     TRUTH_PROGRESSIVE, TRUTH_THERMAL, TRUTH_HOST_NIC,
                     TRUTH_CLUSTER_STORAGE, COMPONENT_DATA, COMPONENT_COMM,
                     COMPONENT_COMPUTE)

# 检测器输出的原因标签
CAUSE_NONE = "normal"
CAUSE_TRANSIENT = "sporadic"             # 偶发一次，忍着
CAUSE_DATA = "data_driven"               # 跟着数据走 → 重分配
CAUSE_PERSISTENT = "persistent_gpu"      # 卡/机器持续慢 → 剔除
CAUSE_PROGRESSIVE = "degrading_gpu"      # 越来越慢 → 剔除
CAUSE_PERIODIC_GPU = "thermal_cycle"     # 卡级周期（温度）→ 不剔
CAUSE_HOST = "host_level"                # 机器级（网卡）→ 谨慎处理
CAUSE_CLUSTER = "cluster_window"         # 时间段/集群事件 → 不剔


@dataclass
class GPUVerdict:
    gpu_id: int
    state: str = CAUSE_NONE
    cause: str = CAUSE_NONE
    confidence: float = 0.0
    component: str = ""
    z_total: float = 0.0
    workload_ratio: float = 1.0
    efficiency_ratio: float = 1.0
    evidence: List[str] = field(default_factory=list)


def mad_z(values: np.ndarray) -> np.ndarray:
    """中位数/MAD 标准化；MAD 退化（全相同）时退回普通标准差。"""
    med = np.median(values)
    mad = np.median(np.abs(values - med))
    if mad < 1e-9:
        sd = values.std()
        if sd < 1e-9:
            return np.zeros_like(values)
        return (values - med) / sd
    return (values - med) / (1.4826 * mad)


class _GPUHistory:
    def __init__(self, cap: int):
        self.total = deque(maxlen=cap)
        self.data = deque(maxlen=cap)
        self.compute = deque(maxlen=cap)
        self.comm = deque(maxlen=cap)
        self.efficiency = deque(maxlen=cap)   # 每 token 耗时（越低越好）
        self.workload = deque(maxlen=cap)
        self.strong_hits = deque(maxlen=cap)  # 1=该步强异常
        self.weak_hits = deque(maxlen=cap)
        self.components = deque(maxlen=cap)

    def reset(self):
        for q in (self.total, self.data, self.compute, self.comm,
                  self.efficiency, self.workload, self.strong_hits,
                  self.weak_hits, self.components):
            q.clear()


class Detector:
    def __init__(self, cfg: SimConfig, num_gpus: int, gpus_per_host: int):
        self.cfg = cfg
        self.num_gpus = num_gpus
        self.gpus_per_host = gpus_per_host
        self.hist: Dict[int, _GPUHistory] = {
            g: _GPUHistory(cfg.history_len) for g in range(num_gpus)}
        self.latest: Dict[int, GPUVerdict] = {
            g: GPUVerdict(g) for g in range(num_gpus)}
        self.cluster_bad_fraction: float = 0.0
        # 每台机器「同机多数卡异常」连续成立的判决次数
        self.host_streak: Dict[int, int] = {}
        self.last_step = -1

    def reset_gpu(self, gpu: int):
        """卡重新加入时清空历史，避免旧数据污染。"""
        self.hist[gpu].reset()
        self.latest[gpu] = GPUVerdict(gpu)

    # ------------------------------------------------------------------
    def update(self, step: int, active_ids: List[int], data_t: np.ndarray,
               compute_t: np.ndarray, comm_t: np.ndarray,
               token_count: np.ndarray) -> Dict[int, GPUVerdict]:
        cfg = self.cfg
        ids = np.array(active_ids)
        total_t = data_t + compute_t + comm_t

        z_data = mad_z(data_t)
        z_compute = mad_z(compute_t)
        z_comm = mad_z(comm_t)
        z_total = mad_z(total_t)
        med_tokens = np.median(token_count)
        workload_ratio = token_count / max(med_tokens, 1e-9)
        # 效率：单位 token 的耗时，与工作量解耦
        efficiency = total_t / np.maximum(token_count, 1)
        med_eff = np.median(efficiency)
        eff_ratio = efficiency / max(med_eff, 1e-9)

        weak_mask = z_total >= cfg.z_weak
        strong_mask = z_total >= cfg.z_strong
        self.cluster_bad_fraction = float(weak_mask.mean())

        for i, g in enumerate(ids):
            g = int(g)
            h = self.hist[g]
            comp = self._attribute_component(z_data[i], z_compute[i], z_comm[i])
            h.total.append(float(total_t[i]))
            h.data.append(float(data_t[i]))
            h.compute.append(float(compute_t[i]))
            h.comm.append(float(comm_t[i]))
            h.efficiency.append(float(efficiency[i]))
            h.workload.append(float(workload_ratio[i]))
            h.strong_hits.append(1 if strong_mask[i] else 0)
            h.weak_hits.append(1 if weak_mask[i] else 0)
            h.components.append(comp)
            v = GPUVerdict(
                gpu_id=g, z_total=float(z_total[i]),
                workload_ratio=float(workload_ratio[i]),
                efficiency_ratio=float(eff_ratio[i]), component=comp)
            self.latest[g] = v

        self.last_step = step
        return self.latest

    @staticmethod
    def _attribute_component(z_d: float, z_c: float, z_m: float) -> str:
        scores = {COMPONENT_DATA: z_d, COMPONENT_COMPUTE: z_c, COMPONENT_COMM: z_m}
        return max(scores, key=scores.get)

    # ------------------------------------------------------------------
    def classify(self, step: int) -> Dict[int, GPUVerdict]:
        """在决策点对所有活动卡做原因分类（每 decide_every 步调用一次）。"""
        cfg = self.cfg
        # 预热期：历史窗口未填满，只给观察标签，绝不触发动作
        if step < cfg.warmup_steps:
            return self.latest
        # 先判断当前是否处于集群级时间窗
        cluster_window = self.cluster_bad_fraction >= cfg.cluster_event_ratio
        host_flags = self._host_anomaly_counts()
        # 更新每台机器的连续异常计数
        num_hosts = self.num_gpus // self.gpus_per_host
        for host in range(num_hosts):
            host_bad, host_total = host_flags.get(host, (0, 0))
            is_host_event = host_total > 0 and \
                host_bad / host_total >= cfg.host_event_ratio
            if is_host_event:
                self.host_streak[host] = self.host_streak.get(host, 0) + 1
            else:
                self.host_streak[host] = 0

        # 跨卡效率归一化基线：h.efficiency 存的是原始「单位 token 耗时」，
        # 必须除以同侪中位数得到效率比（慢卡 >1，快卡 <1），后续判据才成立
        active = [g for g in range(self.num_gpus)
                  if len(self.hist[g].efficiency) >= cfg.persistent_steps]
        if active:
            peer_eff = np.median([
                np.median(list(self.hist[g].efficiency)[-cfg.persistent_steps:])
                for g in active])
            peer_eff = max(float(peer_eff), 1e-9)
        else:
            peer_eff = 1.0

        for g in range(self.num_gpus):
            v = self.latest[g]
            h = self.hist[g]
            if len(h.total) < cfg.persistent_steps:
                continue
            recent_n = min(cfg.persistent_steps, len(h.weak_hits))
            weak = list(h.weak_hits)[-recent_n:]
            strong = list(h.strong_hits)[-recent_n:]
            n_weak, n_strong = sum(weak), sum(strong)
            if n_weak == 0:
                v.state, v.cause, v.confidence = CAUSE_NONE, CAUSE_NONE, 0.0
                v.evidence = []
                continue

            wr = float(np.median(list(h.workload)[-recent_n:]))
            er = float(np.median(list(h.efficiency)[-recent_n:])) / peer_eff
            comp_dom = self._dominant_component(h, recent_n)
            v.evidence = [
                f"近{recent_n}步弱异常{n_weak}次/强异常{n_strong}次",
                f"工作量比={wr:.2f} 效率比={er:.2f}",
                f"主导分量={comp_dom}", f"当前MAD-z={v.z_total:.1f}"]

            # 1) 数据驱动：长窗口内工作量稳定偏高、但效率正常。
            #    用长窗口中位数排除单步采样波动，且要求异常命中率本身不低
            wr_long = float(np.median(list(h.workload)[-cfg.progressive_window:])) \
                if len(h.workload) >= 30 else wr
            eff_long = (float(np.median(list(h.efficiency)[-cfg.progressive_window:]))
                        / peer_eff) if len(h.efficiency) >= 30 else er
            if (wr_long >= 1 + cfg.data_min_gap
                    and abs(eff_long - 1.0) <= cfg.efficiency_band
                    and wr >= cfg.workload_ratio_high):
                self._set(v, CAUSE_DATA, comp_dom, n_weak, recent_n,
                          extra=f"长窗工作量高{wr_long:.2f}而效率{eff_long:.2f}正常→数据倾斜")
                continue

            # 2) 集群级时间窗：超过一半卡一起慢，谁也不剔
            if cluster_window:
                v.state = CAUSE_CLUSTER
                v.cause = CAUSE_CLUSTER
                v.confidence = min(0.5 + 0.1 * n_weak / recent_n, 0.8)
                v.evidence.append(f"{self.cluster_bad_fraction:.0%}卡同时异常→时间段事件")
                continue

            # 3) 机器级：同机多数卡连续多个判决点一起异常（网卡/机器问题）
            host = g // self.gpus_per_host
            host_bad, host_total = host_flags.get(host, (0, 0))
            streak = self.host_streak.get(host, 0)
            host_confirmed = streak >= cfg.host_persistent_windows
            # 机器级剔除需要更久的观察：必须持续超过一个典型周期窗口，
            # 否则周期性网卡争抢在慢相位内会被误判成永久故障
            host_eject_ready = streak >= cfg.host_eject_streak
            if host_total and host_bad / host_total >= cfg.host_event_ratio \
                    and host_confirmed:
                self._classify_host(v, h, recent_n, host_bad, host_total,
                                    comp_dom, streak, host_eject_ready)
                continue

            # 4) 周期性优先（温度/周期争抢）：必须用长窗口，覆盖至少一个完整
            #    慢-恢复循环，否则短窗会把慢相位误判成持续慢
            if self._looks_periodic_long(h):
                v.state = CAUSE_PERIODIC_GPU
                v.cause = CAUSE_PERIODIC_GPU
                v.confidence = min(0.6 + 0.05 * n_weak / recent_n, 0.85)
                v.evidence.append("长窗口呈现≥2个慢-恢复循环→温度类周期降频，不剔除")
                continue

            # 4b) 周期保护：历史上出现过显著连续恢复段（>=15 步不异常）即排除
            #     单向退化/持续故障。真坏卡不会自愈，只拦周期事件慢相位误判
            if self._has_recovery_episode(h):
                v.state = CAUSE_PERIODIC_GPU
                v.cause = CAUSE_PERIODIC_GPU
                v.confidence = 0.7
                v.evidence.append("历史出现连续恢复段，会自愈，按周期事件观察不剔除")
                continue

            # 5) 退化：效率斜率持续向上（越来越慢）。要求当前效率仍显著高于
            #    同侪且斜率在长窗口稳定，避免刚启动/抖动造成的虚高斜率。
            #    额外必要条件：异常期效率相对峰值没有显著回落（单调恶化），
            #    温度周期在慢相位也有正斜率，但它前期必有大幅自愈回落
            slope = self._efficiency_slope(h)
            peak_drawdown = self._efficiency_peak_drawdown(h)
            anomaly_age = self._anomaly_age(h)
            if (slope >= cfg.progressive_min_slope
                    and n_strong >= cfg.min_strong_hits
                    and peak_drawdown < 0.12
                    and anomaly_age >= cfg.min_anomaly_age):
                v.state = CAUSE_PROGRESSIVE
                v.cause = CAUSE_PROGRESSIVE
                conf = self._confidence(n_weak, n_strong, recent_n, slope)
                v.confidence = min(conf + 0.05, 0.98)
                v.evidence.append(
                    f"效率斜率={slope:.4f}/步、异常持续{anomaly_age}步→硬件退化")
                continue

            # 6) 持续慢 vs 偶发。同样要求长窗口没有显著自愈回落，
            #    否则周期事件在慢相位内会满足持续慢的所有短窗条件
            persistent_drawdown = self._efficiency_peak_drawdown(h)
            if (n_strong >= cfg.min_strong_hits
                    and er > 1 + cfg.efficiency_band
                    and persistent_drawdown < 0.12
                    and anomaly_age >= cfg.min_persistent_age):
                v.state = CAUSE_PERSISTENT
                v.cause = CAUSE_PERSISTENT
                v.confidence = self._confidence(n_weak, n_strong, recent_n)
                v.evidence.append(
                    f"效率持续偏低、强异常{anomaly_age}步且无自愈→卡本身慢")
            elif persistent_drawdown >= 0.12:
                v.state = CAUSE_PERIODIC_GPU
                v.cause = CAUSE_PERIODIC_GPU
                v.confidence = 0.7
                v.evidence.append(
                    f"长窗效率回落{persistent_drawdown:.2f}→会自愈，按周期事件观察不剔除")
            elif n_weak <= max(2, recent_n // cfg.sporadic_gap):
                v.state = CAUSE_TRANSIENT
                v.cause = CAUSE_TRANSIENT
                v.confidence = 0.3
                v.evidence.append("异常稀疏→偶发，继续观察")
            else:
                # 证据不足，挂观察态，不触发动作
                v.state = CAUSE_TRANSIENT
                v.cause = CAUSE_TRANSIENT
                v.confidence = 0.5
                v.evidence.append("有异常但证据不充分，保持观察")
        return self.latest

    # ------------------------------------------------------------------
    def _classify_host(self, v: GPUVerdict, h: _GPUHistory, recent_n: int,
                       host_bad: int, host_total: int, comp_dom: str,
                       streak: int, eject_ready: bool):
        cfg = self.cfg
        n_strong = sum(list(h.strong_hits)[-recent_n:])
        n_weak = sum(list(h.weak_hits)[-recent_n:])
        # 效率必须显著差于同侪，排除「只是被多分了活」的卡
        if v.efficiency_ratio < cfg.host_min_efficiency:
            v.state = CAUSE_TRANSIENT
            v.cause = CAUSE_TRANSIENT
            v.confidence = 0.4
            v.evidence.append(
                f"同机{host_bad}/{host_total}卡异常但效率比仅{v.efficiency_ratio:.2f}，证据不足")
            return
        # 机器级周期（网卡被抢）：游程检验能识别出恢复段
        if self._looks_periodic_long(h):
            v.state = CAUSE_HOST
            v.cause = CAUSE_HOST
            v.confidence = min(0.6 + 0.05 * host_bad / host_total, 0.85)
            v.evidence.append(
                f"同机{host_bad}/{host_total}卡异常且周期性恢复→网卡/机器争抢")
            return
        # 持续时间不足以排除周期性原因：保守观察，置信度压到剔除阈值以下
        if not eject_ready:
            v.state = CAUSE_HOST
            v.cause = CAUSE_HOST
            v.confidence = min(0.5 + 0.05 * n_weak / recent_n, 0.8)
            v.evidence.append(
                f"同机{host_bad}/{host_total}卡异常已持续{streak}个判决点，"
                f"未达机器级剔除门槛{cfg.host_eject_streak}，保守观察")
            return
        frac = host_bad / host_total
        v.state = CAUSE_HOST
        v.cause = CAUSE_HOST
        base = self._confidence(n_weak, n_strong, recent_n)
        v.confidence = min(base + 0.1 * frac, 0.95)
        v.evidence.append(f"同机{host_bad}/{host_total}卡持续异常{streak}个判决点"
                          f"且效率比{v.efficiency_ratio:.2f}→机器级故障")

    def _host_anomaly_counts(self) -> Dict[int, tuple]:
        out: Dict[int, list] = {}
        for g in range(self.num_gpus):
            h = self.hist[g]
            if not h.weak_hits:
                continue
            host = g // self.gpus_per_host
            cur = out.setdefault(host, [0, 0])
            cur[1] += 1
            if list(h.weak_hits)[-1] == 1:
                cur[0] += 1
        return {k: tuple(v) for k, v in out.items()}

    def _dominant_component(self, h: _GPUHistory, n: int) -> str:
        comps = list(h.components)[-n:]
        return max(set(comps), key=comps.count)

    def _looks_periodic(self, h: _GPUHistory) -> bool:
        """短窗周期判断：命中序列同时出现成段 1 和明显的 0 段。"""
        seq = "".join(str(x) for x in list(h.weak_hits))
        if len(seq) < 30 or "1" not in seq or "0" not in seq:
            return False
        ones = seq.count("1")
        zeros = seq.count("0")
        if ones / len(seq) < 0.2 or zeros / len(seq) < 0.2:
            return False
        return "00000" in seq and "11" in seq

    def _looks_periodic_long(self, h: _GPUHistory) -> bool:
        """长窗周期判断（主判决用）。

        只分析窗口内第一次异常之后的序列，排除健康期正常的未命中被误当成
        「恢复段」。周期事件必须给出「慢-恢复-慢」的实际交替：
          - 强证据（≥2 个慢段 + ≥2 个恢复段）：直接判周期；
          - 弱证据（≥1 个慢段 + ≥1 个长度 ≥15 的恢复段且命中率 <60%）：
            也判周期并保持不剔除，宁可放过也不把会自愈的卡当坏卡。
        持续慢/退化在异常开始后只有连续命中，给不出成段恢复，不会被误保护。
        """
        seq_full = list(h.weak_hits)[-self.cfg.history_len:]
        if len(seq_full) < 45:
            return False
        first_one = next((i for i, x in enumerate(seq_full) if x == 1), None)
        if first_one is None:
            return False
        seq = seq_full[first_one:]
        text = "".join(str(x) for x in seq)
        import re
        on_runs = re.findall(r"1{8,}", text)
        off_long = re.findall(r"0{15,}", text)
        off_runs = re.findall(r"0{10,}", text)
        hit_rate = sum(seq) / len(seq)
        if len(on_runs) >= 2 and len(off_runs) >= 2:
            return True
        if len(on_runs) >= 1 and len(off_long) >= 1 and hit_rate < 0.6:
            return True
        return False

    def _has_recovery_episode(self, h: _GPUHistory) -> bool:
        """是否存在「异常段 → 连续恢复段 → 再次异常」的自愈闭环。

        只看窗口内第一次异常之后的序列，排除健康期正常的未命中被误当成恢复。
        持续慢/退化卡在异常开始后只有连续命中，给不出这个闭环；
        周期事件（温度/网卡）则必然呈现慢-恢复-慢的交替。
        """
        import re
        seq = list(h.weak_hits)
        if len(seq) < 45:
            return False
        first_one = next((i for i, x in enumerate(seq) if x == 1), None)
        if first_one is None:
            return False
        text = "".join(str(x) for x in seq[first_one:])
        # 异常(>=5) → 恢复(>=15 个 0) → 再次异常(>=5 个 1)
        return re.search(r"1{5,}0{15,}1{5,}", text) is not None

    def _anomaly_age(self, h: _GPUHistory) -> int:
        """从当前往前数，连续弱异常命中的长度（异常已持续多少步）。"""
        seq = list(h.weak_hits)
        age = 0
        for x in reversed(seq):
            if x == 1:
                age += 1
            else:
                break
        return age

    def _efficiency_slope(self, h: _GPUHistory) -> float:
        eff = np.array(list(h.efficiency)[-self.cfg.progressive_window:])
        if len(eff) < 20:
            return 0.0
        x = np.arange(len(eff))
        # 归一化后的最小二乘斜率（每步相对变化）
        slope = np.polyfit(x, eff / max(eff[0], 1e-9), 1)[0]
        return float(slope)

    def _efficiency_peak_drawdown(self, h: _GPUHistory) -> float:
        """效率（单位 token 耗时，越大越慢）相对窗口内峰值的最大回落比例。

        用完整历史窗口（最多 200 步）而不是退化斜率的 60 步短窗：60 步可能
        整个落在周期事件的慢相位上升段内，看不到回落。退化卡的效率在整个
        历史上单调爬升、几乎不回落，drawdown 只有噪声量级；周期事件的
        长窗口必然包含「慢相位峰值 → 恢复回落」，drawdown 显著。
        """
        eff = np.array(list(h.efficiency))
        if len(eff) < 40:
            return 0.0
        running_peak = np.maximum.accumulate(eff)
        drawdown = (running_peak - eff) / np.maximum(running_peak, 1e-9)
        return float(drawdown.max())

    def _confidence(self, n_weak: int, n_strong: int, recent_n: int,
                    slope: float = 0.0) -> float:
        hit_score = 0.45 * (n_weak / recent_n) + 0.35 * min(n_strong / max(
            self.cfg.min_strong_hits, 1), 1.0)
        slope_score = min(slope / (self.cfg.progressive_min_slope * 4), 0.2)
        return float(min(0.5 + hit_score + slope_score, 0.98))

    @staticmethod
    def _set(v: GPUVerdict, cause: str, comp: str, n_weak: int,
             recent_n: int, extra: str = ""):
        v.state = cause
        v.cause = cause
        v.confidence = float(min(0.6 + 0.3 * n_weak / recent_n, 0.92))
        v.evidence = v.evidence + [extra]
