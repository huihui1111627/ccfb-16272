"""剧本：在 3000 步训练中注入题目要求的全部慢因，且多个同时发生。

GPU 编号 0..63，机器 host = gpu // 8，共 8 台机器。
时间线（步）：
  250~   偶发尖峰（两张卡，稀疏随机）
  400~   数据倾斜（3 张卡长期只拿长样本），1250 步倾斜对象切换 → 检验再分配撤回
  700~   两张卡持续慢（机器/硬件问题）→ 剔除；其中一张 1500 步维修后加回
  900~   温度降频（卡级周期性，慢-恢复循环）→ 应忍住不剔
 1100~   硬件退化（越来越慢）→ 应在越过阈值后剔除
 1300~   机器 6 网卡被抢（机器级周期性）→ 机器级观察，不整台剔
 1600~1900 共享存储抖动（全集群时间段事件）→ 不剔任何卡
 2050~   一张其实正常的卡被强噪声打成持续慢 → 误判保护/撤回演示
 2300    剔除位上的新卡上线（换卡加回）
"""
from __future__ import annotations

from typing import List

from .events import (SlowEvent, TRUTH_TRANSIENT, TRUTH_DATA_SKEW,
                     TRUTH_PERSISTENT, TRUTH_PROGRESSIVE, TRUTH_THERMAL,
                     TRUTH_HOST_NIC, TRUTH_CLUSTER_STORAGE,
                     COMPONENT_DATA, COMPONENT_COMPUTE, COMPONENT_COMM)


def build_scenario(total_steps: int = 3000) -> List[SlowEvent]:
    events: List[SlowEvent] = []

    # 1) 偶发尖峰：两张卡，每步个位数概率的单次变慢，噪声来源不同
    events.append(SlowEvent(
        id="transient-7", kind=TRUTH_TRANSIENT, start=250, end=total_steps,
        gpus=[7], component=COMPONENT_COMPUTE, factor=1.9, prob=0.03))
    events.append(SlowEvent(
        id="transient-52", kind=TRUTH_TRANSIENT, start=250, end=total_steps,
        gpus=[52], component=COMPONENT_DATA, factor=1.7, prob=0.025))

    # 2) 数据倾斜：3 张卡被分到长样本；1250 步切换倾斜对象
    skew_a = [3, 18, 41]
    skew_b = [22, 47]
    events.append(SlowEvent(
        id="skew-a", kind=TRUTH_DATA_SKEW, start=400, end=1250,
        gpus=skew_a, component=COMPONENT_DATA, workload_mult=2.0))
    events.append(SlowEvent(
        id="skew-b", kind=TRUTH_DATA_SKEW, start=1250, end=2100,
        gpus=skew_b, component=COMPONENT_DATA, workload_mult=2.0))

    # 3) 持续慢（机器问题）：两张卡；GPU 27 在 1500 维修，GPU 33 一直坏到换卡。
    #    真实硬件故障通常慢 2 倍以上，弱于这个量级的剔除收益抵不过工作量摊薄
    events.append(SlowEvent(
        id="persist-27", kind=TRUTH_PERSISTENT, start=700, end=1500,
        gpus=[27], component=COMPONENT_COMPUTE, factor=2.3))
    events.append(SlowEvent(
        id="persist-33", kind=TRUTH_PERSISTENT, start=700, end=2300,
        gpus=[33], component=COMPONENT_COMPUTE, factor=2.1))

    # 4) 温度降频：卡级周期，慢 120 步、恢复 120 步，带爬升渐变
    events.append(SlowEvent(
        id="thermal-11", kind=TRUTH_THERMAL, start=900, end=total_steps,
        gpus=[11], component=COMPONENT_COMPUTE,
        factor=1.55, period=240, duty=0.5, ramp=15))

    # 5) 硬件退化：每步 +0.004 因子，约 250 步后翻倍，封顶 3 倍
    events.append(SlowEvent(
        id="degrade-44", kind=TRUTH_PROGRESSIVE, start=1100, end=total_steps,
        gpus=[44], component=COMPONENT_COMPUTE,
        slope=0.004, factor_max=3.0))

    # 6) 机器 6（GPU 48..55）网卡被抢：机器级周期，通信分量
    events.append(SlowEvent(
        id="nic-host6", kind=TRUTH_HOST_NIC, start=1300, end=total_steps,
        host=6, gpus=list(range(48, 56)), component=COMPONENT_COMM,
        factor=1.8, period=300, duty=0.4, ramp=10))

    # 7) 共享存储抖动：全集群数据加载变慢的时间窗
    events.append(SlowEvent(
        id="storage-window", kind=TRUTH_CLUSTER_STORAGE,
        start=1600, end=1900, gpus=None,
        component=COMPONENT_DATA, factor=1.45))

    # 8) 误判保护演示：好卡 GPU 60 被强噪声持续打慢 200 步后自行恢复。
    #    若在恢复前被剔除，观察窗复核命中序列中的恢复段后撤回加回。
    events.append(SlowEvent(
        id="false-suspect-60", kind=TRUTH_PERSISTENT, start=2050, end=2250,
        gpus=[60], component=COMPONENT_COMPUTE, factor=1.45))

    return events
