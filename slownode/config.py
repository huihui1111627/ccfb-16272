"""全局配置：集群、计时模型、检测器、控制器、小模型训练的参数。

时间单位统一为毫秒（ms）。所有随机源都走显式传入的 numpy.random.Generator，
保证同一剧本可复现，也保证「干预 run / 对照 run」除干预外完全一致。
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class SimConfig:
    # ---- 训练规模 ----
    total_steps: int = 3000
    num_gpus: int = 64
    gpus_per_host: int = 8            # 64 卡 = 8 台机器，每台 8 卡
    seed: int = 42

    # ---- 小模型（真的在算的缩小模型）----
    vocab_size: int = 2000
    embed_dim: int = 64
    hidden_dim: int = 128
    num_classes: int = 10
    dataset_size: int = 4096
    seq_len_mu: float = 3.0          # log(seq_len) 的均值，exp(3)≈20 token
    seq_len_sigma: float = 0.35
    seq_len_clip: tuple = (8, 96)
    long_pool_ratio: float = 0.25    # 数据集里最长的 1/4 构成「长样本池」

    # ---- 训练超参（全局批大小固定模式）----
    global_batch: int = 512
    base_lr: float = 0.015
    momentum: float = 0.9
    grad_clip: float = 5.0           # 全局梯度范数裁剪，防止个别大步发散
    init_warmup_steps: int = 60      # 训练初期 LR 线性热身
    weight_clip: float = 8.0         # 参数值温和约束，兜底防发散
    batch_policy: str = "global"     # global=保持全局批大小; local=保持每卡批大小
    topology_warmup_steps: int = 15  # 拓扑变化后的 LR 再热身
    lr_scale_clip: float = 2.0       # local 模式下线性缩放规则的截断倍数

    # ---- 计时模型基线（单步，毫秒）----
    base_data_ms: float = 18.0       # 数据加载基线
    base_compute_ms: float = 75.0    # 前向+反向基线
    base_comm_ms: float = 22.0       # 集合通信基线
    gpu_cap_spread: float = 0.015    # 同型号卡固有效能差 ±1.5%
    noise_cv: float = 0.006          # 步间独立抖动（小，模拟系统噪声）
    slow_noise_cv: float = 0.012     # 慢卡的步间抖动略大
    ar_coef: float = 0.85            # 卡级慢漂移的自相关系数（AR(1)）
    ar_sd: float = 0.006             # 卡级慢漂移的稳态标准差
    # 事件叠加时的合并方式：各因子相乘（乘性降速），上限在事件层各自截断
    factor_clip: float = 6.0

    # ---- 检测器（MAD：中位数绝对偏差，抗离群卡）----
    decide_every: int = 5            # 每 5 步做一次完整判决（每步仍更新统计）
    warmup_steps: int = 120          # 预热期：历史不足时只观察不动手
    z_strong: float = 3.5            # 强异常门槛（MAD z-score）
    z_weak: float = 2.0
    history_len: int = 320       # 须覆盖最长周期事件（温度 240 步）的完整循环
    persistent_steps: int = 25       # 持续慢的观察窗口
    min_strong_hits: int = 5         # 窗口内强异常命中次数
    host_persistent_windows: int = 4 # 机器级事件需连续 4 个判决点成立
    host_eject_streak: int = 24      # 机器级剔除需持续 24 个判决点（120 步），
                                     # 超过网卡周期的慢相位，确保不是周期争抢
    progressive_window: int = 60     # 退化斜率的观察窗口
    progressive_min_slope: float = 0.0025   # 每步效率因子涨幅（60 步涨 15%）
    min_anomaly_age: int = 180      # 退化类剔除前异常至少持续的步数
    min_persistent_age: int = 130   # 持续慢类剔除前的最小异常年龄，
                                    # 超过温度慢相位(120步)，保证见过一次可能恢复
    sporadic_gap: int = 40           # 窗口内异常过于稀疏 => 偶发
    workload_ratio_high: float = 1.40       # 工作量明显高于同侪（健康卡 99 分位 ~1.26）
    workload_ratio_low: float = 0.75
    efficiency_band: float = 0.18    # 效率落在同侪 ±18% 内视为正常
    data_min_gap: float = 0.35       # 数据倾斜：长窗工作量须高于中位数 35%
    data_top_fraction: float = 0.10  # 只对工作量最高的 10% 卡考虑重分配
    cluster_event_ratio: float = 0.50       # ≥50% 卡同时异常 => 时间段/集群事件
    host_event_ratio: float = 0.625  # 同机 ≥5/8 卡异常 => 机器级事件
    host_eject_fraction: float = 0.75       # 机器级持续慢：≥75% 卡坏才整台处理

    # ---- 控制器 ----
    confidence_eject: float = 0.90   # 剔除必须达到的置信度
    confidence_rebalance: float = 0.70
    rebalance_evals: int = 50        # 重分配后的观察窗口（覆盖多个判决周期）
    eject_evals: int = 30            # 剔除后的观察窗口
    rebalance_gain: float = 0.10     # 等待占比至少降 10%（相对）才算改善
    eject_gain: float = 0.25
    cooldown_steps: int = 40         # 同一对象两次动作的冷却
    topology_settle_steps: int = 60 # 拓扑变化后所有判决暂停的稳定期
    max_eject_fraction: float = 0.25 # 最多剔除 1/4 卡，超出拒绝
    eject_min_slowdown: float = 1.30 # 慢于中位数 30% 才允许剔除
    host_min_efficiency: float = 1.20 # 机器级判决要求效率比同侪慢 20%

    def to_dict(self) -> dict:
        return asdict(self)
