"""检测器与控制器的单元测试。

运行：python3 -m pytest tests/ -v
（无 pytest 时可直接 python3 tests/test_detector.py）
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from slownode.config import SimConfig
from slownode.detector import (Detector, mad_z, CAUSE_DATA, CAUSE_PERSISTENT,
                               CAUSE_PROGRESSIVE, CAUSE_PERIODIC_GPU,
                               CAUSE_HOST, CAUSE_TRANSIENT)
from slownode.events import SlowEvent, TRUTH_THERMAL, TRUTH_HOST_NIC
from slownode.topology import Topology
from slownode.controller import Controller


def _feed(det, step, active, total_per_gpu, tokens_per_gpu,
          data_share=0.18, comm_share=0.18):
    """直接喂构造的计时数据，绕开计时仿真，便于精确控制单卡行为。"""
    ids = np.array(active)
    total = np.array([total_per_gpu[g] for g in active])
    tokens = np.array([tokens_per_gpu[g] for g in active], dtype=float)
    data_t = total * data_share
    compute_t = total * (1 - data_share - comm_share)
    comm_t = total * comm_share
    det.update(step, list(ids), data_t, compute_t, comm_t, tokens)


def test_mad_z_robust_to_outlier():
    """单个极端离群值不应拉爆其它卡的 z 分。"""
    x = np.ones(64)
    x[0] = 5.0
    z = mad_z(x)
    assert z[0] > 7
    assert np.abs(z[1:]).max() < 1e-9


def test_persistent_slow_gpu_classified_persistent():
    cfg = SimConfig()
    det = Detector(cfg, 64, 8)
    rng = np.random.default_rng(0)
    for step in range(cfg.warmup_steps + cfg.persistent_steps + 5):
        total = {g: 100 * (1 + rng.normal(0, 0.005)) for g in range(64)}
        tokens = {g: 200.0 for g in range(64)}
        total[7] = 175 * (1 + rng.normal(0, 0.01))   # 持续慢 75%
        _feed(det, step, list(range(64)), total, tokens)
        if step % cfg.decide_every == 0:
            det.classify(step)
    det.classify(step)
    v = det.latest[7]
    assert v.cause == CAUSE_PERSISTENT
    assert v.confidence >= cfg.confidence_eject


def test_data_skew_classified_as_data_not_gpu_fault():
    """工作量高但单位 token 效率正常 → 数据原因，绝不进剔除流程。"""
    cfg = SimConfig()
    det = Detector(cfg, 64, 8)
    rng = np.random.default_rng(1)
    for step in range(cfg.warmup_steps + 80):
        # 正常卡 200 token / 100ms；倾斜卡 400 token，单位 token 效率相同，
        # 但因承担更多工作在屏障处有轻微排队，产生弱异常而非效率劣化
        total = {g: 100 * (1 + rng.normal(0, 0.004)) for g in range(64)}
        tokens = {g: 200.0 for g in range(64)}
        total[3] = 210 * (1 + rng.normal(0, 0.004))
        tokens[3] = 400.0
        _feed(det, step, list(range(64)), total, tokens)
        if step % cfg.decide_every == 0:
            det.classify(step)
    det.classify(cfg.warmup_steps + 79)   # 循环结束后补一次判决
    v = det.latest[3]
    assert v.cause == CAUSE_DATA
    assert v.confidence < cfg.confidence_eject


def test_thermal_cycle_never_ejectable():
    """温度周期卡在慢-恢复循环下必须被判周期类，置信度低于剔除阈值。"""
    cfg = SimConfig()
    det = Detector(cfg, 64, 8)
    rng = np.random.default_rng(2)
    period, duty = 240, 120
    for step in range(cfg.warmup_steps + period + 30):
        total = {g: 100 * (1 + rng.normal(0, 0.004)) for g in range(64)}
        tokens = {g: 200.0 for g in range(64)}
        phase = step % period
        if phase < duty:
            total[11] = 155 * (1 + rng.normal(0, 0.006))
        _feed(det, step, list(range(64)), total, tokens)
        if step % cfg.decide_every == 0 and step >= cfg.warmup_steps:
            det.classify(step)
    det.classify(step)
    v = det.latest[11]
    assert v.cause in (CAUSE_PERIODIC_GPU, CAUSE_TRANSIENT)
    assert v.confidence < cfg.confidence_eject


def test_progressive_degradation_detected():
    """效率单调爬升、几乎不回落 → 退化类。"""
    cfg = SimConfig()
    det = Detector(cfg, 64, 8)
    rng = np.random.default_rng(3)
    start = cfg.warmup_steps
    # 异常需持续超过 min_anomaly_age(180) 才能判退化，预留足够序列
    for step in range(start + cfg.min_anomaly_age + 30):
        total = {g: 100 * (1 + rng.normal(0, 0.004)) for g in range(64)}
        tokens = {g: 200.0 for g in range(64)}
        factor = 1.0 + 0.004 * max(0, step - start)
        total[44] = 100 * min(factor, 3.0) * (1 + rng.normal(0, 0.003))
        _feed(det, step, list(range(64)), total, tokens)
        if step % cfg.decide_every == 0:
            det.classify(step)
    det.classify(step)
    v = det.latest[44]
    # 退化或持续慢都属于硬件类剔除原因；关键是不能被判成会自愈的周期类
    assert v.cause in (CAUSE_PROGRESSIVE, CAUSE_PERSISTENT)
    assert v.confidence >= cfg.confidence_eject
    assert det._efficiency_peak_drawdown(det.hist[44]) < 0.12


def test_sporadic_spike_not_persistent():
    """偶发尖峰（稀疏命中）应判偶发或正常，绝不进剔除类原因。"""
    cfg = SimConfig()
    det = Detector(cfg, 64, 8)
    rng = np.random.default_rng(4)
    total_steps = cfg.warmup_steps + 60
    # 最后一个尖峰落在近窗内，保证当前确实有稀疏异常
    spike_steps = {cfg.warmup_steps + 10, total_steps - 3}
    for step in range(total_steps):
        total = {g: 100 * (1 + rng.normal(0, 0.004)) for g in range(64)}
        tokens = {g: 200.0 for g in range(64)}
        if step in spike_steps:
            total[52] = 190.0
        _feed(det, step, list(range(64)), total, tokens)
        if step % cfg.decide_every == 0 and step >= cfg.warmup_steps:
            det.classify(step)
    det.classify(step)
    v = det.latest[52]
    # 偶发/正常/周期都属于「不动手」，关键是不能被判成持续慢或退化（剔除类）
    assert v.cause not in (CAUSE_PERSISTENT, CAUSE_PROGRESSIVE, CAUSE_HOST)
    assert v.confidence < cfg.confidence_eject


def test_eject_budget_cap():
    """剔除数量保护：超过 1/4 卡时拒绝继续剔除。"""
    cfg = SimConfig()
    rng = np.random.default_rng(5)
    topo = Topology(cfg, rng)
    det = Detector(cfg, 64, 8)
    ctrl = Controller(cfg, topo, det)
    # 手动剔到接近上限
    cap = int(64 * cfg.max_eject_fraction)
    for g in range(cap):
        topo.eject(g, 0, "test")
    assert not ctrl._within_eject_budget(1)
    assert ctrl._within_eject_budget(0)


def test_local_batch_sums_to_global():
    """全局批固定：任意权重下本地批之和恒等于 global_batch。"""
    cfg = SimConfig()
    rng = np.random.default_rng(6)
    topo = Topology(cfg, rng)
    for trial in range(20):
        for g in topo.active_ids:
            topo.gpus[g].alloc_weight = float(rng.uniform(0.2, 3.0))
        batches = topo.local_batches()
        assert batches.sum() == cfg.global_batch
        assert (batches >= 1).all()


def test_lr_warmup_and_recovery():
    """拓扑变化后 LR 做再热身并恢复目标值。"""
    cfg = SimConfig()
    rng = np.random.default_rng(7)
    topo = Topology(cfg, rng)
    # 初始热身
    assert topo.lr_for_topology(0, 0) < cfg.base_lr
    assert abs(topo.lr_for_topology(cfg.init_warmup_steps + 5, 0)
               - cfg.base_lr) < 1e-9
    # 拓扑变化后再热身
    lr0 = topo.lr_for_topology(1000, 1000)
    assert lr0 < cfg.base_lr
    lr_end = topo.lr_for_topology(1000 + cfg.topology_warmup_steps, 1000)
    assert abs(lr_end - cfg.base_lr) < 1e-9


def test_local_policy_lr_scales_with_active_count():
    """local 策略下 LR 随活动卡数线性缩放。"""
    cfg = SimConfig(batch_policy="local")
    rng = np.random.default_rng(8)
    topo = Topology(cfg, rng)
    for g in range(8):
        topo.eject(g, 500, "test")
    lr = topo.lr_for_topology(600, 500)
    expected = cfg.base_lr * (56 / 64)
    assert abs(lr - expected) < 1e-9


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
        passed += 1
    print(f"\n{passed}/{len(fns)} 个测试通过")


if __name__ == "__main__":
    _run_all()
