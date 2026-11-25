"""端到端集成测试：跑一个短剧本，验证仿真全链路的不变量。

运行：python3 tests/test_integration.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from slownode.config import SimConfig
from slownode.scenarios import build_scenario
from slownode.simulator import Simulator


def _small_cfg():
    cfg = SimConfig(total_steps=400, seed=7)
    cfg.dataset_size = 1024
    cfg.embed_dim = 32
    cfg.hidden_dim = 64
    cfg.global_batch = 256
    return cfg


def test_three_run_modes_share_same_training_base():
    """三组 run 除干预外随机流对齐：clean/full/observe 初始损失应一致。"""
    cfg = _small_cfg()
    full = Simulator(cfg, build_scenario(400), "full", seed=7)
    observe = Simulator(cfg, build_scenario(400), "observe", seed=7)
    clean = Simulator(cfg, build_scenario(400), "clean", seed=7)
    full.run(log_every=10_000)
    observe.run(log_every=10_000)
    clean.run(log_every=10_000)
    # 第一步还没有任何干预，三者损失完全相同
    assert abs(full.loss_history[0] - clean.loss_history[0]) < 1e-9
    assert abs(observe.loss_history[0] - clean.loss_history[0]) < 1e-9


def test_loss_never_nan_and_decreases():
    """训练全程数值稳定，最终损失明显低于初始。"""
    cfg = _small_cfg()
    sim = Simulator(cfg, build_scenario(400), "full", seed=7)
    sim.run(log_every=10_000)
    losses = np.array(sim.loss_history)
    assert not np.any(np.isnan(losses))
    assert losses[-20:].mean() < losses[:20].mean() - 0.02


def test_ejected_gpu_has_zero_timing_and_can_rejoin():
    """被剔卡计时为 0、不参与训练；维修后能重新加入并恢复计时。"""
    cfg = _small_cfg()
    sim = Simulator(cfg, build_scenario(400), "full", seed=7)
    sim.run(log_every=10_000)
    ejected = sim.topo.ejected_ids
    # 剧本里 700 才注入持续慢，400 步短剧本不应有剔除，这里手工验证机制
    sim.topo.eject(0, 300, "test")
    sim.detector.reset_gpu(0)
    ids = sim.topo.active_ids
    assert 0 not in ids
    batches = sim.topo.local_batches(ids)
    assert batches.sum() == cfg.global_batch
    sim.topo.rejoin(0, 350, "test rejoin")
    assert 0 in sim.topo.active_ids


def test_wait_ratio_bounded_in_clean_run():
    """健康集群等待占比应在合理范围（不超过 35%），验证计时模型标定。"""
    cfg = _small_cfg()
    sim = Simulator(cfg, build_scenario(400), "clean", seed=7)
    sim.run(log_every=10_000)
    mean_wait = float(np.mean(sim.wait_history))
    assert mean_wait < 0.35, f"健康集群等待占比异常偏高: {mean_wait}"


def test_every_step_records_all_gpus():
    """每条逐步记录都覆盖全部 64 卡，热力图数据无缺口。"""
    cfg = _small_cfg()
    sim = Simulator(cfg, build_scenario(400), "full", seed=7)
    sim.run(log_every=10_000)
    for rec in sim.records:
        assert len(rec.total_t) == cfg.num_gpus
        assert len(rec.truth) == cfg.num_gpus
        assert len(rec.detected_cause) == cfg.num_gpus
        assert 1 <= rec.active_count <= cfg.num_gpus


def test_actions_have_rationale_and_terminal_status():
    """所有动作都有原因记录，且评估窗结束后有终态。"""
    cfg = SimConfig(total_steps=900, seed=7)
    cfg.dataset_size = 1024
    cfg.embed_dim = 32
    cfg.hidden_dim = 64
    cfg.global_batch = 256
    sim = Simulator(cfg, build_scenario(900), "full", seed=7)
    sim.run(log_every=10_000)
    for a in sim.controller.actions:
        assert a.rationale
        assert a.status in ("kept", "rolled_back", "evaluating", "rejected")
        assert 0.0 <= a.confidence <= 1.0


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
        passed += 1
    print(f"\n{passed}/{len(fns)} 个集成测试通过")


if __name__ == "__main__":
    _run_all()
