"""把仿真结果导出成页面可直接加载的 JSON（含热力图全量逐步数据）。"""
from __future__ import annotations

import json
from typing import Dict

import numpy as np

from .simulator import Simulator
from .metrics import full_report


def export_run(sim: Simulator) -> dict:
    recs = sim.records
    n = sim.cfg.num_gpus
    # 热力图：每张卡一条逐步序列，整数化压缩体积
    heat = {
        "total": [[int(round(x)) for x in r.total_t] for r in recs],
        "data": [[int(round(x)) for x in r.data_t] for r in recs],
        "compute": [[int(round(x)) for x in r.compute_t] for r in recs],
        "comm": [[int(round(x)) for x in r.comm_t] for r in recs],
        "tokens": [r.tokens for r in recs],
    }
    steps = [r.step for r in recs]
    hosts = [g // sim.cfg.gpus_per_host for g in range(n)]
    return {
        "mode": sim.run_mode,
        "num_gpus": n,
        "gpus_per_host": sim.cfg.gpus_per_host,
        "hosts": hosts,
        "steps": steps,
        "heatmap": heat,
        "wall_ms": [r.wall_ms for r in recs],
        "wait_ratio": [r.wait_ratio for r in recs],
        "wait_ratio_normalized": [r.wait_ratio_normalized for r in recs],
        "loss": [r.loss for r in recs],
        "lr": [r.lr for r in recs],
        "active_count": [r.active_count for r in recs],
        "truth": [r.truth for r in recs],
        "detected_cause": [r.detected_cause for r in recs],
        "confidence": [r.confidence for r in recs],
        "gpu_state": [r.gpu_state for r in recs],
        "actions": [_action(a) for a in sim.controller.actions],
        "topology_log": sim.topo.change_log,
        "summary": sim.summary(),
    }


def _action(a) -> dict:
    d = a.__dict__.copy()
    return d


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    raise TypeError(f"不可序列化的类型 {type(o)}")


def write_experiment_json(path: str, full: Simulator, observe: Simulator,
                          clean: Simulator, cfg) -> None:
    payload = {
        "config": cfg.to_dict(),
        "events": [_event_dict(ev) for ev in full.events.events],
        "runs": {
            "full": export_run(full),
            "observe": export_run(observe),
            "clean": export_run(clean),
        },
        "report": full_report(full, observe, clean),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, default=_json_default)


def _event_dict(ev) -> dict:
    return {k: v for k, v in ev.__dict__.items() if k != "rng"}
