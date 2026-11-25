"""效果评估：把题目要求的 5 个量化指标全部算出来。

  1. 等待时间占比下降（full 组 vs observe 组，逐窗对比）；
  2. 训练到同一损失水平的总墙钟时间（以 clean 组的最终损失为目标）；
  3. 误剔除数：剔除/撤回当时并不处于真坏状态的卡；
  4. 真坏卡发现时延：从事件注入到第一次剔除动作的步数；
  5. 损失曲线衔接：剔除动作前后窗口的损失差与梯度差。
"""
from __future__ import annotations

from typing import List, Dict

import numpy as np

from .simulator import Simulator
from .events import TRUTH_PERSISTENT, TRUTH_PROGRESSIVE

BAD_KINDS = {TRUTH_PERSISTENT, TRUTH_PROGRESSIVE}


def wait_ratio_by_window(sim: Simulator, window: int = 100,
                         normalized: bool = True) -> List[dict]:
    waits = np.array(sim.wait_norm_history if normalized
                     else sim.wait_history)
    out = []
    for s in range(0, len(waits), window):
        chunk = waits[s:s + window]
        if len(chunk):
            out.append({"start": s, "end": s + len(chunk),
                        "mean_wait_ratio": float(chunk.mean())})
    return out


def wait_reduction(full: Simulator, control: Simulator, window: int = 100) -> dict:
    a = wait_ratio_by_window(full, window)
    b = wait_ratio_by_window(control, window)
    rows, gains = [], []
    for ra, rb in zip(a, b):
        base = rb["mean_wait_ratio"]
        gain = (base - ra["mean_wait_ratio"]) / base if base > 1e-9 else 0.0
        rows.append({"start": ra["start"], "end": ra["end"],
                     "full_wait": ra["mean_wait_ratio"],
                     "observe_wait": rb["mean_wait_ratio"],
                     "relative_reduction": gain})
        gains.append(gain)
    full_norm = float(np.mean(full.wait_norm_history))
    ctrl_norm = float(np.mean(control.wait_norm_history))
    return {"note": "等待占比按满配卡数归一化，剔除卡后可跨拓扑公平比较",
            "overall_mean_wait_full": full_norm,
            "overall_mean_wait_observe": ctrl_norm,
            "overall_mean_wait_full_raw": float(np.mean(full.wait_history)),
            "overall_mean_wait_observe_raw": float(np.mean(control.wait_history)),
            "overall_relative_reduction": float(
                (ctrl_norm - full_norm) / max(ctrl_norm, 1e-9)),
            "by_window": rows}


def time_to_target_loss(sim: Simulator, target_loss: float,
                        smooth: int = 20) -> dict:
    losses = np.array(sim.loss_history)
    kernel = np.ones(smooth) / smooth
    sm = np.convolve(losses, kernel, mode="valid")
    hit = np.where(sm <= target_loss)[0]
    if len(hit) == 0:
        return {"target": target_loss, "reached": False,
                "steps_needed": None, "wall_ms_needed": None,
                "best_smoothed_loss": float(sm.min())}
    idx = int(hit[0]) + smooth - 1
    wall = float(np.sum(sim.wall_history[:idx + 1]))
    return {"target": target_loss, "reached": True,
            "steps_needed": idx, "wall_ms_needed": round(wall, 1)}


def ejection_audit(sim: Simulator) -> dict:
    """逐条剔除动作对照 ground truth，统计误剔除与发现时延。"""
    mgr = sim.events
    false_ejects, true_ejects, delays = [], [], []
    for rec in sim.controller.actions:
        if rec.kind not in ("eject", "host_eject"):
            continue
        bad_now = mgr.persistent_bad_gpus(rec.step)
        for g in rec.gpus:
            if g in bad_now:
                true_ejects.append({"action": rec.action_id, "gpu": g,
                                    "step": rec.step})
            else:
                false_ejects.append({"action": rec.action_id, "gpu": g,
                                     "step": rec.step, "cause": rec.cause})
    # 发现时延：每个真坏事件从 start 到其卡第一次被剔除（只统计事件发生后的动作）
    for ev in mgr.events:
        if ev.kind not in BAD_KINDS or not ev.gpus:
            continue
        eject_steps = [r.step for r in sim.controller.actions
                       if r.kind in ("eject", "host_eject")
                       and r.step >= ev.start
                       and set(r.gpus) & set(ev.gpus)]
        if eject_steps:
            first = min(eject_steps)
            delays.append({"event": ev.id, "gpus": ev.gpus,
                           "injected_step": ev.start,
                           "detected_step": first,
                           "delay_steps": first - ev.start,
                           "delay_wall_ms": round(float(np.sum(
                               sim.wall_history[ev.start:first + 1])), 1)})
        else:
            delays.append({"event": ev.id, "gpus": ev.gpus,
                           "injected_step": ev.start, "detected_step": None,
                           "delay_steps": None, "delay_wall_ms": None})
    return {"false_ejects": false_ejects,
            "false_eject_count": len(false_ejects),
            "true_ejects": true_ejects,
            "true_eject_count": len(true_ejects),
            "detection_delays": delays,
            "rolled_back_ejections": [a.action_id for a in sim.controller.actions
                                      if a.kind == "eject"
                                      and a.status == "rolled_back"]}


def loss_continuity(sim: Simulator) -> List[dict]:
    """每次拓扑变化（剔除/加回）前后损失是否衔接。"""
    out = []
    losses = np.array(sim.loss_history)
    w = 10
    for change in sim.topo.change_log:
        if change["action"] not in ("eject", "rejoin"):
            continue
        s = change["step"]
        if s - w < 0 or s + w >= len(losses):
            continue
        before = losses[s - w:s]
        after = losses[s + 1:s + 1 + w]
        out.append({
            "step": s, "action": change["action"],
            "gpu": change.get("gpu"),
            "loss_before": round(float(before.mean()), 5),
            "loss_after": round(float(after.mean()), 5),
            "abs_delta": round(float(after.mean() - before.mean()), 5),
            "relative_delta_pct": round(float(
                (after.mean() - before.mean()) / max(before.mean(), 1e-9) * 100), 2),
            "std_before": round(float(before.std()), 5),
            "std_after": round(float(after.std()), 5)})
    return out


def full_report(full: Simulator, observe: Simulator, clean: Simulator) -> dict:
    target = float(np.mean(clean.loss_history[-50:]))
    return {
        "target_loss_from_clean_run": round(target, 5),
        "wall_total": {
            "full_ms": round(float(np.sum(full.wall_history)), 1),
            "observe_ms": round(float(np.sum(observe.wall_history)), 1),
            "clean_ms": round(float(np.sum(clean.wall_history)), 1),
            "full_vs_observe_saving_pct": round(float(
                (np.sum(observe.wall_history) - np.sum(full.wall_history))
                / np.sum(observe.wall_history) * 100), 2)},
        "wait_ratio": wait_reduction(full, observe),
        "time_to_target": {
            "full": time_to_target_loss(full, target),
            "observe": time_to_target_loss(observe, target),
            "clean": time_to_target_loss(clean, target)},
        "ejection_audit": ejection_audit(full),
        "loss_continuity": loss_continuity(full),
        "action_counts": {
            "rebalance": sum(1 for a in full.controller.actions
                             if a.kind == "rebalance"),
            "rebalance_rollback": sum(1 for a in full.controller.actions
                                      if a.status == "rolled_back"
                                      and a.kind == "rebalance"),
            "eject": sum(1 for a in full.controller.actions
                         if a.kind in ("eject", "host_eject")),
            "eject_rollback": sum(1 for a in full.controller.actions
                                  if a.kind == "eject"
                                  and a.status == "rolled_back"),
            "rejoin": sum(1 for a in full.controller.actions
                          if a.kind == "rejoin"),
            "rejected": sum(1 for a in full.controller.actions
                            if "rejected" in a.kind)},
        "final_loss": {
            "full": round(float(np.mean(full.loss_history[-50:])), 5),
            "observe": round(float(np.mean(observe.loss_history[-50:])), 5),
            "clean": round(target, 5)}}
