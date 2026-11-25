"""实验入口：同一剧本跑 full / observe / clean 三组并导出页面数据。

用法：
  python3 run_experiment.py                      # 默认 3000 步
  python3 run_experiment.py --steps 800 --quick  # 快速冒烟
  python3 run_experiment.py --policy local       # 每卡批大小固定 + LR 线性缩放
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from slownode.config import SimConfig
from slownode.scenarios import build_scenario
from slownode.simulator import Simulator
from slownode.reporting import write_experiment_json
from slownode.metrics import full_report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--policy", choices=["global", "local"], default="global")
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "output", "experiment.json"))
    ap.add_argument("--quick", action="store_true",
                    help="冒烟模式：缩小模型和数据，快速验证流程")
    args = ap.parse_args()

    cfg = SimConfig(total_steps=args.steps, seed=args.seed,
                    batch_policy=args.policy)
    if args.quick:
        cfg.dataset_size = 1024
        cfg.embed_dim = 32
        cfg.hidden_dim = 64
        cfg.global_batch = 256
    events = build_scenario(args.steps)

    t0 = time.time()
    print("=== Run 1/3: full（检测 + 处理全开）===")
    full = Simulator(cfg, events, run_mode="full", seed=args.seed)
    full.run()

    print("=== Run 2/3: observe（只检测不处理，对照组）===")
    observe = Simulator(cfg, build_scenario(args.steps),
                        run_mode="observe", seed=args.seed)
    observe.run()

    print("=== Run 3/3: clean（无慢因，健康基线）===")
    clean = Simulator(cfg, build_scenario(args.steps),
                      run_mode="clean", seed=args.seed)
    clean.run()

    report = full_report(full, observe, clean)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    write_experiment_json(args.out, full, observe, clean, cfg)
    with open(args.out.replace(".json", "_report.json"), "w",
              encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n仿真墙钟耗时 {time.time() - t0:.1f}s")
    print("=" * 68)
    print("评估报告")
    print("=" * 68)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n页面数据已写入 {args.out}")
    print("查看页面：python3 serve.py  然后浏览器打开 http://localhost:8000")


if __name__ == "__main__":
    main()
