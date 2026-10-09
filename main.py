# -*- coding: utf-8 -*-
"""主入口。

用法（在项目根目录 canteen-moead/ 下）：
  python main.py            # 完整流程：10 次独立实验 + 全部图表
  python main.py --plots    # 只重新出图（用已保存的实验结果）
  python main.py --tests    # 只跑正确性自检 + 冒烟测试
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def run_tests():
    import subprocess

    py = sys.executable
    for script in ["tests/sanity_tests.py", "tests/smoke_test.py"]:
        print(f"\n>>> 运行 {script}")
        r = subprocess.run([py, os.path.join(HERE, script)], cwd=HERE)
        if r.returncode != 0:
            sys.exit(f"{script} 未通过")


def run_experiments():
    from experiments.run_experiments import run_one, agg, SEEDS, OUT  # noqa: F401
    import experiments.run_experiments as rex

    rex.__name__ = "__main__"
    # 直接复用实验模块的 __main__ 逻辑
    import numpy as np  # noqa: F401
    import json

    rows = []
    for seed in rex.SEEDS:
        row, Ax, AF = rex.run_one(seed)
        np.save(os.path.join(rex.OUT, f"archive_seed{seed}_X.npy"), Ax)
        np.save(os.path.join(rex.OUT, f"archive_seed{seed}_F.npy"), AF)
        rows.append(row)
        print(f"seed={seed:2d} | 档案 {row['archive_size']:4d} | HV {row['hv']:.4f} | "
              f"IGD {row['igd']:.5f} | Spacing {row['spacing']:.4f} | "
              f"联合前沿 {row['on_merged_front']:4d} | {row['time_s']:5.1f}s")
    keys = ["archive_size", "time_s", "hv", "igd", "spacing", "on_merged_front",
            "total_replacements", "n_evals", "f1_min", "f1_max",
            "f2_min", "f2_max", "f3_min", "f3_max"]
    summary = {k: rex.agg([r[k] for r in rows]) for k in keys}
    summary["hv_baseline_random_sampling"] = rex.hv_3d(rex.REF_FRONT, rex.HV_REF)
    results = {"seeds": rex.SEEDS, "runs": rows, "summary": summary,
               "hv_ref": list(rex.HV_REF),
               "norm_lo": rex.LO.tolist(), "norm_hi": rex.HI.tolist()}
    with open(os.path.join(rex.OUT, "experiment_results.json"), "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
    print("\n===== 10 次运行聚合统计 =====")
    for k, v in summary.items():
        if isinstance(v, dict):
            print(f"  {k:20s} mean={v['mean']:.5f} std={v['std']:.5f} "
                  f"min={v['min']:.5f} max={v['max']:.5f}")
    print(f"结果已写入 {os.path.join(rex.OUT, 'experiment_results.json')}")


def run_plots():
    from visualization.moead_plots import make_all_figures

    make_all_figures()


if __name__ == "__main__":
    if "--tests" in sys.argv:
        run_tests()
    elif "--plots" in sys.argv:
        run_plots()
    else:
        run_experiments()
        run_plots()
