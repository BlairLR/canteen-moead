# -*- coding: utf-8 -*-
"""实验入口：10 次独立 MOEA/D 运行 + 指标统计。

每次运行：
  - 不同随机种子（1..10）
  - 记录：档案规模、各目标值域、HV、IGD、Spacing、耗时、评价次数、替换次数
  - 每 track_every 代记录一次 HV 收敛轨迹

指标统一在"固定归一化空间"计算：归一化边界取第一阶段 6000 随机采样的
分量 min/max（保证 10 次运行之间、以及与随机采样基线之间可比）。
HV 参考点 ref=(1.05, 1.05, 1.05)；IGD 参考集=第一阶段 2421 点非支配前沿。

输出：
  output/moead/archive_seed{S}_X.npy / _F.npy   各次运行的 Pareto 解集
  output/moead/experiment_results.json          全部统计（含聚合 mean/std/min/max）
"""

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # experiments/ 的上一级即项目根
sys.path.insert(0, ROOT)

from metrics import hv_3d, igd, normalize_fixed, spacing  # noqa: E402
from moead import MOEAD, MOEADConfig  # noqa: E402
from src.campus_model import evaluate  # noqa: E402

OUT = os.path.join(ROOT, "output", "moead")
os.makedirs(OUT, exist_ok=True)

SEEDS = [1, 2, 3]
HV_REF = (1.05, 1.05, 1.05)

# ---- 第一阶段数据：固定归一化边界 + IGD 参考前沿 ----
stage1_F = np.load(os.path.join(ROOT, "output", "samples_F.npy"))
stage1_pf = np.load(os.path.join(ROOT, "output", "samples_pf_mask.npy"))
LO, HI = stage1_F.min(axis=0), stage1_F.max(axis=0)
REF_FRONT = normalize_fixed(stage1_F[stage1_pf], LO, HI)
print(f"固定归一化边界 lo={np.round(LO, 3).tolist()} hi={np.round(HI, 3).tolist()}")
print(f"IGD 参考前沿（阶段一随机采样非支配集）: {len(REF_FRONT)} 点")


def run_one(seed: int) -> dict:
    cfg = MOEADConfig(seed=seed)
    traj = []

    def on_gen(gen, archive, n_repl):
        Fn = normalize_fixed(archive.F, LO, HI)
        traj.append([gen, hv_3d(Fn, HV_REF)])

    t0 = time.perf_counter()
    res = MOEAD(cfg, evaluate).run(on_generation=on_gen)
    dt = time.perf_counter() - t0

    Ax, AF = res["archive_X"], res["archive_F"]
    AFn = normalize_fixed(AF, LO, HI)

    # 与阶段一参考前沿合并后的"联合前沿"：衡量 MOEA/D 改进随机采样前沿的程度
    merged = np.vstack([REF_FRONT, AFn])
    from moead.dominance import dominated_by

    on_merged = int((~dominated_by(AFn, merged)).sum())
    ref_survive = int((~dominated_by(REF_FRONT, merged)).sum())

    return {
        "seed": seed,
        "n_evals": int(res["n_evals"]),
        "total_replacements": int(res["total_replacements"]),
        "archive_size": int(len(AF)),
        "time_s": round(dt, 2),
        "f1_min": float(AF[:, 0].min()), "f1_max": float(AF[:, 0].max()),
        "f2_min": float(AF[:, 1].min()), "f2_max": float(AF[:, 1].max()),
        "f3_min": float(AF[:, 2].min()), "f3_max": float(AF[:, 2].max()),
        "hv": float(hv_3d(AFn, HV_REF)),
        "igd": float(igd(AFn, REF_FRONT)),
        "spacing": float(spacing(AFn)),
        "on_merged_front": on_merged,
        "ref_survivors": ref_survive,
        "hv_trajectory": traj,
    }, Ax, AF


def agg(values):
    v = np.asarray(values, dtype=float)
    return {"mean": float(v.mean()), "std": float(v.std(ddof=1)),
            "min": float(v.min()), "max": float(v.max())}


if __name__ == "__main__":
    rows = []
    for seed in SEEDS:
        row, Ax, AF = run_one(seed)
        np.save(os.path.join(OUT, f"archive_seed{seed}_X.npy"), Ax)
        np.save(os.path.join(OUT, f"archive_seed{seed}_F.npy"), AF)
        rows.append(row)
        print(f"seed={seed:2d} | 档案 {row['archive_size']:4d} | HV {row['hv']:.4f} | "
              f"IGD {row['igd']:.5f} | Spacing {row['spacing']:.4f} | "
              f"联合前沿 {row['on_merged_front']:4d} | {row['time_s']:5.1f}s | "
              f"替换 {row['total_replacements']}")

    keys_scalar = ["archive_size", "time_s", "hv", "igd", "spacing",
                   "on_merged_front", "total_replacements", "n_evals",
                   "f1_min", "f1_max", "f2_min", "f2_max", "f3_min", "f3_max"]
    summary = {k: agg([r[k] for r in rows]) for k in keys_scalar}

    # 随机采样基线（阶段一前沿本身）的 HV，作为对照
    hv_baseline = hv_3d(REF_FRONT, HV_REF)
    summary["hv_baseline_random_sampling"] = hv_baseline

    results = {"seeds": SEEDS, "runs": rows, "summary": summary,
               "hv_ref": list(HV_REF),
               "norm_lo": LO.tolist(), "norm_hi": HI.tolist()}
    with open(os.path.join(OUT, "experiment_results.json"), "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)

    print("\n===== 3 次运行聚合统计 =====")
    for k, v in summary.items():
        if isinstance(v, dict):
            print(f"  {k:20s} mean={v['mean']:.5f} std={v['std']:.5f} "
                  f"min={v['min']:.5f} max={v['max']:.5f}")
    print(f"\n随机采样基线 HV = {hv_baseline:.4f}")
    print(f"结果已写入 {os.path.join(OUT, 'experiment_results.json')}")
