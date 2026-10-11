# -*- coding: utf-8 -*-
"""南湖版选址实验（单次运行）：阶段0 采样 + MOEA/D 1×200（A3 多边形搜索空间）。

数据 provenance：
  [真实数据 real]         21 点位坐标（4 竞争 + 10 宿舍 + 7 教学，WGS-84）+ A3 边界多边形

约束机制（moead/ 核心零改动）：
  - 阶段0/参考前沿采样：src.nanhu_campus.sample_in_polygon（拒绝法直接生成
    多边形内部均匀点，无投影、无边界聚集）；
  - MOEA/D 运行：evaluate 用 build_constrained_evaluate 包装——界外候选返回
    有界罚值（可行域各目标最大估计 ×1.1），罚解被一切可行解严格支配，
    外部档案保持干净，动态归一化上界稳定。约 30% 评价预算耗于界外候选。
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from metrics import hv_3d, igd, normalize_fixed, spacing, nondominated_mask  # noqa: E402
from moead import MOEAD, MOEADConfig  # noqa: E402
from moead.dominance import dominated_by  # noqa: E402
from src import nanhu_campus as nc  # noqa: E402

OUT = os.path.join(ROOT, "output", "moead_nanhu")
os.makedirs(OUT, exist_ok=True)

N_SAMPLES = 6000
SEEDS = [42]                 # 单次运行（主入口固定种子，仅用于学习演示）
HV_REF = (1.05, 1.05, 1.05)


def stage0(model, evaluate, out_dir: str | None = None) -> dict:
    """临时验证采样（A3 多边形内直接采样，方法同阶段一）。

    Parameters
    ----------
    model : NanhuModel
    evaluate : callable
    out_dir : str, optional — 输出目录，默认用模块级 OUT。
    """
    _dir = out_dir if out_dir is not None else OUT
    rng = np.random.default_rng(42)
    pts = nc.sample_in_polygon(N_SAMPLES, model.polygon_local, rng=rng)
    F = evaluate(pts)
    df = pd.DataFrame(F, columns=["f1", "f2", "f3"])
    spearman = df.rank().corr(method="pearson")
    pf = nondominated_mask(F)          # 非支配掩码（O(N·log|front|)，与 ~dominated_by(F,F) 等价）
    np.save(os.path.join(_dir, "stage0_samples_pts.npy"), pts)
    np.save(os.path.join(_dir, "stage0_samples_F.npy"), F)
    np.save(os.path.join(_dir, "stage0_samples_pf.npy"), pf)
    stats = {
        "n_samples": N_SAMPLES,
        "spearman": {f"{a}-{b}": float(spearman.loc[a, b])
                     for a, b in [("f1", "f2"), ("f1", "f3"), ("f2", "f3")]},
        "front_size": int(pf.sum()),
        "ranges": {n: [float(F[:, k].min()), float(F[:, k].max())] for k, n in enumerate(["f1", "f2", "f3"])},
    }
    with open(os.path.join(_dir, "stage0_stats.json"), "w", encoding="utf-8") as fh:
        json.dump(stats, fh, ensure_ascii=False, indent=2)
    return stats


def run_one(seed: int, model, evaluate, LO, HI, REF_FRONT):
    cfg = MOEADConfig(x_bounds=model.bounds_x, y_bounds=model.bounds_y, seed=seed)
    traj = []

    def on_gen(gen, archive, n_repl):
        Fn = normalize_fixed(archive.F, LO, HI)
        traj.append([gen, hv_3d(Fn, HV_REF)])

    t0 = time.perf_counter()
    res = MOEAD(cfg, evaluate).run(on_generation=on_gen)
    dt = time.perf_counter() - t0

    AF = res["archive_F"]
    AFn = normalize_fixed(AF, LO, HI)
    n_ref = len(REF_FRONT)
    merged_front = nondominated_mask(np.vstack([REF_FRONT, AFn]))  # 联合非支配前沿一次求出（与 ~dominated_by 等价）
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
        "on_merged_front": int(merged_front[n_ref:].sum()),
        "ref_survivors": int(merged_front[:n_ref].sum()),
        "hv_trajectory": traj,
    }, res["archive_X"], AF


def agg(values):
    v = np.asarray(values, dtype=float)
    return {"mean": float(v.mean()), "std": float(v.std(ddof=1)),
            "min": float(v.min()), "max": float(v.max())}


def run_experiment(evaluate_fn=None, out_dir=None, model=None):
    """运行一次完整的南湖选址实验（阶段0 + MOEA/D + 结果保存）。

    这是主实验入口，供命令行和 Flask 网页共用。

    Parameters
    ----------
    evaluate_fn : callable, optional
        外部评价函数（如区域模拟地价版）。未提供则使用默认高斯 f3。
    out_dir : str, optional
        输出目录。未提供则使用 output/moead_nanhu。
    model : NanhuModel, optional
        校园模型。未提供则自动构建。

    Returns
    -------
    dict — 实验结果（含 summary、runs、model 元数据等）。
    """
    nonlocal_model = model if model is not None else nc.build_model()
    _OUT = out_dir if out_dir is not None else OUT
    os.makedirs(_OUT, exist_ok=True)

    if evaluate_fn is not None:
        evaluate, caps = evaluate_fn, None
        # 如果外部 evaluate 是 (fn, caps) 元组则拆包
        if isinstance(evaluate_fn, (list, tuple)) and len(evaluate_fn) == 2:
            evaluate, caps = evaluate_fn
        # 否则从外部 evaluate 自己推导 caps（不需严格匹配）
    else:
        evaluate, caps = nc.build_constrained_evaluate(nonlocal_model)

    # 界外候选评价比例统计（计数包装器，不改变评价结果）
    eval_stats = {"n_out": 0, "n_total": 0}

    def counting_evaluate(X):
        Xa = np.atleast_2d(np.asarray(X, dtype=float))
        n_out = int((~nc.is_inside_polygon(nonlocal_model.polygon_local, Xa)).sum())
        eval_stats["n_out"] += n_out
        eval_stats["n_total"] += int(len(Xa))
        return evaluate(X)

    print(f"模型构造完成: A3 多边形搜索空间，外包矩形 x[{nonlocal_model.bounds_x[0]:.1f},{nonlocal_model.bounds_x[1]:.1f}] "
          f"y[{nonlocal_model.bounds_y[0]:.1f},{nonlocal_model.bounds_y[1]:.1f}]"
          f" | 需求点 {len(nonlocal_model.demand_xy)} 个 {nonlocal_model.demand_ids}"
          f" | 竞争点 {len(nonlocal_model.competition_xy)} 个 {nonlocal_model.competition_ids}"
          f" | caps={'N/A (external)' if caps is None else np.round(caps, 2).tolist()}", flush=True)

    s0 = stage0(nonlocal_model, evaluate, out_dir=_OUT)
    print(f"阶段0 临时验证（A3 内直接采样 n={N_SAMPLES}）: 前沿 {s0['front_size']}/{N_SAMPLES} | "
          f"Spearman {json.dumps({k: round(v, 3) for k, v in s0['spearman'].items()})}", flush=True)

    stage0_F = np.load(os.path.join(_OUT, "stage0_samples_F.npy"))
    stage0_pf = np.load(os.path.join(_OUT, "stage0_samples_pf.npy"))
    LO, HI = stage0_F.min(axis=0), stage0_F.max(axis=0)
    REF_FRONT = normalize_fixed(stage0_F[stage0_pf], LO, HI)

    rows = []
    for seed in SEEDS:
        eval_stats["n_out"] = 0
        eval_stats["n_total"] = 0
        row, Ax, AF = run_one(seed, nonlocal_model, counting_evaluate, LO, HI, REF_FRONT)
        row["n_out_evals"] = eval_stats["n_out"]
        row["n_total_evals"] = eval_stats["n_total"]
        row["out_eval_frac"] = eval_stats["n_out"] / max(1, eval_stats["n_total"])
        np.save(os.path.join(_OUT, f"archive_seed{seed}_X.npy"), Ax)
        np.save(os.path.join(_OUT, f"archive_seed{seed}_F.npy"), AF)
        rows.append(row)
        print(f"seed={seed:2d} | 档案 {row['archive_size']:5d} | HV {row['hv']:.4f} | "
              f"IGD {row['igd']:.5f} | Spacing {row['spacing']:.4f} | "
              f"联合前沿 {row['on_merged_front']:5d} | 界外评价 {row['out_eval_frac']:.1%} | "
              f"{row['time_s']:6.1f}s", flush=True)

    keys = ["archive_size", "time_s", "hv", "igd", "spacing", "on_merged_front",
            "total_replacements", "n_evals", "out_eval_frac",
            "f1_min", "f1_max", "f2_min", "f2_max", "f3_min", "f3_max"]
    summary = {k: agg([r[k] for r in rows]) for k in keys}
    summary["hv_baseline_random_sampling"] = hv_3d(REF_FRONT, HV_REF)

    results = {
        "provenance": nonlocal_model.provenance,
        "model": {
            "search_space": "A3 Polygon（data/nanhu_boundary_polygon.json）",
            "bounds_x": list(nonlocal_model.bounds_x), "bounds_y": list(nonlocal_model.bounds_y),
            "n_polygon_vertices": int(len(nonlocal_model.polygon_local)),
            "demand_ids": nonlocal_model.demand_ids, "competition_ids": nonlocal_model.competition_ids,
            "penalty_caps": [float(c) for c in caps] if caps is not None else None,
        },
        "stage0": s0,
        "seeds": SEEDS, "runs": rows, "summary": summary,
        "hv_ref": list(HV_REF),
        "norm_lo": LO.tolist(), "norm_hi": HI.tolist(),
    }
    with open(os.path.join(_OUT, "experiment_results.json"), "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)

    print("\n===== 单次运行结果 =====", flush=True)
    for k, v in summary.items():
        if isinstance(v, dict):
            print(f"  {k:20s} mean={v['mean']:.5f} std={v['std']:.5f} "
                  f"min={v['min']:.5f} max={v['max']:.5f}", flush=True)
    print(f"随机采样基线 HV = {summary['hv_baseline_random_sampling']:.4f}", flush=True)
    print(f"结果已写入 {_OUT}", flush=True)

    return results


if __name__ == "__main__":
    run_experiment()
