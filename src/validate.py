# -*- coding: utf-8 -*-
"""
阶段一验证实验：不运行 MOEA/D，仅对目标空间做随机采样分析。

实验内容：
  1. 在校园范围内均匀随机采样 N=6000 个候选位置（>5000 要求）
  2. 计算每个位置的 (f1, f2, f3)
  3. Spearman 相关矩阵 —— 检测目标间相关性 / 潜在冗余
  4. 三维 Pareto 非支配前沿提取 + 三组成对二维前沿
  5. 冲突性与 trade-off 幅度量化
  6. 输出统计结果（JSON + 控制台）与 5 张图

运行：python validate.py（在 src 目录下，或保证 campus_model 可导入）
"""

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from campus_model import (  # noqa: E402
    CAMPUS_X,
    CAMPUS_Y,
    COST_BASE,
    COST_HOTSPOTS,
    DORMITORIES,
    EXISTING_CANTEENS,
    N_DORM,
    SERVICE_RADIUS,
    evaluate,
)

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
os.makedirs(OUT_DIR, exist_ok=True)

N_SAMPLES = 6000
SEED = 42
OBJ_NAMES = ["f1 平均距离(m)", "f2 竞争重叠度", "f3 土地成本(万元)"]


# ---------------------------------------------------------------------
# Pareto 工具
# ---------------------------------------------------------------------
def pareto_mask(F: np.ndarray) -> np.ndarray:
    """返回布尔掩码：三目标（全最小化）下的非支配点。

    O(N²) 逐点检查：若存在任一点在所有目标上 ≤ i 且至少一个目标 < i，则 i 被支配。
    """
    N = F.shape[0]
    keep = np.ones(N, dtype=bool)
    for i in range(N):
        if not keep[i]:
            continue
        dominates_i = np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1)
        if dominates_i.any():
            keep[i] = False
    return keep


def pareto_mask_2d(F2: np.ndarray) -> np.ndarray:
    """二维非支配前沿（两个目标均最小化），O(N log N) 扫描法。"""
    a, b = F2[:, 0], F2[:, 1]
    order = np.lexsort((b, a))
    keep = np.zeros(len(F2), dtype=bool)
    best_b = np.inf
    for i in order:
        if b[i] < best_b:
            keep[i] = True
            best_b = b[i]
    return keep


# ---------------------------------------------------------------------
# 1. 采样与目标计算
# ---------------------------------------------------------------------
rng = np.random.default_rng(SEED)
pts = np.column_stack(
    [
        rng.uniform(CAMPUS_X[0], CAMPUS_X[1], N_SAMPLES),
        rng.uniform(CAMPUS_Y[0], CAMPUS_Y[1], N_SAMPLES),
    ]
)
F = evaluate(pts)
print(f"采样 {N_SAMPLES} 个候选位置（seed={SEED}），目标矩阵 {F.shape}")

# ---------------------------------------------------------------------
# 2. 相关性分析（Spearman，对非线性单调关系稳健）
# ---------------------------------------------------------------------
df = pd.DataFrame(F, columns=["f1", "f2", "f3"])
spearman = df.rank().corr(method="pearson")  #秩变换后的 Pearson = Spearman
pearson = df.corr(method="pearson")
print("\nSpearman 相关矩阵：")
print(spearman.round(4).to_string())

# ---------------------------------------------------------------------
# 3. Pareto 前沿
# ---------------------------------------------------------------------
pf = pareto_mask(F)
F_pf = F[pf]
print(f"\n三目标非支配前沿点数：{pf.sum()} / {N_SAMPLES}（被支配比例 {(1 - pf.mean()) * 100:.1f}%）")

# 各目标全样本范围 vs 前沿范围
ranges = {}
for k, name in enumerate(["f1", "f2", "f3"]):
    ranges[name] = {
        "full_min": float(F[:, k].min()),
        "full_max": float(F[:, k].max()),
        "pf_min": float(F_pf[:, k].min()),
        "pf_max": float(F_pf[:, k].max()),
    }
    r = ranges[name]
    print(
        f"  {name}: 全样本 [{r['full_min']:.2f}, {r['full_max']:.2f}]"
        f"  前沿 [{r['pf_min']:.2f}, {r['pf_max']:.2f}]"
    )

# ---------------------------------------------------------------------
# 4. 成对二维前沿：冲突性与 trade-off 幅度
# ---------------------------------------------------------------------
pairs = [(0, 1, "f1-f2"), (0, 2, "f1-f3"), (1, 2, "f2-f3")]
pair_stats = {}
for i, j, label in pairs:
    F2 = F[:, [i, j]]
    m2 = pareto_mask_2d(F2)
    n_pf2 = int(m2.sum())
    a_rng = float(F2[m2, 0].max() - F2[m2, 0].min())
    b_rng = float(F2[m2, 1].max() - F2[m2, 1].min())
    a_full = float(F2[:, 0].max() - F2[:, 0].min())
    b_full = float(F2[:, 1].max() - F2[:, 1].min())
    rho = float(spearman.iloc[i, j])
    pair_stats[label] = {
        "front_size": n_pf2,
        "a_range_on_front": a_rng,
        "b_range_on_front": b_rng,
        "a_range_ratio": a_rng / a_full,
        "b_range_ratio": b_rng / b_full,
        "spearman": rho,
    }
    print(
        f"\n[{label}] Spearman ρ = {rho:+.4f} | 2D前沿点数 = {n_pf2}"
        f"\n  前沿上目标A跨度 {a_rng:.2f}（占全程 {a_rng / a_full * 100:.1f}%），"
        f"目标B跨度 {b_rng:.2f}（占全程 {b_rng / b_full * 100:.1f}%）"
    )

# ---------------------------------------------------------------------
# 5. 参考点诊断：单目标最优解在其他目标上的表现（冲突的直接证据）
# ---------------------------------------------------------------------
diagnostics = {}
single_opt = {}
for k, name in enumerate(["f1", "f2", "f3"]):
    i_star = int(F[:, k].argmin())
    single_opt[name] = {
        "x": float(pts[i_star, 0]),
        "y": float(pts[i_star, 1]),
        "f1": float(F[i_star, 0]),
        "f2": float(F[i_star, 1]),
        "f3": float(F[i_star, 2]),
    }
diag_txt = []
for name, d in single_opt.items():
    others = {k: v for k, v in d.items() if k not in ("x", "y", name)}
    worst = max(
        ranges[k]["full_max"] for k in ["f1", "f2", "f3"]
    )
    line = f"argmin {name} @ ({d['x']:.0f},{d['y']:.0f}) -> f1={d['f1']:.1f}, f2={d['f2']:.3f}, f3={d['f3']:.1f}"
    print(line)
    diag_txt.append(line)

# 前沿上目标间的边际替代率（近似）：f1 每变化 100m，f3 变化多少
o = np.argsort(F_pf[:, 0])
F_sorted = F_pf[o]
if len(F_sorted) > 10:
    q = np.linspace(0, 1, 6).round(3)
    idxq = (q * (len(F_sorted) - 1)).astype(int)
    quant = F_sorted[idxq]
    print("\n前沿按 f1 五分位的 (f1, f2, f3) 分布：")
    for row in quant:
        print(f"  f1={row[0]:7.1f}  f2={row[1]:.4f}  f3={row[2]:7.1f}")

# ---------------------------------------------------------------------
# 6. 保存统计结果
# ---------------------------------------------------------------------
stats = {
    "n_samples": N_SAMPLES,
    "seed": SEED,
    "n_dorm": N_DORM,
    "n_canteen": len(EXISTING_CANTEENS),
    "service_radius": SERVICE_RADIUS,
    "cost_base": COST_BASE,
    "cost_hotspots": [list(h) for h in COST_HOTSPOTS],
    "spearman": {label: float(spearman.iloc[i, j]) for i, j, label in pairs},
    "pearson": {label: float(pearson.iloc[i, j]) for i, j, label in pairs},
    "pareto_front_size": int(pf.sum()),
    "dominated_fraction": float(1 - pf.mean()),
    "objective_ranges": ranges,
    "pair_stats": pair_stats,
    "single_objective_optima": single_opt,
    "front_quantiles": (
        [[float(v) for v in row] for row in quant] if len(F_sorted) > 10 else None
    ),
}
with open(os.path.join(OUT_DIR, "validation_stats.json"), "w", encoding="utf-8") as fh:
    json.dump(stats, fh, ensure_ascii=False, indent=2)
print(f"\n统计结果已写入 {os.path.join(OUT_DIR, 'validation_stats.json')}")

# 供绘图脚本复用
np.save(os.path.join(OUT_DIR, "samples_pts.npy"), pts)
np.save(os.path.join(OUT_DIR, "samples_F.npy"), F)
np.save(os.path.join(OUT_DIR, "samples_pf_mask.npy"), pf)
print("采样数据已保存（samples_*.npy），供 make_figures.py 使用。")
