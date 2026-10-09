# -*- coding: utf-8 -*-
"""南湖 A3 稳定性实验可视化（只读取 output/moead_nanhu/ 现有结果，不重算实验）。

生成 output/moead_nanhu/figures/ 下 5 张图：
  fig1_a3_boundary_points.png   A3 边界 + 21 个模型点（+四门锚点）
  fig2_pareto_decision_space.png 10 seeds Pareto 解空间分布（按种子着色）
  fig3_pareto_front_3d.png      三目标 Pareto 前沿（10 seeds 合并，按种子着色）
  fig4_hv_convergence.png       10 seeds HV 收敛曲线（均值±std 带）
  fig5_cost_heatmap_pareto.png  B2 成本场热力图 + Pareto 解（按 f1 着色）
"""

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from src import nanhu_campus as nc  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

RES = os.path.join(ROOT, "output", "moead_nanhu")
FIG = os.path.join(RES, "figures")
os.makedirs(FIG, exist_ok=True)

with open(os.path.join(RES, "experiment_results.json"), encoding="utf-8") as fh:
    RES_J = json.load(fh)
SEEDS = RES_J["runs"] and [r["seed"] for r in RES_J["runs"]]
model = nc.build_model()
rng = np.random.default_rng(0)

COLORS = plt.cm.tab10(np.linspace(0, 1, 10))
seed_color = {s: COLORS[i] for i, s in enumerate(SEEDS)}

# 各种子档案（决策 + 目标）
ARCH = {}
for r in RES_J["runs"]:
    s = r["seed"]
    ARCH[s] = {
        "X": np.load(os.path.join(RES, f"archive_seed{s}_X.npy")),
        "F": np.load(os.path.join(RES, f"archive_seed{s}_F.npy")),
    }

SUB = 1500  # 每种子抽样上限（渲染用）
for s in ARCH:
    if len(ARCH[s]["X"]) > SUB:
        idx = rng.choice(len(ARCH[s]["X"]), SUB, replace=False)
        ARCH[s]["Xs"], ARCH[s]["Fs"] = ARCH[s]["X"][idx], ARCH[s]["F"][idx]
    else:
        ARCH[s]["Xs"], ARCH[s]["Fs"] = ARCH[s]["X"], ARCH[s]["F"]

poly = model.polygon_local


def draw_polygon(ax, **kw):
    pp = np.vstack([poly, poly[:1]])
    ax.plot(pp[:, 0], pp[:, 1], **kw)


# =====================================================================
# 图 1：A3 边界 + 21 个模型点（+ 四门锚点）
# =====================================================================
fig, ax = plt.subplots(figsize=(11, 9))
ax.fill(poly[:, 0], poly[:, 1], color="#e8f5e9", zorder=1)
draw_polygon(ax, color="#1565c0", linewidth=2.2, zorder=4,
             label=f"A3 边界（{len(poly)} 顶点, {nc.shoelace_area(poly)/1e6:.3f} km²）")
comp = {pid: xy for pid, xy in zip(model.competition_ids, model.competition_xy)}
dem = {pid: xy for pid, xy in zip(model.demand_ids, model.demand_xy)}
for pid, (x, y) in comp.items():
    ax.plot(x, y, "^", color="#d32f2f", markersize=13, markeredgecolor="k", zorder=6)
    ax.annotate(f"{pid} {model.points[[p.pid for p in model.points].index(pid)].name}",
                (x, y), textcoords="offset points", xytext=(8, 5), fontsize=11,
                fontweight="bold", color="#b71c1c")
for pid, (x, y) in dem.items():
    ax.plot(x, y, "s", color="#1565c0", markersize=11, markeredgecolor="k", zorder=6)
    ax.annotate(pid, (x, y), textcoords="offset points", xytext=(8, 5),
                fontsize=11, fontweight="bold", color="#0d47a1")
ref = nc.load_reference()
for g in ref["official_gate_anchors"]:
    gx, gy = nc.wgs2local(g["latitude"], g["longitude"])
    ax.plot(gx, gy, "*", color="#f57c00", markersize=14, markeredgecolor="k", zorder=5)
    ax.annotate(g["name"] + "锚", (gx, gy), textcoords="offset points", xytext=(6, -12),
                fontsize=9, color="#e65100")
ax.set_xlabel("局部 x（米, 东向）")
ax.set_ylabel("局部 y（米, 北向）")
ax.set_aspect("equal")
ax.set_title("图1 A3 边界 + 模型点（红▲=食堂竞争点, 蓝■=宿舍需求点, 橙★=四门锚点）")
ax.legend(loc="upper right", fontsize=10)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig1_a3_boundary_points.png"), dpi=140)
plt.close(fig)
print("fig1 完成", flush=True)

# =====================================================================
# 图 2：10 seeds Pareto 解空间分布（决策空间）
# =====================================================================
fig, ax = plt.subplots(figsize=(11, 9))
ax.fill(poly[:, 0], poly[:, 1], color="#f1f8e9", zorder=1)
draw_polygon(ax, color="#455a64", linewidth=1.8, zorder=4)
for s in SEEDS:
    Xs = ARCH[s]["Xs"]
    ax.plot(Xs[:, 0], Xs[:, 1], ".", color=seed_color[s], markersize=2.5,
            alpha=0.55, label=f"seed={s} (n={len(ARCH[s]['X'])})", zorder=5)
for pid, (x, y) in {**comp, **dem}.items():
    ax.plot(x, y, "^" if pid in comp else "s", color="k", markersize=8, zorder=7)
    ax.annotate(pid, (x, y), textcoords="offset points", xytext=(5, 4), fontsize=10, fontweight="bold")
ax.set_xlabel("局部 x（米）")
ax.set_ylabel("局部 y（米）")
ax.set_aspect("equal")
ax.set_title("图2 10 seeds Pareto 解空间分布（每种子抽样≤1500 点渲染）")
ax.legend(loc="upper left", fontsize=8, markerscale=3, ncol=2)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig2_pareto_decision_space.png"), dpi=140)
plt.close(fig)
print("fig2 完成", flush=True)

# =====================================================================
# 图 3：三目标 Pareto 前沿（10 seeds 合并，按种子着色）
# =====================================================================
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

fig = plt.figure(figsize=(12, 9.5))
ax3 = fig.add_subplot(111, projection="3d")
for s in SEEDS:
    F = ARCH[s]["Fs"]
    ax3.scatter(F[:, 0], F[:, 2], F[:, 1], s=3, color=seed_color[s], alpha=0.5,
                label=f"seed={s}")
ax3.set_xlabel("f1 平均距离（米）")
ax3.set_ylabel("f3 成本（万元）")
ax3.set_zlabel("f2 竞争重叠度")
ax3.set_title("图3 三目标 Pareto 前沿（10 seeds 合并，按种子着色）")
ax3.legend(loc="upper left", fontsize=8, markerscale=4)
ax3.view_init(elev=22, azim=-58)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig3_pareto_front_3d.png"), dpi=140)
plt.close(fig)
print("fig3 完成", flush=True)

# =====================================================================
# 图 4：10 seeds HV 收敛曲线（均值 ± std）
# =====================================================================
trajs = [np.array(r["hv_trajectory"]) for r in RES_J["runs"]]
gens = trajs[0][:, 0]
HVs = np.array([t[:, 1] for t in trajs])  # (10, T)
mean, std = HVs.mean(axis=0), HVs.std(axis=0)
base = RES_J["summary"]["hv_baseline_random_sampling"]

fig, ax = plt.subplots(figsize=(10, 6.5))
ax.fill_between(gens, mean - std, mean + std, alpha=0.25, color="#1e88e5", label="±1 std")
ax.plot(gens, mean, "o-", color="#1e88e5", markersize=4, label="10 seeds HV 均值")
for s, t in zip(SEEDS, trajs):
    ax.plot(t[:, 0], t[:, 1], "-", color=seed_color[s], alpha=0.55, linewidth=1)
ax.axhline(base, color="k", linestyle="--", linewidth=1.2, label=f"随机采样基线 HV={base:.4f}")
ax.set_xlabel("代数")
ax.set_ylabel("HV（归一化空间, ref=1.05³）")
ax.set_title(f"图4 10 seeds HV 收敛曲线（均值±std；终值 {mean[-1]:.4f}±{std[-1]:.4f}）")
ax.legend(loc="lower right", fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig4_hv_convergence.png"), dpi=140)
plt.close(fig)
print("fig4 完成", flush=True)

# =====================================================================
# 图 5：B2 成本场热力图 + Pareto 解（按 f1 着色）
# =====================================================================
gx, gy = np.meshgrid(
    np.linspace(model.bounds_x[0] - 40, model.bounds_x[1] + 40, 420),
    np.linspace(model.bounds_y[0] - 40, model.bounds_y[1] + 40, 420),
)
pts_grid = np.column_stack([gx.ravel(), gy.ravel()])
f3_grid = np.full(len(pts_grid), nc.COST_BASE)
for hcx, hcy, A, s in model.cost_hotspots:
    f3_grid += A * np.exp(-(((pts_grid[:, 0] - hcx) ** 2 + (pts_grid[:, 1] - hcy) ** 2)) / (2.0 * s**2))
f3_grid = f3_grid.reshape(gx.shape)

fig, ax = plt.subplots(figsize=(11, 9))
cf = ax.contourf(gx, gy, f3_grid, levels=14, cmap="YlOrRd", alpha=0.9)
cb = fig.colorbar(cf, ax=ax, shrink=0.85)
cb.set_label("B2 provisional 合成成本 f3（万元）——非真实地价")
ax.contour(gx, gy, f3_grid, levels=[120, 240], colors="k", linewidths=0.8,
           linestyles=["--", "-"], alpha=0.5)
draw_polygon(ax, color="#1565c0", linewidth=2, zorder=4)
# 合并全部种子 Pareto 解（抽样），按 f1 着色
allX = np.vstack([ARCH[s]["Xs"] for s in SEEDS])
allF = np.vstack([ARCH[s]["Fs"] for s in SEEDS])
sc = ax.scatter(allX[:, 0], allX[:, 1], c=allF[:, 0], cmap="viridis", s=4,
                alpha=0.75, edgecolors="none", zorder=6)
cb2 = fig.colorbar(sc, ax=ax, shrink=0.85, pad=0.02)
cb2.set_label("Pareto 解 f1（米）")
for pid, (x, y) in {**comp, **dem}.items():
    ax.plot(x, y, "^" if pid in comp else "s", color="k", markersize=8, zorder=7)
    ax.annotate(pid, (x, y), textcoords="offset points", xytext=(5, 4),
                fontsize=10, fontweight="bold", color="white")
ax.set_xlabel("局部 x（米）")
ax.set_ylabel("局部 y（米）")
ax.set_aspect("equal")
ax.set_title("图5 B2 合成成本场热力图 + 10 seeds Pareto 解（黑▲=食堂 黑■=宿舍）")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig5_cost_heatmap_pareto.png"), dpi=140)
plt.close(fig)
print("fig5 完成", flush=True)

print(f"\n全部图输出至 {FIG}")
