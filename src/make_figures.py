# -*- coding: utf-8 -*-
"""
阶段一图表生成：5 张验证图。
  1. campus_map.png    校园地图：成本场 + 宿舍 + 已有食堂 + 服务半径
  2. scatter_f1f2.png  f1-f2 散点（灰=全部采样，彩=3D前沿，黑线=2D成对前沿）
  3. scatter_f1f3.png  f1-f3 散点
  4. scatter_f2f3.png  f2-f3 散点
  5. pareto_3d.png     三目标 Pareto 分布 3D 图（含底面投影）

运行：python make_figures.py
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from campus_model import (  # noqa: E402
    CAMPUS_X,
    CAMPUS_Y,
    DORMITORIES,
    EXISTING_CANTEENS,
    SERVICE_RADIUS,
    evaluate,
)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "output")

pts = np.load(os.path.join(OUT, "samples_pts.npy"))
F = np.load(os.path.join(OUT, "samples_F.npy"))
pf = np.load(os.path.join(OUT, "samples_pf_mask.npy"))
F_pf = F[pf]

import json  # noqa: E402

with open(os.path.join(OUT, "validation_stats.json"), encoding="utf-8") as fh:
    STATS = json.load(fh)


def pareto2_mask(F2):
    a, b = F2[:, 0], F2[:, 1]
    order = np.lexsort((b, a))
    keep = np.zeros(len(F2), dtype=bool)
    best = np.inf
    for i in order:
        if b[i] < best:
            keep[i] = True
            best = b[i]
    return keep


# =====================================================================
# 图 1：校园地图
# =====================================================================
fig, ax = plt.subplots(figsize=(11, 7.5))
gx, gy = np.meshgrid(
    np.linspace(CAMPUS_X[0], CAMPUS_X[1], 400),
    np.linspace(CAMPUS_Y[0], CAMPUS_Y[1], 280),
)
G = evaluate(np.column_stack([gx.ravel(), gy.ravel()])).reshape(gx.shape + (3,))[:, :, 2]

# 成本场填充：绿=便宜，红=昂贵
cf = ax.contourf(gx, gy, G, levels=[80, 120, 160, 200, 240, 280, 320], cmap="RdYlGn_r", alpha=0.85)
cbar = fig.colorbar(cf, ax=ax, shrink=0.9)
cbar.set_label("土地成本 f3（万元）")

# 等值线：120 = 低/中成本分界，240 = 中/高成本分界
ax.contour(gx, gy, G, levels=[120, 240], colors="k", linewidths=[0.8, 1.4], linestyles=["--", "-"])
ax.clabel(ax.contour(gx, gy, G, levels=[120, 240], colors="k", linewidths=0.8, linestyles="--"), fmt="%d")

# 已有食堂服务半径圈（f2 的直观含义）
for cx, cy in EXISTING_CANTEENS:
    circ = plt.Circle((cx, cy), SERVICE_RADIUS, fill=False, color="darkred", linestyle=":", linewidth=1.2, alpha=0.8)
    ax.add_patch(circ)

# 宿舍
ax.scatter(DORMITORIES[:, 0], DORMITORIES[:, 1], marker="s", s=70, c="royalblue", edgecolors="k", linewidths=0.6, zorder=5, label="宿舍（16 栋，等权需求点）")
# 已有食堂
ax.scatter(EXISTING_CANTEENS[:, 0], EXISTING_CANTEENS[:, 1], marker="^", s=160, c="red", edgecolors="k", linewidths=0.8, zorder=6, label="已有食堂（3 个）")
for k, (cx, cy) in enumerate(EXISTING_CANTEENS):
    ax.annotate(f"C{k + 1}", (cx, cy), textcoords="offset points", xytext=(8, 8), fontsize=12, fontweight="bold", color="darkred")

# 生活区标注
ax.annotate("西生活区", (220, 315), textcoords="offset points", xytext=(-72, 60), fontsize=12, color="navy",
            arrowprops=dict(arrowstyle="->", color="navy", lw=1))
ax.annotate("东生活区", (772, 328), textcoords="offset points", xytext=(24, 55), fontsize=12, color="navy",
            arrowprops=dict(arrowstyle="->", color="navy", lw=1))
ax.annotate("北生活区", (495, 595), textcoords="offset points", xytext=(55, 22), fontsize=12, color="navy",
            arrowprops=dict(arrowstyle="->", color="navy", lw=1))
ax.annotate("教学行政核心区\n（高成本热点）", (500, 350), fontsize=11, color="darkgreen", ha="center", fontweight="bold")
ax.annotate("滨水景观区\n（高成本热点）", (880, 110), fontsize=10, color="darkgreen", ha="center")

ax.set_xlim(CAMPUS_X)
ax.set_ylim(CAMPUS_Y)
ax.set_xlabel("x（米）")
ax.set_ylabel("y（米）")
ax.set_title("模拟校园地图：宿舍分布、已有食堂及其服务场（虚线圈, σ=150m）、土地成本场", fontsize=13)
ax.legend(loc="lower left", framealpha=0.9)
ax.set_aspect("equal")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "campus_map.png"), dpi=150)
plt.close(fig)
print("campus_map.png 完成")

# =====================================================================
# 图 2-4：成对目标散点图
# =====================================================================
PAIR_CFG = [
    (0, 1, "scatter_f1f2.png", "f1 平均距离（米）", "f2 竞争重叠度",
     "f1-f2：平均距离 vs 竞争重叠度", "f3"),
    (0, 2, "scatter_f1f3.png", "f1 平均距离（米）", "f3 土地成本（万元）",
     "f1-f3：平均距离 vs 土地成本", "f2"),
    (1, 2, "scatter_f2f3.png", "f2 竞争重叠度", "f3 土地成本（万元）",
     "f2-f3：竞争重叠度 vs 土地成本", "f1"),
]

for i, j, fname, xl, yl, title, third in PAIR_CFG:
    k_third = 3 - i - j  # 三个目标索引 {0,1,2} 中剩下的那个
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(F[:, i], F[:, j], s=4, c="lightgray", alpha=0.45, edgecolors="none", label=f"全部随机采样（n={len(F)}）")
    sc = ax.scatter(F_pf[:, i], F_pf[:, j], s=10, c=F_pf[:, k_third], cmap="viridis", alpha=0.9, edgecolors="none", label=f"三目标Pareto前沿（n={pf.sum()}）")
    cbar = fig.colorbar(sc, ax=ax, shrink=0.85)
    cbar.set_label(f"f{k_third + 1} 的取值（前沿点着色）")

    # 2D 成对前沿
    m2 = pareto2_mask(F[:, [i, j]])
    F2f = F[m2][:, [i, j]]
    o = np.argsort(F2f[:, 0])
    ax.plot(F2f[o, 0], F2f[o, 1], "k-", linewidth=1.6, label=f"二维成对前沿（n={m2.sum()}）")
    ax.scatter(F2f[:, 0], F2f[:, 1], s=26, c="k", zorder=6)

    rho = STATS["spearman"][f"f{i + 1}-f{j + 1}"]
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    ax.set_title(f"{title}   (Spearman ρ = {rho:+.3f})", fontsize=13)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname), dpi=150)
    plt.close(fig)
    print(f"{fname} 完成")

# =====================================================================
# 图 5：三目标 Pareto 分布 3D 图
# =====================================================================
from mpl_toolkits.mplot3d import Axes3D  # noqa: E402,F401

fig = plt.figure(figsize=(10.5, 8.5))
ax3 = fig.add_subplot(111, projection="3d")
p = ax3.scatter(F_pf[:, 0], F_pf[:, 2], F_pf[:, 1], c=F_pf[:, 0], cmap="coolwarm", s=10, alpha=0.75, depthshade=True)
ax3.scatter(F[:, 0], F[:, 2], F[:, 1], c="lightgray", s=3, alpha=0.15, depthshade=False)

ax3.set_xlabel("f1 平均距离（米）")
ax3.set_zlabel("f2 竞争重叠度")
ax3.set_ylabel("f3 土地成本（万元）")
ax3.set_title(f"三目标 Pareto 前沿分布（n={pf.sum()}，按 f1 着色；灰点为全部采样）", fontsize=13)
ax3.view_init(elev=22, azim=-58)
fig.colorbar(p, ax=ax3, shrink=0.6, label="f1（米）")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "pareto_3d.png"), dpi=150)
plt.close(fig)
print("pareto_3d.png 完成")

print(f"\n全部图表输出至 {OUT}")
