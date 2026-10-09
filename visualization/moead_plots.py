# -*- coding: utf-8 -*-
"""可视化模块：MOEA/D 第一版结果图（9 张）。

依赖 experiments/run_experiments.py 的输出（archive_seed*.npy 与
experiment_results.json）以及第一阶段采样数据。
"""

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # visualization/ 的上一级即项目根
sys.path.insert(0, ROOT)

from metrics import normalize_fixed  # noqa: E402
from src.campus_model import (  # noqa: E402
    CAMPUS_X,
    CAMPUS_Y,
    DORMITORIES,
    EXISTING_CANTEENS,
    SERVICE_RADIUS,
    evaluate,
)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

OUT = os.path.join(ROOT, "output", "moead")
STAGE1 = os.path.join(ROOT, "output")


def load_all():
    with open(os.path.join(OUT, "experiment_results.json"), encoding="utf-8") as fh:
        results = json.load(fh)
    runs = {}
    for r in results["runs"]:
        s = r["seed"]
        runs[s] = {
            "X": np.load(os.path.join(OUT, f"archive_seed{s}_X.npy")),
            "F": np.load(os.path.join(OUT, f"archive_seed{s}_F.npy")),
            "row": r,
        }
    stage1_F = np.load(os.path.join(STAGE1, "samples_F.npy"))
    stage1_pf = np.load(os.path.join(STAGE1, "samples_pf_mask.npy"))
    lo, hi = stage1_F.min(axis=0), stage1_F.max(axis=0)
    return results, runs, stage1_F, stage1_F[stage1_pf], (lo, hi)


def campus_base(ax, show_cost=True):
    """校园底图：成本场 + 宿舍 + 已有食堂。"""
    if show_cost:
        gx, gy = np.meshgrid(
            np.linspace(CAMPUS_X[0], CAMPUS_X[1], 300),
            np.linspace(CAMPUS_Y[0], CAMPUS_Y[1], 210),
        )
        G = evaluate(np.column_stack([gx.ravel(), gy.ravel()])).reshape(gx.shape + (3,))[:, :, 2]
        ax.contourf(gx, gy, G, levels=[80, 120, 160, 200, 240, 280, 320],
                    cmap="RdYlGn_r", alpha=0.55)
        ax.contour(gx, gy, G, levels=[120, 240], colors="k",
                   linewidths=[0.6, 1.0], linestyles=["--", "-"], alpha=0.6)
    ax.scatter(DORMITORIES[:, 0], DORMITORIES[:, 1], marker="s", s=55,
               c="royalblue", edgecolors="k", linewidths=0.5, zorder=5, label="宿舍（16 栋）")
    for cx, cy in EXISTING_CANTEENS:
        ax.add_patch(plt.Circle((cx, cy), SERVICE_RADIUS, fill=False,
                                color="darkred", linestyle=":", linewidth=1.0, alpha=0.7))
    ax.scatter(EXISTING_CANTEENS[:, 0], EXISTING_CANTEENS[:, 1], marker="^", s=130,
               c="red", edgecolors="k", linewidths=0.7, zorder=6, label="已有食堂（3 个）")
    ax.set_xlim(CAMPUS_X)
    ax.set_ylim(CAMPUS_Y)
    ax.set_aspect("equal")
    ax.set_xlabel("x（米）")
    ax.set_ylabel("y（米）")


def fig_campus(results, runs, best_seed):
    # 图 1：宿舍
    fig, ax = plt.subplots(figsize=(10, 7))
    campus_base(ax, show_cost=False)
    for k, name in [(0, "西生活区"), (1, "东生活区"), (2, "北生活区")]:
        pass
    ax.annotate("西生活区", (220, 315), xytext=(-60, 50), textcoords="offset points",
                fontsize=11, color="navy", arrowprops=dict(arrowstyle="->", color="navy"))
    ax.annotate("东生活区", (772, 328), xytext=(20, 45), textcoords="offset points",
                fontsize=11, color="navy", arrowprops=dict(arrowstyle="->", color="navy"))
    ax.annotate("北生活区", (495, 595), xytext=(45, 18), textcoords="offset points",
                fontsize=11, color="navy", arrowprops=dict(arrowstyle="->", color="navy"))
    ax.set_title("图1 校园地图 + 16 个宿舍（等权需求点）")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1_campus_dorms.png"), dpi=150)
    plt.close(fig)

    # 图 2：已有食堂 + 服务场
    fig, ax = plt.subplots(figsize=(10, 7))
    campus_base(ax, show_cost=False)
    ax.scatter(DORMITORIES[:, 0], DORMITORIES[:, 1], marker="s", s=25,
               c="lightgray", edgecolors="none", zorder=4)
    ax.set_title("图2 校园地图 + 3 个已有食堂及其服务场（虚线圈 σ=150m）")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_campus_canteens.png"), dpi=150)
    plt.close(fig)
    print("fig1/fig2 完成")

    # 图 3：Pareto 候选位置（最佳运行）
    Ax = runs[best_seed]["X"]
    fig, ax = plt.subplots(figsize=(10.5, 7.2))
    campus_base(ax, show_cost=True)
    ax.scatter(Ax[:, 0], Ax[:, 1], marker="o", s=22, c="limegreen",
               edgecolors="darkgreen", linewidths=0.4, alpha=0.85, zorder=7,
               label=f"Pareto 候选新食堂位置（seed={best_seed}, n={len(Ax)}）")
    ax.set_title(f"图3 校园地图 + Pareto 候选位置（MOEA/D 外部档案）")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_campus_pareto.png"), dpi=150)
    plt.close(fig)
    print("fig3 完成")

    # 图 8：Pareto 解按 f1 渐变
    F = runs[best_seed]["F"]
    order = np.argsort(F[:, 0])
    sc = ax  # 占位
    fig, ax = plt.subplots(figsize=(10.5, 7.2))
    campus_base(ax, show_cost=False)
    p = ax.scatter(Ax[:, 0], Ax[:, 1], marker="o", s=26, c=F[:, 0],
                   cmap="viridis", edgecolors="k", linewidths=0.25, zorder=7)
    cb = fig.colorbar(p, ax=ax, shrink=0.8)
    cb.set_label("f1 平均距离（米）——绿=近 黄=远")
    ax.set_title(f"图8 Pareto 候选位置按 f1 渐变（seed={best_seed}）")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig8_campus_gradient.png"), dpi=150)
    plt.close(fig)
    print("fig8 完成")


def fig_pairs(results, runs, best_seed, stage1_front, bounds):
    Ax, AF = runs[best_seed]["X"], runs[best_seed]["F"]
    lo, hi = bounds
    ref_n = normalize_fixed(stage1_front, lo, hi)

    cfgs = [
        (0, 1, "fig4_pareto_f1f2.png", "f1 平均距离（米）", "f2 竞争重叠度"),
        (0, 2, "fig5_pareto_f1f3.png", "f1 平均距离（米）", "f3 土地成本（万元）"),
        (1, 2, "fig6_pareto_f2f3.png", "f2 竞争重叠度", "f3 土地成本（万元）"),
    ]
    for i, j, fname, xl, yl in cfgs:
        k3 = 3 - i - j
        fig, ax = plt.subplots(figsize=(9, 7))
        ax.scatter(ref_n[:, i], ref_n[:, j], s=4, c="lightgray", alpha=0.5,
                   edgecolors="none", label=f"阶段一随机采样前沿（n={len(ref_n)}）")
        p = ax.scatter(AF[:, i], AF[:, j], s=14, c=AF[:, k3], cmap="viridis",
                       alpha=0.9, edgecolors="none", zorder=5,
                       label=f"MOEA/D Pareto 解集（n={len(AF)}）")
        fig.colorbar(p, ax=ax, shrink=0.85, label=f"f{k3 + 1}（归一化前原始值着色）")
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        # 坐标改用原始值更直观：反归一化刻度
        ticks = np.linspace(0, 1, 6)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{lo[i] + t * (hi[i] - lo[i]):.0f}" if i != 1 else f"{lo[i] + t * (hi[i] - lo[i]):.2f}" for t in ticks])
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{lo[j] + t * (hi[j] - lo[j]):.0f}" if j != 1 else f"{lo[j] + t * (hi[j] - lo[j]):.2f}" for t in ticks])
        ax.set_title(f"{fname[:-4].replace('_', ' ')}（seed={best_seed}，灰=采样参考前沿）")
        ax.legend(loc="upper right")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, fname), dpi=150)
        plt.close(fig)
        print(f"{fname} 完成")


def fig_3d(results, runs, best_seed, stage1_front, bounds):
    AF = runs[best_seed]["F"]
    ref_n = normalize_fixed(stage1_front, *bounds)
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(10.5, 8.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(ref_n[:, 0], ref_n[:, 2], ref_n[:, 1], c="lightgray", s=3,
               alpha=0.2, depthshade=False, label="阶段一随机采样前沿")
    p = ax.scatter(AF[:, 0], AF[:, 2], AF[:, 1], c=AF[:, 0], cmap="coolwarm",
                   s=12, alpha=0.85, depthshade=True)
    ax.set_xlabel("f1 平均距离（米）")
    ax.set_ylabel("f3 土地成本（万元）")
    ax.set_zlabel("f2 竞争重叠度")
    ax.set_title(f"图7 三目标 Pareto 结果 3D 图（MOEA/D, n={len(AF)}，按 f1 着色）")
    ax.view_init(elev=22, azim=-58)
    fig.colorbar(p, ax=ax, shrink=0.6, label="f1（米）")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig7_pareto_3d.png"), dpi=150)
    plt.close(fig)
    print("fig7 完成")


def fig_convergence(results):
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    for r in results["runs"]:
        t = np.array(r["hv_trajectory"])
        ax.plot(t[:, 0], t[:, 1], alpha=0.75, label=f"seed={r['seed']}" if r["seed"] in (1,) else None)
    base = results["summary"]["hv_baseline_random_sampling"]
    ax.axhline(base, color="k", linestyle="--", linewidth=1.2,
               label=f"随机采样基线 HV={base:.4f}")
    ax.set_xlabel("代数")
    ax.set_ylabel("HV（归一化空间, ref=1.05³）")
    ax.set_title("图9 MOEA/D 收敛轨迹：10 次独立运行的 HV 随代数变化")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig9_convergence.png"), dpi=150)
    plt.close(fig)
    print("fig9 完成")


def make_all_figures():
    results, runs, stage1_F, stage1_front, bounds = load_all()
    best_seed = max(results["runs"], key=lambda r: r["hv"])["seed"]
    print(f"图表使用最佳 HV 运行：seed={best_seed}")
    fig_campus(results, runs, best_seed)
    fig_pairs(results, runs, best_seed, stage1_front, bounds)
    fig_3d(results, runs, best_seed, stage1_front, bounds)
    fig_convergence(results)
    print(f"全部图表已输出至 {OUT}")


if __name__ == "__main__":
    make_all_figures()
