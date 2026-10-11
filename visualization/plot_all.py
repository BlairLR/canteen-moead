# -*- coding: utf-8 -*-
"""南湖 21 点统一可视化入口。

约束：
  1. 5~10 秒完成，否则宁可减少图表数
  2. 严禁自动重新运行优化算法
  3. 缺少数据时跳过并报告，不阻塞
  4. 不重新计算 HV/IGD，不蒙特卡洛，不网格搜索
  5. 校园地图优先使用现有 PNG，不联网
  6. 静态 2D 图，无动画/交互 3D
  7. 读取数据后尽量复用，不重复加载
"""

import json
import os
import random
import sys
import time
from math import pi

import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PROJ, "output")
sys.path.insert(0, PROJ)

from src.nanhu_campus import wgs2local

# ── 常量 ──
SEED = 42
MAX_SCATTER_PTS = 500
RNG = random.Random(SEED)
AX_LABELS = {0: "f1 Distance (m)", 1: "f2 Overlap", 2: "f3 Cost (¥)"}
RESULTS_DIR = os.path.join(OUT, "moead_nanhu")


def _downsample(arr, k=MAX_SCATTER_PTS):
    n = len(arr)
    return arr if n <= k else arr[sorted(RNG.sample(range(n), k))]


def _try_plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception:
        return None


def _seed_paths(rd, seed=SEED):
    """返回指定种子的 (archive_X_path, archive_F_path) 或 (None, None)。"""
    xp = os.path.join(rd, f"archive_seed{seed}_X.npy")
    fp = os.path.join(rd, f"archive_seed{seed}_F.npy")
    return (xp, fp) if os.path.isfile(xp) and os.path.isfile(fp) else (None, None)


# ── 图 1: HV 收敛 ──
def fig_hv_convergence(rd, save_path):
    fp = os.path.join(rd, "experiment_results.json")
    if not os.path.isfile(fp):
        return "跳过：缺少 experiment_results.json"
    with open(fp, "r", encoding="utf-8") as f:
        runs = json.load(f).get("runs", [])
    trajs = [(r["seed"], np.array(r["hv_trajectory"]))
             for r in runs if r.get("hv_trajectory")]
    if not trajs:
        return "跳过：无 HV 轨迹"
    plt = _try_plt()
    if plt is None:
        return "跳过：matplotlib 导入失败"
    _, ax = plt.subplots(figsize=(8, 4.5))
    for seed, arr in trajs:
        ax.plot(arr[:, 0], arr[:, 1], marker=".", label=f"Seed {seed}")
    ax.set_xlabel("Generation"); ax.set_ylabel("HV")
    ax.set_title("HV Convergence (Nanhu 1x200)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()


# ── 图 2: Pareto 2D ──
def fig_pareto_2d(rd, dim_pair, save_path):
    xp, fp = _seed_paths(rd)
    if not xp:
        return "跳过：缺少 archive 档案"
    F = np.load(fp, mmap_mode="r")
    plt = _try_plt()
    if plt is None:
        return "跳过：matplotlib 导入失败"
    _, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(_downsample(F[:, dim_pair[0]]), _downsample(F[:, dim_pair[1]]),
               s=4, alpha=0.4)
    ax.set_xlabel(AX_LABELS[dim_pair[0]]); ax.set_ylabel(AX_LABELS[dim_pair[1]])
    ax.set_title(f"Pareto Front (seed {SEED})")
    ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()


# ── 图 3: A3 边界 + 21 点 ──
def fig_a3_boundary_points(save_path):
    poly_fp = os.path.join(PROJ, "data", "nanhu_boundary_polygon.json")
    csv_fp = os.path.join(PROJ, "data", "nanhu_points.csv")
    if not os.path.isfile(poly_fp) or not os.path.isfile(csv_fp):
        return "跳过：缺少边界/CSV 数据"
    import csv
    with open(poly_fp, "r", encoding="utf-8") as f:
        vertices_wgs84 = json.load(f).get("vertices_wgs84", [])
    if not vertices_wgs84:
        return "跳过：JSON 缺 vertices_wgs84"
    poly_local = np.array([wgs2local(float(lat), float(lon)) for lat, lon in vertices_wgs84])
    pts_local, pt_types = [], []
    with open(csv_fp, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            xy = wgs2local(float(row["latitude"]), float(row["longitude"]))
            pts_local.append(xy); pt_types.append(row["type"])
    pts_local = np.array(pts_local)
    plt = _try_plt()
    if plt is None:
        return "跳过：matplotlib 导入失败"
    _, ax = plt.subplots(figsize=(7, 7))
    pc = np.vstack([poly_local, poly_local[0]])
    ax.plot(pc[:, 0], pc[:, 1], "k-", lw=1.5, label="A3 Polygon")
    ax.fill(pc[:, 0], pc[:, 1], alpha=0.05, color="gray")
    colors = {"competition": "#e74c3c", "dorm": "#3498db", "teaching": "#2ecc71"}
    markers = {"competition": "s", "dorm": "o", "teaching": "^"}
    for typ in set(pt_types):
        mask = [t == typ for t in pt_types]
        ax.scatter(pts_local[mask, 0], pts_local[mask, 1],
                   c=colors[typ], marker=markers[typ], s=50, edgecolors="k",
                   linewidths=0.5, label=typ, zorder=5)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title("A3 Polygon & 21 Points")
    ax.set_aspect((poly_local[:, 0].max()-poly_local[:, 0].min()) /
                  (poly_local[:, 1].max()-poly_local[:, 1].min()))
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()


# ── 图 4: 决策空间 Pareto ──
def fig_decision_space(rd, save_path):
    """决策变量 (x,y) 空间的 Pareto 解集 + A3 边界 + 21 点"""
    xp, _ = _seed_paths(rd)
    if not xp:
        return "跳过：缺少 archive_X.npy"
    X = np.load(xp, mmap_mode="r")
    plt = _try_plt()
    if plt is None:
        return "跳过：matplotlib 导入失败"
    _, ax = plt.subplots(figsize=(7, 6))
    # A3 polygon
    poly_fp = os.path.join(PROJ, "data", "nanhu_boundary_polygon.json")
    if os.path.isfile(poly_fp):
        with open(poly_fp, "r", encoding="utf-8") as f:
            vtx = json.load(f).get("vertices_wgs84", [])
        if vtx:
            pl = np.array([wgs2local(float(lat), float(lon)) for lat, lon in vtx])
            pc = np.vstack([pl, pl[0]])
            ax.plot(pc[:, 0], pc[:, 1], "k-", lw=1, alpha=0.4)
            ax.fill(pc[:, 0], pc[:, 1], alpha=0.03, color="gray")
    # Pareto 解
    ax.scatter(_downsample(X)[:, 0], _downsample(X)[:, 1],
               s=3, alpha=0.3)
    # 21 点
    csv_fp = os.path.join(PROJ, "data", "nanhu_points.csv")
    if os.path.isfile(csv_fp):
        import csv
        pts, typs = [], []
        with open(csv_fp, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                xy = wgs2local(float(row["latitude"]), float(row["longitude"]))
                pts.append(xy); typs.append(row["type"])
        if pts:
            pts = np.array(pts)
            colors = {"competition": "#e74c3c", "dorm": "#3498db", "teaching": "#2ecc71"}
            markers = {"competition": "s", "dorm": "o", "teaching": "^"}
            for typ in set(typs):
                mask = [t == typ for t in typs]
                ax.scatter(pts[mask, 0], pts[mask, 1],
                           c=colors[typ], marker=markers[typ], s=40, edgecolors="k",
                           linewidths=0.5, label=f"Pt-{typ}", zorder=5)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title("Pareto Decision Space")
    ax.legend(fontsize=6, markerscale=2); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()


# ── 图 5: 3D Pareto ──
def fig_pareto_3d(rd, save_path, max_pts=500):
    xp, fp = _seed_paths(rd)
    if not xp:
        return "跳过：缺少 archive 档案"
    F = np.load(fp, mmap_mode="r")
    plt = _try_plt()
    if plt is None:
        return "跳过：matplotlib 导入失败"
    import mpl_toolkits.mplot3d  # noqa
    _, ax = plt.subplots(figsize=(8, 6), subplot_kw={"projection": "3d"})
    arr = _downsample(F, max_pts)
    ax.scatter(arr[:, 0], arr[:, 1], arr[:, 2], s=3, alpha=0.3)
    ax.set_xlabel(AX_LABELS[0]); ax.set_ylabel(AX_LABELS[1]); ax.set_zlabel(AX_LABELS[2])
    ax.set_title("Pareto Front 3D")
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()


# ── 图 6: 新食堂候选方案 ──
def fig_canteen_schemes(rd, save_path):
    """从 Pareto 档案选取 4 个候选，位置图 + 目标值雷达图"""
    if not os.path.isdir(rd):
        return "跳过：无实验数据"
    xp, fp = _seed_paths(rd)
    if not xp:
        return "跳过：缺少 archive 档案"
    X, F = np.load(xp, mmap_mode="r"), np.load(fp, mmap_mode="r")
    idxs = [np.argmin(F[:, 0]), np.argmin(F[:, 1]), np.argmin(F[:, 2]),
            np.argmin(F[:, 0]/F[:, 0].max() + F[:, 1]/F[:, 1].max() + F[:, 2]/F[:, 2].max())]
    names = ["Min f1 (prox.)", "Min f2 (overlap)", "Min f3 (cost)", "Compromise"]
    cand_xy = {n: X[i] for n, i in zip(names, idxs)}
    cand_f = {n: F[i] for n, i in zip(names, idxs)}
    scheme_colors = ["#e67e22", "#9b59b6", "#1abc9c", "#34495e"]
    scheme_markers = ["D", "v", "s", "o"]
    import csv
    poly_local, pts_local, pt_types = None, [], []
    pf = os.path.join(PROJ, "data", "nanhu_boundary_polygon.json")
    cf = os.path.join(PROJ, "data", "nanhu_points.csv")
    if os.path.isfile(pf):
        with open(pf, "r", encoding="utf-8") as f:
            vtx = json.load(f).get("vertices_wgs84", [])
            if vtx:
                poly_local = np.array([wgs2local(float(lat), float(lon)) for lat, lon in vtx])
    if os.path.isfile(cf):
        with open(cf, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                xy = wgs2local(float(row["latitude"]), float(row["longitude"]))
                pts_local.append(xy); pt_types.append(row["type"])
        pts_local = np.array(pts_local) if pts_local else np.empty((0, 2))
    plt = _try_plt()
    if plt is None:
        return "跳过：matplotlib 导入失败"
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5.5),
                                    gridspec_kw={"width_ratios": [1, 1.1]})
    # ─ 左图：候选位置 ─
    if poly_local is not None:
        pc = np.vstack([poly_local, poly_local[0]])
        ax0.plot(pc[:, 0], pc[:, 1], "k-", lw=1, alpha=0.3)
        ax0.fill(pc[:, 0], pc[:, 1], alpha=0.03, color="gray")
    colors_pt = {"competition": "#e74c3c", "dorm": "#3498db", "teaching": "#2ecc71"}
    for typ in set(pt_types):
        mask = [t == typ for t in pt_types]
        ax0.scatter(pts_local[mask, 0], pts_local[mask, 1],
                    c=colors_pt[typ], s=30, alpha=0.6, label=typ)
    for i, (nm, xy) in enumerate(cand_xy.items()):
        ax0.scatter(xy[0], xy[1], c=scheme_colors[i], marker=scheme_markers[i],
                    s=120, edgecolors="k", linewidths=1, zorder=6, label=nm)
    ax0.set_xlabel("x (m)"); ax0.set_ylabel("y (m)")
    ax0.set_title("Candidate Canteen Locations")
    ax0.legend(fontsize=7); ax0.grid(True, alpha=0.3)
    # ─ 右图：雷达 ─
    categories = [AX_LABELS[0], AX_LABELS[1], AX_LABELS[2]]
    angles = [n / 3.0 * 2 * pi for n in range(3)] + [0]
    ax1 = fig.add_subplot(1, 2, 2, polar=True)
    for i, (nm, fv) in enumerate(cand_f.items()):
        vals = [fv[0], fv[1], fv[2]]
        mx = [max(v[0] for v in cand_f.values()), max(v[1] for v in cand_f.values()),
              max(v[2] for v in cand_f.values())]
        normed = [vals[k] / max(mx[k], 1e-12) for k in range(3)] + [vals[0] / max(mx[0], 1e-12)]
        ax1.plot(angles, normed, "o-", lw=1.5, label=nm, color=scheme_colors[i])
        ax1.fill(angles, normed, alpha=0.05, color=scheme_colors[i])
    ax1.set_xticks(angles[:-1]); ax1.set_xticklabels(categories, fontsize=8)
    ax1.set_ylim(0, 1.1)
    ax1.set_title("Objective Comparison (norm.)", fontsize=10)
    ax1.legend(fontsize=7, loc="upper right", bbox_to_anchor=(1.3, 1.0))
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()


# ── 图 7: 地价空间分布 ──
def fig_land_cost_map(rd, save_path):
    """A3 多边形内部的合成成本空间分布 + 校园点位 + Pareto 候选位置"""
    poly_fp = os.path.join(PROJ, "data", "nanhu_boundary_polygon.json")
    csv_fp = os.path.join(PROJ, "data", "nanhu_points.csv")
    if not os.path.isfile(poly_fp) or not os.path.isfile(csv_fp):
        return "跳过：缺少边界/CSV 数据"

    import csv
    from matplotlib.path import Path

    # --- 加载边界 ---
    with open(poly_fp, "r", encoding="utf-8") as f:
        vertices_wgs84 = json.load(f).get("vertices_wgs84", [])
    if not vertices_wgs84:
        return "跳过：JSON 缺 vertices_wgs84"
    poly_local = np.array([wgs2local(float(lat), float(lon)) for lat, lon in vertices_wgs84])

    # --- 加载点位 ---
    pts_local, pt_types = [], []
    with open(csv_fp, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            xy = wgs2local(float(row["latitude"]), float(row["longitude"]))
            pts_local.append(xy); pt_types.append(row["type"])
    pts_local = np.array(pts_local)

    # --- 计算成本（无热点时的常量基准） ---
    COST_BASE = 80.0
    bx_lo, bx_hi = poly_local[:, 0].min(), poly_local[:, 0].max()
    by_lo, by_hi = poly_local[:, 1].min(), poly_local[:, 1].max()
    nx, ny = 120, 120
    xs = np.linspace(bx_lo, bx_hi, nx)
    ys = np.linspace(by_lo, by_hi, ny)
    Xg, Yg = np.meshgrid(xs, ys)
    grid_pts = np.column_stack([Xg.ravel(), Yg.ravel()])

    # --- 多边形掩码 ---
    poly_path = Path(poly_local)
    inside = poly_path.contains_points(grid_pts).reshape(ny, nx)

    # --- 计算成本 ---
    cost = np.full(grid_pts.shape[0], COST_BASE)
    cost_grid = cost.reshape(ny, nx)
    cost_grid[~inside] = np.nan   # 多边形外部留白

    # --- 加载 Pareto 候选解 ---
    xp, _ = _seed_paths(rd)
    pareto_X = np.load(xp, mmap_mode="r") if xp else None

    # --- 绘图 ---
    plt = _try_plt()
    if plt is None:
        return "跳过：matplotlib 导入失败"
    _, ax = plt.subplots(figsize=(8, 7))

    # 成本色阶
    cmin, cmax = float(np.nanmin(cost_grid)), float(np.nanmax(cost_grid))
    if cmax - cmin < 1e-6:
        # 均匀成本场（无控制点）→ 单色填充并标注基准值
        ax.imshow(np.where(inside, cost_grid, np.nan),
                   extent=(xs[0], xs[-1], ys[0], ys[-1]),
                   origin="lower", cmap="YlOrRd", aspect="auto")
        ax.text(0.5, 0.02, f"Uniform cost = {COST_BASE:.0f} a.u. (no control points)",
                transform=ax.transAxes, ha="center", fontsize=11,
                color="#555", style="italic")
    else:
        levels = np.linspace(cmin, cmax, 25)
        cf = ax.contourf(Xg, Yg, cost_grid, levels=levels, cmap="YlOrRd", extend="both")
        cbar = plt.colorbar(cf, ax=ax, shrink=0.8)
        cbar.set_label("Simulated Cost (a.u.)")

    # A3 边界
    pc = np.vstack([poly_local, poly_local[0]])
    ax.plot(pc[:, 0], pc[:, 1], "k-", lw=1.5, label="A3 Boundary")

    # 校园点位
    colors_pt = {"competition": "#e74c3c", "dorm": "#3498db", "teaching": "#2ecc71"}
    markers_pt = {"competition": "s", "dorm": "o", "teaching": "^"}
    point_labels = {"competition": "Existing canteen", "dorm": "Dorm", "teaching": "Teaching"}
    for typ in sorted(set(pt_types)):
        mask = [t == typ for t in pt_types]
        ax.scatter(pts_local[mask, 0], pts_local[mask, 1],
                   c=colors_pt[typ], marker=markers_pt[typ], s=50,
                   edgecolors="k", linewidths=0.5, label=point_labels[typ], zorder=5)

    # Pareto 候选解
    if pareto_X is not None:
        ax.scatter(pareto_X[:, 0], pareto_X[:, 1], s=3, alpha=0.2,
                   color="white", marker="x", label=f"Pareto candidate (seed={SEED})")

    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title("Nanhu Campus — Simulated Land Cost Map")
    ax.set_aspect((bx_hi - bx_lo) / (by_hi - by_lo))
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.2)
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()


# ── 主入口 ──
def main():
    t0 = time.perf_counter()
    rd = os.path.join(OUT, "moead_nanhu")
    charts, skipped = [], []

    def go(fig_fn, save_path, label, *args):
        err = fig_fn(*args, save_path) if args else fig_fn(save_path)
        if err is None:
            charts.append(save_path)
        else:
            skipped.append((label, err))

    go(fig_hv_convergence, os.path.join(OUT, "moead_nanhu", "hv_convergence.png"),
       "HV Convergence", rd)
    go(fig_pareto_2d, os.path.join(OUT, "pareto_f1f2_nanhu.png"), "Pareto f1-f2", rd, (0, 1))
    go(fig_a3_boundary_points, os.path.join(OUT, "a3_boundary_points.png"), "A3 Boundary")
    go(fig_decision_space, os.path.join(OUT, "decision_space.png"), "Decision Space", rd)
    go(fig_pareto_3d, os.path.join(OUT, "pareto_3d_nanhu.png"), "Pareto 3D", rd)
    go(fig_canteen_schemes, os.path.join(OUT, "canteen_schemes.png"), "Canteen Schemes", rd)
    go(fig_land_cost_map, os.path.join(OUT, "land_cost_map.png"), "Land Cost Map", rd)
    elapsed = time.perf_counter() - t0
    print("=" * 55)
    print("▌ 绘图报告")
    print("=" * 55)
    print(f" 总耗时:         {elapsed:.2f} s")
    print(f" 成功生成:       {len(charts)} 张")
    for c in charts:
        print(f"   [OK] {os.path.relpath(c, PROJ)}")
    print(f" 跳过的图表:     {len(skipped)} 张")
    for name, reason in skipped:
        print(f"   ⏭️  {name}: {reason}")
    print(f" 是否重新运行 MOEA/D:  否")
    print("=" * 55)


if __name__ == "__main__":
    main()