# -*- coding: utf-8 -*-
"""交互式连续地价场 — 由用户点击地图设置价格控制点，通过高斯径向基函数生成平滑价格场。

数学公式
--------
给定 N 个控制点 (x_i, y_i, p_i)，全局影响半径 σ，
以及基础价格 p_base，在任意位置 (x, y) 的价格为：

    w_i(x,y) = exp(-||(x,y) - (x_i,y_i)||² / (2σ²))
    w_base    = ε = 0.01                # 基础价格全局基准权重（常数）
    price(x,y) = (p_base * w_base + Σ p_i * w_i(x,y))
                 ─────────────────────────────────────
                 (w_base + Σ w_i(x,y))

即：所有控制点和基础价格的**加权平均**，权重随距离指数衰减。
· 紧贴控制点（d≪σ，w≈1）→ 价格近似该控制点价格
· 远离所有控制点（d≫3σ，w≪ε）→ 价格趋于 p_base
· 多个控制点互相影响 → 价格在两个高点之间平滑过渡
· ε=0.01 保证了在 3σ 以外控制点影响可忽略

用法
----
    from src.interactive_pricing import make_interactive_evaluate

    control_points = [{"x": -100.0, "y": 430.0, "price": 240.0}, ...]
    f3_fn = make_interactive_evaluate(control_points, base_price=80.0, sigma=150.0)

    prices = f3_fn(xy_array)    # (k,) 价格 × assumed_floor_area
"""

import json
import os
import sys
from typing import Optional

import numpy as np

# 确保项目根目录在 sys.path 中（独立运行或测试时）
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.nanhu_campus import (
    build_model, build_constrained_evaluate,
    is_inside_polygon, sample_in_polygon,
)

# ---------------------------------------------------------------------------
# 数值常数
# ---------------------------------------------------------------------------
PRICE_MIN = 0.0
PRICE_MAX = 500.0
BASE_WEIGHT = 0.01  # 基础价格全局恒定权重（保证 3σ 外控制点影响可忽略）
ASSUMED_FLOOR_AREA = 1.0  # 占地面积系数


# ---------------------------------------------------------------------------
# 核心价格场计算（向量化）
# ---------------------------------------------------------------------------

def _bounding_radius(bounds_x, bounds_y) -> float:
    """搜索空间外包矩形对角线半长，用于 base 权重的特征尺度。"""
    corners = np.array([
        [bounds_x[0], bounds_y[0]],
        [bounds_x[0], bounds_y[1]],
        [bounds_x[1], bounds_y[0]],
        [bounds_x[1], bounds_y[1]],
    ])
    cx = 0.5 * (bounds_x[0] + bounds_x[1])
    cy = 0.5 * (bounds_y[0] + bounds_y[1])
    return float(np.max(np.hypot(corners[:, 0] - cx, corners[:, 1] - cy)))


def compute_prices(xy: np.ndarray,
                   control_points: list[dict],
                   base_price: float = 80.0,
                   sigma: float = 150.0,
                   bounds_x=(0, 0),
                   bounds_y=(0, 0)) -> np.ndarray:
    """对一组坐标计算连续地价场。

    Parameters
    ----------
    xy : (k, 2) ndarray — 局部米制坐标。
    control_points : list[dict] — 每个含 {"x": float, "y": float, "price": float}。
    base_price : float — 基础价格（BASE）。
    sigma : float — 全局高斯影响半径（米）。
    bounds_x, bounds_y : tuple — 搜索空间外包矩形，用于 base 权重。

    Returns
    -------
    prices : (k,) ndarray — 每个点的模拟价格，范围 [0, PRICE_MAX]。
    """
    xy = np.atleast_2d(np.asarray(xy, dtype=float))
    k = len(xy)

    if not control_points:
        return np.full(k, float(base_price))

    # 提取控制点数组
    cp = np.array([[p["x"], p["y"]] for p in control_points], dtype=float)
    pv = np.array([p["price"] for p in control_points], dtype=float)

    # 计算 base 权重（常数，保证远离控制点时回归 BASE）
    w_base = BASE_WEIGHT

    # 计算每个控制点对各目标点的权重
    # cp: (m, 2), xy: (k, 2) → dx: (m, k)
    dx = cp[:, None, 0] - xy[None, :, 0]  # (m, k)
    dy = cp[:, None, 1] - xy[None, :, 1]
    d2 = dx**2 + dy**2  # (m, k)

    if sigma > 0:
        w = np.exp(-d2 / (2.0 * sigma**2))  # (m, k)
    else:
        # sigma=0: 只取最近控制点的价格（最近邻）
        nearest = np.argmin(d2, axis=0)  # (k,)
        prices = pv[nearest]
        return np.clip(prices, PRICE_MIN, PRICE_MAX)

    # 加权平均
    total_weight = w_base + w.sum(axis=0)  # (k,)
    weighted_sum = base_price * w_base + (pv[:, None] * w).sum(axis=0)  # (k,)
    prices = weighted_sum / total_weight

    return np.clip(prices, PRICE_MIN, PRICE_MAX)


# ---------------------------------------------------------------------------
# f3 闭包生成
# ---------------------------------------------------------------------------

def make_interactive_evaluate(control_points: list[dict],
                              base_price: float = 80.0,
                              sigma: float = 150.0,
                              bounds_x=(0, 0),
                              bounds_y=(0, 0),
                              floor_area: float = ASSUMED_FLOOR_AREA):
    """返回一个闭包 evaluate(X) → (k,) 价格 × floor_area。

    这是 f3 的成本部分，用于传入 MOEA/D 评价函数。
    """
    def evaluate(X: np.ndarray) -> np.ndarray:
        xy = np.atleast_2d(np.asarray(X, dtype=float))
        prices = compute_prices(xy, control_points, base_price, sigma,
                                bounds_x, bounds_y)
        return prices * floor_area

    return evaluate


# ---------------------------------------------------------------------------
# 带 A3 约束的全目标函数
# ---------------------------------------------------------------------------

def build_constrained_interactive_evaluate(model,
                                           control_points: list[dict],
                                           base_price: float = 80.0,
                                           sigma: float = 150.0,
                                           margin: float = 1.1):
    """带 A3 多边形约束的目标函数（f1/f2 同默认，f3 用交互式地价场）。

    界内 → 真实 (f1,f2,f3)；界外 → 有界罚值。
    """
    base_eval = build_constrained_evaluate(model)

    # 构建交互式 f3
    f3_fn = make_interactive_evaluate(
        control_points, base_price, sigma,
        model.bounds_x, model.bounds_y,
    )

    # 用 base 评价 probe 确定罚值上界
    probe = sample_in_polygon(2000, model.polygon_local)
    probe_F = base_eval(probe)
    # 替换 f3 为交互式价格
    probe_F[:, 2] = f3_fn(probe)
    caps = probe_F.max(axis=0) * margin

    # 预构建标准 f1/f2
    base = __import__("src.nanhu_campus", fromlist=[""]).build_evaluate(model)
    demand = model.demand_xy
    comp = model.competition_xy

    def evaluate(X: np.ndarray) -> np.ndarray:
        Xa = np.atleast_2d(np.asarray(X, dtype=float))
        x, y = Xa[:, 0], Xa[:, 1]

        # f1 / f2 同默认
        dx = demand[:, None, 0] - x[None, :]
        dy = demand[:, None, 1] - y[None, :]
        f1 = np.hypot(dx, dy).mean(axis=0)
        cx = comp[:, None, 0] - x[None, :]
        cy = comp[:, None, 1] - y[None, :]
        f2 = np.exp(-(cx**2 + cy**2) / (2.0 * (__import__("src.nanhu_campus",
            fromlist=[""]).SERVICE_RADIUS_M)**2)).mean(axis=0)

        # f3 用交互式地价场
        f3 = f3_fn(Xa)

        F = np.column_stack([f1, f2, f3])

        # A3 约束
        out = ~is_inside_polygon(model.polygon_local, Xa)
        if out.any():
            F[out] = caps
        return F

    return evaluate


# ---------------------------------------------------------------------------
# 网格价格场计算（供热力图/等值线）
# ---------------------------------------------------------------------------

def compute_price_grid(model,
                       control_points: list[dict],
                       base_price: float = 80.0,
                       sigma: float = 150.0,
                       resolution: int = 50):
    """在 A3 外包矩形上计算规则网格的价格场。

    Returns
    -------
    xs : (resolution,) ndarray — 等间距 x 坐标
    ys : (resolution,) ndarray — 等间距 y 坐标
    values : (resolution, resolution) ndarray — 各网格点的价格
    mask_inside : (resolution, resolution) ndarray — True 表示在 A3 内
    """
    xs = np.linspace(model.bounds_x[0], model.bounds_x[1], resolution)
    ys = np.linspace(model.bounds_y[0], model.bounds_y[1], resolution)
    xx, yy = np.meshgrid(xs, ys)
    grid_pts = np.column_stack([xx.ravel(), yy.ravel()])

    prices = compute_prices(grid_pts, control_points, base_price, sigma,
                            model.bounds_x, model.bounds_y)
    values = prices.reshape(xx.shape)

    # A3 多边形掩码
    mask_inside = is_inside_polygon(model.polygon_local, grid_pts).reshape(xx.shape)

    return xs, ys, values, mask_inside


# ---------------------------------------------------------------------------
# 等值线计算
# ---------------------------------------------------------------------------

def compute_contours(xs, ys, values, mask_inside, n_levels: int = 8):
    """用 matplotlib 计算等值线，返回可序列化的线段列表。

    Returns
    -------
    levels : list[float] — 等值线价格层级
    contours : list[list[list[float, float]]] — 每条等值线的顶点列表
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return [], []

    Z = np.ma.array(values, mask=~mask_inside)

    vmin, vmax = values[mask_inside].min(), values[mask_inside].max()
    if vmax - vmin < 1:
        return [], []

    levels = np.linspace(vmin, vmax, n_levels)

    fig, ax = plt.subplots(figsize=(1, 1))
    cs = ax.contour(xs, ys, Z, levels=levels)
    plt.close(fig)

    contour_data = []
    for i, level in enumerate(cs.levels):
        for seg in cs.allsegs[i]:
            if len(seg) >= 2:
                contour_data.append({
                    "level": float(level),
                    "points": [[float(p[0]), float(p[1])] for p in seg],
                })

    return [float(l) for l in levels], contour_data


# ---------------------------------------------------------------------------
# 自测
# ---------------------------------------------------------------------------

def _test():
    """轻量自测：验证价格场的连续性、可复现性和边界行为。"""
    print("=" * 50)
    print("交互式地价场 — 自测")
    print("=" * 50)

    model = build_model()
    bounds_x, bounds_y = model.bounds_x, model.bounds_y

    # 1. 无控制点时返回 BASE
    f3_fn = make_interactive_evaluate([], base_price=80.0, sigma=150.0,
                                       bounds_x=bounds_x, bounds_y=bounds_y)
    p = f3_fn(np.array([[0.0, 500.0]]))
    assert abs(p[0] - 80.0) < 1e-6, f"空控制点: {p[0]}"
    print(f"[PASS] 无控制点 → BASE (80.0) = {p[0]}")

    # 2. 单控制点：紧贴 ≈ 控制点价格，远离趋于 BASE
    pts = np.array([
        [-102.5, 434.2],           # 紧贴 H1 中心
        [-102.48 + 450, 434.16],   # 3σ = 450m 外
        [-102.48 + 900, 434.16],   # 6σ = 900m 外
    ])
    cp = [{"x": -102.48, "y": 434.16, "price": 240.0}]
    f3_fn2 = make_interactive_evaluate(cp, base_price=80.0, sigma=150.0,
                                        bounds_x=bounds_x, bounds_y=bounds_y)
    p2 = f3_fn2(pts)
    print(f"  单控制点衰减: 紧贴={p2[0]:.1f}  3σ处={p2[1]:.1f}  6σ处={p2[2]:.1f}")
    assert abs(p2[0] - 240.0) < 5.0, f"紧贴点: {p2[0]} 应 ≈ 240"
    assert p2[2] < 100.0, f"900m 处: {p2[2]} 应 < 100 (趋于 BASE=80)"
    assert p2[0] > p2[1] > p2[2], f"应单调递减: {p2}"
    print("[PASS] 单控制点单调衰减 OK")

    # 3. 多控制点平滑
    cp2 = [
        {"x": -102.48, "y": 434.16, "price": 300.0},
        {"x": 123.51, "y": 753.61, "price": 100.0},
    ]
    mid = np.array([[(-102.48 + 123.51) / 2, (434.16 + 753.61) / 2]])
    f3_fn3 = make_interactive_evaluate(cp2, base_price=80.0, sigma=200.0,
                                        bounds_x=bounds_x, bounds_y=bounds_y)
    p3 = f3_fn3(mid)
    assert 100.0 < p3[0] < 300.0, f"两控制点中点: {p3[0]} 应介于 100-300"
    print(f"[PASS] 多控制点平滑过渡: 中点价格={p3[0]:.1f}")

    # 4. 网格计算
    xs, ys, values, mask = compute_price_grid(model, cp2, base_price=80.0,
                                               sigma=200.0, resolution=20)
    assert values.shape == (20, 20), f"网格形状: {values.shape}"
    assert mask.sum() > 0, "掩码应有 True 值"
    print(f"[PASS] 网格计算: {values.shape}, A3 内 {mask.sum()}/{20*20} 格点")

    # 5. 等值线计算
    levels, contours = compute_contours(xs, ys, values, mask, n_levels=6)
    assert len(levels) > 0 or len(contours) == 0, f"等值线: {len(levels)} 层级"
    print(f"[PASS] 等值线: {len(levels)} 条层级, {len(contours)} 条线段")

    print("\n全部交互式地价场测试通过 OK")


if __name__ == "__main__":
    _test()