# -*- coding: utf-8 -*-
"""标量化函数（子问题定义）。可切换：Tchebycheff（第一版默认）/ PBI（预留）。

约定：输入 F_norm 为**归一化后**的目标值 (k, m)，且 ≥ 0；
权重 λ 位于单纯形上（和为 1，分量 ≥ 0），可含零分量（极端子问题退化为
单目标优化，属正常情况）。
"""

import numpy as np


def tchebycheff(F_norm: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Tchebycheff 标量化：g(x|λ, z*) = max_i λ_i · |f_i - z*_i|。

    归一化后 z* 对应 0、f' ≥ 0，故 |f' - 0| = f'。
    输入 F_norm (k, m)、W (nw, m)，返回 G (k, nw)：
    G[a, b] = 子代 a 在权重 b 下的标量化值。
    """
    F_norm = np.atleast_2d(F_norm)
    W = np.atleast_2d(W)
    return np.max(F_norm[:, None, :] * W[None, :, :], axis=2)


def pbi(F_norm: np.ndarray, W: np.ndarray, theta: float = 5.0) -> np.ndarray:
    """PBI（Penalty Boundary Intersection）：g = d1 + θ·d2。

    d1 = (f'·λ̂)：解在权重方向上的投影长度；
    d2 = ‖f' - d1·λ̂‖：解到权重方向法平面的垂直距离。
    返回 (k, nw)。
    """
    F_norm = np.atleast_2d(F_norm)
    W = np.atleast_2d(W)
    norms = np.linalg.norm(W, axis=1)                    # (nw,)
    norms = np.where(norms < 1e-12, 1.0, norms)          # 数值保护
    L = W / norms[None, :]                               # 单位权重 (nw, m)
    d1 = F_norm @ L.T                                    # (k, nw)
    proj = d1[:, :, None] * L[None, :, :]                # (k, nw, m)
    d2 = np.linalg.norm(F_norm[:, None, :] - proj, axis=2)  # (k, nw)
    return d1 + theta * d2


def scalarize(name: str, F_norm: np.ndarray, W: np.ndarray, theta: float = 5.0) -> np.ndarray:
    """统一入口：按名字切换标量化函数。"""
    if name == "tch":
        return tchebycheff(F_norm, W)
    if name == "pbi":
        return pbi(F_norm, W, theta)
    raise ValueError(f"未知标量化函数: {name}（可选 'tch' / 'pbi'）")
