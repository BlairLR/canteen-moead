# -*- coding: utf-8 -*-
"""单纯形格点（simplex-lattice）权重向量生成与邻域构造。

三目标、H=12 时，权重 (i/H, j/H, k/H) 且 i+j+k=H，共 C(14,2)=91 个，
即 MOEA/D 的 91 个子问题（种群规模 N=91）。
"""

import itertools

import numpy as np


def simplex_lattice_weights(H: int, n_obj: int = 3) -> np.ndarray:
    """生成单纯形格点权重矩阵 W (N, n_obj)，每行和为 1。

    三目标：枚举 (i, j, H-i-j)，i,j ≥ 0。
    """
    if n_obj != 3:
        raise NotImplementedError("第一版仅实现三目标（n_obj=3）")
    weights = []
    for i in range(H + 1):
        for j in range(H + 1 - i):
            k = H - i - j
            weights.append((i / H, j / H, k / H))
    W = np.array(weights, dtype=float)
    assert np.allclose(W.sum(axis=1), 1.0), "权重向量必须和为 1"
    return W


def neighborhoods(W: np.ndarray, T: int) -> np.ndarray:
    """基于权重向量欧氏距离构造邻域。

    返回 B (N, T)：B[i] 为子问题 i 的 T 个最近邻索引（按距离升序，
    含自身——距离 0 排第一，这是 MOEA/D 的常见约定）。
    """
    D = np.linalg.norm(W[:, None, :] - W[None, :, :], axis=2)  # (N, N)
    B = np.argsort(D, axis=1)[:, :T]
    return B


def expected_population_size(H: int, n_obj: int = 3) -> int:
    """N = C(H + n_obj - 1, n_obj - 1)。"""
    return int(np.prod([H + n_obj - 1 - k for k in range(n_obj - 1)]) // np.prod(range(1, n_obj)))
