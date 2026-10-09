# -*- coding: utf-8 -*-
"""Pareto 支配判定（向量化、分块，避免大矩阵爆内存）。

dominated_by(A, B)[a] = True  ⟺  存在 b ∈ B 使 b 在所有目标上 ≤ a
                                    且至少一个目标上 < a（严格支配）。
"""


def dominated_by(A, B, chunk: int = 384) -> "np.ndarray":
    import numpy as np

    A = np.atleast_2d(A)
    B = np.atleast_2d(B)
    out = np.zeros(len(A), dtype=bool)
    for s in range(0, len(A), chunk):
        a = A[s : s + chunk]                              # (c, m)
        le = (B[None, :, :] <= a[:, None, :]).all(axis=2)  # (c, |B|)
        lt = (B[None, :, :] < a[:, None, :]).any(axis=2)   # (c, |B|)
        out[s : s + chunk] = (le & lt).any(axis=1)
    return out
