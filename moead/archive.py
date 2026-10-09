# -*- coding: utf-8 -*-
"""外部 Pareto 档案（External Archive）。

职责：
  1. 保存迄今发现的非支配解（决策变量 + 目标值）；
  2. 新解入档时删除被支配解（批内互斥 + 被档案支配 + 档案中被新解支配）；
  3. 删除重复解（决策空间按 dedup_decimals 舍入去重；
     目标空间按 6 位小数舍入去重——目标值完全相同的解视为重复）。

性能：每代批量 add_batch 一次；支配检查向量化 + 分块。
"""

import numpy as np

from .dominance import dominated_by


class ParetoArchive:
    def __init__(self, dedup_decimals: int = 6):
        self.dec = dedup_decimals
        self.X = np.empty((0, 2))   # 决策变量
        self.F = np.empty((0, 3))   # 目标值

    def __len__(self):
        return len(self.X)

    def add_batch(self, Xb: np.ndarray, Fb: np.ndarray) -> int:
        """批量加入候选解，返回实际入档数量。"""
        Xb = np.atleast_2d(np.asarray(Xb, dtype=float))
        Fb = np.atleast_2d(np.asarray(Fb, dtype=float))
        if len(Xb) == 0:
            return 0

        # 1) 重复解剔除：决策空间 + 目标空间双重舍入键
        keys = np.column_stack(
            [np.round(Xb, self.dec), np.round(Fb, 6)]
        )
        _, idx = np.unique(keys, axis=0, return_index=True)
        Xb, Fb = Xb[idx], Fb[idx]

        # 2) 批内互支配过滤
        keep = ~dominated_by(Fb, Fb)
        Xb, Fb = Xb[keep], Fb[keep]
        if len(Xb) == 0:
            return 0

        # 3) 被现有档案支配的剔除
        if len(self.F) > 0:
            keep = ~dominated_by(Fb, self.F)
            Xb, Fb = Xb[keep], Fb[keep]
            if len(Xb) == 0:
                return 0

        # 4) 档案中被新解支配的剔除
        if len(self.F) > 0:
            keep_arch = ~dominated_by(self.F, Fb)
            self.X, self.F = self.X[keep_arch], self.F[keep_arch]

        # 5) 追加幸存者
        self.X = np.vstack([self.X, Xb])
        self.F = np.vstack([self.F, Fb])
        return len(Xb)
