# -*- coding: utf-8 -*-
"""动态目标归一化。

三个目标尺度完全不同（f1~数百米，f2~[0,1]，f3~数百万元），Tchebycheff
标量化 g = max_i λ_i·|f_i - z_i*| 若直接用原始值，f3 项会数值上完全压制
f2 项。因此在标量化之前先做归一化：

    f'_i = (f_i - z*_i) / (z_nad_i - z*_i)

其中：
  z*   理想点：所有已评价解的分量运行最小值（单调不增）；
  z_nad 最低点（nadir）估计：所有已评价解的分量运行最大值（单调不减）。

两者都来自"当前种群 + 迄今所有子代"，即基于当前搜索状态的动态估计。
初始种群是均匀随机采样，其 max 已接近问题的全域最差值（第一阶段的
6000 点验证：f1∈[269,650]，f2∈[0.003,0.387]，f3∈[80,320]），因此
z_nad 从第一代起就是稳定的分母，不会随代数剧烈波动；
z* 会随搜索进展缓慢下降（只变好不变差），归一化后的 Tchebycheff
比较在同一对 (child, member) 之间总是用同一套归一化参数即时计算，
保证比较的一致性。

数值保护：若某目标尺度 < eps（退化），回退到 fallback_scale。
"""

import numpy as np


class DynamicNormalizer:
    """基于理想点 z* 与最低点估计 z_nad 的动态归一化器。"""

    def __init__(self, fallback_scale=(400.0, 0.4, 250.0), eps: float = 1e-12):
        self.fallback_scale = np.asarray(fallback_scale, dtype=float)
        self.eps = eps
        self.z_star: np.ndarray | None = None   # 理想点（运行最小）
        self.z_nad: np.ndarray | None = None    # 最低点估计（运行最大）

    def update(self, F: np.ndarray) -> None:
        """用一批新评价解 (k, n_obj) 更新 z* 与 z_nad（只收紧，不放松）。"""
        F = np.atleast_2d(np.asarray(F, dtype=float))
        mn, mx = F.min(axis=0), F.max(axis=0)
        self.z_star = mn if self.z_star is None else np.minimum(self.z_star, mn)
        self.z_nad = mx if self.z_nad is None else np.maximum(self.z_nad, mx)

    def normalize(self, F: np.ndarray) -> np.ndarray:
        """返回归一化目标值 (k, n_obj)，理论上 ≥ 0（因为 f ≥ z*）。"""
        if self.z_star is None or self.z_nad is None:
            raise RuntimeError("Normalizer 尚未 update，无法归一化")
        F = np.atleast_2d(np.asarray(F, dtype=float))
        scale = self.z_nad - self.z_star
        scale = np.where(scale < self.eps, self.fallback_scale, scale)
        return (F - self.z_star) / scale
