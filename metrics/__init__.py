# -*- coding: utf-8 -*-
"""性能指标包：HV（精确 3D）、IGD、Spacing，及固定边界归一化工具。

所有指标均在**归一化目标空间**上计算（用第一阶段 6000 随机采样的
分量 min/max 作为固定边界），消除量纲差异对指标的影响。
"""

from .indicators import hv_3d, igd, spacing, normalize_fixed, nondominated_mask

__all__ = ["hv_3d", "igd", "spacing", "normalize_fixed", "nondominated_mask"]
