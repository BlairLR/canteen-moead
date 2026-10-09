# -*- coding: utf-8 -*-
"""MOEA/D 全部可配置参数集中于此（第一版）。修改参数只改这个文件。"""

from dataclasses import dataclass


@dataclass
class MOEADConfig:
    # ---- 问题（决策空间） ----
    x_bounds: tuple = (0.0, 1000.0)   # x ∈ [0, 1000] m
    y_bounds: tuple = (0.0, 700.0)    # y ∈ [0, 700] m
    n_var: int = 2
    n_obj: int = 3

    # ---- 权重向量 / 种群 ----
    H: int = 12        # 单纯形格点参数，三目标时 N = C(H+2,2) = 91 个子问题
    T: int = 18        # 邻域大小（≈ 0.2N）

    # ---- 遗传算子 ----
    pc: float = 0.9    # SBX 交叉概率
    eta_c: float = 15.0    # SBX 分布指数
    pm: float = 0.5    # 多项式变异概率（按每个决策变量计算，= 1/n_var）
    eta_m: float = 20.0    # 多项式变异分布指数

    # ---- 标量化 ----
    scalarization: str = "tch"   # "tch"（第一版）| "pbi"（预留接口）
    pbi_theta: float = 5.0       # PBI 罚参数（备用）

    # ---- 终止条件 ----
    max_gen: int = 200   # 最大代数（用户加速设定，原 500）

    # ---- 动态归一化 ----
    fallback_scale: tuple = (400.0, 0.4, 250.0)  # 尺度退化时的兜底值（约等于问题量级）
    norm_eps: float = 1e-12

    # ---- 外部档案 ----
    dedup_decimals: int = 6    # 决策空间去重精度（1e-6 m）

    # ---- 运行 ----
    seed: int = 42
    track_every: int = 20     # 每隔多少代触发一次轨迹回调（用户加速设定，原 100）
