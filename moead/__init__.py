# -*- coding: utf-8 -*-
"""MOEA/D 第一版工程实现包。

模块划分：
  config.py        全部参数集中配置
  weights.py       单纯形格点权重向量生成 + 邻域构造
  normalization.py 动态目标归一化（理想点 z* + 最低点估计 z_nad）
  scalarization.py 标量化函数（Tchebycheff / PBI，可切换）
  operators.py     SBX 交叉 + 多项式变异（含边界修复）
  dominance.py     Pareto 支配判定（向量化）
  archive.py       外部 Pareto 档案（非支配过滤 + 去重）
  moead.py         MOEA/D 主循环
"""

from .config import MOEADConfig
from .moead import MOEAD

__all__ = ["MOEADConfig", "MOEAD"]
