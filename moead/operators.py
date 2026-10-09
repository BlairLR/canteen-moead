# -*- coding: utf-8 -*-
"""实数编码遗传算子：SBX 交叉 + 多项式变异（Deb 标准公式）。

决策变量只有 2 维，为避免 numpy 标量开销，这里用纯 Python 浮点实现。
所有算子保证子代落在边界内（越界即截断修复，对应盒约束）。
"""

import numpy as np


def sbx_crossover(p1, p2, bounds, pc: float, eta_c: float, rng: np.random.Generator):
    """模拟二进制交叉（SBX）。返回一个子代（长度 n_var 的 list）。

    每个变量：u ~ U(0,1)，
      u ≤ 0.5:  β = (2u)^(1/(η_c+1))；否则 β = (1/(2(1-u)))^(1/(η_c+1))
      c1 = 0.5[(p1+p2) - β·|p2-p1|]，c2 = 0.5[(p1+p2) + β·|p2-p1|]
    以 0.5 概率取 c1 / c2。以 pc 概率整体执行交叉，否则复制 p1。
    """
    child = list(p1)
    if rng.random() >= pc:
        # 不交叉：直接复制父代 1（变异仍会作用）
        return _clip(child, bounds)
    for k in range(len(p1)):
        if abs(p1[k] - p2[k]) > 1e-14:
            u = rng.random()
            if u <= 0.5:
                beta = (2.0 * u) ** (1.0 / (eta_c + 1.0))
            else:
                beta = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta_c + 1.0))
            avg = 0.5 * (p1[k] + p2[k])
            diff = 0.5 * beta * abs(p2[k] - p1[k])
            c1, c2 = avg - diff, avg + diff
            child[k] = c1 if rng.random() < 0.5 else c2
        else:
            child[k] = p1[k] if rng.random() < 0.5 else p2[k]
    return _clip(child, bounds)


def polynomial_mutation(x, bounds, pm: float, eta_m: float, rng: np.random.Generator):
    """多项式变异（就地修改并返回）。每个变量以 pm 概率变异。

    Deb 标准公式（含边界感知的 δ 分布）：
      δ1 = (x-lb)/(ub-lb)，δ2 = (ub-x)/(ub-lb)
      u ≤ 0.5: δq = (2u + (1-2u)(1-δ1)^(η+1))^(1/(η+1)) - 1
      u >  0.5: δq = 1 - (2(1-u) + 2(u-0.5)(1-δ2)^(η+1))^(1/(η+1))
      x' = x + δq·(ub-lb)，再截断回边界。
    """
    for k in range(len(x)):
        if rng.random() >= pm:
            continue
        lo, hi = bounds[k]
        if hi - lo < 1e-14:
            continue
        delta1 = (x[k] - lo) / (hi - lo)
        delta2 = (hi - x[k]) / (hi - lo)
        u = rng.random()
        if u <= 0.5:
            dq = (2.0 * u + (1.0 - 2.0 * u) * (1.0 - delta1) ** (eta_m + 1.0)) ** (
                1.0 / (eta_m + 1.0)
            ) - 1.0
        else:
            dq = 1.0 - (
                2.0 * (1.0 - u) + 2.0 * (u - 0.5) * (1.0 - delta2) ** (eta_m + 1.0)
            ) ** (1.0 / (eta_m + 1.0))
        x[k] = x[k] + dq * (hi - lo)
    return _clip(x, bounds)


def _clip(x, bounds):
    for k in range(len(x)):
        lo, hi = bounds[k]
        x[k] = min(max(x[k], lo), hi)
    return x
