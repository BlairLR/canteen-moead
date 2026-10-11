# -*- coding: utf-8 -*-
"""MOEA/D 主循环（Zhang & Li 2007 经典流程）。

每一代：
  for 每个子问题 i（顺序 0..N-1）:
    1. 从邻域 B(i) 中随机选两个父代
    2. SBX 交叉 + 多项式变异 → 一个子代
    3. 评价子代，更新归一化器（z* / z_nad）
    4. 邻域更新：对每个 j ∈ B(i)，若子代在 λ_j 下的标量化值
       不劣于当前成员 j，则替换之（经典 MOEA/D 无替换上限）
  每代结束后子代批量进入外部档案；按 track_every 触发轨迹回调。
"""

import numpy as np

from .archive import ParetoArchive
from .config import MOEADConfig
from .normalization import DynamicNormalizer
from .operators import polynomial_mutation, sbx_crossover
from .scalarization import scalarize
from .weights import neighborhoods, simplex_lattice_weights


class MOEAD:
    """面向学习目的的清晰 MOEA/D 实现（三目标、两决策变量、盒约束）。

    Parameters
    ----------
    cfg : MOEADConfig
        集中配置（见 moead/config.py）。
    evaluate_fn : callable
        目标函数：输入 (k, 2) 决策变量矩阵，返回 (k, 3) 目标值矩阵
        （本项目中即 src/nanhu_campus.py 的 evaluate）。
    """

    def __init__(self, cfg: MOEADConfig, evaluate_fn):
        if cfg.n_obj != 3 or cfg.n_var != 2:
            raise ValueError("第一版按 3 目标 / 2 决策变量问题编写")
        self.cfg = cfg
        self.evaluate = evaluate_fn

    def run(self, on_generation=None):
        """执行完整 MOEA/D。

        on_generation(gen, archive, n_replacements)：每 track_every 代回调一次，
        用于外部计算收敛轨迹（如 HV）。回调内不得修改档案。
        返回结果 dict。
        """
        cfg = self.cfg
        rng = np.random.default_rng(cfg.seed)

        # ---- 权重向量与邻域 ----
        W = simplex_lattice_weights(cfg.H, cfg.n_obj)      # (N, 3)
        N = len(W)
        B = neighborhoods(W, cfg.T)                        # (N, T)
        bounds = [cfg.x_bounds, cfg.y_bounds]

        # ---- 初始化种群 ----
        X = np.column_stack(
            [
                rng.uniform(cfg.x_bounds[0], cfg.x_bounds[1], N),
                rng.uniform(cfg.y_bounds[0], cfg.y_bounds[1], N),
            ]
        )
        F = self.evaluate(X)
        n_evals = N

        # ---- 归一化器与外部档案 ----
        norm = DynamicNormalizer(cfg.fallback_scale, cfg.norm_eps)
        norm.update(F)
        archive = ParetoArchive(cfg.dedup_decimals)
        archive.add_batch(X, F)

        history = []
        total_replacements = 0

        for gen in range(1, cfg.max_gen + 1):
            gen_replacements = 0
            children_X, children_F = [], []

            for i in range(N):
                # 1) 邻域内选两个父代
                j1, j2 = rng.choice(B[i], size=2, replace=False)
                p1, p2 = X[j1], X[j2]

                # 2) SBX + 变异 → 子代
                child = sbx_crossover(p1, p2, bounds, cfg.pc, cfg.eta_c, rng)
                child = polynomial_mutation(child, bounds, cfg.pm, cfg.eta_m, rng)

                # 3) 评价并更新归一化（z* 只收紧）
                Fc = self.evaluate(np.asarray(child)[None, :])   # (1, 3)
                n_evals += 1
                norm.update(Fc)

                # 4) 邻域更新：子代与各邻居在各自权重下比较
                #    （归一化参数取当前最新值，保证同一对比较的一致性）
                Fc_n = norm.normalize(Fc)                       # (1, 3)
                Fj_n = norm.normalize(F[B[i]])                  # (T, 3)
                g_child = scalarize(cfg.scalarization, Fc_n, W[B[i]], cfg.pbi_theta)  # (1, T)
                # 邻居各自在自己权重下的标量化值（逐权重直接计算，避免 T×T 广播）
                if cfg.scalarization == "tch":
                    g_member_diag = np.max(Fj_n * W[B[i]], axis=1)      # (T,)
                else:  # "pbi"
                    Wi = W[B[i]]                                         # (T, 3)
                    nrm = np.linalg.norm(Wi, axis=1)
                    nrm = np.where(nrm < 1e-12, 1.0, nrm)
                    Li = Wi / nrm[:, None]                               # (T, 3)
                    d1 = np.sum(Fj_n * Li, axis=1)                       # (T,)
                    proj = d1[:, None] * Li                              # (T, 3)
                    d2 = np.linalg.norm(Fj_n - proj, axis=1)             # (T,)
                    g_member_diag = d1 + cfg.pbi_theta * d2              # (T,)

                better = g_child[0] <= g_member_diag
                if better.any():
                    idx = B[i][better]
                    X[idx] = child
                    F[idx] = Fc[0]
                    gen_replacements += int(better.sum())

                children_X.append(child)
                children_F.append(Fc[0])

            # 每代子代批量入档
            archive.add_batch(np.asarray(children_X), np.asarray(children_F))
            total_replacements += gen_replacements
            history.append({"gen": gen, "archive_size": len(archive), "replacements": gen_replacements})

            if on_generation is not None and (gen % cfg.track_every == 0 or gen == cfg.max_gen):
                on_generation(gen, archive, gen_replacements)

        return {
            "pop_X": X,
            "pop_F": F,
            "archive_X": archive.X,
            "archive_F": archive.F,
            "weights": W,
            "neighborhoods": B,
            "n_evals": n_evals,
            "total_replacements": total_replacements,
            "history": history,
            "normalizer": norm,
        }
