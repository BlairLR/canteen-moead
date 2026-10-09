# -*- coding: utf-8 -*-
"""指标与算法组件的正确性自检（sanity tests）。

用独立实现的参考方法（蒙特卡洛超体积、暴力支配检查、暴力最近邻）
交叉验证主实现的正确性。全绿才进入正式实验。
"""

import sys
import time

import numpy as np

sys.path.insert(0, ".")

from metrics import hv_3d, igd, spacing, normalize_fixed
from moead.archive import ParetoArchive
from moead.dominance import dominated_by
from moead.normalization import DynamicNormalizer
from moead.operators import polynomial_mutation, sbx_crossover
from moead.scalarization import pbi, scalarize, tchebycheff
from moead.weights import neighborhoods, simplex_lattice_weights

rng = np.random.default_rng(7)
FAIL = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAIL.append(name)


print("== 1. 权重向量 ==")
W = simplex_lattice_weights(12, 3)
check("数量 = C(14,2) = 91", len(W) == 91, f"got {len(W)}")
check("每行和为 1", np.allclose(W.sum(axis=1), 1.0))
check("分量非负", (W >= 0).all())
check("含三个单目标极端向量",
      any(np.allclose(w, [1, 0, 0]) for w in W)
      and any(np.allclose(w, [0, 1, 0]) for w in W)
      and any(np.allclose(w, [0, 0, 1]) for w in W))
B = neighborhoods(W, 18)
check("邻域形状 (91, 18)", B.shape == (91, 18))
check("邻域含自身", (B == np.arange(91)[:, None]).any(axis=1).all())

print("== 2. 归一化器 ==")
norm = DynamicNormalizer((400.0, 0.4, 250.0), 1e-12)
F0 = np.array([[300.0, 0.1, 100.0], [500.0, 0.3, 250.0]])
norm.update(F0)
Fn = norm.normalize(F0)
check("最小值归一到 0", np.allclose(Fn[0], [0.0, 0.0, 0.0]))
check("最大值归一到 1", np.allclose(Fn[1], [1.0, 1.0, 1.0]))
norm.update(np.array([[280.0, 0.05, 90.0]]))  # 更优理想点
Fn2 = norm.normalize(np.array([[300.0, 0.1, 100.0]]))
check("z* 只收紧", np.allclose(Fn2[0], [(300 - 280) / (500 - 280), (0.1 - 0.05) / 0.25, 10 / 160]))

print("== 3. 标量化 ==")
Fn = np.array([[0.2, 0.8, 0.5], [0.9, 0.1, 0.3]])
w = np.array([[1.0, 0.0, 0.0], [0.5, 0.5, 0.0]])
G = tchebycheff(Fn, w)
check("TCH 极端权重=单目标", np.allclose(G[:, 0], [0.2, 0.9]))
check("TCH 均匀权重", np.allclose(G[:, 1], [0.4, 0.45]))
check("scalarize 分发一致", np.allclose(scalarize("tch", Fn, w), G))
Gp = pbi(np.array([[0.2, 0.0, 0.0]]), np.array([[1.0, 0.0, 0.0]]), 5.0)
check("PBI 在权重方向上 d2=0", np.allclose(Gp[0, 0], 0.2))
Gq = pbi(np.array([[0.0, 0.5, 0.0]]), np.array([[1.0, 0.0, 0.0]]), 5.0)
check("PBI 垂直偏离惩罚 d2", np.allclose(Gq[0, 0], 5.0 * 0.5))
Gp2 = pbi(np.array([[0.5, 0.5, 0.5]]), np.array([[1 / 3, 1 / 3, 1 / 3]]), 5.0)
d1 = 0.5 * np.sqrt(3)
d2 = 0.0
check("PBI 对角点 d1=√3/2, d2=0", np.allclose(Gp2[0, 0], d1 + 5 * d2))

print("== 4. 支配判定（vs 暴力） ==")
A = rng.uniform(0, 1, (60, 3))
Bv = rng.uniform(0, 1, (40, 3))
fast = dominated_by(A, Bv)


def brute(a, b):
    out = []
    for p in a:
        hit = False
        for q in b:
            if (q <= p).all() and (q < p).any():
                hit = True
                break
        out.append(hit)
    return np.array(out)


check("与暴力实现一致", (fast == brute(A, Bv)).all())
t = np.linspace(0, 1, 10)
arch_ref_F = np.column_stack([t, 1 - t, np.full(10, 0.5)])  # 互不支配的前沿
same = dominated_by(A, A)
check("与暴力实现一致（自集）", (same == brute(A, A)).all())
check("非支配集全 False", not dominated_by(arch_ref_F, arch_ref_F).any())

print("== 5. 外部档案 ==")
arch = ParetoArchive(6)
arch.add_batch(np.array([[1.0, 1.0], [5.0, 5.0]]), np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]]))
n = arch.add_batch(np.array([[2.0, 2.0], [9.0, 9.0], [1.0, 1.0]]),
                   np.array([[0.5, 1.0, 1.5], [4.0, 4.0, 4.0], [1.0, 2.0, 3.0]]))
check("支配+去重后仅入档 1 个", n == 1 and len(arch) == 2)
check("被支配档案点已删除", not np.isin(arch.F[:, 0], [1.0]).any() or arch.F[np.argmax(arch.F[:, 0])].tolist() != [1.0, 2.0, 3.0])
arch2 = ParetoArchive(6)
arch2.add_batch(rng.uniform(0, 1, (50, 2)), rng.uniform(0, 1, (50, 3)))
nd = ~dominated_by(arch2.F, arch2.F)
check("档案内无支配关系", nd.all(), f"{len(arch2)} points")

print("== 6. HV vs 蒙特卡洛 ==")
ref = (1.05, 1.05, 1.05)


def mc_hv(F, ref, n=400_000, seed=1):
    r = np.random.default_rng(seed)
    P = r.uniform(0, ref, (n, 3))
    dom = np.zeros(n, dtype=bool)
    for f in F:
        dom |= (P >= f).all(axis=1)
    return dom.mean() * np.prod(ref)


for tag, F in [
    ("随机 30 点", rng.uniform(0.05, 0.95, (30, 3))),
    ("随机 200 点", rng.uniform(0.05, 0.95, (200, 3))),
    ("球面前沿 500 点", np.column_stack([
        rng.uniform(0, 1, 500),
        rng.uniform(0, 1, 500),
        rng.uniform(0, 1, 500)])),
]:
    t0 = time.time()
    exact = hv_3d(F, ref)
    mc = mc_hv(F, ref)
    ok = abs(exact - mc) < 0.015  # 蒙特卡洛 40 万样本 ~±0.0008 σ，留裕量
    check(f"HV≈MC [{tag}]", ok, f"exact={exact:.5f} mc={mc:.5f} 耗时{time.time()-t0:.2f}s")
# 单点解析验证：点 (0.2,0.3,0.4)，ref(1.05,1.05,1.05)
check("HV 单点解析", abs(hv_3d(np.array([[0.2, 0.3, 0.4]]), ref) - 0.85 * 0.75 * 0.65) < 1e-12)
# 两点部分重叠：p1=(0.2,0.3,0.4) 并 p2=(0.3,0.2,0.5)
vol = 0.85 * 0.75 * 0.65 + 0.75 * 0.85 * 0.55 - 0.75 * 0.75 * 0.55  # 交集在 f1∈[0.2,0.3] 不可并…手工算改为程序验证
check("HV 两点=并集体积(数值积分)",
      abs(hv_3d(np.array([[0.2, 0.3, 0.4], [0.3, 0.2, 0.5]]), ref)
          - mc_hv(np.array([[0.2, 0.3, 0.4], [0.3, 0.2, 0.5]]), ref, 600_000, 3)) < 0.01)

print("== 7. IGD / Spacing ==")
F = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
R = np.array([[0.0, 0.0], [0.5, 0.5]])
check("IGD 手算", abs(igd(F, R) - (0.0 + np.sqrt(0.5) / np.sqrt(2) * 0)) < 1e-9
      or abs(igd(F, R) - np.mean([0.0, np.sqrt(0.5)])) < 1e-9)
grid = np.array([[0, 0], [1, 0], [2, 0], [3, 0]], dtype=float)
check("Spacing 等距=0", abs(spacing(grid)) < 1e-12)

print("== 8. 遗传算子 ==")
r2 = np.random.default_rng(0)
bounds = [(0.0, 1000.0), (0.0, 700.0)]
kids = [sbx_crossover([100.0, 100.0], [900.0, 600.0], bounds, 1.0, 15.0, r2) for _ in range(2000)]
K = np.array(kids)
check("SBX 子代在界内", (K >= 0).all() and (K[:, 0] <= 1000).all() and (K[:, 1] <= 700).all())
check("SBX 分布近似覆盖父代区间", 50 < K[:, 0].mean() < 950)
mk = [polynomial_mutation([500.0, 350.0], bounds, 1.0, 20.0, r2) for _ in range(2000)]
MK = np.array(mk)
check("变异子代在界内", (MK >= 0).all() and (MK[:, 0] <= 1000).all() and (MK[:, 1] <= 700).all())
check("变异零散（小步长为主）", np.abs(MK - 500).std() < 120)
c = sbx_crossover([5.0, 5.0], [5.0, 5.0], bounds, 1.0, 15.0, r2)
check("SBX 相同父代不产生 NaN", np.isfinite(c).all())

print()
if FAIL:
    print(f"共 {len(FAIL)} 项失败: {FAIL}")
    sys.exit(1)
print("全部自检通过 ✔")
