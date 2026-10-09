# -*- coding: utf-8 -*-
"""性能指标实现。

1) hv_3d：精确三维超体积（f1 升序扫描 + 二维阶梯面积增量法）。
   复杂度 O(K²) 最坏，K=档案规模（千级）秒级完成。
2) igd：反向世代距离 = 参考点到前沿的最小欧氏距离均值（分块向量化）。
3) spacing：前沿点最近邻距离的标准差（分布均匀性）。
4) normalize_fixed：按外部给定 (lo, hi) 边界归一化（实验统一用第一阶段
   6000 随机采样的 min/max，保证 10 次运行、以及与随机采样基线可比）。
"""

import bisect
import math

import numpy as np

try:  # scipy 可用时用 cKDTree 加速最近邻（精确，非近似）；缺失则退化为原向量化实现
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover
    cKDTree = None


# ---------------------------------------------------------------------
# 归一化
# ---------------------------------------------------------------------
def normalize_fixed(F: np.ndarray, lo, hi) -> np.ndarray:
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    scale = np.where((hi - lo) < 1e-12, 1.0, hi - lo)
    return (np.asarray(F, float) - lo) / scale


# ---------------------------------------------------------------------
# 精确 3D 超体积
# ---------------------------------------------------------------------
def _staircase_insert(bps, keys, p2, p3, R2, R3):
    """向二维阶梯覆盖区插入锚定在 (R2, R3) 的矩形 [p2,R2]×[p3,R3]。

    bps: [f2, f3] 断点列表，f2 严格升序、f3 严格降序（最小表示）。
    覆盖区：u ∈ [bps[i].f2, bps[i+1].f2) 的下界为 bps[i].f3；
            u ∈ [last.f2, R2) 的下界为 last.f3。
    返回新增覆盖面积（可能为 0），并就地更新 bps/keys。
    """
    if p2 >= R2 or p3 >= R3:
        return 0.0
    pos = bisect.bisect_left(keys, p2)
    # p2 右侧起旧下界：前一个断点的 f3；若无覆盖则 R3（该区域原本为空）
    g_old = bps[pos - 1][1] if pos > 0 else R3
    gained = 0.0
    left = p2
    k = pos
    while True:
        right = bps[k][0] if k < len(bps) else R2
        if g_old > p3:
            gained += (g_old - p3) * (right - left)
        left = right
        if k >= len(bps) or right >= R2 or bps[k][1] <= p3:
            break
        g_old = bps[k][1]
        k += 1

    if gained <= 0.0:
        return 0.0  # 完全被已有覆盖吸收，无需改阶梯

    # 更新阶梯：插入/替换 (p2, p3)，删除其后被覆盖（f3 ≥ p3）的断点
    if pos < len(bps) and keys[pos] == p2:
        bps[pos] = [p2, p3]  # 替换（p3 必更小，否则 gained=0）
        end = pos + 1
        while end < len(bps) and bps[end][1] >= p3:
            end += 1
        del bps[pos + 1 : end]
        del keys[pos + 1 : end]
    else:
        bps.insert(pos, [p2, p3])
        keys.insert(pos, p2)
        end = pos + 1
        while end < len(bps) and bps[end][1] >= p3:
            end += 1
        del bps[pos + 1 : end]
        del keys[pos + 1 : end]
    return gained


def hv_3d(F: np.ndarray, ref) -> float:
    """精确计算三维超体积（最小化，ref 为被支配参考点，如 (1.05,1.05,1.05)）。

    算法：按 f1 升序扫描；每点把其 (f2,f3) 矩形并入二维阶梯覆盖区，
    新增面积 × (ref_f1 - f1) 即该点对超体积的贡献（并集无重复计数）。
    """
    F = np.atleast_2d(np.asarray(F, float))
    ref = np.asarray(ref, float)
    pts = F[(F[:, 0] <= ref[0]) & (F[:, 1] <= ref[1]) & (F[:, 2] <= ref[2])]
    if len(pts) == 0:
        return 0.0
    order = np.argsort(pts[:, 0], kind="stable")
    bps, keys = [], []
    hv = 0.0
    for i in order:
        p2, p3 = float(pts[i, 1]), float(pts[i, 2])
        gained = _staircase_insert(bps, keys, p2, p3, ref[1], ref[2])
        if gained > 0.0:
            hv += gained * (ref[0] - float(pts[i, 0]))
    return hv


# ---------------------------------------------------------------------
# IGD
# ---------------------------------------------------------------------
def igd(F: np.ndarray, ref_points: np.ndarray, chunk: int = 512) -> float:
    """IGD = mean_{r ∈ ref_points} min_{f ∈ F} ‖f - r‖₂。

    scipy 可用时用 cKDTree 精确最近邻（O(N·log K)），结果与分块向量化一致；
    缺失则退化为原 O(|R|·|F|) 分块实现。
    """
    F = np.atleast_2d(np.asarray(F, float))
    R = np.atleast_2d(np.asarray(ref_points, float))
    if cKDTree is not None and len(F) > 0:
        tree = cKDTree(F)
        d, _ = tree.query(R, k=1)           # 每个 R 点到最近 F 点的精确欧氏距离
        return float(np.asarray(d).mean())
    ds = []
    for s in range(0, len(R), chunk):
        d = np.linalg.norm(R[s : s + chunk, None, :] - F[None, :, :], axis=2)
        ds.append(d.min(axis=1))
    return float(np.concatenate(ds).mean())


# ---------------------------------------------------------------------
# Spacing
# ---------------------------------------------------------------------
def spacing(F: np.ndarray) -> float:
    """Schott's Spacing：前沿点最近邻距离的标准差（越小越均匀）。

    scipy 可用时用 cKDTree 精确最近邻：query(F, k=2)，第二近邻即最近非自身点
    （与原分块“填对角 inf 排除自身”等价；重复点→0 一致）；缺失则退化为原 O(K²) 分块。
    """
    F = np.atleast_2d(np.asarray(F, float))
    K = len(F)
    if K < 2:
        return 0.0
    if cKDTree is not None:
        tree = cKDTree(F)
        d, _ = tree.query(F, k=2)               # d[:,0]=自身(0), d[:,1]=最近非自身
        nn = np.asarray(d)[:, 1]
        d_bar = nn.mean()
        return float(math.sqrt(((nn - d_bar) ** 2).sum() / (K - 1)))
    d = np.full(K, np.inf)
    for s in range(0, K, 256):
        block = F[s : s + 256]
        D = np.linalg.norm(block[:, None, :] - F[None, :, :], axis=2)
        np.fill_diagonal(D[:, s : s + 256], np.inf)  # 排除自身
        d[s : s + 256] = np.minimum(d[s : s + 256], D.min(axis=1))
    d_bar = d.mean()
    return float(math.sqrt(((d - d_bar) ** 2).sum() / (K - 1)))


# ---------------------------------------------------------------------
# 非支配前沿掩码（精确，约 O(N·log|front|)）
# ---------------------------------------------------------------------
def _front2d_min(a, b):
    """二维最小化非支配掩码。要求已按 a 升序、b 升序（同 a 内 b 升序）排列。
    返回 bool 掩码（True=非支配），与 strict-Pareto ``~dominated_by`` 二维情形一致。
    点 k 被支配 ⟺ 存在更早的点 l 满足 a[l]≤a[k]、b[l]≤b[k] 且严格。
    因已按 a 升序：a[l]≤a[k] 自动成立；严格性来自 a[l]<a[k] 或（a 相等时）b[l]<b[k]。
    """
    n = len(a)
    if n == 0:
        return np.zeros(0, bool)
    mask = np.ones(n, bool)
    running_min_strict = math.inf   # 严格更小 a 的已处理点中最小 b
    cur_a_min_b = math.inf          # 当前 a 值组内已处理点最小 b
    prev = None
    for k in range(n):
        if prev is not None and a[k] != prev:
            running_min_strict = min(running_min_strict, cur_a_min_b)
            cur_a_min_b = math.inf
        if running_min_strict <= b[k]:      # 严格更小 a 且 b<=当前 → 支配
            mask[k] = False
        elif cur_a_min_b < b[k]:            # 同 a 组内更小 b → 支配
            mask[k] = False
        if b[k] < cur_a_min_b:
            cur_a_min_b = b[k]
        prev = a[k]
    return mask


def nondominated_mask(F: np.ndarray) -> np.ndarray:
    """精确三目标（最小化）非支配前沿掩码，True=非支配。

    与 ``~moead.dominance.dominated_by(F, F)`` 逐点结果一致，但用 f1 分组扫描 +
    (f2,f3) 运行最小值结构，避免 O(N²) 全配对广播，复杂度约 O(N·log|front|)。
    关键不变性：结构只保留严格更小 f1 的非支配点——被支配的点若有支配能力，
    其支配者（在结构中）同样支配目标点（传递性），故只需查结构即可。
    """
    F = np.atleast_2d(np.asarray(F, float))
    n, m = F.shape
    if n == 0:
        return np.zeros(0, bool)
    if n == 1:
        return np.array([True])
    if m == 1:
        return F[:, 0] == F[:, 0].min()
    if m == 2:
        order = np.lexsort((F[:, 1], F[:, 0]))
        Fo = F[order]
        nd = _front2d_min(Fo[:, 0], Fo[:, 1])
        mask = np.zeros(n, bool)
        mask[order] = nd
        return mask
    if m > 3:  # 本项目恒为 3 目标；m>3 退化为精确 O(N²) 兜底（不触发）
        le = (F[None, :, :] <= F[:, None, :]).all(axis=2)
        eq = (F[None, :, :] == F[:, None, :]).all(axis=2)
        return ~((le & ~eq).any(axis=1))
    # ---- m == 3：f1 分组扫描 ----
    order = np.lexsort((F[:, 2], F[:, 1], F[:, 0]))
    Fo = F[order]
    nondom_sorted = np.zeros(n, bool)
    s_a = np.empty(0)     # 结构中按 f2 升序
    s_b = np.empty(0)     # 配对 f3
    s_minb = np.empty(0)  # f3 前缀最小值
    i = 0
    while i < n:
        f1v = Fo[i, 0]
        j = i + 1
        while j < n and Fo[j, 0] == f1v:
            j += 1
        grp = Fo[i:j]
        g = len(grp)
        # 1) 查结构（严格更小 f1 支配者）
        survives = np.ones(g, bool)
        if s_a.size > 0:
            pos = np.searchsorted(s_a, grp[:, 1], side="right") - 1
            valid = pos >= 0
            minb_at = np.where(valid, s_minb[np.clip(pos, 0, s_minb.size - 1)], math.inf)
            survives = ~(valid & (minb_at <= grp[:, 2]))
        # 2) 组内二维非支配（处理 f1 相等）
        idx = np.where(survives)[0]
        if idx.size > 0:
            sub = grp[idx]
            sub_nd = _front2d_min(sub[:, 1], sub[:, 2])
            winners = idx[sub_nd]
        else:
            winners = np.empty(0, int)
        nondom_sorted[i + winners] = True
        # 3) 获胜者并入结构
        if winners.size > 0:
            cat_a = np.concatenate([s_a, grp[winners, 1]])
            cat_b = np.concatenate([s_b, grp[winners, 2]])
            o2 = np.argsort(cat_a, kind="stable")
            s_a = cat_a[o2]
            s_b = cat_b[o2]
            s_minb = np.minimum.accumulate(s_b)
        i = j
    mask = np.zeros(n, bool)
    mask[order] = nondom_sorted
    return mask
