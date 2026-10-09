# -*- coding: utf-8 -*-
"""东北大学南湖校区数据层：WGS-84 点位 → 局部米制坐标 → A3 多边形搜索空间。

数据 provenance（写入一切报告时必须区分三类）：
  [真实数据 real]         21 个点位坐标（4 竞争 + 10 宿舍 + 7 教学，WGS-84）：data/nanhu_points.csv
  [真实边界 real-ish]     A3 多边形 = 高德底图描边 + 四门锚点配准（工程近似 ±20-40m）
                          data/nanhu_boundary_polygon.json（用户已审核批准）
  [临时假设 provisional]  f3 成本场 = 合成高斯（用户决定 B2），非真实土地成本

搜索空间（A2 → A3 变更，用户 2026-10-08 批准）：
  旧 A2：7 点范围 ±150m 外包框（已弃用）
  新 A3：候选点必须位于 A3 多边形内。实现方式：
    - cfg.bounds 取多边形外包矩形（供 SBX/变异截断）；
    - moead.MOEAD(repair_fn=...) 修复钩子将越界候选投影回多边形
      （最近边界点投影，保证种群/档案的 X-F 一致）；
    - MOEA/D 核心算法逻辑零改动（钩子默认 None）。

用法：
  python src/nanhu_campus.py        # 打印就绪检查报告
  from src.nanhu_campus import build_model, build_evaluate, repair_to_polygon
"""

import csv
import json
import math
import os
from dataclasses import dataclass

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POINTS_CSV = os.path.join(ROOT, "data", "nanhu_points.csv")
REFERENCE_JSON = os.path.join(ROOT, "data", "nanhu_reference.json")
POLYGON_JSON = os.path.join(ROOT, "data", "nanhu_boundary_polygon.json")

# ---- 冻结局部米制坐标系（与 A3 配准一致，勿改动） ----
LAT0, LON0 = 41.758119, 123.412376
M_LAT, M_LON = 111068.8, 83150.3

# ---- provisional synthetic 成本场参数（用户决定 B2，非真实地价） ----
SERVICE_RADIUS_M = 150.0                    # f2 服务半径 σs：沿用第一版定义
COST_BASE = 80.0                            # 万元
COST_HOTSPOT_AMPLITUDES = (240.0, 170.0)    # 热点1/热点2 峰值增量
COST_HOTSPOT_SIGMAS = (160.0, 100.0)


class MissingCoordinatesError(RuntimeError):
    """南湖点位/边界数据不完整，无法进入选址计算时抛出。"""


@dataclass
class NanhuPoint:
    pid: str
    name: str
    ptype: str  # competition / demand
    latitude: float | None
    longitude: float | None
    coordinate_system: str
    representative_point_definition: str
    source: str
    confidence: str
    error_estimate: str
    status: str
    notes: str


@dataclass
class NanhuModel:
    """南湖选址模型数据（A3 多边形搜索空间，局部米制坐标）。"""
    points: list                      # NanhuPoint 列表 [真实数据]
    polygon_local: np.ndarray         # (n,2) A3 边界顶点（局部米制）
    polygon_wgs84: np.ndarray         # (n,2) A3 边界顶点（WGS-84）
    bounds_x: tuple                   # 多边形外包矩形（SBX/变异截断用）
    bounds_y: tuple
    demand_xy: np.ndarray             # (n,2) 需求点（宿舍+教学）局部米制 [真实数据转换]
    competition_xy: np.ndarray        # (m,2) 竞争点局部米制 [真实数据转换]
    cost_hotspots: list               # [(cx,cy,A,s)] provisional synthetic
    demand_ids: list                  # 全部需求点 id（dorm + teaching）
    demand_dorm_ids: list             # 宿舍需求点 id（D01-D10，等权；分类权重为后续扩展接口）
    demand_teaching_ids: list         # 教学区需求点 id（T01-T07，等权；分类权重为后续扩展接口）
    competition_ids: list

    @property
    def provenance(self) -> dict:
        return {
            "points": "real（上层收口，WGS-84）",
            "search_space": "A3 Polygon（高德底图描边+四门配准，工程近似 ±20-40m，用户已批准）",
            "cost_field": "provisional synthetic（决定 B2，非真实地价）",
            "service_radius": f"{SERVICE_RADIUS_M:.0f} m（沿用第一版定义）",
        }


def wgs2local(lat, lon):
    return (lon - LON0) * M_LON, (lat - LAT0) * M_LAT


def local2wgs(x, y):
    return LAT0 + y / M_LAT, LON0 + x / M_LON


def load_points(path: str = POINTS_CSV) -> list[NanhuPoint]:
    """读取点位 CSV。空坐标字段解析为 None（显式缺失，不做任何猜测）。"""
    points = []
    with open(path, encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            lat = row["latitude"].strip()
            lon = row["longitude"].strip()
            points.append(
                NanhuPoint(
                    pid=row["id"].strip(),
                    name=row["name"].strip(),
                    ptype=row["type"].strip(),
                    latitude=float(lat) if lat else None,
                    longitude=float(lon) if lon else None,
                    coordinate_system=row["coordinate_system"].strip(),
                    representative_point_definition=row["representative_point_definition"].strip(),
                    source=row["source"].strip(),
                    confidence=row["confidence"].strip(),
                    error_estimate=row["error_estimate"].strip(),
                    status=row["status"].strip(),
                    notes=row["notes"].strip(),
                )
            )
    return points


def load_reference() -> dict:
    with open(REFERENCE_JSON, encoding="utf-8") as fh:
        return json.load(fh)


def load_polygon(path: str = POLYGON_JSON):
    """读取 A3 边界多边形（唯一事实来源 = WGS-84 顶点；局部坐标由帧转换重建）。"""
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    if not raw.get("approved"):
        raise MissingCoordinatesError(f"A3 边界未获批准: {path}")
    poly_wgs = np.asarray(raw["vertices_wgs84"], dtype=float)
    poly_local = np.array([wgs2local(la, lo) for la, lo in poly_wgs])
    return raw, poly_wgs, poly_local


def validate_structure(points: list[NanhuPoint]) -> list[str]:
    """校验点位硬性约束，返回违规列表（空=通过）。

    三类别：competition（C01-C04，C04 必须保留）、dorm（D01-D10，全部有坐标）、
    teaching（T01-T07，必须全部有坐标）。坐标系禁止混用。
    """
    violations = []
    ids = [p.pid for p in points]
    if len(ids) != len(set(ids)):
        violations.append(f"点位编号重复: {[i for i in ids if ids.count(i) > 1]}")

    by_id = {p.pid: p for p in points}

    # C 类竞争点：C01-C04，C04 必须保留
    have_c = sorted(i for i in ids if i.startswith("C"))
    expect_c = [f"C{k:02d}" for k in range(1, 5)]
    if have_c != expect_c:
        violations.append(f"C 类点位编号异常: 期望 {expect_c}, 实际 {have_c}")
    if "C04" not in by_id:
        violations.append("违反硬性约束: C04 学生城食堂必须保留（当前缺失）")

    # D 类宿舍需求点：D01-D10（全部有坐标）
    have_d = sorted(i for i in ids if i.startswith("D"))
    expect_d = [f"D{k:02d}" for k in range(1, 11)]
    if have_d != expect_d:
        violations.append(f"D 类点位编号异常: 期望 {expect_d}, 实际 {have_d}")
    for did in expect_d:
        p = by_id.get(did)
        if p is None:
            continue
        if p.ptype != "dorm":
            violations.append(f"{did} 类型错误: 应为 dorm, 实际 {p.ptype}")
        if p.latitude is None and p.status != "missing":
            violations.append(f"{did} 无坐标但状态未标记 missing")

    # T 类教学区需求点：T01-T07，必须全部有坐标（不允许缺失占位）
    have_t = sorted(i for i in ids if i.startswith("T"))
    expect_t = [f"T{k:02d}" for k in range(1, 8)]
    if have_t != expect_t:
        violations.append(f"T 类点位编号异常: 期望 {expect_t}, 实际 {have_t}")
    for tid in expect_t:
        p = by_id.get(tid)
        if p is not None:
            if p.ptype != "teaching":
                violations.append(f"{tid} 类型错误: 应为 teaching, 实际 {p.ptype}")
            if p.latitude is None:
                violations.append(f"{tid} 教学点缺坐标（教学区不允许缺失占位）")

    if not any(p.ptype == "competition" for p in points):
        violations.append("缺少 competition（食堂竞争点）")
    if not any(p.ptype in ("dorm", "teaching") for p in points):
        violations.append("缺少需求点（dorm/teaching）")
    systems = {p.coordinate_system for p in points if p.latitude is not None}
    if len(systems) > 1:
        violations.append(f"坐标系混用: {systems}")
    return violations


# ---------------------------------------------------------------------
# A3 多边形几何：包含判定 + 最近边界点修复
# ---------------------------------------------------------------------
def is_inside_polygon(poly: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """射线法包含判定（全向量化，无 Python 顶点循环）。pts (k,2) → bool (k,)。

    逐边计算射线穿越条件并按边 XOR 累加——与原逐边循环 ``res ^= cond`` 逐点
    等价（XOR 结合律），仅消除 ``for i in range(n)`` 的 Python 开销。
    """
    poly = np.asarray(poly, float)
    pts = np.atleast_2d(np.asarray(pts, float))
    n = len(poly)
    i = np.arange(n)
    j = (i - 1) % n
    x, y = poly[:, 0], poly[:, 1]
    X, Y = pts[:, 0], pts[:, 1]
    yi = y[:, None]; yj = y[j][:, None]; xi = x[:, None]; xj = x[j][:, None]
    Xb = X[None, :]; Yb = Y[None, :]
    cond = ((yi > Yb) != (yj > Yb)) & (
        Xb < (xj - xi) * (Yb - yi) / (yj - yi + 1e-12) + xi
    )
    return np.bitwise_xor.reduce(cond, axis=0)


def repair_to_polygon(pts: np.ndarray, poly: np.ndarray | None = None) -> np.ndarray:
    """约束修复：越界点投影到 A3 多边形内（最近边界点），界内点原样返回。

    输入 (k,2) → 输出 (k,2)。供 moead.MOEAD 的 repair_fn 钩子使用。
    """
    if poly is None:
        _, _, poly_local = load_polygon()
    poly = np.asarray(poly, float)
    pts = np.atleast_2d(np.asarray(pts, dtype=float)).copy()
    inside = is_inside_polygon(poly, pts)
    if inside.all():
        return pts
    n = len(poly)
    centroid = poly.mean(axis=0)
    for i in np.where(~inside)[0]:
        p = pts[i]
        best_d, best_q = math.inf, None
        for k in range(n):
            a, bb = poly[k], poly[(k + 1) % n]
            ab = bb - a
            denom = ab @ ab
            t = 0.0 if denom < 1e-18 else max(0.0, min(1.0, ((p - a) @ ab) / denom))
            q = a + t * ab
            d = float(np.linalg.norm(p - q))
            if d < best_d:
                best_d, best_q = d, q
        # 投影点恰落在边界线上，射线法对边界点判定不稳定——
        # 沿"边界点→形心"方向内推，直至严格进入界内
        q = np.asarray(best_q, dtype=float)
        dirv = centroid - q
        nd = float(np.linalg.norm(dirv))
        dn = dirv / nd if nd > 1e-12 else np.array([0.0, 1.0])
        fixed = q
        for step in (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0):
            cand = q + dn * step
            if bool(is_inside_polygon(poly, cand[None, :])[0]):
                fixed = cand
                break
        pts[i] = fixed
    return pts


def build_model() -> NanhuModel:
    """构造南湖模型：A3 多边形搜索空间 + 点位 + provisional 成本场。

    需求点 = dorm（D01-D10）+ teaching（T01-T07）两类（当前等权、逐点计入 f1；
    类别权重为未来扩展接口）。全部 17 个需求点均有坐标，无缺失。
    竞争点/需求点缺坐标属于数据错误，直接抛错。
    """
    points = load_points()
    violations = validate_structure(points)
    if violations:
        raise MissingCoordinatesError(f"南湖数据存在结构违规: {violations}")
    hard_missing = [p.pid for p in points
                    if p.ptype in ("competition", "teaching")
                    and (p.latitude is None or p.longitude is None)]
    if hard_missing:
        raise MissingCoordinatesError(f"竞争点/教学点坐标缺失（禁止缺失）: {hard_missing}")

    _, poly_wgs, poly_local = load_polygon()
    bounds_x = (float(poly_local[:, 0].min()), float(poly_local[:, 0].max()))
    bounds_y = (float(poly_local[:, 1].min()), float(poly_local[:, 1].max()))

    located = [p for p in points if p.latitude is not None]
    xy = np.array([wgs2local(p.latitude, p.longitude) for p in located])
    demand_idx = [i for i, p in enumerate(located) if p.ptype in ("dorm", "teaching")]
    comp_idx = [i for i, p in enumerate(located) if p.ptype == "competition"]
    demand_xy = xy[demand_idx]
    competition_xy = xy[comp_idx]
    demand_ids = [located[i].pid for i in demand_idx]
    demand_dorm_ids = [located[i].pid for i in demand_idx if located[i].ptype == "dorm"]
    demand_teaching_ids = [located[i].pid for i in demand_idx if located[i].ptype == "teaching"]
    competition_ids = [located[i].pid for i in comp_idx]

    # provisional synthetic 成本场（B2 规则不变，随点位/帧重算）：
    #   热点1 = 已定位点位收口质心（活动质心）；热点2 = 热点1 关于搜索框中心镜像
    centroid = xy.mean(axis=0)
    cx_c = 0.5 * (bounds_x[0] + bounds_x[1])
    cy_c = 0.5 * (bounds_y[0] + bounds_y[1])
    h1 = (float(centroid[0]), float(centroid[1]),
          COST_HOTSPOT_AMPLITUDES[0], COST_HOTSPOT_SIGMAS[0])
    h2 = (2 * cx_c - h1[0], 2 * cy_c - h1[1],
          COST_HOTSPOT_AMPLITUDES[1], COST_HOTSPOT_SIGMAS[1])

    return NanhuModel(
        points=points,
        polygon_local=poly_local, polygon_wgs84=poly_wgs,
        bounds_x=bounds_x, bounds_y=bounds_y,
        demand_xy=demand_xy, competition_xy=competition_xy,
        cost_hotspots=[h1, h2],
        demand_ids=demand_ids,
        demand_dorm_ids=demand_dorm_ids,
        demand_teaching_ids=demand_teaching_ids,
        competition_ids=competition_ids,
    )


def build_evaluate(model: NanhuModel | None = None):
    """返回与旧版 virtual campus 同签名的目标函数（f1/f2/f3，最小化）。

    f1/f2/f3 的数学定义与第一版完全一致（本次接入未改任何目标函数）。
    """
    if model is None:
        model = build_model()
    demand = model.demand_xy
    comp = model.competition_xy

    def evaluate(X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        x, y = X[:, 0], X[:, 1]
        dx = demand[:, None, 0] - x[None, :]
        dy = demand[:, None, 1] - y[None, :]
        f1 = np.hypot(dx, dy).mean(axis=0)
        cx = comp[:, None, 0] - x[None, :]
        cy = comp[:, None, 1] - y[None, :]
        f2 = np.exp(-(cx**2 + cy**2) / (2.0 * SERVICE_RADIUS_M**2)).mean(axis=0)
        f3 = np.full(X.shape[0], COST_BASE)
        for hcx, hcy, A, s in model.cost_hotspots:
            f3 = f3 + A * np.exp(-(((x - hcx) ** 2) + ((y - hcy) ** 2)) / (2.0 * s**2))
        return np.column_stack([f1, f2, f3])

    return evaluate


def shoelace_area(poly: np.ndarray) -> float:
    """多边形面积（鞋带公式，局部米制）。"""
    poly = np.asarray(poly, float)
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def sample_in_polygon(n: int, poly: np.ndarray | None = None, rng: np.random.Generator | None = None) -> np.ndarray:
    """直接生成 A3 多边形内部均匀随机候选点（拒绝法）。

    外包矩形均匀抽样的界外点**直接重抽**（不投影到边界），因此：
      - 返回点 100% 严格界内；
      - 分布为多边形内均匀分布，无"边界聚集"偏差。
    拒绝率 ≈ 1 - 多边形面积/外包矩形面积（当前约 30%）。
    """
    if poly is None:
        _, _, poly = load_polygon()
    poly = np.asarray(poly, float)
    lo, hi = poly.min(axis=0), poly.max(axis=0)
    rng = np.random.default_rng() if rng is None else rng
    out = np.empty((0, 2))
    while len(out) < n:
        cand = np.column_stack([rng.uniform(lo[0], hi[0], n), rng.uniform(lo[1], hi[1], n)])
        ok = cand[is_inside_polygon(poly, cand)]
        out = np.vstack([out, ok])
    return out[:n]


def boundary_distances(poly: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """每个点到多边形边界的最近距离（点-线段距离，向量化）。"""
    poly = np.asarray(poly, float)
    pts = np.atleast_2d(np.asarray(pts, float))
    n = len(poly)
    dmin = np.full(len(pts), np.inf)
    for k in range(n):
        a, b = poly[k], poly[(k + 1) % n]
        ab = b - a
        denom = ab @ ab
        t = np.clip(((pts - a) @ ab) / (denom if denom > 1e-18 else 1.0), 0.0, 1.0)
        proj = a + t[:, None] * ab
        dmin = np.minimum(dmin, np.linalg.norm(pts - proj, axis=1))
    return dmin


def build_constrained_evaluate(model: NanhuModel, margin: float = 1.1):
    """带 A3 约束的目标函数（供 experiments/run_nanhu.py 的 MOEA/D 使用）。

    界内点 → 真实 (f1,f2,f3)；界外点 → 罚值 = 可行域各目标最大估计值 × margin。
    罚值有界且三个分量同时劣化，保证：
      1) 界外解被任何可行解严格支配 → 外部档案永远干净；
      2) z_nad 上界稳定 → 动态归一化不被破坏（TCH 比较有效）。
    界外候选在 MOEA/D 中只占用评价预算（约 30%），其决策变量不会进入档案。
    """
    base = build_evaluate(model)
    probe = sample_in_polygon(2000, model.polygon_local)
    caps = base(probe).max(axis=0) * margin

    def evaluate(X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        F = base(X)
        out = ~is_inside_polygon(model.polygon_local, X)
        if out.any():
            F[out] = caps
        return F

    return evaluate, caps


def readiness_report() -> dict:
    """就绪检查：点位/结构/A3 边界/包含性/修复自检。"""
    ref = load_reference()
    report = {"reference": ref, "blockers": [], "ready": False}

    try:
        points = load_points()
    except Exception as e:  # noqa: BLE001
        report["blockers"].append(f"点位文件不可读: {e}")
        return report
    report["points"] = points
    report["structure_violations"] = validate_structure(points)
    report["blockers"].extend(report["structure_violations"])
    missing = [p.pid for p in points if p.status != "ok" or p.latitude is None]
    if missing:
        report["blockers"].append(f"点位坐标缺失: {', '.join(missing)}")

    try:
        raw, poly_wgs, poly_local = load_polygon()
        report["polygon"] = {"n_vertices": int(len(poly_local)),
                             "file": POLYGON_JSON,
                             "area_km2": raw.get("area_km2")}
    except Exception as e:  # noqa: BLE001
        report["blockers"].append(f"A3 边界不可用: {e}")
        return report

    try:
        model = build_model()
        report["model"] = model
    except Exception as e:  # noqa: BLE001
        report["blockers"].append(f"模型构造失败: {e}")
        return report

    # 包含性 + 修复自检
    allxy = np.vstack([model.demand_xy, model.competition_xy])
    ins = is_inside_polygon(model.polygon_local, allxy)
    report["points_inside"] = {**{pid: bool(v) for pid, v in zip(model.demand_ids, ins[:len(model.demand_ids)])},
                               **{pid: bool(v) for pid, v in zip(model.competition_ids, ins[len(model.demand_ids):])}}
    if not ins.all():
        report["blockers"].append(
            "存在界外模型点: " + ", ".join(pid for pid, v in report["points_inside"].items() if not v))

    # 直接采样自检（拒绝法，非投影）：候选点应 100% 严格界内且无边界聚集
    rng = np.random.default_rng(0)
    n_test = 3000
    sample = sample_in_polygon(n_test, model.polygon_local, rng=rng)
    inside_rate = float(is_inside_polygon(model.polygon_local, sample).mean())
    d = boundary_distances(model.polygon_local, sample)
    band = 20.0
    band_frac = float((d <= band).mean())
    perimeter = float(np.linalg.norm(np.diff(np.vstack([model.polygon_local, model.polygon_local[:1]]), axis=0), axis=1).sum())
    theory = min(1.0, perimeter * band / (raw_area := shoelace_area(model.polygon_local)))
    report["sampling_test"] = {
        "n": n_test, "inside_rate": inside_rate,
        "raw_accept_rate_hint": "拒绝法仅保留界内点，返回点 100% 界内",
        "band20m_frac": band_frac, "band20m_theory": theory,
        "dist_min_m": float(d.min()), "dist_mean_m": float(d.mean()),
    }
    if inside_rate < 1.0:
        report["blockers"].append(f"直接采样存在界外点（{inside_rate:.2%}）")
    if band_frac > theory * 1.5:
        report["blockers"].append("采样呈边界聚集（带内占比超理论 50%）")

    report["ready"] = not report["blockers"]
    return report


if __name__ == "__main__":
    rep = readiness_report()
    print("== 东北大学南湖校区 选址计算就绪检查（A3 搜索空间）==")
    if "points" in rep:
        print(f"点位定义: {len(rep['points'])} 个（C01-C04, D01-D10, T01-T07）[真实数据]")
        for p in rep["points"]:
            print(f"  {p.pid} {p.name:<8s} [{p.ptype:<11s}] WGS-84({p.latitude:.6f}, {p.longitude:.6f})")
    if "polygon" in rep:
        print(f"\nA3 边界: {rep['polygon']['n_vertices']} 顶点, {rep['polygon']['area_km2']} km²"
              f"（data/nanhu_boundary_polygon.json，用户已批准）")
    if "model" in rep:
        m = rep["model"]
        print(f"搜索空间: A3 多边形外包矩形 x{tuple(round(v,1) for v in m.bounds_x)} "
              f"y{tuple(round(v,1) for v in m.bounds_y)}（候选点经拒绝法直接采样约束在多边形内）")
        print("模型点局部坐标（米）:")
        for ids, arr in [(m.demand_ids, m.demand_xy), (m.competition_ids, m.competition_xy)]:
            for pid, (x, y) in zip(ids, arr):
                print(f"  {pid:<4s} ({x:7.1f}, {y:7.1f})")
        print("provisional synthetic 成本热点 [决定 B2]:")
        for k, (cx, cy, A, s) in enumerate(m.cost_hotspots, 1):
            print(f"  热点{k}: 中心({cx:.1f}, {cy:.1f}) +{A:.0f}万 σ={s:.0f}m")
    if "points_inside" in rep:
        bad = [k for k, v in rep["points_inside"].items() if not v]
        print(f"\n21 点包含性: {'全部界内 ✔' if not bad else '界外: ' + str(bad)}")
    if "sampling_test" in rep:
        st = rep["sampling_test"]
        print(f"\n直接采样自检（拒绝法，无投影）: n={st['n']}，界内率 {st['inside_rate']:.1%}")
        print(f"边界聚集检查: 距边界≤20m 带内占比 {st['band20m_frac']:.1%}"
              f"（均匀分布理论值 {st['band20m_theory']:.1%}），"
              f"样本-边界最近距离 min={st['dist_min_m']:.1f}m mean={st['dist_mean_m']:.1f}m")
    print(f"结构约束校验: {'通过' if not rep.get('structure_violations') else rep['structure_violations']}")
    print(f"可进入选址计算: {'是' if rep['ready'] else '否'}")
    for bl in rep["blockers"]:
        print(f"  [BLOCKER] {bl}")
    if rep["ready"]:
        print("\ndata provenance: " + "; ".join(f"{k}={v}" for k, v in rep["model"].provenance.items()))
