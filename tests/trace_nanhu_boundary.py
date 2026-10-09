# -*- coding: utf-8 -*-
"""一次性研究脚本：南湖校区 A3 临时真实边界提取与配准。

流程：
  1. 从 data/nanhu_campus_map.jpg 程序化提取校园外轮廓（颜色分类 + matplotlib 等值线 + Douglas-Peucker）
  2. 四个正式 WGS-84 校门锚点做像素→局部米制仿射配准（最小二乘，报告残差）
  3. 多边形顶点转换（局部米制 + WGS-84），面积/范围
  4. 7 个工程点包含性检查（必要时质心缩放外扩 buffer 并记录）
  5. GCJ-02→WGS-84 一致性验证（eviltransform 逆变换 vs 上层锚点）
结果打印并保存 output/georef/boundary_trace.json。不修改任何模型文件。
"""

import json
import math
import os

import numpy as np
from PIL import Image

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys  # noqa: E402

sys.path.insert(0, ROOT)
from src import nanhu_campus as nc  # noqa: E402

OUT = os.path.join(ROOT, "output", "georef")
os.makedirs(OUT, exist_ok=True)

IMG = os.path.join(ROOT, "data", "nanhu_campus_map.jpg")

im = Image.open(IMG).convert("RGB")
arr = np.asarray(im, float)

# ---- 正式 WGS-84 锚点（上层提供）与图中像素位置（视觉读取，±15px） ----
ANCHORS = {
    "南门": {"wgs": (41.759173, 123.409959), "pix": (408.0, 648.0)},
    "西门": {"wgs": (41.763915, 123.405851), "pix": (172.0, 325.0)},
    "东门": {"wgs": (41.763453, 123.419148), "pix": (855.0, 325.0)},
    "北门": {"wgs": (41.767916, 123.413592), "pix": (568.0, 100.0)},
}
GCJ_CANDIDATES = {  # 上一轮高德公开 POI 候选（GCJ-02），用于一致性验证
    "南门": (41.761609, 123.416105),
    "西门": (41.766358, 123.411994),
}

# ---------------------------------------------------------------------
# 0. 模型局部坐标系（与 src/nanhu_campus.build_model 完全一致）
# ---------------------------------------------------------------------
model = nc.build_model()
LAT0, LON0 = model.lat0, model.lon0
M_LAT, M_LON = model.m_per_deg_lat, model.m_per_deg_lon


def wgs2local(lat, lon):
    return (lon - LON0) * M_LON, (lat - LAT0) * M_LAT


def local2wgs(x, y):
    return LAT0 + y / M_LAT, LON0 + x / M_LON


anchors_local = {k: wgs2local(*v["wgs"]) for k, v in ANCHORS.items()}
print("锚点局部坐标（米，模型帧）:")
for k, (x, y) in anchors_local.items():
    print(f"  {k}: ({x:.1f}, {y:.1f})")

# ---------------------------------------------------------------------
# 1. GCJ-02 → WGS-84 一致性验证（eviltransform 标准算法）
# ---------------------------------------------------------------------
A_EARTH, EE = 6378245.0, 0.00669342162296594323


def _tf_x(x, y):
    r = 300.0 + x + 2.0 * y + 0.1 * x * x + 0.1 * x * y + 0.1 * math.sqrt(abs(x))
    r += (20.0 * math.sin(6.0 * x * math.pi) + 20.0 * math.sin(2.0 * x * math.pi)) * 2.0 / 3.0
    r += (20.0 * math.sin(x * math.pi) + 40.0 * math.sin(x / 3.0 * math.pi)) * 2.0 / 3.0
    r += (150.0 * math.sin(x / 12.0 * math.pi) + 300.0 * math.sin(x / 30.0 * math.pi)) * 2.0 / 3.0
    return r


def _tf_y(x, y):
    r = -100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y + 0.1 * x * y + 0.2 * math.sqrt(abs(x))
    r += (20.0 * math.sin(6.0 * x * math.pi) + 20.0 * math.sin(2.0 * x * math.pi)) * 2.0 / 3.0
    r += (20.0 * math.sin(y * math.pi) + 40.0 * math.sin(y / 3.0 * math.pi)) * 2.0 / 3.0
    r += (160.0 * math.sin(y / 12.0 * math.pi) + 320.0 * math.sin(y * math.pi / 30.0)) * 2.0 / 3.0
    return r


def wgs84_to_gcj(lat, lon):
    dlat = _tf_y(lon - 105.0, lat - 35.0)
    dlon = _tf_x(lon - 105.0, lat - 35.0)
    radlat = lat / 180.0 * math.pi
    magic = 1 - EE * math.sin(radlat) ** 2
    sq = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((A_EARTH * (1 - EE)) / (magic * sq) * math.pi)
    dlon = (dlon * 180.0) / (A_EARTH / sq * math.cos(radlat) * math.pi)
    return lat + dlat, lon + dlon


def gcj2wgs(lat, lon, iters=3):
    wlat, wlon = lat, lon
    for _ in range(iters):
        glat, glon = wgs84_to_gcj(wlat, wlon)
        wlat += lat - glat
        wlon += lon - glon
    return wlat, wlon


print("\nGCJ-02 候选 → WGS-84（eviltransform）与上层正式锚点对比:")
gcj_check = {}
for k, (glat, glon) in GCJ_CANDIDATES.items():
    wlat, wlon = gcj2wgs(glat, glon)
    alat, alon = ANCHORS[k]["wgs"]
    d = math.hypot((wlat - alat) * 111068.8, (wlon - alon) * 83150.3)
    gcj_check[k] = {"gcj": [glat, glon], "gcj2wgs": [wlat, wlon], "delta_m_vs_anchor": round(d, 1)}
    print(f"  {k}: GCJ({glat},{glon}) -> WGS({wlat:.6f},{wlon:.6f}) vs 锚点({alat},{alon}) 差 {d:.1f} m")

# ---------------------------------------------------------------------
# 2. 校园轮廓提取：flood-fill 背景法（背景浅绿与校园深绿颜色不连续，
#    道路/建筑/草坪都不会被泛洪，从而避免颜色分类误判）
# ---------------------------------------------------------------------
from PIL import ImageDraw  # noqa: E402

flood = im.copy()
ImageDraw.floodfill(flood, (3, 3), (255, 0, 255), thresh=25)
f = np.asarray(flood, int)
# 被泛洪成紫色的像素 = 背景
is_bg = (f[:, :, 0] == 255) & (f[:, :, 1] == 0) & (f[:, :, 2] == 255)
mask = (~is_bg).astype(float)
print(f"背景泛洪占比 {is_bg.mean():.1%}（原图四角颜色 {arr[3,3].astype(int).tolist()}）")
mask[0:175, 0:215] = 0.0   # 图例框（白底不会被泛洪，需手动排除）
mask[35:130, 290:380] = 0.0  # 指北针（黑色，同上）

cs = plt.contour(mask, levels=[0.5])
segs = cs.allsegs[0]
plt.close("all")
outer = max(segs, key=len)
print(f"轮廓段数 {len(segs)}，最长段 {len(outer)} 点")


def simplify(pts, tol):
    pts = np.asarray(pts, float)
    keep = np.zeros(len(pts), bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        ax, ay = pts[i]
        bx, by = pts[j]
        sx, sy = bx - ax, by - ay
        L = math.hypot(sx, sy)
        px, py = pts[i + 1 : j, 0], pts[i + 1 : j, 1]
        if L == 0:
            d = np.hypot(px - ax, py - ay)
        else:
            d = np.abs(sx * (py - ay) - sy * (px - ax)) / L
        k = int(np.argmax(d))
        if d[k] > tol:
            m = i + 1 + k
            keep[m] = True
            stack.append((i, m))
            stack.append((m, j))
    return pts[keep]


tol = 3.0
poly_pix = simplify(outer, tol)
while len(poly_pix) > 60:
    tol *= 1.3
    poly_pix = simplify(outer, tol)
print(f"Douglas-Peucker: tol={tol:.1f}px -> {len(poly_pix)} 顶点")

# ---------------------------------------------------------------------
# 3. 仿射配准：像素 → 局部米制（4 锚点最小二乘）
# ---------------------------------------------------------------------
P = np.array([ANCHORS[k]["pix"] for k in ANCHORS])
Q = np.array([anchors_local[k] for k in ANCHORS])
M = np.zeros((8, 6))
b = np.zeros(8)
for i, (p, q_) in enumerate(zip(P, Q)):
    M[2 * i] = [p[0], p[1], 1, 0, 0, 0]
    M[2 * i + 1] = [0, 0, 0, p[0], p[1], 1]
    b[2 * i], b[2 * i + 1] = q_
coef, *_ = np.linalg.lstsq(M, b, rcond=None)
A2 = np.array([[coef[0], coef[1]], [coef[3], coef[4]]])
t2 = np.array([coef[2], coef[5]])


def pix2local(p):
    return A2 @ np.asarray(p) + t2


res = {k: float(np.linalg.norm(pix2local(np.array(ANCHORS[k]["pix"])) - np.array(anchors_local[k])))
       for k in ANCHORS}
print("仿射残差（米）:", {k: round(v, 1) for k, v in res.items()})

poly_local = np.array([pix2local(p) for p in poly_pix])

# ---------------------------------------------------------------------
# 4. 面积 / 范围 / 包含性
# ---------------------------------------------------------------------
def shoelace(poly):
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def inside(poly, pts):
    x, y = poly[:, 0], poly[:, 1]
    n = len(poly)
    X, Y = pts[:, 0], pts[:, 1]
    res = np.zeros(len(pts), bool)
    for i in range(n):
        j = (i - 1) % n
        cond = ((y[i] > Y) != (y[j] > Y)) & (
            X < (x[j] - x[i]) * (Y - y[i]) / (y[j] - y[i] + 1e-12) + x[i]
        )
        res ^= cond
    return res


area_m2 = shoelace(poly_local)
ext_x = (float(poly_local[:, 0].min()), float(poly_local[:, 0].max()))
ext_y = (float(poly_local[:, 1].min()), float(poly_local[:, 1].max()))
print(f"\n面积 {area_m2/1e6:.3f} km² | 东西 {ext_x[1]-ext_x[0]:.0f} m | 南北 {ext_y[1]-ext_y[0]:.0f} m")

pts = nc.load_points()
names = [p.pid for p in pts]
wgs = np.array([(p.latitude, p.longitude) for p in pts])
pts_local = np.array([wgs2local(la, lo) for la, lo in wgs])
ins = inside(poly_local, pts_local)

need_buffer = not ins.all()
if need_buffer:
    cen = poly_local.mean(axis=0)
    lo_s, hi_s = 1.0, 1.5
    for _ in range(40):
        mid = (lo_s + hi_s) / 2
        if inside(cen + (poly_local - cen) * mid, pts_local).all():
            hi_s = mid
        else:
            lo_s = mid
    s = hi_s
    poly_final = cen + (poly_local - cen) * s
    area_final = shoelace(poly_final)
    print(f"存在界外点 -> 质心缩放外扩 x{s:.4f}（+{(s-1)*100:.1f}%），面积 {area_final/1e6:.3f} km²")
else:
    s = 1.0
    poly_final = poly_local
    area_final = area_m2
    print("全部点已在边界内，无需外扩")

ins_f = inside(poly_final, pts_local)
print("\n包含性检查:")
for nm, xy, ok in zip(names, pts_local, ins_f):
    print(f"  {nm}: ({xy[0]:7.1f}, {xy[1]:7.1f}) {'内' if ok else '外'}")

# ---------------------------------------------------------------------
# 5. 顶点 WGS-84 与存档
# ---------------------------------------------------------------------
verts_wgs = [local2wgs(x, y) for x, y in poly_final]
out = {
    "anchors": {k: {"wgs": list(v["wgs"]), "pix": list(v["pix"]), "local": list(anchors_local[k]),
                    "affine_residual_m": res[k]} for k, v in ANCHORS.items()},
    "gcj_check": gcj_check,
    "affine": {"matrix": A2.tolist(), "translation": t2.tolist()},
    "pixel_tolerance": tol,
    "n_vertices": int(len(poly_final)),
    "buffer_scale": s,
    "area_km2": area_final / 1e6,
    "extent_x_m": list(ext_x), "extent_y_m": list(ext_y),
    "polygon_local_m": [[round(float(x), 1), round(float(y), 1)] for x, y in poly_final],
    "polygon_wgs84": [[round(la, 6), round(lo, 6)] for la, lo in verts_wgs],
    "points_inside": {nm: bool(ok) for nm, ok in zip(names, ins_f)},
}
with open(os.path.join(OUT, "boundary_trace.json"), "w", encoding="utf-8") as fh:
    json.dump(out, fh, ensure_ascii=False, indent=1)
print(f"\n已保存 {os.path.join(OUT, 'boundary_trace.json')}")
