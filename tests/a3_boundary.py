# -*- coding: utf-8 -*-
"""A3 校园边界草案 v2：基于高德地图截图 (data/nanhu_campus_map.png, 1884x1560, GCJ-02 渲染)。

方法：
  1. 上层四门 WGS-84 锚点经 eviltransform 正变换 → GCJ-02；
  2. 图上可辨认的三个门 POI 图标像素（北门/东门/南门）↔ GCJ-02 做仿射拟合（3 点精确解）；
  3. 手工沿地图外围描 21 顶点不规则多边形（像素）→ 仿射 → GCJ-02 → WGS-84；
  4. 新基准 7 模型点 + 3 辅助点包含性检查；
  5. 定量验证：模型点投影像素 vs 图上同名 POI 标签像素的距离；
  6. 叠加图 output/georef/a3_overlay.png 供目视核验。
注意：AMap 截图渲染于 GCJ-02；WGS-84 锚点必须先转 GCJ-02 才能与图上像素对齐。
"""

import json
import math
import os
import sys

import numpy as np
from PIL import Image

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.nanhu_campus import load_points  # noqa: E402

OUT = os.path.join(ROOT, "output", "georef")
os.makedirs(OUT, exist_ok=True)
IMG = os.path.join(ROOT, "data", "nanhu_campus_map.png")

# ---- eviltransform（与 trace_nanhu_boundary.py 相同实现） ----
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
    return (lat + (dlat * 180.0) / ((A_EARTH * (1 - EE)) / (magic * sq) * math.pi),
            lon + (dlon * 180.0) / (A_EARTH / sq * math.cos(radlat) * math.pi))


def gcj2wgs(lat, lon, iters=3):
    wlat, wlon = lat, lon
    for _ in range(iters):
        glat, glon = wgs84_to_gcj(wlat, wlon)
        wlat += lat - glat
        wlon += lon - glon
    return wlat, wlon


# ---- 四门：WGS-84（上层正式锚点）→ GCJ-02 → 图上 POI 图标像素（视觉读取 ±20px） ----
GATES_WGS = {
    "南门": (41.759173, 123.409959),
    "西门": (41.763915, 123.405851),
    "东门": (41.763453, 123.419148),
    "北门": (41.767916, 123.413592),
}
GATES_GCJ = {k: wgs84_to_gcj(*v) for k, v in GATES_WGS.items()}
GATE_PIXELS = {  # 仅三个门在图上可可靠辨认（南门取校园正南门 POI）
    "北门": (1068.0, 60.0),
    "东门": (1875.0, 763.0),
    "南门": (652.0, 1468.0),
}

# ---- 手工描边的校园外轮廓（顺时针，像素；依据道路/围墙/地物，±30px） ----
POLY_PIX = [
    (1068, 60),    # 北门
    (1420, 180),   # 知行路南侧
    (1650, 430),   # 东北角（科苑小区以西）
    (1740, 620),   # 五四体育场以东
    (1875, 763),   # 东门
    (1760, 890),
    (1620, 1060),  # 同方广场蓝铂佳苑以西
    (1560, 1260),  # 材料学院以东
    (1430, 1425),  # 东南角（沈水路北）
    (1050, 1487),  # 沈水路
    (652, 1468),   # 南门
    (430, 1445),   # 沈水路（云享公园以西界外）
    (230, 1385),   # 西南角（文体西路小区以北）
    (110, 1265),   # 1923咖啡/国际学术交流中心
    (62, 1050),    # 学生第十宿舍以西
    (55, 830),     # 西墙
    (90, 640),     # 蜜雪冰城以西界外
    (55, 555),     # 西北（八舍C区以北仍在校内）
    (200, 505),    # 北门家属区以南
    (450, 460),    # 幼儿园/滨湖园以西界外
    (750, 300),    # 知行路折向东北
]

# ---- 定量验证标签（图上 POI 文字像素，视觉读取 ±15px） ----
VALIDATION_LABELS = {
    "C02": ("二食堂", (865, 1170)),
    "C03": ("东大餐饮中心", (925, 1400)),
    "D03": ("学生第三宿舍A区", (1293, 1400)),
    "D08": ("学生第八宿舍C区", (220, 522)),
    "C01": ("1食堂", (1180, 1290)),
}

# ---- 局部米制帧（锚定南门 GCJ，用于面积/距离；GCJ 局部度量与 WGS 一致） ----
LAT_ANCH, LON_ANCH = GATES_GCJ["南门"]
M_LAT, M_LON = 111068.8, 83150.3


def gcj2local(lat, lon):
    return (lon - LON_ANCH) * M_LON, (lat - LAT_ANCH) * M_LAT


# ---------------------------------------------------------------------
# 1. 三门仿射拟合：像素 → GCJ-02（度）。3 点精确解。
# ---------------------------------------------------------------------
P = np.array([GATE_PIXELS[k] for k in ("北门", "东门", "南门")])
Q = np.array([GATES_GCJ[k] for k in ("北门", "东门", "南门")])
M = np.zeros((6, 6))
b = np.zeros(6)
for i, (p, q_) in enumerate(zip(P, Q)):
    M[2 * i] = [p[0], p[1], 1, 0, 0, 0]
    M[2 * i + 1] = [0, 0, 0, p[0], p[1], 1]
    b[2 * i], b[2 * i + 1] = q_
coef = np.linalg.solve(M, b)
A2 = np.array([[coef[0], coef[1]], [coef[3], coef[4]]])
t2 = np.array([coef[2], coef[5]])
A2inv = np.linalg.inv(A2)
scale = float(np.sqrt(abs(np.linalg.det(A2)))) * M_LON  # 近似 m/px
print(f"仿射（像素→GCJ°）| 比例 ≈ {scale:.3f} m/px（比例尺目测 ~0.7-0.9）")


def pix2gcj(p):
    return A2 @ np.asarray(p, float) + t2


def gcj2pix(g):
    return A2inv @ (np.asarray(g, float) - t2)


# 西门预测像素（图上未找到西门标注，用于核对落点）
ximen_pred = gcj2pix(GATES_GCJ["西门"])
print(f"西门预测像素: ({ximen_pred[0]:.0f}, {ximen_pred[1]:.0f})（图上西墙一带应在其附近）")

# ---------------------------------------------------------------------
# 2. 多边形 → GCJ 局部米制；验证标签距离
# ---------------------------------------------------------------------
poly_local = np.array([gcj2local(*pix2gcj(p)) for p in POLY_PIX])


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


area = shoelace(poly_local)
ext = (poly_local[:, 0].min(), poly_local[:, 0].max(),
       poly_local[:, 1].min(), poly_local[:, 1].max())
print(f"\nA3 多边形: {len(POLY_PIX)} 顶点 | 面积 {area/1e6:.3f} km² | "
      f"东西 {ext[1]-ext[0]:.0f} m | 南北 {ext[3]-ext[2]:.0f} m")

pts = load_points()
names = [p.pid for p in pts]
wgs7 = np.array([(p.latitude, p.longitude) for p in pts])
gcj7 = np.array([wgs84_to_gcj(la, lo) for la, lo in wgs7])
loc7 = np.array([gcj2local(la, lo) for la, lo in gcj7])
ins7 = inside(poly_local, loc7)

AUX = {"一舍(校验)": (41.760656, 123.413818), "二舍(校验)": (41.760620, 123.412308),
       "十舍(校验)": (41.761856, 123.406278)}
aux_names = list(AUX)
aux_gcj = np.array([wgs84_to_gcj(*AUX[k]) for k in aux_names])
aux_loc = np.array([gcj2local(la, lo) for la, lo in aux_gcj])
ins_aux = inside(poly_local, aux_loc)

print("\n模型点包含性（A3 草案）:")
for nm, xy, ok in zip(names, loc7, ins7):
    print(f"  {nm}: ({xy[0]:7.1f}, {xy[1]:7.1f}) {'内 ✔' if ok else '外 ✘'}")
print("辅助校验点:")
for nm, xy, ok in zip(aux_names, aux_loc, ins_aux):
    print(f"  {nm}: {'内' if ok else '外'}")

print("\n定量验证（模型点投影像素 vs 图上同名标签像素）:")
val_results = {}
for pid, (label, lpix) in VALIDATION_LABELS.items():
    i = names.index(pid)
    ppix = gcj2pix(gcj7[i])
    d = float(np.hypot(ppix[0] - lpix[0], ppix[1] - lpix[1])) * scale
    val_results[pid] = {"label": label, "label_pix": list(lpix),
                        "proj_pix": [round(float(ppix[0]), 1), round(float(ppix[1]), 1)],
                        "dist_m": round(d, 1)}
    print(f"  {pid} vs {label}: 投影({ppix[0]:.0f},{ppix[1]:.0f}) vs 标签{lpix} → {d:.1f} m")

# ---------------------------------------------------------------------
# 3. 叠加图
# ---------------------------------------------------------------------
im = Image.open(IMG).convert("RGB")
fig, ax = plt.subplots(figsize=(14, 11.5))
ax.imshow(np.asarray(im.convert("RGB")))
pp = np.array(POLY_PIX, float)
ax.plot(np.append(pp[:, 0], pp[0, 0]), np.append(pp[:, 1], pp[0, 1]),
        "b-", linewidth=2, label="A3 边界草案（手工描边+三门配准）")
for k, v in GATE_PIXELS.items():
    ax.plot(*v, "r^", markersize=12, markeredgecolor="k")
    ax.annotate(k + "锚", v, textcoords="offset points", xytext=(7, 4),
                color="darkred", fontsize=11, fontweight="bold")
for nm, xy, ok in zip(names, loc7, ins7):
    px = gcj2pix(gcj7[names.index(nm)])
    ax.plot(*px, "o", color="lime" if ok else "magenta", markersize=11,
            markeredgecolor="k", markeredgewidth=1, zorder=6)
    ax.annotate(nm, px, textcoords="offset points", xytext=(8, -4),
                color="black", fontsize=11, fontweight="bold")
for nm in aux_names:
    i = aux_names.index(nm)
    px = gcj2pix(aux_gcj[i])
    ax.plot(*px, "s", color="yellow", markersize=7, markeredgecolor="k", zorder=6)
for pid, (label, lpix) in VALIDATION_LABELS.items():
    ax.plot(*lpix, "o", markerfacecolor="none", markeredgecolor="orange",
            markersize=13, markeredgewidth=1.6, zorder=5)
ax.set_title("A3 边界草案 + 新基准 7 点（绿=界内 品红=界外；黄=辅助点；橙圈=图上同名标签）")
ax.legend(loc="upper right", fontsize=10)
ax.set_xlim(0, 1884)
ax.set_ylim(1560, 0)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "a3_overlay.png"), dpi=130)
plt.close(fig)
print(f"\n叠加图已保存 {os.path.join(OUT, 'a3_overlay.png')}")

# ---------------------------------------------------------------------
# 4. 存档
# ---------------------------------------------------------------------
verts_wgs = [gcj2wgs(*pix2gcj(p)) for p in POLY_PIX]
out = {
    "map": {"file": "data/nanhu_campus_map.png", "size": [1884, 1560],
            "renderer": "高德地图截图（GCJ-02 渲染）", "sha256_16": "73f6b9112060b2c6"},
    "gates_gcj": {k: list(v) for k, v in GATES_GCJ.items()},
    "gates_pixel": {k: list(v) for k, v in GATE_PIXELS.items()},
    "affine_pix2gcj": {"matrix": A2.tolist(), "translation": t2.tolist()},
    "scale_m_per_px": round(scale, 3),
    "ximen_predicted_pixel": [round(float(ximen_pred[0]), 1), round(float(ximen_pred[1]), 1)],
    "n_vertices": len(POLY_PIX),
    "area_km2": area / 1e6,
    "extent_local_m": {"x": [round(ext[0], 1), round(ext[1], 1)],
                       "y": [round(ext[2], 1), round(ext[3], 1)],
                       "ew_m": round(ext[1] - ext[0]), "ns_m": round(ext[3] - ext[2])},
    "polygon_pixel": [[int(x), int(y)] for x, y in POLY_PIX],
    "polygon_wgs84": [[round(la, 6), round(lo, 6)] for la, lo in verts_wgs],
    "model_points_inside": {nm: bool(ok) for nm, ok in zip(names, ins7)},
    "aux_inside": {nm: bool(ok) for nm, ok in zip(aux_names, ins_aux)},
    "validation": val_results,
}
with open(os.path.join(OUT, "a3_boundary.json"), "w", encoding="utf-8") as fh:
    json.dump(out, fh, ensure_ascii=False, indent=1)
print(f"边界数据已保存 {os.path.join(OUT, 'a3_boundary.json')}")
