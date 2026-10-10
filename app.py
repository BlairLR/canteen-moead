# -*- coding: utf-8 -*-
"""Flask 后端：交互式 MOEA/D 选址实验的 API 服务器。

启动: python app.py                → http://127.0.0.1:5200
"""

import json
import os
import sys
import threading
import time
import uuid
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from flask import Flask, jsonify, request, send_from_directory

from src.nanhu_campus import (
    build_model,
    local2wgs, wgs2local,
    LAT0, LON0, M_LAT, M_LON,
)
from src.interactive_pricing import (
    compute_price_grid, compute_contours,
    make_interactive_evaluate, ASSUMED_FLOOR_AREA,
)

# --- 必须放在 Flask app 创建之前，避免循环导入 ---
WEB_OUT = os.path.join(HERE, "output", "moead_nanhu_web")
os.makedirs(WEB_OUT, exist_ok=True)

app = Flask(__name__)

# ---------------------------------------------------------------------------
# 模型全局缓存（只构建一次）
# ---------------------------------------------------------------------------
_MODEL = None

def _get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = build_model()
    return _MODEL


# ---------------------------------------------------------------------------
# 后台作业管理器
# ---------------------------------------------------------------------------
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()

def _update_job(job_id: str, updates: dict):
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id].update(updates)


# ---------------------------------------------------------------------------
# 交互式地价场 — 后台运行函数
# ---------------------------------------------------------------------------

def _run_interactive_job(job_id: str, control_points: list, base_price: float,
                          sigma: float):
    """在后台线程中运行 MOEA/D 实验（交互式地价场模式）。"""
    try:
        model = _get_model()
        _update_job(job_id, {"status": "running", "progress": "stage0"})

        # 构建带交互式地价的评价函数
        from experiments.run_nanhu import run_experiment

        # 直接用交互式 evaluate 闭包
        def interactive_evaluate(X):
            # f1/f2 同默认
            from src.nanhu_campus import is_inside_polygon, SERVICE_RADIUS_M
            from src.nanhu_campus import build_model as _bm
            m = _get_model()
            demand = m.demand_xy
            comp = m.competition_xy
            Xa = np.atleast_2d(np.asarray(X, dtype=float))
            x, y = Xa[:, 0], Xa[:, 1]
            dx = demand[:, None, 0] - x[None, :]
            dy = demand[:, None, 1] - y[None, :]
            f1 = np.hypot(dx, dy).mean(axis=0)
            cx = comp[:, None, 0] - x[None, :]
            cy = comp[:, None, 1] - y[None, :]
            f2 = np.exp(-(cx**2 + cy**2) / (2.0 * SERVICE_RADIUS_M**2)).mean(axis=0)
            f3 = make_interactive_evaluate(
                control_points, base_price, sigma,
                m.bounds_x, m.bounds_y, ASSUMED_FLOOR_AREA)(Xa)
            F = np.column_stack([f1, f2, f3])
            out = ~is_inside_polygon(m.polygon_local, Xa)
            if out.any():
                probe = __import__("src.nanhu_campus", fromlist=[""]).sample_in_polygon(
                    2000, m.polygon_local)
                caps = np.column_stack([
                    np.hypot(demand[:, None, 0] - probe[None, :, 0],
                             demand[:, None, 1] - probe[None, :, 1]).mean(axis=0),
                    np.exp(-((comp[:, None, 0] - probe[None, :, 0])**2 +
                             (comp[:, None, 1] - probe[None, :, 1])**2)
                           / (2.0 * SERVICE_RADIUS_M**2)).mean(axis=0),
                    make_interactive_evaluate(
                        control_points, base_price, sigma,
                        m.bounds_x, m.bounds_y, ASSUMED_FLOOR_AREA)(probe),
                ]).max(axis=0) * 1.1
                F[out] = caps
            return F

        # 创建独立输出目录
        out_dir = os.path.join(WEB_OUT, job_id)
        os.makedirs(out_dir, exist_ok=True)

        # 保存价格配置快照
        config_snapshot = {
            "mode": "interactive_pricing",
            "control_points": control_points,
            "base_price": base_price,
            "sigma": sigma,
            "assumed_floor_area": ASSUMED_FLOOR_AREA,
            "coordinate_system": "米制局部坐标（x=东向, y=北向）",
        }
        with open(os.path.join(out_dir, "pricing_config.json"), "w", encoding="utf-8") as f:
            json.dump(config_snapshot, f, ensure_ascii=False, indent=2)

        _update_job(job_id, {"progress": "optimizing"})

        results = run_experiment(
            evaluate_fn=interactive_evaluate,
            out_dir=out_dir,
            model=model,
        )

        _update_job(job_id, {"progress": "postprocess"})

        # 加载归档数据（降采样供前端展示）
        Af = np.load(os.path.join(out_dir, "archive_seed42_F.npy"))
        Ax = np.load(os.path.join(out_dir, "archive_seed42_X.npy"))

        from metrics import nondominated_mask
        mask = nondominated_mask(Af)
        pf_f = Af[mask]
        pf_x = Ax[mask]
        n_pf = len(pf_f)

        max_pts = 500
        if n_pf > max_pts:
            idx = np.linspace(0, n_pf - 1, max_pts, dtype=int)
            pf_f = pf_f[idx]
            pf_x = pf_x[idx]

        pf_wgs = np.array([local2wgs(x, y) for x, y in pf_x])

        pareto_data = []
        for i in range(len(pf_f)):
            pareto_data.append({
                "x_local": float(pf_x[i, 0]), "y_local": float(pf_x[i, 1]),
                "lat": float(pf_wgs[i, 0]), "lon": float(pf_wgs[i, 1]),
                "f1": float(pf_f[i, 0]), "f2": float(pf_f[i, 1]), "f3": float(pf_f[i, 2]),
            })

        result_data = {
            "archive_size": int(len(Af)),
            "pareto_size": int(n_pf),
            "pareto": pareto_data,
            "hv": float(results["summary"]["hv"]["mean"]),
            "igd": float(results["summary"]["igd"]["mean"]),
            "time_s": float(results["summary"]["time_s"]["mean"]),
            "f1_range": [float(results["summary"]["f1_min"]["mean"]),
                         float(results["summary"]["f1_max"]["mean"])],
            "f2_range": [float(results["summary"]["f2_min"]["mean"]),
                         float(results["summary"]["f2_max"]["mean"])],
            "f3_range": [float(results["summary"]["f3_min"]["mean"]),
                         float(results["summary"]["f3_max"]["mean"])],
            "mode": "interactive_pricing",
            "job_id": job_id,
        }

        _update_job(job_id, {"status": "done", "progress": "done", "result": result_data})

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        _update_job(job_id, {"status": "error", "error": str(e), "traceback": tb})


# ---------------------------------------------------------------------------
# API 路由
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory(os.path.join(HERE, "templates"), "index.html")


@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory(os.path.join(HERE, "static"), filename)


@app.route("/data/<path:filename>")
def data_files(filename):
    """服务 data/ 目录下的静态文件（校园底图等）。"""
    return send_from_directory(os.path.join(HERE, "data"), filename)


@app.route("/api/model")
def api_model():
    """返回校园模型数据（边界多边形、点位、默认控制点）。"""
    model = _get_model()

    # 多边形顶点（局部 + WGS-84）
    poly_local = model.polygon_local.tolist()
    poly_wgs = model.polygon_wgs84.tolist()

    # 所有点（局部 + WGS-84）
    points = []
    for p in model.points:
        if p.latitude is not None and p.longitude is not None:
            x, y = wgs2local(p.latitude, p.longitude)
        else:
            x, y = None, None
        points.append({
            "id": p.pid, "name": p.name, "type": p.ptype,
            "lat": p.latitude, "lon": p.longitude,
            "x_local": x, "y_local": y,
        })

    return jsonify({
        "bounds_x": list(model.bounds_x),
        "bounds_y": list(model.bounds_y),
        "polygon_local": poly_local,
        "polygon_wgs84": poly_wgs,
        "points": points,
        "affine_transform": {
            # 局部坐标 → WGS-84: lat = a*y + b*x + b_lat, lon = c*y + d*x + b_lon
            "a": 1.0 / M_LAT, "b": 0.0, "c": 0.0, "d": 1.0 / M_LON,
            "b_lat": LAT0, "b_lon": LON0,
        },
        "assumed_floor_area": 1.0,
        "default_control_points": [
            {
                "id": "H1", "name": "中心热区",
                "lat": 41.76202793, "lon": 123.41114353,
                "x_local": -102.48, "y_local": 434.16,
                "default_price": 240.0,
                "description": "21 点质心附近区域",
            },
            {
                "id": "H2", "name": "镜像热区",
                "lat": 41.76490407, "lon": 123.41386138,
                "x_local": 123.51, "y_local": 753.61,
                "default_price": 170.0,
                "description": "质心关于A3外包框中心的镜像区域",
            },
        ],
        "coordinate_system": "米制局部坐标（x=东向, y=北向）",
        "note": "所有价格为模拟值（a.u.），不是真实地价",
        # 校园底图信息：21 顶点 LSQ 仿射配准（残差 0.1m RMS）
        # 像素→WGS-84 映射：lon = 5.8450e-6*px + 1.1936e-6*py + 123.407278
        #                   lat = -7.887e-7*px - 6.554e-6*py + 41.769152
        # 四个角点：NW(0,0)=(41.768395,123.405863) NE(1883,0)=(41.768211,123.419537)
        #           SW(0,1559)=(41.758669,123.405185) SE(1883,1559)=(41.758486,123.418859)
        "map_image": {
            "url": "/data/nanhu_campus_map.png",
            "width_px": 1884,
            "height_px": 1560,
            "bounds_sw": [41.758486, 123.405185],
            "bounds_ne": [41.768395, 123.419537],
            "provenance": "Gaode 底图截图，21 顶点 LSQ 仿射配准（残差 0.1m RMS）",
            "pixel_scale_m_per_px": 0.56,
        },
    })


@app.route("/api/status/<job_id>")
def api_status(job_id):
    """查询任务状态和结果。"""
    with _jobs_lock:
        job = _jobs.get(job_id)

    if job is None:
        return jsonify({"status": "not_found"}), 404

    resp = {
        "status": job["status"],
        "progress": job.get("progress", ""),
    }
    if job["status"] == "error":
        resp["error"] = job.get("error", "未知错误")
    if "result" in job:
        resp["result"] = job["result"]

    return jsonify(resp)


@app.route("/api/experiments", methods=["GET"])
def api_list_experiments():
    """列出保存的实验。"""
    exps = []
    if os.path.isdir(WEB_OUT):
        for d in sorted(os.listdir(WEB_OUT)):
            dpath = os.path.join(WEB_OUT, d)
            if os.path.isdir(dpath) and os.path.isfile(os.path.join(dpath, "experiment_results.json")):
                with open(os.path.join(dpath, "experiment_results.json"), "r", encoding="utf-8") as f:
                    results = json.load(f)
                pricing = {}
                pcfg_path = os.path.join(dpath, "pricing_config.json")
                if os.path.isfile(pcfg_path):
                    with open(pcfg_path, "r", encoding="utf-8") as f:
                        pricing = json.load(f)

                exps.append({
                    "id": d,
                    "created_at": os.path.getmtime(dpath),
                    "hv": results["summary"]["hv"]["mean"],
                    "pricing": pricing,
                    "archive_size": results["runs"][0]["archive_size"],
                })
    return jsonify(sorted(exps, key=lambda x: x["created_at"], reverse=True))


@app.route("/api/experiments/save", methods=["POST"])
def api_save_experiment():
    """标记一个 job 结果为实验 A 或 B（硬链接 / 复制快照）。"""
    data = request.get_json(force=True)
    label = data.get("label", "A").upper()
    job_id = data.get("job_id", "")

    if label not in ("A", "B"):
        return jsonify({"status": "error", "error": "标签只能是 A 或 B"}), 400

    src = os.path.join(WEB_OUT, job_id)
    if not os.path.isdir(src):
        return jsonify({"status": "error", "error": f"job {job_id} 不存在"}), 404

    dst = os.path.join(WEB_OUT, f"experiment_{label}")
    # 删除旧快照
    if os.path.isdir(dst):
        import shutil
        shutil.rmtree(dst)
    import shutil
    shutil.copytree(src, dst)

    return jsonify({"status": "ok", "label": label, "path": dst})


@app.route("/api/experiments/compare")
def api_compare():
    """比较实验 A 和 B 的数据。"""
    a_dir = os.path.join(WEB_OUT, "experiment_A")
    b_dir = os.path.join(WEB_OUT, "experiment_B")

    def _load_exp(exp_dir, label):
        if not os.path.isdir(exp_dir):
            return None
        results_path = os.path.join(exp_dir, "experiment_results.json")
        config_path = os.path.join(exp_dir, "pricing_config.json")
        if not os.path.isfile(results_path):
            return None

        with open(results_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        pricing = {}
        if os.path.isfile(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                pricing = json.load(f)

        Af_path = os.path.join(exp_dir, "archive_seed42_F.npy")
        Ax_path = os.path.join(exp_dir, "archive_seed42_X.npy")
        pf_data = []
        if os.path.isfile(Af_path) and os.path.isfile(Ax_path):
            Af = np.load(Af_path)
            Ax = np.load(Ax_path)
            from metrics import nondominated_mask
            mask = nondominated_mask(Af)
            pf_f = Af[mask][:500]
            pf_x = Ax[mask][:500]
            for i in range(len(pf_f)):
                lat, lon = local2wgs(pf_x[i, 0], pf_x[i, 1])
                pf_data.append({
                    "x_local": float(pf_x[i, 0]), "y_local": float(pf_x[i, 1]),
                    "lat": float(lat), "lon": float(lon),
                    "f1": float(pf_f[i, 0]), "f2": float(pf_f[i, 1]), "f3": float(pf_f[i, 2]),
                })

        return {
            "label": label,
            "pricing": pricing,
            "hv": results["summary"]["hv"]["mean"],
            "igd": results["summary"]["igd"]["mean"],
            "archive_size": results["runs"][0]["archive_size"],
            "f1_range": [results["summary"]["f1_min"]["mean"],
                         results["summary"]["f1_max"]["mean"]],
            "f2_range": [results["summary"]["f2_min"]["mean"],
                         results["summary"]["f2_max"]["mean"]],
            "f3_range": [results["summary"]["f3_min"]["mean"],
                         results["summary"]["f3_max"]["mean"]],
            "pareto": pf_data,
        }

    a_data = _load_exp(a_dir, "A")
    b_data = _load_exp(b_dir, "B")

    return jsonify({"A": a_data, "B": b_data})


# ---------------------------------------------------------------------------
# 交互式地价场 API
# ---------------------------------------------------------------------------

@app.route("/api/interactive-pricing/run", methods=["POST"])
def api_interactive_run():
    """提交交互式地价模式优化任务。"""
    data = request.get_json(force=True)
    control_points = data.get("control_points", [])
    base_price = float(data.get("base_price", 80.0))
    sigma = float(data.get("sigma", 150.0))

    # 验证控制点
    for i, cp in enumerate(control_points):
        if not isinstance(cp, dict) or "x" not in cp or "y" not in cp or "price" not in cp:
            return jsonify({"status": "error",
                            "errors": [f"控制点 {i} 缺少 x/y/price 字段"]}), 400
        p = float(cp["price"])
        if p < 0 or p > 500:
            return jsonify({"status": "error",
                            "errors": [f"控制点 {i} 价格 {p} 超出范围 [0, 500]"]}), 400

    job_id = str(uuid.uuid4())[:8]
    with _jobs_lock:
        _jobs[job_id] = {
            "status": "queued",
            "progress": "",
            "mode": "interactive_pricing",
            "control_points": control_points,
            "base_price": base_price,
            "sigma": sigma,
            "created_at": time.time(),
        }

    t = threading.Thread(target=_run_interactive_job,
                         args=(job_id, control_points, base_price, sigma),
                         daemon=True)
    t.start()

    return jsonify({"status": "queued", "job_id": job_id})


@app.route("/api/interactive-pricing/field", methods=["POST"])
def api_interactive_field():
    """计算交互式地价场的网格 + 等值线（供前端热力图显示）。"""
    data = request.get_json(force=True)
    control_points = data.get("control_points", [])
    base_price = float(data.get("base_price", 80.0))
    sigma = float(data.get("sigma", 150.0))
    resolution = int(data.get("resolution", 50))

    model = _get_model()

    # 计算网格
    xs, ys, values, mask_inside = compute_price_grid(
        model, control_points, base_price, sigma, resolution)

    # 计算等值线
    levels, contours = compute_contours(xs, ys, values, mask_inside, n_levels=6)

    # 将 2D 网格转为可序列化
    vmin = float(values[mask_inside].min()) if mask_inside.any() else 0
    vmax = float(values[mask_inside].max()) if mask_inside.any() else 100

    return jsonify({
        "xs": xs.tolist(),
        "ys": ys.tolist(),
        "values": values.tolist(),
        "mask": mask_inside.tolist(),
        "vmin": vmin,
        "vmax": vmax,
        "contour_levels": levels,
        "contours": contours,
    })


@app.route("/api/interactive-pricing/price-at", methods=["POST"])
def api_interactive_price_at():
    """查询指定坐标点的价格（用于点击地图显示价格）。"""
    data = request.get_json(force=True)
    x = float(data["x"])
    y = float(data["y"])
    control_points = data.get("control_points", [])
    base_price = float(data.get("base_price", 80.0))
    sigma = float(data.get("sigma", 150.0))

    model = _get_model()
    price = make_interactive_evaluate(
        control_points, base_price, sigma,
        model.bounds_x, model.bounds_y, ASSUMED_FLOOR_AREA
    )(np.array([[x, y]]))[0]

    return jsonify({
        "x": x,
        "y": y,
        "price": float(price),
        "base_price": base_price,
    })


# ---------------------------------------------------------------------------
# 启动
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 预构建模型（预热）
    print("加载校园模型...", flush=True)
    _get_model()
    print(f"Flask 服务器启动: http://127.0.0.1:5200")
    app.run(debug=False, host="127.0.0.1", port=5200)