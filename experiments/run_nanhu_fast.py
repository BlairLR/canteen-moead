# -*- coding: utf-8 -*-
"""Fast experiment config (independent output, does not overwrite 10-seed history).

- Quick verify: 1 seed x 50 gen
- Demo experiment: 5 seeds x 100 gen
- Stage 0 still uses 6000 samples; HV trajectory interval 50 gen
- Output dir output/moead_nanhu_fast/, never touches output/moead_nanhu/
- Reuses optimized front extraction / merged front / igd / spacing (see metrics/indicators.py)
- Does not modify MOEAD/, objectives, A3 boundary, 21 points, MOEAD core, constraint mechanism
"""
import json, os, sys, time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src import nanhu_campus as nc
from moead import MOEAD, MOEADConfig
from metrics import hv_3d, igd, normalize_fixed, spacing, nondominated_mask

FAST_ROOT = os.path.join(ROOT, "output", "moead_nanhu_fast")
N_SAMPLES = 6000
HV_REF = (1.05, 1.05, 1.05)
TRACK_EVERY = 50
QUICK_SEEDS, QUICK_GEN = [1], 50
DEMO_SEEDS, DEMO_GEN = list(range(1, 6)), 100


def agg(values):
    v = np.asarray(values, dtype=float)
    return {"mean": float(v.mean()), "std": float(v.std(ddof=1)),
            "min": float(v.min()), "max": float(v.max())}


def stage0(model, evaluate, out, log):
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(42)
    t0 = time.perf_counter()
    pts = nc.sample_in_polygon(N_SAMPLES, model.polygon_local, rng=rng)
    F = evaluate(pts)
    pf = nondominated_mask(F)
    df = pd.DataFrame(F, columns=["f1", "f2", "f3"])
    spearman = df.rank().corr(method="pearson")
    np.save(os.path.join(out, "stage0_samples_pts.npy"), pts)
    np.save(os.path.join(out, "stage0_samples_F.npy"), F)
    np.save(os.path.join(out, "stage0_samples_pf.npy"), pf)
    stats = {
        "n_samples": N_SAMPLES,
        "spearman": {f"{a}-{b}": float(spearman.loc[a, b]) for a, b in [("f1","f2"),("f1","f3"),("f2","f3")]},
        "front_size": int(pf.sum()),
        "ranges": {n: [float(F[:,k].min()), float(F[:,k].max())] for k,n in enumerate(["f1","f2","f3"])},
    }
    with open(os.path.join(out, "stage0_stats.json"), "w", encoding="utf-8") as fh:
        json.dump(stats, fh, ensure_ascii=False, indent=2)
    dt = time.perf_counter() - t0
    log.append(f"[stage0] n={N_SAMPLES} front={int(pf.sum())} time={dt:.3f}s")
    return stats


def run_one(seed, model, evaluate, LO, HI, REF_FRONT, max_gen, out, log):
    cfg = MOEADConfig(x_bounds=model.bounds_x, y_bounds=model.bounds_y,
                      seed=seed, max_gen=max_gen, track_every=TRACK_EVERY)
    traj = []
    def on_gen(gen, archive, n_repl):
        Fn = normalize_fixed(archive.F, LO, HI)
        traj.append([gen, hv_3d(Fn, HV_REF)])
    t0 = time.perf_counter()
    res = MOEAD(cfg, evaluate).run(on_generation=on_gen)
    dt = time.perf_counter() - t0
    AF = res["archive_F"]; AFn = normalize_fixed(AF, LO, HI)
    n_ref = len(REF_FRONT)
    merged_front = nondominated_mask(np.vstack([REF_FRONT, AFn]))
    row = {
        "seed": seed, "max_gen": max_gen, "n_evals": int(res["n_evals"]),
        "total_replacements": int(res["total_replacements"]), "archive_size": int(len(AF)),
        "time_s": round(dt, 3),
        "f1_min": float(AF[:,0].min()), "f1_max": float(AF[:,0].max()),
        "f2_min": float(AF[:,1].min()), "f2_max": float(AF[:,1].max()),
        "f3_min": float(AF[:,2].min()), "f3_max": float(AF[:,2].max()),
        "hv": float(hv_3d(AFn, HV_REF)), "igd": float(igd(AFn, REF_FRONT)),
        "spacing": float(spacing(AFn)),
        "on_merged_front": int(merged_front[n_ref:].sum()),
        "ref_survivors": int(merged_front[:n_ref].sum()),
        "hv_trajectory": traj,
    }
    np.save(os.path.join(out, f"archive_seed{seed}_X.npy"), res["archive_X"])
    np.save(os.path.join(out, f"archive_seed{seed}_F.npy"), AF)
    return row, res["archive_X"], AF


def check_polygons(model, out, seeds):
    ok = True
    for s in seeds:
        Ax = np.load(os.path.join(out, f"archive_seed{s}_X.npy"))
        inside = nc.is_inside_polygon(model.polygon_local, Ax)
        if not inside.all():
            ok = False
            print(f"[WARN] seed={s} has {int((~inside).sum())} archive points outside A3 polygon", flush=True)
    return ok


def run_experiment(model, evaluate, caps, seeds, max_gen, label, subdir, log):
    out = os.path.join(FAST_ROOT, subdir)
    os.makedirs(out, exist_ok=True)
    log.append(f"=== {label}: seeds={seeds} max_gen={max_gen} track_every={TRACK_EVERY} out={out} ===")
    s0 = stage0(model, evaluate, out, log)
    stage0_F = np.load(os.path.join(out, "stage0_samples_F.npy"))
    stage0_pf = np.load(os.path.join(out, "stage0_samples_pf.npy"))
    LO, HI = stage0_F.min(axis=0), stage0_F.max(axis=0)
    REF_FRONT = normalize_fixed(stage0_F[stage0_pf], LO, HI)
    t_all = time.perf_counter()
    rows = []
    for seed in seeds:
        row, Ax, AF = run_one(seed, model, evaluate, LO, HI, REF_FRONT, max_gen, out, log)
        rows.append(row)
        log.append(f"[{label}] seed={seed} archive={row['archive_size']} HV={row['hv']:.4f} "
                   f"IGD={row['igd']:.5f} Spacing={row['spacing']:.4f} on_merged={row['on_merged_front']} "
                   f"n_evals={row['n_evals']} time={row['time_s']:.2f}s")
    keys = ["archive_size","time_s","hv","igd","spacing","on_merged_front","total_replacements","n_evals",
            "f1_min","f1_max","f2_min","f2_max","f3_min","f3_max"]
    summary = {k: agg([r[k] for r in rows]) for k in keys}
    summary["hv_baseline_random_sampling"] = hv_3d(REF_FRONT, HV_REF)
    poly_ok = check_polygons(model, out, seeds)
    dt_all = time.perf_counter() - t_all
    results = {
        "provenance": model.provenance,
        "config": {"label": label, "seeds": seeds, "max_gen": max_gen, "track_every": TRACK_EVERY,
                   "n_samples": N_SAMPLES, "hv_ref": list(HV_REF),
                   "output_dir": out, "total_time_s": round(dt_all, 2),
                   "archive_all_inside_polygon": bool(poly_ok)},
        "model": {
            "search_space": "A3 Polygon (data/nanhu_boundary_polygon.json)",
            "bounds_x": list(model.bounds_x), "bounds_y": list(model.bounds_y),
            "n_polygon_vertices": int(len(model.polygon_local)),
            "demand_ids": model.demand_ids, "competition_ids": model.competition_ids,
            "n_demand": int(len(model.demand_ids)), "n_competition": int(len(model.competition_ids)),
            "n_total_points": int(len(model.demand_ids) + len(model.competition_ids)),
            "penalty_caps": [float(c) for c in caps],
        },
        "stage0": s0, "runs": rows, "summary": summary, "hv_ref": list(HV_REF),
        "norm_lo": LO.tolist(), "norm_hi": HI.tolist(),
    }
    with open(os.path.join(out, "experiment_results.json"), "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
    log.append(f"[{label}] done archive_all_inside_polygon={poly_ok} total_time={dt_all:.2f}s "
               f"n_evals_mean={summary['n_evals']['mean']:.0f} HV_mean={summary['hv']['mean']:.4f}")
    return results, poly_ok


if __name__ == "__main__":
    log = []
    model = nc.build_model()
    evaluate, caps = nc.build_constrained_evaluate(model)
    log.append(f"Model: demand {len(model.demand_ids)} comp {len(model.competition_ids)} "
               f"total {len(model.demand_ids)+len(model.competition_ids)} caps={np.round(caps,2).tolist()}")
    print(log[-1], flush=True)

    q_res, q_ok = run_experiment(model, evaluate, caps, QUICK_SEEDS, QUICK_GEN, "quick_1x50", "quick_1x50", log)
    if not q_ok:
        log.append("[FAIL] Quick experiment constraint check failed - stopping")
        os.makedirs(FAST_ROOT, exist_ok=True)
        with open(os.path.join(FAST_ROOT, "experiment_log.txt"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(log))
        print("\n".join(log), flush=True)
        sys.exit(1)

    d_res, d_ok = run_experiment(model, evaluate, caps, DEMO_SEEDS, DEMO_GEN, "demo_5x100", "demo_5x100", log)
    if d_ok:
        log.append("[OK] Demo experiment constraint check passed")
    else:
        log.append("[FAIL] Demo experiment constraint check failed")

    os.makedirs(FAST_ROOT, exist_ok=True)
    with open(os.path.join(FAST_ROOT, "experiment_log.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(log))
    print("\n".join(log), flush=True)
    sys.exit(0 if d_ok else 2)
