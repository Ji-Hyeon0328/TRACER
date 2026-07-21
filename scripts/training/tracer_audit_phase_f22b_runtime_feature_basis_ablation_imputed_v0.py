#!/usr/bin/env python3
import csv
import json
import math
from pathlib import Path
from collections import defaultdict
from statistics import mean

import numpy as np

IN_CSV = Path("datasets/phase_f/f19_d7_robust_beta_calibration_dataset_v0.csv")
F20_MODEL = Path("models/phase_f/f20_robust_runtime_safe_beta_calibrator_ridge_v0.json")

OUT_CSV = Path("datasets/phase_f/f22b_runtime_feature_basis_ablation_imputed_v0.csv")
OUT_MD = Path("reports/phase_f/f22b_runtime_feature_basis_ablation_imputed_summary_v0.md")

TARGETS = [
    "target_beta_motion_robust_true_metric",
    "target_beta_stability_robust_true_metric",
    "target_beta_energy_robust_true_metric",
]

ALPHA = 1.0

def ff(x):
    try:
        if x is None or x == "":
            return None
        v = float(x)
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return None

def val(r, k, default=0.0):
    v = ff(r.get(k))
    return default if v is None else v

def load_feature_names():
    with open(F20_MODEL) as f:
        m = json.load(f)
    names = list(m.get("feature_names", []))
    if not names:
        raise RuntimeError(f"no feature_names found in {F20_MODEL}")
    return names

def target_ok(r):
    return all(ff(r.get(k)) is not None for k in TARGETS)

def build_matrix(rows, features, derived_fn=None, derived_names=None):
    valid = [r for r in rows if target_ok(r)]
    X = []

    for r in valid:
        xs = []
        for k in features:
            v = ff(r.get(k))
            xs.append(np.nan if v is None else v)
        if derived_fn:
            xs.extend(derived_fn(r))
        X.append(xs)

    X = np.asarray(X, dtype=float)
    feature_names = list(features) + list(derived_names or [])

    finite_frac = np.isfinite(X).mean(axis=0)
    keep = finite_frac > 0.50
    X = X[:, keep]
    kept_names = [n for n, k in zip(feature_names, keep) if k]

    return X, valid, kept_names

def build_targets(rows):
    return np.asarray([[ff(r[k]) for k in TARGETS] for r in rows], dtype=float)

def normalize_simplex(P):
    P = np.clip(P, 0.0, None)
    s = P.sum(axis=1, keepdims=True)
    return np.where(s > 1e-12, P / s, np.ones_like(P) / 3.0)

def ridge_fit_predict(Xtr, Ytr, Xte, alpha=1.0):
    mu = np.nanmean(Xtr, axis=0)
    mu = np.where(np.isfinite(mu), mu, 0.0)

    Xtr_i = np.where(np.isfinite(Xtr), Xtr, mu)
    Xte_i = np.where(np.isfinite(Xte), Xte, mu)

    sd = Xtr_i.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)

    Xtr_n = (Xtr_i - mu) / sd
    Xte_n = (Xte_i - mu) / sd

    Xb = np.concatenate([np.ones((Xtr_n.shape[0], 1)), Xtr_n], axis=1)
    Xtb = np.concatenate([np.ones((Xte_n.shape[0], 1)), Xte_n], axis=1)

    reg = np.eye(Xb.shape[1]) * alpha
    reg[0, 0] = 0.0

    W = np.linalg.solve(Xb.T @ Xb + reg, Xb.T @ Ytr)
    return normalize_simplex(Xtb @ W)

def rmse_vec(Y, P):
    return np.sqrt(((P - Y) ** 2).mean(axis=0))

def l1_mean(Y, P):
    return np.abs(P - Y).sum(axis=1).mean()

def eval_suite(name, rows, features, derived_fn=None, derived_names=None):
    X, vr, kept_names = build_matrix(rows, features, derived_fn, derived_names)
    Y = build_targets(vr)
    tags = np.asarray([r.get("f19_base_tag", "unknown") for r in vr])
    uniq = sorted(set(tags))

    results = []
    for hold in uniq:
        tr = tags != hold
        te = tags == hold
        if tr.sum() == 0 or te.sum() == 0:
            continue

        P = ridge_fit_predict(X[tr], Y[tr], X[te], ALPHA)
        yt = Y[te].mean(axis=0)
        pt = P.mean(axis=0)
        rmse = rmse_vec(Y[te], P)

        results.append({
            "suite": name,
            "heldout": hold,
            "n_features": len(kept_names),
            "n_train": int(tr.sum()),
            "n_test": int(te.sum()),
            "target_motion": yt[0],
            "target_stability": yt[1],
            "target_energy": yt[2],
            "pred_motion": pt[0],
            "pred_stability": pt[1],
            "pred_energy": pt[2],
            "loto_l1": float(l1_mean(Y[te], P)),
            "rmse_motion": float(rmse[0]),
            "rmse_stability": float(rmse[1]),
            "rmse_energy": float(rmse[2]),
            "kept_features": ";".join(kept_names),
        })

    return results

def keep_existing(cols, all_features):
    return [c for c in cols if c in all_features]

def derived_lateral(r):
    y = val(r, "y")
    ym = val(r, "y_mean")
    yaw = val(r, "ref_yaw_rate_mean")
    lat = val(r, "rollout_mean_abs_y", val(r, "mean_abs_y_context"))
    stab = val(r, "stability_score")
    energy = val(r, "energy_score")
    motion = val(r, "motion_score")
    vx = val(r, "ref_vx_mean")
    clr = val(r, "ref_clearance_mean")

    ay = abs(y)
    aym = abs(ym)

    return [
        ay,
        y * y,
        ym * ym,
        aym,
        y * yaw,
        ym * yaw,
        lat * lat,
        y * lat,
        ay * lat,
        y * stab,
        ay * stab,
        y * energy,
        ay * energy,
        y * motion,
        ay * motion,
        vx * y,
        vx * ay,
        clr * lat,
    ]

DERIVED_NAMES = [
    "abs_y", "y2", "y_mean2", "abs_y_mean",
    "y_x_ref_yaw", "ymean_x_ref_yaw",
    "lat2", "y_x_lat", "abs_y_x_lat",
    "y_x_stability", "abs_y_x_stability",
    "y_x_energy", "abs_y_x_energy",
    "y_x_motion", "abs_y_x_motion",
    "vx_x_y", "vx_x_abs_y", "clearance_x_lat",
]

rows = list(csv.DictReader(open(IN_CSV)))
if not rows:
    raise SystemExit(f"empty input: {IN_CSV}")

all_features = load_feature_names()

suites = {
    "d7_beta_only": keep_existing([
        "pred_beta_motion", "pred_beta_stability", "pred_beta_energy",
        "raw_beta_motion", "raw_beta_stability", "raw_beta_energy",
        "prior_beta_motion", "prior_beta_stability", "prior_beta_energy",
        "actual_beta_motion", "actual_beta_stability", "actual_beta_energy",
    ], all_features),
    "lateral_runtime": keep_existing([
        "y", "y_mean",
        "rollout_mean_abs_y", "rollout_max_abs_y",
        "mean_abs_y_context", "max_abs_y_context",
        "ref_yaw_rate_mean", "stability_score", "hold_drift",
    ], all_features),
    "score_proxy": keep_existing([
        "motion_score", "stability_score", "energy_score",
        "effort_proxy", "hold_drift",
    ], all_features),
    "ram_ref": keep_existing([
        "ram_slip_proxy_mean", "ram_roughness_proxy_mean", "ram_sigma_mean",
        "ref_vx_mean", "ref_yaw_rate_mean", "ref_clearance_mean",
    ], all_features),
    "all_runtime_safe": list(all_features),
}

all_results = []
for name, feats in suites.items():
    if not feats:
        print(f"[TRACER][WARN] skip empty suite {name}")
        continue
    all_results.extend(eval_suite(name, rows, feats))

all_results.extend(
    eval_suite(
        "all_runtime_safe_plus_lateral_basis",
        rows,
        list(all_features),
        derived_fn=derived_lateral,
        derived_names=DERIVED_NAMES,
    )
)

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
fields = list(all_results[0].keys())
with open(OUT_CSV, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(all_results)

by_suite = defaultdict(list)
for r in all_results:
    by_suite[r["suite"]].append(r)

summary = []
for suite, rs in by_suite.items():
    l1s = [float(r["loto_l1"]) for r in rs]
    summary.append({
        "suite": suite,
        "n_features": int(rs[0]["n_features"]),
        "mean_loto_l1": mean(l1s),
        "max_loto_l1": max(l1s),
        "worst_condition": max(rs, key=lambda x: x["loto_l1"])["heldout"],
        "best_condition": min(rs, key=lambda x: x["loto_l1"])["heldout"],
    })

summary.sort(key=lambda r: r["mean_loto_l1"])

OUT_MD.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_MD, "w") as f:
    f.write("# TRACER Phase-F22b Runtime Feature Basis Ablation with Imputation v0\n\n")
    f.write("This repeats F22 with train-mean feature imputation so all_runtime_safe and nonlinear lateral-basis suites are evaluated instead of being skipped by strict row filtering.\n\n")
    f.write(f"- input: `{IN_CSV}`\n")
    f.write(f"- F20 model feature source: `{F20_MODEL}`\n")
    f.write(f"- output csv: `{OUT_CSV}`\n")
    f.write(f"- alpha: `{ALPHA}`\n")
    f.write(f"- F20 feature count: `{len(all_features)}`\n\n")

    f.write("## Suite summary\n\n")
    f.write("| rank | suite | n_features | mean LOTO L1 | max LOTO L1 | worst condition | best condition |\n")
    f.write("|---:|---|---:|---:|---:|---|---|\n")
    for i, s in enumerate(summary, 1):
        f.write(
            f"| {i} | {s['suite']} | {s['n_features']} | "
            f"{s['mean_loto_l1']:.6f} | {s['max_loto_l1']:.6f} | "
            f"{s['worst_condition']} | {s['best_condition']} |\n"
        )

    f.write("\n## Per-condition LOTO for top suites\n\n")
    top_names = [s["suite"] for s in summary[:3]]
    f.write("| suite | heldout | target beta | pred beta | LOTO L1 |\n")
    f.write("|---|---|---|---|---:|\n")
    for suite in top_names:
        for r in sorted(by_suite[suite], key=lambda x: x["heldout"]):
            tgt = f"({r['target_motion']:.4f}, {r['target_stability']:.4f}, {r['target_energy']:.4f})"
            pred = f"({r['pred_motion']:.4f}, {r['pred_stability']:.4f}, {r['pred_energy']:.4f})"
            f.write(f"| {suite} | {r['heldout']} | {tgt} | {pred} | {r['loto_l1']:.6f} |\n")

    f.write("\n## Safe interpretation\n\n")
    f.write("If all_runtime_safe_plus_lateral_basis substantially improves over all_runtime_safe, nonlinear lateral interactions are useful. If not, the remaining bottleneck is missing RAM/context information rather than simply the linear ridge basis.\n")

print(f"[TRACER] wrote {OUT_CSV}")
print(f"[TRACER] wrote {OUT_MD}")
print("[TRACER] suite summary:")
for s in summary:
    print(f"  {s['suite']}: n_features={s['n_features']} mean_loto_l1={s['mean_loto_l1']:.6f}, max={s['max_loto_l1']:.6f}, worst={s['worst_condition']}")
