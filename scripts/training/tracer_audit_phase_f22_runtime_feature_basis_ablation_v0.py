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

OUT_CSV = Path("datasets/phase_f/f22_runtime_feature_basis_ablation_v0.csv")
OUT_MD = Path("reports/phase_f/f22_runtime_feature_basis_ablation_summary_v0.md")

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

def load_feature_names():
    with open(F20_MODEL) as f:
        m = json.load(f)
    return list(m.get("feature_names", []))

def ridge_fit_predict(Xtr, Ytr, Xte, alpha=1.0):
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0)
    sd[sd < 1e-12] = 1.0

    Xtrn = (Xtr - mu) / sd
    Xten = (Xte - mu) / sd

    Xb = np.concatenate([np.ones((Xtrn.shape[0], 1)), Xtrn], axis=1)
    Xtb = np.concatenate([np.ones((Xten.shape[0], 1)), Xten], axis=1)

    reg = np.eye(Xb.shape[1]) * alpha
    reg[0, 0] = 0.0

    W = np.linalg.solve(Xb.T @ Xb + reg, Xb.T @ Ytr)
    P = Xtb @ W

    P = np.clip(P, 0.0, None)
    s = P.sum(axis=1, keepdims=True)
    P = np.where(s > 1e-12, P / s, np.ones_like(P) / 3.0)
    return P

def build_matrix(rows, features, derived=None):
    X = []
    valid_rows = []
    for r in rows:
        vals = []
        ok = True
        for k in features:
            v = ff(r.get(k))
            if v is None:
                ok = False
                break
            vals.append(v)

        if ok and derived:
            extra = derived(r)
            if extra is None:
                ok = False
            else:
                vals.extend(extra)

        if ok:
            X.append(vals)
            valid_rows.append(r)

    return np.asarray(X, dtype=float), valid_rows

def build_targets(rows):
    Y = []
    for r in rows:
        vals = [ff(r.get(k)) for k in TARGETS]
        if any(v is None for v in vals):
            raise RuntimeError("missing target")
        Y.append(vals)
    return np.asarray(Y, dtype=float)

def l1_mean(Y, P):
    return np.abs(P - Y).sum(axis=1).mean()

def rmse_vec(Y, P):
    return np.sqrt(((P - Y) ** 2).mean(axis=0))

def eval_suite(name, rows, features, derived=None, derived_feature_names=None):
    X, vr = build_matrix(rows, features, derived=derived)
    if X.shape[0] == 0:
        return []

    Y = build_targets(vr)
    tags = np.asarray([r.get("f19_base_tag", "unknown") for r in vr])
    uniq = sorted(set(tags))

    out = []
    for hold in uniq:
        tr = tags != hold
        te = tags == hold

        if tr.sum() == 0 or te.sum() == 0:
            continue

        P = ridge_fit_predict(X[tr], Y[tr], X[te], alpha=ALPHA)
        yt = Y[te].mean(axis=0)
        pt = P.mean(axis=0)

        out.append({
            "suite": name,
            "heldout": hold,
            "n_features": len(features) + (len(derived_feature_names or []) if derived else 0),
            "n_train": int(tr.sum()),
            "n_test": int(te.sum()),
            "target_motion": yt[0],
            "target_stability": yt[1],
            "target_energy": yt[2],
            "pred_motion": pt[0],
            "pred_stability": pt[1],
            "pred_energy": pt[2],
            "loto_l1": float(l1_mean(Y[te], P)),
            "rmse_motion": float(rmse_vec(Y[te], P)[0]),
            "rmse_stability": float(rmse_vec(Y[te], P)[1]),
            "rmse_energy": float(rmse_vec(Y[te], P)[2]),
        })

    return out

rows = list(csv.DictReader(open(IN_CSV)))
if not rows:
    raise SystemExit(f"empty input: {IN_CSV}")

f20_features = load_feature_names()

def keep_existing(cols):
    return [c for c in cols if c in f20_features]

suites = {}

suites["d7_beta_only"] = keep_existing([
    "pred_beta_motion", "pred_beta_stability", "pred_beta_energy",
    "raw_beta_motion", "raw_beta_stability", "raw_beta_energy",
    "prior_beta_motion", "prior_beta_stability", "prior_beta_energy",
    "actual_beta_motion", "actual_beta_stability", "actual_beta_energy",
])

suites["lateral_runtime"] = keep_existing([
    "y", "y_mean",
    "rollout_mean_abs_y", "rollout_max_abs_y",
    "mean_abs_y_context", "max_abs_y_context",
    "ref_yaw_rate_mean", "stability_score", "hold_drift",
])

suites["score_proxy"] = keep_existing([
    "motion_score", "stability_score", "energy_score",
    "effort_proxy", "hold_drift",
])

suites["ram_ref"] = keep_existing([
    "ram_slip_proxy_mean", "ram_roughness_proxy_mean", "ram_sigma_mean",
    "ref_vx_mean", "ref_yaw_rate_mean", "ref_clearance_mean",
])

suites["all_runtime_safe"] = list(f20_features)

base_aug_features = list(f20_features)

def derived_lateral(r):
    y = ff(r.get("y"))
    ym = ff(r.get("y_mean"))
    yaw = ff(r.get("ref_yaw_rate_mean"))
    lat = ff(r.get("rollout_mean_abs_y")) or ff(r.get("mean_abs_y_context"))
    stab = ff(r.get("stability_score"))
    energy = ff(r.get("energy_score"))
    motion = ff(r.get("motion_score"))

    if y is None or ym is None or yaw is None:
        return None

    ay = abs(y)
    aym = abs(ym)
    vals = [
        ay,
        y * y,
        ym * ym,
        aym,
        y * yaw,
        ym * yaw,
    ]

    if lat is not None:
        vals += [lat * lat, y * lat, ay * lat]
    else:
        vals += [0.0, 0.0, 0.0]

    if stab is not None:
        vals += [y * stab, ay * stab]
    else:
        vals += [0.0, 0.0]

    if energy is not None:
        vals += [y * energy, ay * energy]
    else:
        vals += [0.0, 0.0]

    if motion is not None:
        vals += [y * motion, ay * motion]
    else:
        vals += [0.0, 0.0]

    return vals

derived_names = [
    "abs_y", "y2", "y_mean2", "abs_y_mean",
    "y_x_ref_yaw", "ymean_x_ref_yaw",
    "lat2", "y_x_lat", "abs_y_x_lat",
    "y_x_stability", "abs_y_x_stability",
    "y_x_energy", "abs_y_x_energy",
    "y_x_motion", "abs_y_x_motion",
]

all_results = []
for name, feats in suites.items():
    if not feats:
        continue
    all_results.extend(eval_suite(name, rows, feats))

all_results.extend(
    eval_suite(
        "all_runtime_safe_plus_lateral_basis",
        rows,
        base_aug_features,
        derived=derived_lateral,
        derived_feature_names=derived_names,
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
        "n_features": rs[0]["n_features"],
        "mean_loto_l1": mean(l1s),
        "max_loto_l1": max(l1s),
        "best_condition": min(rs, key=lambda x: x["loto_l1"])["heldout"],
        "worst_condition": max(rs, key=lambda x: x["loto_l1"])["heldout"],
    })

summary.sort(key=lambda r: r["mean_loto_l1"])

OUT_MD.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_MD, "w") as f:
    f.write("# TRACER Phase-F22 Runtime Feature Basis Ablation v0\n\n")
    f.write("This tests whether the F20 robust beta errors are mainly due to weak feature expressiveness or linear basis limitations.\n\n")
    f.write(f"- input: `{IN_CSV}`\n")
    f.write(f"- F20 model feature source: `{F20_MODEL}`\n")
    f.write(f"- output csv: `{OUT_CSV}`\n")
    f.write(f"- alpha: `{ALPHA}`\n\n")

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
    f.write("If the lateral-basis suite improves strongly over all_runtime_safe, the current runtime signal is useful but needs nonlinear/interaction features. If it does not improve, the bottleneck is likely missing context/RAM information rather than just the linear ridge basis.\n")

print(f"[TRACER] wrote {OUT_CSV}")
print(f"[TRACER] wrote {OUT_MD}")
print("[TRACER] suite summary:")
for s in summary:
    print(f"  {s['suite']}: mean_loto_l1={s['mean_loto_l1']:.6f}, max={s['max_loto_l1']:.6f}, worst={s['worst_condition']}")
