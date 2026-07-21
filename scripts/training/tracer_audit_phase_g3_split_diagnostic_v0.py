#!/usr/bin/env python3
import csv
import math
from pathlib import Path
from collections import defaultdict
from statistics import mean

import numpy as np

IN_CSV = Path("datasets/phase_g/g1_enriched_runtime_feature_table_v0.csv")
G2_CSV = Path("datasets/phase_g/g2_enriched_feature_beta_diagnostic_v0.csv")
OUT_CSV = Path("datasets/phase_g/g3_split_diagnostic_v0.csv")
OUT_MD = Path("reports/phase_g/g3_split_diagnostic_summary_v0.md")

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

def is_numeric_col(rows, col):
    return sum(ff(r.get(col)) is not None for r in rows) >= max(3, len(rows) // 3)

def bad_as_input(col):
    c = col.lower()
    if c in ["tag", "base_tag", "manifest", "d7_csv", "context_mode"]:
        return True
    if c.startswith("target_"):
        return True
    if "true_seed" in c or "dominant_beta" in c:
        return True
    if "summary_csv" in c or "true_csv" in c:
        return True
    return False

def leakage_col(col):
    c = col.lower()
    return c.startswith("rollout_true_metric_") or c.startswith("robust_group_true_metric_") or c.startswith("group_")

def normalize_simplex(P):
    P = np.clip(P, 0.0, None)
    s = P.sum(axis=1, keepdims=True)
    return np.where(s > 1e-12, P / s, np.ones_like(P) / 3.0)

def ridge_fit_predict(Xtr, Ytr, Xte):
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

    reg = np.eye(Xb.shape[1]) * ALPHA
    reg[0, 0] = 0.0

    W = np.linalg.solve(Xb.T @ Xb + reg, Xb.T @ Ytr)
    return normalize_simplex(Xtb @ W)

def build_XY(rows, features):
    X, Y, tags, base_tags = [], [], [], []
    for r in rows:
        y = [ff(r.get(t)) for t in TARGETS]
        if any(v is None for v in y):
            continue
        x = []
        for c in features:
            v = ff(r.get(c))
            x.append(np.nan if v is None else v)
        X.append(x)
        Y.append(y)
        tags.append(r.get("tag", "unknown"))
        base_tags.append(r.get("base_tag", "unknown"))
    return np.asarray(X, dtype=float), np.asarray(Y, dtype=float), np.asarray(tags), np.asarray(base_tags)

def l1_mean(Y, P):
    return float(np.abs(Y - P).sum(axis=1).mean())

def eval_leave_one_condition(rows, features, suite):
    X, Y, tags, base_tags = build_XY(rows, features)
    out = []
    for hold in sorted(set(base_tags)):
        tr = base_tags != hold
        te = base_tags == hold
        P = ridge_fit_predict(X[tr], Y[tr], X[te])
        out.append({
            "suite": suite,
            "split": "leave_one_condition",
            "heldout": hold,
            "n_train": int(tr.sum()),
            "n_test": int(te.sum()),
            "l1": l1_mean(Y[te], P),
            "target_beta": tuple(Y[te].mean(axis=0)),
            "pred_beta": tuple(P.mean(axis=0)),
        })
    return out

def eval_leave_one_rollout(rows, features, suite):
    X, Y, tags, base_tags = build_XY(rows, features)
    out = []
    for hold in sorted(set(tags)):
        tr = tags != hold
        te = tags == hold
        P = ridge_fit_predict(X[tr], Y[tr], X[te])
        out.append({
            "suite": suite,
            "split": "leave_one_rollout",
            "heldout": hold,
            "base_tag": str(base_tags[te][0]),
            "n_train": int(tr.sum()),
            "n_test": int(te.sum()),
            "l1": l1_mean(Y[te], P),
            "target_beta": tuple(Y[te].mean(axis=0)),
            "pred_beta": tuple(P.mean(axis=0)),
        })
    return out

def has_any(col, keys):
    c = col.lower()
    return any(k in c for k in keys)

rows = list(csv.DictReader(open(IN_CSV)))
if not rows:
    raise SystemExit(f"empty input: {IN_CSV}")

all_cols = list(rows[0].keys())
numeric_cols = [c for c in all_cols if not bad_as_input(c) and is_numeric_col(rows, c)]

tracking_energy_proxy = [
    c for c in numeric_cols
    if c in [
        "vx_tracking_error_g1",
        "vx_tracking_abs_error_g1",
        "vx_tracking_ratio_g1",
        "power_per_vx_g1",
        "contact_per_vx_g1",
        "lateral_per_forward_g1",
        "imu_per_vx_g1",
        "x_progress_g1",
        "y_change_g1",
        "abs_y_change_g1",
    ]
]

ram_ref_context = [
    c for c in numeric_cols
    if (
        c.startswith("ram_")
        or c.startswith("ref_")
        or c.startswith("context_frac_")
        or c.startswith("y_")
        or c.startswith("abs_y_")
        or c.startswith("x_")
    )
    and not leakage_col(c)
]

online_safe = [
    c for c in numeric_cols
    if not leakage_col(c)
    and not has_any(c, [
        "rollout_true_metric",
        "robust_group_true_metric",
        "joint_abs_power",
        "contact_force",
        "imu_ang_vel",
        "base_vx",
        "base_vy",
    ])
]

true_metric_upper = [
    c for c in numeric_cols
    if c.startswith("rollout_true_metric_") or c in tracking_energy_proxy
]

suites = {
    "tracking_energy_proxy_g3": tracking_energy_proxy,
    "ram_ref_context_g3": ram_ref_context,
    "online_safe_g3": online_safe,
    "true_metric_upper_bound_g3_leaky": true_metric_upper,
}

results = []
for suite, feats in suites.items():
    feats = sorted(set(feats))
    print(f"[TRACER] suite={suite} features={len(feats)}")
    results.extend(eval_leave_one_condition(rows, feats, suite))
    results.extend(eval_leave_one_rollout(rows, feats, suite))

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
fields = ["suite", "split", "heldout", "base_tag", "n_train", "n_test", "l1", "target_beta", "pred_beta"]
with open(OUT_CSV, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    for r in results:
        rr = dict(r)
        rr["target_beta"] = "(" + ", ".join(f"{x:.4f}" for x in r["target_beta"]) + ")"
        rr["pred_beta"] = "(" + ", ".join(f"{x:.4f}" for x in r["pred_beta"]) + ")"
        rr.setdefault("base_tag", "")
        w.writerow(rr)

by = defaultdict(list)
for r in results:
    by[(r["suite"], r["split"])].append(r)

summary = []
for (suite, split), rs in by.items():
    l1s = [r["l1"] for r in rs]
    summary.append({
        "suite": suite,
        "split": split,
        "mean_l1": mean(l1s),
        "max_l1": max(l1s),
        "worst": max(rs, key=lambda r: r["l1"])["heldout"],
        "best": min(rs, key=lambda r: r["l1"])["heldout"],
    })

summary.sort(key=lambda r: (r["suite"], r["split"]))

OUT_MD.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_MD, "w") as f:
    f.write("# TRACER Phase-G3 Split Diagnostic v0\n\n")
    f.write("This compares leave-one-condition-out vs leave-one-rollout-out validation on G1 enriched runtime features.\n\n")
    f.write(f"- input: `{IN_CSV}`\n")
    f.write(f"- output csv: `{OUT_CSV}`\n")
    f.write(f"- rollouts: `{len(rows)}`\n")
    f.write(f"- alpha: `{ALPHA}`\n\n")

    f.write("## Split summary\n\n")
    f.write("| suite | split | mean L1 | max L1 | worst | best |\n")
    f.write("|---|---|---:|---:|---|---|\n")
    for s in summary:
        f.write(f"| {s['suite']} | {s['split']} | {s['mean_l1']:.6f} | {s['max_l1']:.6f} | {s['worst']} | {s['best']} |\n")

    f.write("\n## Interpretation guide\n\n")
    f.write("- If leave-one-rollout is much better than leave-one-condition, the bottleneck is condition coverage / extrapolation.\n")
    f.write("- If leave-one-rollout is also poor, the features do not explain rollout-level variability.\n")
    f.write("- If even the leaky upper bound is poor under leave-one-condition, collecting anchor conditions near p060/p030 is more valuable than adding another regressor.\n")

print(f"[TRACER] wrote {OUT_CSV}")
print(f"[TRACER] wrote {OUT_MD}")
print("[TRACER] split summary:")
for s in summary:
    print(f"  {s['suite']} / {s['split']}: mean={s['mean_l1']:.6f}, max={s['max_l1']:.6f}, worst={s['worst']}")
