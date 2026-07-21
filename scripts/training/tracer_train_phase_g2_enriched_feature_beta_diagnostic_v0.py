#!/usr/bin/env python3
import csv
import math
from pathlib import Path
from collections import defaultdict
from statistics import mean

import numpy as np

IN_CSV = Path("datasets/phase_g/g1_enriched_runtime_feature_table_v0.csv")
OUT_CSV = Path("datasets/phase_g/g2_enriched_feature_beta_diagnostic_v0.csv")
OUT_MD = Path("reports/phase_g/g2_enriched_feature_beta_diagnostic_summary_v0.md")

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
    n = 0
    for r in rows:
        if ff(r.get(col)) is not None:
            n += 1
    return n >= max(3, len(rows) // 3)

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
    X = []
    Y = []
    tags = []
    valid = []

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
        tags.append(r.get("base_tag", "unknown"))
        valid.append(r)

    return np.asarray(X, dtype=float), np.asarray(Y, dtype=float), np.asarray(tags), valid

def l1_mean(Y, P):
    return np.abs(Y - P).sum(axis=1).mean()

def rmse_vec(Y, P):
    return np.sqrt(((Y - P) ** 2).mean(axis=0))

def eval_suite(name, rows, features):
    if not features:
        return []

    X, Y, tags, valid = build_XY(rows, features)
    uniq = sorted(set(tags))

    out = []
    for hold in uniq:
        tr = tags != hold
        te = tags == hold
        if tr.sum() == 0 or te.sum() == 0:
            continue

        P = ridge_fit_predict(X[tr], Y[tr], X[te])
        yt = Y[te].mean(axis=0)
        pt = P.mean(axis=0)
        rmse = rmse_vec(Y[te], P)

        out.append({
            "suite": name,
            "heldout": hold,
            "n_features": len(features),
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
            "features": ";".join(features),
        })

    return out

def has_any(col, keys):
    c = col.lower()
    return any(k in c for k in keys)

def bad_as_input(col):
    c = col.lower()

    if c in ["tag", "base_tag", "manifest", "d7_csv", "context_mode"]:
        return True
    if c.startswith("target_"):
        return True
    if "true_seed" in c:
        return True
    if "dominant_beta" in c:
        return True
    if "summary_csv" in c or "true_csv" in c:
        return True
    return False

def leakage_col(col):
    c = col.lower()
    if c.startswith("rollout_true_metric_"):
        return True
    if c.startswith("robust_group_true_metric_"):
        return True
    if c.startswith("group_"):
        return True
    return False

rows = list(csv.DictReader(open(IN_CSV)))
if not rows:
    raise SystemExit(f"empty input: {IN_CSV}")

all_cols = list(rows[0].keys())
numeric_cols = [c for c in all_cols if not bad_as_input(c) and is_numeric_col(rows, c)]

d7_beta_cols = [
    c for c in numeric_cols
    if has_any(c, ["pred_beta_", "raw_beta_", "prior_beta_", "actual_beta_", "err_beta_"])
    and not c.startswith("target_")
]

ram_ref_context_cols = [
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

tracking_energy_proxy_cols = [
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

online_safe_enriched_cols = [
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

# This suite is intentionally diagnostic only: these columns are not deployable as-is.
true_metric_upper_bound_cols = [
    c for c in numeric_cols
    if c.startswith("rollout_true_metric_")
    or c in tracking_energy_proxy_cols
]

all_nonleak_enriched_cols = [
    c for c in numeric_cols
    if not leakage_col(c)
    and not c.startswith("rollout_true_metric_")
    and not c.startswith("robust_group_true_metric_")
]

suites = {
    "d7_beta_only_g2": d7_beta_cols,
    "ram_ref_context_g2": ram_ref_context_cols,
    "tracking_energy_proxy_g2_diagnostic": tracking_energy_proxy_cols,
    "online_safe_enriched_g2": online_safe_enriched_cols,
    "all_nonleak_enriched_g2": all_nonleak_enriched_cols,
    "true_metric_upper_bound_g2_leaky": true_metric_upper_bound_cols,
}

results = []
for name, feats in suites.items():
    feats = sorted(set(feats))
    print(f"[TRACER] suite={name} features={len(feats)}")
    results.extend(eval_suite(name, rows, feats))

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
fields = list(results[0].keys())
with open(OUT_CSV, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(results)

by_suite = defaultdict(list)
for r in results:
    by_suite[r["suite"]].append(r)

summary = []
for s, rs in by_suite.items():
    l1s = [float(r["loto_l1"]) for r in rs]
    summary.append({
        "suite": s,
        "n_features": int(rs[0]["n_features"]),
        "mean_loto_l1": mean(l1s),
        "max_loto_l1": max(l1s),
        "worst_condition": max(rs, key=lambda x: float(x["loto_l1"]))["heldout"],
        "best_condition": min(rs, key=lambda x: float(x["loto_l1"]))["heldout"],
    })

summary.sort(key=lambda r: r["mean_loto_l1"])

OUT_MD.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_MD, "w") as f:
    f.write("# TRACER Phase-G2 Enriched Feature Beta Diagnostic v0\n\n")
    f.write("This evaluates enriched rollout-level RAM/context feature suites for robust true-metric beta prediction using leave-one-base-condition-out validation.\n\n")
    f.write(f"- input: `{IN_CSV}`\n")
    f.write(f"- output csv: `{OUT_CSV}`\n")
    f.write(f"- rollout rows: `{len(rows)}`\n")
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
    top = [s["suite"] for s in summary[:4]]
    f.write("| suite | heldout | target beta | pred beta | LOTO L1 |\n")
    f.write("|---|---|---|---|---:|\n")
    for suite in top:
        for r in sorted(by_suite[suite], key=lambda x: x["heldout"]):
            tgt = f"({float(r['target_motion']):.4f}, {float(r['target_stability']):.4f}, {float(r['target_energy']):.4f})"
            pred = f"({float(r['pred_motion']):.4f}, {float(r['pred_stability']):.4f}, {float(r['pred_energy']):.4f})"
            f.write(f"| {suite} | {r['heldout']} | {tgt} | {pred} | {float(r['loto_l1']):.6f} |\n")

    f.write("\n## Suite meanings\n\n")
    f.write("- `online_safe_enriched_g2`: excludes direct rollout true metrics and target labels.\n")
    f.write("- `tracking_energy_proxy_g2_diagnostic`: uses features such as velocity tracking ratio and power/contact per velocity. These are diagnostic now, but can become online temporal-window features later.\n")
    f.write("- `true_metric_upper_bound_g2_leaky`: intentionally uses rollout true metrics. This is an upper bound only and must not be deployed.\n\n")

    f.write("## Safe interpretation\n\n")
    f.write("If online-safe or tracking-proxy suites improve over F20/F22b, Phase-G features are useful. If only the leaky true-metric upper bound improves, the current runtime logs still lack deployable explanatory features.\n")

print(f"[TRACER] wrote {OUT_CSV}")
print(f"[TRACER] wrote {OUT_MD}")
print("[TRACER] suite summary:")
for s in summary:
    print(
        f"  {s['suite']}: features={s['n_features']} "
        f"mean_loto_l1={s['mean_loto_l1']:.6f} "
        f"max={s['max_loto_l1']:.6f} worst={s['worst_condition']}"
    )
