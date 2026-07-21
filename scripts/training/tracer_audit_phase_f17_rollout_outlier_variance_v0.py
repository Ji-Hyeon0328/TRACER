#!/usr/bin/env python3
import csv
import math
import re
from pathlib import Path
from collections import defaultdict
from statistics import mean, pstdev

IN_CSV = Path("datasets/phase_f/f4_true_metric_training_table_v0.csv")
OUT_CSV = Path("datasets/phase_f/f17_rollout_outlier_variance_audit_v0.csv")
OUT_MD = Path("reports/phase_f/f17_rollout_outlier_variance_audit_summary_v0.md")

BETA_FLOOR = 0.05

METRICS = [
    "base_vx_mean",
    "base_vy_abs_mean",
    "joint_abs_power_mean",
    "imu_ang_vel_norm_mean",
    "contact_force_z_sum_mean",
]

def base_tag(tag):
    return re.sub(r"_r[0-9]+$", "", str(tag))

def ff(x, default=None):
    try:
        if x is None or x == "":
            return default
        v = float(x)
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return default

def good_high(vals):
    lo, hi = min(vals), max(vals)
    if abs(hi - lo) < 1e-12:
        return [0.5 for _ in vals]
    return [(v - lo) / (hi - lo) for v in vals]

def good_low(vals):
    lo, hi = min(vals), max(vals)
    if abs(hi - lo) < 1e-12:
        return [0.5 for _ in vals]
    return [(hi - v) / (hi - lo) for v in vals]

def build_group_betas(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[base_tag(r["tag"])].append(r)

    group_rows = []
    for g in sorted(groups):
        rs = groups[g]
        gr = {"base_tag": g, "n_rollouts": len(rs)}
        for k in METRICS:
            vals = [ff(r.get(k)) for r in rs]
            vals = [v for v in vals if v is not None]
            if not vals:
                gr[f"{k}_mean"] = None
            else:
                gr[f"{k}_mean"] = mean(vals)
        group_rows.append(gr)

    # Keep only groups with complete metrics.
    group_rows = [
        g for g in group_rows
        if all(g.get(f"{k}_mean") is not None for k in METRICS)
    ]

    vx = [g["base_vx_mean_mean"] for g in group_rows]
    lat = [g["base_vy_abs_mean_mean"] for g in group_rows]
    power = [g["joint_abs_power_mean_mean"] for g in group_rows]
    imu = [g["imu_ang_vel_norm_mean_mean"] for g in group_rows]
    contact = [g["contact_force_z_sum_mean_mean"] for g in group_rows]

    motion_s = good_high(vx)
    lat_s = good_low(lat)
    energy_s = good_low(power)
    imu_s = good_low(imu)
    contact_s = good_low(contact)

    out = {}
    for i, g in enumerate(group_rows):
        stability = 0.45 * lat_s[i] + 0.40 * imu_s[i] + 0.15 * contact_s[i]

        wm = motion_s[i] + BETA_FLOOR
        ws = stability + BETA_FLOOR
        we = energy_s[i] + BETA_FLOOR
        z = wm + ws + we

        out[g["base_tag"]] = {
            "beta_motion": wm / z,
            "beta_stability": ws / z,
            "beta_energy": we / z,
            "motion_score": motion_s[i],
            "stability_score": stability,
            "energy_score": energy_s[i],
        }

    return out

def l1(a, b):
    return abs(a["beta_motion"] - b["beta_motion"]) + abs(a["beta_stability"] - b["beta_stability"]) + abs(a["beta_energy"] - b["beta_energy"])

rows = list(csv.DictReader(open(IN_CSV)))
if not rows:
    raise SystemExit(f"empty input: {IN_CSV}")

for r in rows:
    r["base_tag"] = base_tag(r.get("tag", "unknown"))

groups = defaultdict(list)
for r in rows:
    groups[r["base_tag"]].append(r)

full_beta = build_group_betas(rows)

# Group metric stats
stats = {}
for g, rs in groups.items():
    stats[g] = {}
    for k in METRICS:
        vals = [ff(r.get(k)) for r in rs]
        vals = [v for v in vals if v is not None]
        mu = mean(vals) if vals else None
        sd = pstdev(vals) if vals and len(vals) > 1 else 0.0
        stats[g][k] = {"mean": mu, "std": sd}

out_rows = []
for idx, r in enumerate(rows):
    g = r["base_tag"]

    z_abs_vals = []
    metric_details = {}
    for k in METRICS:
        v = ff(r.get(k))
        mu = stats[g][k]["mean"]
        sd = stats[g][k]["std"]
        if v is None or mu is None or sd <= 1e-12:
            z = 0.0
        else:
            z = (v - mu) / sd
        metric_details[f"{k}_z"] = z
        metric_details[f"{k}_abs_z"] = abs(z)
        z_abs_vals.append(abs(z))

    rows_minus = [rr for j, rr in enumerate(rows) if j != idx]
    loo_beta = build_group_betas(rows_minus)

    if g in full_beta and g in loo_beta:
        beta_delta_l1 = l1(full_beta[g], loo_beta[g])
        loo_b = loo_beta[g]
        full_b = full_beta[g]
    else:
        beta_delta_l1 = 0.0
        loo_b = {"beta_motion": "", "beta_stability": "", "beta_energy": ""}
        full_b = full_beta.get(g, {"beta_motion": "", "beta_stability": "", "beta_energy": ""})

    out = {
        "tag": r.get("tag", ""),
        "base_tag": g,
        "n_group_rollouts": len(groups[g]),
        "mean_abs_metric_z": mean(z_abs_vals) if z_abs_vals else 0.0,
        "max_abs_metric_z": max(z_abs_vals) if z_abs_vals else 0.0,
        "beta_leave_one_out_delta_l1": beta_delta_l1,
        "full_beta_motion": full_b["beta_motion"],
        "full_beta_stability": full_b["beta_stability"],
        "full_beta_energy": full_b["beta_energy"],
        "loo_beta_motion": loo_b["beta_motion"],
        "loo_beta_stability": loo_b["beta_stability"],
        "loo_beta_energy": loo_b["beta_energy"],
    }

    for k in METRICS:
        out[k] = r.get(k, "")
    out.update(metric_details)
    out_rows.append(out)

out_rows.sort(key=lambda r: (float(r["beta_leave_one_out_delta_l1"]), float(r["max_abs_metric_z"])), reverse=True)

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
fields = list(out_rows[0].keys())
with open(OUT_CSV, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(out_rows)

# Group summary
group_summary = []
for g in sorted(groups):
    rs = groups[g]
    deltas = [float(r["beta_leave_one_out_delta_l1"]) for r in out_rows if r["base_tag"] == g]
    maxzs = [float(r["max_abs_metric_z"]) for r in out_rows if r["base_tag"] == g]
    group_summary.append({
        "base_tag": g,
        "n": len(rs),
        "beta_delta_l1_mean": mean(deltas) if deltas else 0.0,
        "beta_delta_l1_max": max(deltas) if deltas else 0.0,
        "max_abs_metric_z_max": max(maxzs) if maxzs else 0.0,
        "beta_motion": full_beta[g]["beta_motion"],
        "beta_stability": full_beta[g]["beta_stability"],
        "beta_energy": full_beta[g]["beta_energy"],
    })

OUT_MD.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_MD, "w") as f:
    f.write("# TRACER Phase-F17 Rollout Outlier / Variance Audit v0\n\n")
    f.write("This audits per-rollout metric outliers and how much each rollout changes its grouped true-metric beta target if left out.\n\n")
    f.write(f"- input: `{IN_CSV}`\n")
    f.write(f"- output csv: `{OUT_CSV}`\n")
    f.write(f"- rollouts: `{len(rows)}`\n")
    f.write(f"- groups: `{len(groups)}`\n\n")

    f.write("## Group-level sensitivity\n\n")
    f.write("| base_tag | n | beta | mean LOO beta delta L1 | max LOO beta delta L1 | max metric abs z |\n")
    f.write("|---|---:|---|---:|---:|---:|\n")
    for s in sorted(group_summary, key=lambda x: x["beta_delta_l1_max"], reverse=True):
        beta = f"({s['beta_motion']:.4f}, {s['beta_stability']:.4f}, {s['beta_energy']:.4f})"
        f.write(
            f"| {s['base_tag']} | {s['n']} | {beta} | "
            f"{s['beta_delta_l1_mean']:.6f} | {s['beta_delta_l1_max']:.6f} | {s['max_abs_metric_z_max']:.4f} |\n"
        )

    f.write("\n## Top rollout influences\n\n")
    f.write("| rank | tag | base_tag | LOO beta delta L1 | max abs z | mean abs z | vx | lat_abs | power | imu | contact |\n")
    f.write("|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for i, r in enumerate(out_rows[:15], 1):
        f.write(
            f"| {i} | {r['tag']} | {r['base_tag']} | "
            f"{float(r['beta_leave_one_out_delta_l1']):.6f} | "
            f"{float(r['max_abs_metric_z']):.4f} | "
            f"{float(r['mean_abs_metric_z']):.4f} | "
            f"{float(r['base_vx_mean']):.6f} | "
            f"{float(r['base_vy_abs_mean']):.6f} | "
            f"{float(r['joint_abs_power_mean']):.6f} | "
            f"{float(r['imu_ang_vel_norm_mean']):.6f} | "
            f"{float(r['contact_force_z_sum_mean']):.6f} |\n"
        )

    f.write("\n## Safe interpretation\n\n")
    f.write("Large leave-one-out beta delta means the grouped beta target is sensitive to a single rollout. If an extreme rollout also has large metric z-scores, treat that condition as unstable/noisy and collect more repeats or use a robust aggregation rule before training a deployable Objective Selector.\n")

print(f"[TRACER] wrote {OUT_CSV}")
print(f"[TRACER] wrote {OUT_MD}")
print("[TRACER] top rollout influences:")
for r in out_rows[:10]:
    print(
        f"  {r['tag']}: base={r['base_tag']} "
        f"delta_l1={float(r['beta_leave_one_out_delta_l1']):.6f} "
        f"max_abs_z={float(r['max_abs_metric_z']):.4f}"
    )
