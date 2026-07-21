#!/usr/bin/env python3
import csv
import math
import re
from pathlib import Path
from statistics import mean, pstdev
from collections import defaultdict

IN_CSV = Path("datasets/phase_f/f4_true_metric_training_table_v0.csv")
OUT_GROUP_CSV = Path("datasets/phase_f/f12_grouped_true_metric_beta_targets_v0.csv")
OUT_ROLLOUT_CSV = Path("datasets/phase_f/f12_rollout_with_group_beta_targets_v0.csv")
OUT_MD = Path("reports/phase_f/f12_grouped_true_metric_beta_targets_summary_v0.md")

BETA_FLOOR = 0.05

METRICS = [
    "base_vx_mean",
    "base_vy_abs_mean",
    "joint_abs_power_mean",
    "imu_ang_vel_norm_mean",
    "contact_force_z_sum_mean",
]

def infer_reset_y(tag):
    if tag == "clean":
        return 0.0
    m = re.match(r"m([0-9]+)$", tag)
    if m:
        return -float(m.group(1)) / 100.0
    m = re.match(r"p([0-9]+)$", tag)
    if m:
        return float(m.group(1)) / 100.0
    return ""

def base_tag(tag):
    return re.sub(r"_r[0-9]+$", "", tag)

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

def fmt(x):
    if x is None or x == "":
        return ""
    return f"{float(x):.6f}"

rows = list(csv.DictReader(open(IN_CSV)))
if not rows:
    raise SystemExit(f"empty input: {IN_CSV}")

for r in rows:
    r["base_tag"] = base_tag(r.get("tag", "unknown"))

groups = defaultdict(list)
for r in rows:
    groups[r["base_tag"]].append(r)

group_rows = []
for g, rs in sorted(groups.items()):
    out = {
        "base_tag": g,
        "reset_y": infer_reset_y(g),
        "n_rollouts": len(rs),
    }

    for k in METRICS:
        vals = [ff(r.get(k)) for r in rs]
        vals = [v for v in vals if v is not None]
        if not vals:
            out[f"{k}_mean"] = ""
            out[f"{k}_std"] = ""
            out[f"{k}_min"] = ""
            out[f"{k}_max"] = ""
        else:
            out[f"{k}_mean"] = mean(vals)
            out[f"{k}_std"] = pstdev(vals) if len(vals) > 1 else 0.0
            out[f"{k}_min"] = min(vals)
            out[f"{k}_max"] = max(vals)

    group_rows.append(out)

# Group-level normalized objective scores
vx = [float(r["base_vx_mean_mean"]) for r in group_rows]
lat = [float(r["base_vy_abs_mean_mean"]) for r in group_rows]
power = [float(r["joint_abs_power_mean_mean"]) for r in group_rows]
imu = [float(r["imu_ang_vel_norm_mean_mean"]) for r in group_rows]
contact = [float(r["contact_force_z_sum_mean_mean"]) for r in group_rows]

motion_s = good_high(vx)
lat_s = good_low(lat)
energy_s = good_low(power)
imu_s = good_low(imu)
contact_s = good_low(contact)

for i, r in enumerate(group_rows):
    r["group_motion_score"] = motion_s[i]
    r["group_lateral_score"] = lat_s[i]
    r["group_energy_score"] = energy_s[i]
    r["group_imu_stability_score"] = imu_s[i]
    r["group_contact_score"] = contact_s[i]

    stability = 0.45 * lat_s[i] + 0.40 * imu_s[i] + 0.15 * contact_s[i]
    r["group_stability_score"] = stability
    r["group_balanced_score"] = (
        0.34 * motion_s[i] +
        0.33 * stability +
        0.33 * energy_s[i]
    )

    wm = motion_s[i] + BETA_FLOOR
    ws = stability + BETA_FLOOR
    we = energy_s[i] + BETA_FLOOR
    z = wm + ws + we

    r["group_beta_motion_true_seed"] = wm / z
    r["group_beta_stability_true_seed"] = ws / z
    r["group_beta_energy_true_seed"] = we / z

    vals = {
        "motion": r["group_beta_motion_true_seed"],
        "stability": r["group_beta_stability_true_seed"],
        "energy": r["group_beta_energy_true_seed"],
    }
    r["group_dominant_beta_true_seed"] = max(vals, key=vals.get)
    r["beta_floor"] = BETA_FLOOR

group_by_tag = {r["base_tag"]: r for r in group_rows}

rollout_rows = []
for r in rows:
    g = group_by_tag[r["base_tag"]]
    out = dict(r)
    for k, v in g.items():
        out[f"group_{k}"] = v
    rollout_rows.append(out)

OUT_GROUP_CSV.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_GROUP_CSV, "w", newline="") as f:
    fields = list(group_rows[0].keys())
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(group_rows)

OUT_ROLLOUT_CSV.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_ROLLOUT_CSV, "w", newline="") as f:
    fields = list(rollout_rows[0].keys())
    for r in rollout_rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rollout_rows)

OUT_MD.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_MD, "w") as f:
    f.write("# TRACER Phase-F12 Grouped True-Metric Beta Targets v0\n\n")
    f.write("This aggregates Phase-F4 rollout-level true metrics by base condition before deriving beta target seeds. This reduces repeat-level target noise from min-max normalization.\n\n")
    f.write(f"- input: `{IN_CSV}`\n")
    f.write(f"- grouped output: `{OUT_GROUP_CSV}`\n")
    f.write(f"- rollout annotated output: `{OUT_ROLLOUT_CSV}`\n")
    f.write(f"- groups: `{len(group_rows)}`\n")
    f.write(f"- rollouts: `{len(rows)}`\n")
    f.write(f"- beta_floor: `{BETA_FLOOR}`\n\n")

    f.write("## Group beta targets\n\n")
    f.write("| base_tag | reset_y | n | vx_mean | lat_abs_mean | power_mean | imu_mean | contact_mean | motion | stability | energy | beta |\n")
    f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n")
    for r in group_rows:
        beta = (
            f"({float(r['group_beta_motion_true_seed']):.4f}, "
            f"{float(r['group_beta_stability_true_seed']):.4f}, "
            f"{float(r['group_beta_energy_true_seed']):.4f})"
        )
        f.write(
            f"| {r['base_tag']} | {r['reset_y']} | {r['n_rollouts']} | "
            f"{fmt(r['base_vx_mean_mean'])} | "
            f"{fmt(r['base_vy_abs_mean_mean'])} | "
            f"{fmt(r['joint_abs_power_mean_mean'])} | "
            f"{fmt(r['imu_ang_vel_norm_mean_mean'])} | "
            f"{fmt(r['contact_force_z_sum_mean_mean'])} | "
            f"{fmt(r['group_motion_score'])} | "
            f"{fmt(r['group_stability_score'])} | "
            f"{fmt(r['group_energy_score'])} | "
            f"{beta} |\n"
        )

    f.write("\n## Safe interpretation\n\n")
    f.write("This is still a bootstrap target table, not a completed IRL Objective Selector. It is more stable than per-rollout F6 because repeated rollouts are aggregated by base condition before target construction.\n")

print(f"[TRACER] wrote {OUT_GROUP_CSV}")
print(f"[TRACER] wrote {OUT_ROLLOUT_CSV}")
print(f"[TRACER] wrote {OUT_MD}")
for r in group_rows:
    print(
        f"[TRACER] {r['base_tag']}: "
        f"beta=({float(r['group_beta_motion_true_seed']):.4f}, "
        f"{float(r['group_beta_stability_true_seed']):.4f}, "
        f"{float(r['group_beta_energy_true_seed']):.4f}), "
        f"dominant={r['group_dominant_beta_true_seed']}"
    )
