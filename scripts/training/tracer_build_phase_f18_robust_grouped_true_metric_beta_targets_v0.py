#!/usr/bin/env python3
import csv
import math
import re
from pathlib import Path
from statistics import mean, median, pstdev
from collections import defaultdict

IN_CSV = Path("datasets/phase_f/f4_true_metric_training_table_v0.csv")
OUT_GROUP_CSV = Path("datasets/phase_f/f18_robust_grouped_true_metric_beta_targets_v0.csv")
OUT_ROLLOUT_CSV = Path("datasets/phase_f/f18_rollout_with_robust_group_beta_targets_v0.csv")
OUT_MD = Path("reports/phase_f/f18_robust_grouped_true_metric_beta_targets_summary_v0.md")

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

def beta_from_scores(motion, stability, energy):
    wm = motion + BETA_FLOOR
    ws = stability + BETA_FLOOR
    we = energy + BETA_FLOOR
    z = wm + ws + we
    return wm / z, ws / z, we / z

def l1(a, b):
    return sum(abs(float(x) - float(y)) for x, y in zip(a, b))

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
for g in sorted(groups):
    rs = groups[g]
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
            out[f"{k}_median"] = ""
            out[f"{k}_std"] = ""
            out[f"{k}_min"] = ""
            out[f"{k}_max"] = ""
        else:
            out[f"{k}_mean"] = mean(vals)
            out[f"{k}_median"] = median(vals)
            out[f"{k}_std"] = pstdev(vals) if len(vals) > 1 else 0.0
            out[f"{k}_min"] = min(vals)
            out[f"{k}_max"] = max(vals)

    group_rows.append(out)

def attach_beta(prefix, stat_key):
    vx = [float(r[f"base_vx_mean_{stat_key}"]) for r in group_rows]
    lat = [float(r[f"base_vy_abs_mean_{stat_key}"]) for r in group_rows]
    power = [float(r[f"joint_abs_power_mean_{stat_key}"]) for r in group_rows]
    imu = [float(r[f"imu_ang_vel_norm_mean_{stat_key}"]) for r in group_rows]
    contact = [float(r[f"contact_force_z_sum_mean_{stat_key}"]) for r in group_rows]

    motion_s = good_high(vx)
    lat_s = good_low(lat)
    energy_s = good_low(power)
    imu_s = good_low(imu)
    contact_s = good_low(contact)

    for i, r in enumerate(group_rows):
        stability = 0.45 * lat_s[i] + 0.40 * imu_s[i] + 0.15 * contact_s[i]
        bm, bs, be = beta_from_scores(motion_s[i], stability, energy_s[i])

        r[f"{prefix}_motion_score"] = motion_s[i]
        r[f"{prefix}_stability_score"] = stability
        r[f"{prefix}_energy_score"] = energy_s[i]
        r[f"{prefix}_beta_motion_true_seed"] = bm
        r[f"{prefix}_beta_stability_true_seed"] = bs
        r[f"{prefix}_beta_energy_true_seed"] = be

        vals = {"motion": bm, "stability": bs, "energy": be}
        r[f"{prefix}_dominant_beta_true_seed"] = max(vals, key=vals.get)

attach_beta("mean", "mean")
attach_beta("robust_median", "median")

for r in group_rows:
    mean_beta = (
        r["mean_beta_motion_true_seed"],
        r["mean_beta_stability_true_seed"],
        r["mean_beta_energy_true_seed"],
    )
    med_beta = (
        r["robust_median_beta_motion_true_seed"],
        r["robust_median_beta_stability_true_seed"],
        r["robust_median_beta_energy_true_seed"],
    )
    r["mean_vs_robust_beta_l1"] = l1(mean_beta, med_beta)
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
    f.write("# TRACER Phase-F18 Robust Grouped True-Metric Beta Targets v0\n\n")
    f.write("This builds robust grouped beta targets using group median metrics instead of mean metrics, reducing sensitivity to single-run outliers.\n\n")
    f.write(f"- input: `{IN_CSV}`\n")
    f.write(f"- grouped output: `{OUT_GROUP_CSV}`\n")
    f.write(f"- rollout annotated output: `{OUT_ROLLOUT_CSV}`\n")
    f.write(f"- groups: `{len(group_rows)}`\n")
    f.write(f"- rollouts: `{len(rows)}`\n")
    f.write(f"- beta_floor: `{BETA_FLOOR}`\n\n")

    f.write("## Robust beta targets\n\n")
    f.write("| base_tag | reset_y | n | mean beta | robust median beta | L1 shift | robust dominant |\n")
    f.write("|---|---:|---:|---|---|---:|---|\n")
    for r in sorted(group_rows, key=lambda x: float(x["mean_vs_robust_beta_l1"]), reverse=True):
        mean_beta = (
            f"({float(r['mean_beta_motion_true_seed']):.4f}, "
            f"{float(r['mean_beta_stability_true_seed']):.4f}, "
            f"{float(r['mean_beta_energy_true_seed']):.4f})"
        )
        med_beta = (
            f"({float(r['robust_median_beta_motion_true_seed']):.4f}, "
            f"{float(r['robust_median_beta_stability_true_seed']):.4f}, "
            f"{float(r['robust_median_beta_energy_true_seed']):.4f})"
        )
        f.write(
            f"| {r['base_tag']} | {r['reset_y']} | {r['n_rollouts']} | "
            f"{mean_beta} | {med_beta} | "
            f"{float(r['mean_vs_robust_beta_l1']):.6f} | "
            f"{r['robust_median_dominant_beta_true_seed']} |\n"
        )

    f.write("\n## Median metrics used for robust target\n\n")
    f.write("| base_tag | vx_med | lat_abs_med | power_med | imu_med | contact_med | motion | stability | energy |\n")
    f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for r in group_rows:
        f.write(
            f"| {r['base_tag']} | "
            f"{fmt(r['base_vx_mean_median'])} | "
            f"{fmt(r['base_vy_abs_mean_median'])} | "
            f"{fmt(r['joint_abs_power_mean_median'])} | "
            f"{fmt(r['imu_ang_vel_norm_mean_median'])} | "
            f"{fmt(r['contact_force_z_sum_mean_median'])} | "
            f"{fmt(r['robust_median_motion_score'])} | "
            f"{fmt(r['robust_median_stability_score'])} | "
            f"{fmt(r['robust_median_energy_score'])} |\n"
        )

    f.write("\n## Safe interpretation\n\n")
    f.write("This is not a final IRL Objective Selector. It is a robust bootstrap target used to reduce mean-target sensitivity to abnormal rollouts such as very low-velocity/contact episodes.\n")

print(f"[TRACER] wrote {OUT_GROUP_CSV}")
print(f"[TRACER] wrote {OUT_ROLLOUT_CSV}")
print(f"[TRACER] wrote {OUT_MD}")
for r in sorted(group_rows, key=lambda x: float(x["mean_vs_robust_beta_l1"]), reverse=True):
    print(
        f"[TRACER] {r['base_tag']}: "
        f"mean_beta=({float(r['mean_beta_motion_true_seed']):.4f}, "
        f"{float(r['mean_beta_stability_true_seed']):.4f}, "
        f"{float(r['mean_beta_energy_true_seed']):.4f}) "
        f"robust_beta=({float(r['robust_median_beta_motion_true_seed']):.4f}, "
        f"{float(r['robust_median_beta_stability_true_seed']):.4f}, "
        f"{float(r['robust_median_beta_energy_true_seed']):.4f}) "
        f"L1={float(r['mean_vs_robust_beta_l1']):.6f}"
    )
