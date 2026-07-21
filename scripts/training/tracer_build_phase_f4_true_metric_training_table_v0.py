#!/usr/bin/env python3
import csv
import re
from pathlib import Path

IN_DIR = Path("datasets/phase_f")
OUT_CSV = Path("datasets/phase_f/f4_true_metric_training_table_v0.csv")
OUT_MD = Path("reports/phase_f/f4_true_metric_training_table_summary_v0.md")

summary_files = sorted(IN_DIR.glob("f3_*_true_metric_summary.csv"))

rows = []
for p in summary_files:
    m = re.match(r"f3_(.+?)_\d{8}_\d{6}_true_metric_summary\.csv", p.name)
    tag = m.group(1) if m else "unknown"

    with open(p) as f:
        r = next(csv.DictReader(f))

    if tag == "clean":
        reset_y = 0.0
    elif tag == "m030":
        reset_y = -0.30
    elif tag == "p030":
        reset_y = 0.30
    else:
        reset_y = ""

    out = dict(r)
    out["tag"] = tag
    out["reset_y"] = reset_y
    out["summary_csv"] = str(p)

    # Initial normalized objective-style proxies.
    # Lower energy/stability/lateral/contact agitation is better.
    def ff(k):
        try:
            return float(out.get(k, ""))
        except Exception:
            return 0.0

    out["motion_proxy"] = ff("base_vx_mean")
    out["lateral_drift_proxy"] = abs(ff("base_vy_abs_mean"))
    out["energy_proxy"] = ff("joint_abs_power_mean")
    out["stability_proxy"] = ff("imu_ang_vel_norm_mean")
    out["contact_load_proxy"] = ff("contact_force_z_sum_mean")

    rows.append(out)

if not rows:
    raise SystemExit("no F3 summary files found")

preferred = [
    "tag", "reset_y", "manifest", "true_csv", "summary_csv", "n_rows",
    "base_dx", "base_dy", "base_vx_mean", "base_vx_std", "base_vy_abs_mean",
    "joint_abs_power_mean", "joint_abs_power_max",
    "joint_abs_effort_mean", "joint_sq_effort_mean",
    "imu_ang_vel_norm_mean", "imu_ang_vel_norm_max",
    "imu_acc_norm_mean", "imu_acc_norm_max",
    "contact_count_mean",
    "contact_force_z_sum_mean", "contact_force_z_sum_max",
    "contact_force_norm_sum_mean", "contact_force_norm_sum_max",
    "motion_proxy", "lateral_drift_proxy", "energy_proxy", "stability_proxy", "contact_load_proxy",
]

fields = preferred + sorted(set().union(*[r.keys() for r in rows]) - set(preferred))

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_CSV, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for r in rows:
        w.writerow(r)

OUT_MD.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_MD, "w") as f:
    f.write("# TRACER Phase-F4 True Metric Training Table v0\n\n")
    f.write("This table aggregates Phase-F3 true metric rollout summaries into a compact training/evaluation table for Objective Selector and RAM follow-up work.\n\n")
    f.write(f"- input summaries: `{len(summary_files)}`\n")
    f.write(f"- output csv: `{OUT_CSV}`\n\n")
    f.write("| tag | reset_y | n_rows | base_vx_mean | base_vy_abs_mean | joint_abs_power_mean | imu_ang_vel_norm_mean | contact_force_z_sum_mean |\n")
    f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
    for r in rows:
        f.write(
            f"| {r.get('tag')} | {r.get('reset_y')} | {r.get('n_rows')} | "
            f"{r.get('base_vx_mean')} | {r.get('base_vy_abs_mean')} | "
            f"{r.get('joint_abs_power_mean')} | {r.get('imu_ang_vel_norm_mean')} | "
            f"{r.get('contact_force_z_sum_mean')} |\n"
        )
    f.write("\n## Safe interpretation\n\n")
    f.write("This is not yet a learned IRL Objective Selector dataset. It is the first true-metric rollout table that connects high-level deployment conditions with physical proxies from ROS1/Gazebo: energy, IMU agitation, contact load, velocity, and lateral drift.\n")

print(f"[TRACER] wrote {OUT_CSV}")
print(f"[TRACER] wrote {OUT_MD}")
print(f"[TRACER] rows={len(rows)}")
