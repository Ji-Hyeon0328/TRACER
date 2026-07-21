#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path
from statistics import mean, pstdev

def ffloat(x, default=None):
    try:
        if x is None or x == "":
            return default
        return float(x)
    except Exception:
        return default

def vals(rows, key):
    out = []
    for r in rows:
        v = ffloat(r.get(key), None)
        if v is not None and math.isfinite(v):
            out.append(v)
    return out

def stat(rows, key):
    v = vals(rows, key)
    if not v:
        return {"n": 0, "mean": "", "std": "", "min": "", "max": ""}
    return {
        "n": len(v),
        "mean": mean(v),
        "std": pstdev(v) if len(v) > 1 else 0.0,
        "min": min(v),
        "max": max(v),
    }

def norm_cols(rows, cols, out_key):
    out = []
    for r in rows:
        xs = []
        ok = True
        for c in cols:
            v = ffloat(r.get(c), None)
            if v is None:
                ok = False
                break
            xs.append(v)
        if ok:
            out.append(math.sqrt(sum(x*x for x in xs)))
    return out

def stat_list(v):
    if not v:
        return {"n": 0, "mean": "", "std": "", "min": "", "max": ""}
    return {
        "n": len(v),
        "mean": mean(v),
        "std": pstdev(v) if len(v) > 1 else 0.0,
        "min": min(v),
        "max": max(v),
    }

def fmt(x):
    if x == "":
        return ""
    if isinstance(x, int):
        return str(x)
    return f"{float(x):.6f}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--true-csv", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.true_csv)))
    if not rows:
        raise SystemExit(f"empty true metric csv: {args.true_csv}")

    base_x = vals(rows, "base_x")
    base_y = vals(rows, "base_y")
    t_ros = vals(rows, "t_ros")

    imu_ang_norm = norm_cols(rows, ["imu_wx", "imu_wy", "imu_wz"], "imu_ang_norm")
    imu_acc_norm = norm_cols(rows, ["imu_ax", "imu_ay", "imu_az"], "imu_acc_norm")

    fields = [
        "manifest",
        "true_csv",
        "n_rows",
        "t_ros_start",
        "t_ros_end",
        "base_x_start",
        "base_x_end",
        "base_y_start",
        "base_y_end",
        "base_dx",
        "base_dy",
        "base_vx_mean",
        "base_vx_std",
        "base_vy_mean",
        "base_vy_abs_mean",
        "joint_abs_power_mean",
        "joint_abs_power_max",
        "joint_abs_effort_mean",
        "joint_sq_effort_mean",
        "imu_ang_vel_norm_mean",
        "imu_ang_vel_norm_max",
        "imu_acc_norm_mean",
        "imu_acc_norm_max",
        "contact_count_mean",
        "contact_force_z_sum_mean",
        "contact_force_z_sum_max",
        "contact_force_norm_sum_mean",
        "contact_force_norm_sum_max",
    ]

    sx = stat(rows, "base_x")
    sy = stat(rows, "base_y")
    svx = stat(rows, "base_vx")
    svy = stat(rows, "base_vy")
    sp = stat(rows, "joint_abs_power_sum")
    se = stat(rows, "joint_abs_effort_sum")
    se2 = stat(rows, "joint_sq_effort_sum")
    sc = stat(rows, "contact_count_fz_gt_1")
    sfz = stat(rows, "contact_force_z_sum")
    sfn = stat(rows, "contact_force_norm_sum")
    sang = stat_list(imu_ang_norm)
    sacc = stat_list(imu_acc_norm)

    vy_abs = [abs(v) for v in vals(rows, "base_vy")]

    out = {
        "manifest": args.manifest,
        "true_csv": args.true_csv,
        "n_rows": len(rows),
        "t_ros_start": min(t_ros) if t_ros else "",
        "t_ros_end": max(t_ros) if t_ros else "",
        "base_x_start": base_x[0] if base_x else "",
        "base_x_end": base_x[-1] if base_x else "",
        "base_y_start": base_y[0] if base_y else "",
        "base_y_end": base_y[-1] if base_y else "",
        "base_dx": (base_x[-1] - base_x[0]) if len(base_x) >= 2 else "",
        "base_dy": (base_y[-1] - base_y[0]) if len(base_y) >= 2 else "",
        "base_vx_mean": svx["mean"],
        "base_vx_std": svx["std"],
        "base_vy_mean": svy["mean"],
        "base_vy_abs_mean": mean(vy_abs) if vy_abs else "",
        "joint_abs_power_mean": sp["mean"],
        "joint_abs_power_max": sp["max"],
        "joint_abs_effort_mean": se["mean"],
        "joint_sq_effort_mean": se2["mean"],
        "imu_ang_vel_norm_mean": sang["mean"],
        "imu_ang_vel_norm_max": sang["max"],
        "imu_acc_norm_mean": sacc["mean"],
        "imu_acc_norm_max": sacc["max"],
        "contact_count_mean": sc["mean"],
        "contact_force_z_sum_mean": sfz["mean"],
        "contact_force_z_sum_max": sfz["max"],
        "contact_force_norm_sum_mean": sfn["mean"],
        "contact_force_norm_sum_max": sfn["max"],
    }

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerow(out)

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    with open(out_md, "w") as f:
        f.write("# TRACER Phase-F2 True Metric Summary v0\n\n")
        f.write("This summarizes the Phase-F1 ROS1 true metric logger smoke CSV into rollout-level physical metrics for future Objective Selector / RAM training.\n\n")
        f.write(f"- manifest: `{args.manifest}`\n")
        f.write(f"- true_csv: `{args.true_csv}`\n")
        f.write(f"- out_csv: `{out_csv}`\n")
        f.write(f"- n_rows: `{len(rows)}`\n\n")

        f.write("## Summary\n\n")
        f.write("| metric | value |\n")
        f.write("|---|---:|\n")
        for k in fields[2:]:
            f.write(f"| {k} | {fmt(out[k])} |\n")

        f.write("\n## Interpretation\n\n")
        f.write("- `joint_abs_power_mean` is the current energy proxy: mean of sum(abs(effort_i * velocity_i)).\n")
        f.write("- `imu_ang_vel_norm_mean/max` provides an IMU-based stability/agitation proxy.\n")
        f.write("- `contact_force_z_sum_mean/max` and `contact_count_mean` provide a contact/load proxy.\n")
        f.write("- `base_vx_mean` and `base_vy_abs_mean` provide motion tracking and lateral drift proxy signals.\n")

    print(f"[TRACER] wrote {out_csv}")
    print(f"[TRACER] wrote {out_md}")
    print(f"[TRACER] rows={len(rows)}")
    print(f"[TRACER] joint_abs_power_mean={fmt(out['joint_abs_power_mean'])}")
    print(f"[TRACER] imu_ang_vel_norm_mean={fmt(out['imu_ang_vel_norm_mean'])}")
    print(f"[TRACER] contact_force_z_sum_mean={fmt(out['contact_force_z_sum_mean'])}")

if __name__ == "__main__":
    main()
