#!/usr/bin/env python
from __future__ import print_function

import os
import sys
import csv
import glob
import math


def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def avg(vals):
    if not vals:
        return None
    return sum(vals) / float(len(vals))


def summarize_one(path):
    walk = []
    modes = {}

    with open(path, "r") as f:
        r = csv.DictReader(f)
        for row in r:
            phase = row.get("phase", "")
            mode = row.get("mode", "")
            if phase == "walk":
                walk.append(row)
                modes[mode] = modes.get(mode, 0) + 1

    if len(walk) < 2:
        return None

    first = walk[0]
    last = walk[-1]

    t0 = safe_float(first.get("ros_time", 0.0))
    t1 = safe_float(last.get("ros_time", 0.0))
    duration = max(1e-6, t1 - t0)

    x0 = safe_float(first.get("base_x", 0.0))
    y0 = safe_float(first.get("base_y", 0.0))
    z0 = safe_float(first.get("base_z", 0.0))

    x1 = safe_float(last.get("base_x", 0.0))
    y1 = safe_float(last.get("base_y", 0.0))
    z1 = safe_float(last.get("base_z", 0.0))

    dx = x1 - x0
    dy = y1 - y0
    dz = z1 - z0

    dist_xy = math.sqrt(dx * dx + dy * dy)
    avg_xy_speed = dist_xy / duration

    max_step_jump = 0.0
    prev = None
    for row in walk:
        x = safe_float(row.get("base_x", 0.0))
        y = safe_float(row.get("base_y", 0.0))
        if prev is not None:
            jx = x - prev[0]
            jy = y - prev[1]
            jump = math.sqrt(jx * jx + jy * jy)
            if jump > max_step_jump:
                max_step_jump = jump
        prev = (x, y)

    teleport_flag = 1 if max_step_jump > 0.50 else 0
    valid_for_report = 0 if teleport_flag else 1

    def avg_field(name):
        vals = []
        for row in walk:
            if name in row and row[name] not in ("", None):
                vals.append(safe_float(row[name]))
        return avg(vals)

    estop_count = sum(1 for row in walk if row.get("emergency_stop", "") == "True")
    recovery_count = sum(1 for row in walk if row.get("mode", "") == "recovery")

    return {
        "file": os.path.basename(path),
        "walk_rows": len(walk),
        "duration_sec": duration,
        "dx": dx,
        "dy": dy,
        "dz": dz,
        "xy_displacement_m": dist_xy,
        "avg_xy_speed_mps": avg_xy_speed,
        "max_step_jump_m": max_step_jump,
        "teleport_flag": teleport_flag,
        "valid_for_report": valid_for_report,
        "vx_axis_start": safe_float(first.get("vx_axis", 0.0)),
        "vx_axis_end": safe_float(last.get("vx_axis", 0.0)),
        "avg_v_cmd_mps": avg_field("v_cmd"),
        "avg_v_meas_mps": avg_field("v_meas"),
        "avg_rho_v": avg_field("rho_v"),
        "avg_rho_v_inst": avg_field("rho_v_inst"),
        "avg_rho_v_mean": avg_field("rho_v_mean"),
        "avg_sigma_v": avg_field("sigma_v"),
        "estop_rows": estop_count,
        "recovery_rows": recovery_count,
        "mode_nominal": modes.get("nominal", 0),
        "mode_cautious_mismatch": modes.get("cautious_mismatch", 0),
        "mode_conservative_mismatch": modes.get("conservative_mismatch", 0),
        "mode_conservative_posture": modes.get("conservative_posture", 0),
        "mode_recovery": modes.get("recovery", 0),
    }


def main():
    if len(sys.argv) < 2:
        log_dir = "/root/TRACER/logs"
    else:
        log_dir = sys.argv[1]

    paths = sorted(glob.glob(os.path.join(log_dir, "tracer_closed_loop_*.csv")))

    summaries = []
    for p in paths:
        s = summarize_one(p)
        if s is not None:
            summaries.append(s)

    if not summaries:
        print("No valid walk logs found.")
        return

    fields = [
        "file",
        "walk_rows",
        "duration_sec",
        "dx",
        "dy",
        "dz",
        "xy_displacement_m",
        "avg_xy_speed_mps",
        "max_step_jump_m",
        "teleport_flag",
        "valid_for_report",
        "vx_axis_start",
        "vx_axis_end",
        "avg_v_cmd_mps",
        "avg_v_meas_mps",
        "avg_rho_v",
        "avg_rho_v_inst",
        "avg_rho_v_mean",
        "avg_sigma_v",
        "estop_rows",
        "recovery_rows",
        "mode_nominal",
        "mode_cautious_mismatch",
        "mode_conservative_mismatch",
        "mode_conservative_posture",
        "mode_recovery",
    ]

    out_path = os.path.join(log_dir, "tracer_closed_loop_summary.csv")
    with open(out_path, "w") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for s in summaries:
            w.writerow(s)

    print("Wrote:", out_path)
    print("num logs:", len(summaries))

    # Print the last 10 summaries in a compact text form.
    print("")
    print("Recent logs:")
    for s in summaries[-10:]:
        print("%s | dur=%.1f s | dist=%.3f m | speed=%.4f m/s | "
              "v_cmd=%.4f | v_meas=%.4f | rho=%.3f | sigma=%.3f | "
              "N/C/CM=%d/%d/%d | estop=%d rec=%d | jump=%.3f valid=%d" % (
                  s["file"],
                  s["duration_sec"],
                  s["xy_displacement_m"],
                  s["avg_xy_speed_mps"],
                  s["avg_v_cmd_mps"] if s["avg_v_cmd_mps"] is not None else -1.0,
                  s["avg_v_meas_mps"] if s["avg_v_meas_mps"] is not None else -1.0,
                  s["avg_rho_v_mean"] if s["avg_rho_v_mean"] is not None else -1.0,
                  s["avg_sigma_v"] if s["avg_sigma_v"] is not None else -1.0,
                  s["mode_nominal"],
                  s["mode_cautious_mismatch"],
                  s["mode_conservative_mismatch"],
                  s["estop_rows"],
                  s["recovery_rows"],
                  s["max_step_jump_m"],
                  s["valid_for_report"],
              ))


if __name__ == "__main__":
    main()
