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
    rows = []
    modes = {}

    with open(path, "r") as f:
        r = csv.DictReader(f)
        for row in r:
            if row.get("phase") in ("track", "reached"):
                rows.append(row)
                mode = row.get("mode", "")
                modes[mode] = modes.get(mode, 0) + 1

    if len(rows) < 2:
        return None

    first = rows[0]
    last = rows[-1]

    t0 = safe_float(first.get("ros_time", 0.0))
    t1 = safe_float(last.get("ros_time", 0.0))
    duration = max(1e-6, t1 - t0)

    gx = safe_float(last.get("goal_x", 0.0))
    gy = safe_float(last.get("goal_y", 0.0))

    x0 = safe_float(first.get("x", 0.0))
    y0 = safe_float(first.get("y", 0.0))
    x1 = safe_float(last.get("x", 0.0))
    y1 = safe_float(last.get("y", 0.0))

    start_dist = math.sqrt((gx - x0) ** 2 + (gy - y0) ** 2)
    end_dist = math.sqrt((gx - x1) ** 2 + (gy - y1) ** 2)
    distance_reduction = start_dist - end_dist

    dx = x1 - x0
    dy = y1 - y0
    travel_distance = math.sqrt(dx * dx + dy * dy)

    def avg_field(name):
        vals = []
        for row in rows:
            if name in row and row[name] not in ("", None):
                vals.append(safe_float(row[name]))
        return avg(vals)

    reached = 1 if any(
        row.get("phase") == "reached" or row.get("reached") == "True"
        for row in rows
    ) else 0

    emergency_stop_rows = sum(
        1 for row in rows if row.get("emergency_stop") == "True"
    )

    avg_raw_vx = avg_field("raw_vx_axis")
    avg_vx = avg_field("vx_axis")

    modulation_ratio = None
    if avg_raw_vx is not None and abs(avg_raw_vx) > 1e-6 and avg_vx is not None:
        modulation_ratio = avg_vx / avg_raw_vx

    valid_for_report = 1
    if emergency_stop_rows > 0:
        valid_for_report = 0
    if distance_reduction <= 0.0:
        valid_for_report = 0
    if duration < 10.0:
        valid_for_report = 0

    return {
        "file": os.path.basename(path),
        "valid_for_report": valid_for_report,
        "duration_sec": duration,
        "goal_x": gx,
        "goal_y": gy,
        "start_x": x0,
        "start_y": y0,
        "end_x": x1,
        "end_y": y1,
        "start_dist": start_dist,
        "end_dist": end_dist,
        "distance_reduction": distance_reduction,
        "travel_distance": travel_distance,
        "reached": reached,
        "emergency_stop_rows": emergency_stop_rows,
        "avg_raw_vx_axis": avg_raw_vx,
        "avg_vx_axis": avg_vx,
        "vx_modulation_ratio": modulation_ratio,
        "avg_raw_yaw_axis": avg_field("raw_yaw_axis"),
        "avg_yaw_axis": avg_field("yaw_axis"),
        "avg_rho_v_mean": avg_field("rho_v_mean"),
        "avg_sigma_v": avg_field("sigma_v"),
        "avg_beta_v": avg_field("beta_v"),
        "avg_beta_s": avg_field("beta_s"),
        "avg_beta_e": avg_field("beta_e"),
        "mode_nominal": modes.get("nominal", 0),
        "mode_cautious_mismatch": modes.get("cautious_mismatch", 0),
        "mode_conservative_mismatch": modes.get("conservative_mismatch", 0),
        "mode_conservative_posture": modes.get("conservative_posture", 0),
        "mode_recovery": modes.get("recovery", 0),
        "mode_reached": modes.get("reached", 0),
    }


def main():
    if len(sys.argv) < 2:
        log_dir = "/root/TRACER/logs"
    else:
        log_dir = sys.argv[1]

    paths = sorted(glob.glob(os.path.join(log_dir, "tracer_goal_tracer_*.csv")))

    summaries = []
    for p in paths:
        s = summarize_one(p)
        if s is not None:
            summaries.append(s)

    if not summaries:
        print("No valid goal-tracer logs found.")
        return

    fields = [
        "file",
        "valid_for_report",
        "duration_sec",
        "goal_x",
        "goal_y",
        "start_x",
        "start_y",
        "end_x",
        "end_y",
        "start_dist",
        "end_dist",
        "distance_reduction",
        "travel_distance",
        "reached",
        "emergency_stop_rows",
        "avg_raw_vx_axis",
        "avg_vx_axis",
        "vx_modulation_ratio",
        "avg_raw_yaw_axis",
        "avg_yaw_axis",
        "avg_rho_v_mean",
        "avg_sigma_v",
        "avg_beta_v",
        "avg_beta_s",
        "avg_beta_e",
        "mode_nominal",
        "mode_cautious_mismatch",
        "mode_conservative_mismatch",
        "mode_conservative_posture",
        "mode_recovery",
        "mode_reached",
    ]

    out_path = os.path.join(log_dir, "tracer_goal_tracer_summary.csv")

    with open(out_path, "w") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for s in summaries:
            w.writerow(s)

    print("Wrote:", out_path)
    print("num logs:", len(summaries))

    reached_count = sum(s["reached"] for s in summaries)
    print("reached:", reached_count, "/", len(summaries))

    print("")
    print("Recent goal-tracer logs:")
    for s in summaries[-10:]:
        mod_ratio = s["vx_modulation_ratio"]
        if mod_ratio is None:
            mod_ratio = -1.0

        print("%s | valid=%d | dur=%.1f s | start=%.3f end=%.3f red=%.3f | "
              "travel=%.3f | reached=%d | raw_vx=%.4f vx=%.4f ratio=%.3f | "
              "rho=%.3f sigma=%.3f | N/C/CM=%d/%d/%d | estop=%d" % (
                  s["file"],
                  s["valid_for_report"],
                  s["duration_sec"],
                  s["start_dist"],
                  s["end_dist"],
                  s["distance_reduction"],
                  s["travel_distance"],
                  s["reached"],
                  s["avg_raw_vx_axis"] if s["avg_raw_vx_axis"] is not None else -1.0,
                  s["avg_vx_axis"] if s["avg_vx_axis"] is not None else -1.0,
                  mod_ratio,
                  s["avg_rho_v_mean"] if s["avg_rho_v_mean"] is not None else -1.0,
                  s["avg_sigma_v"] if s["avg_sigma_v"] is not None else -1.0,
                  s["mode_nominal"],
                  s["mode_cautious_mismatch"],
                  s["mode_conservative_mismatch"],
                  s["emergency_stop_rows"],
              ))


if __name__ == "__main__":
    main()
