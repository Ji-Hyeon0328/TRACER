#!/usr/bin/env python
from __future__ import print_function

import sys
import csv
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


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_goal_tracer_csv.py <csv>")
        sys.exit(1)

    path = sys.argv[1]

    rows = []
    modes = {}
    with open(path, "r") as f:
        r = csv.DictReader(f)
        for row in r:
            if row.get("phase") in ("track", "reached"):
                rows.append(row)
                m = row.get("mode", "")
                modes[m] = modes.get(m, 0) + 1

    if len(rows) < 2:
        print("Not enough rows.")
        return

    first = rows[0]
    last = rows[-1]

    t0 = safe_float(first["ros_time"])
    t1 = safe_float(last["ros_time"])
    dt = max(1e-6, t1 - t0)

    x0 = safe_float(first["x"])
    y0 = safe_float(first["y"])
    x1 = safe_float(last["x"])
    y1 = safe_float(last["y"])

    gx = safe_float(last["goal_x"])
    gy = safe_float(last["goal_y"])

    d0 = math.sqrt((gx - x0) ** 2 + (gy - y0) ** 2)
    d1 = math.sqrt((gx - x1) ** 2 + (gy - y1) ** 2)

    dx = x1 - x0
    dy = y1 - y0
    travel = math.sqrt(dx * dx + dy * dy)

    def avg_field(name):
        vals = []
        for row in rows:
            if name in row and row[name] not in ("", None):
                vals.append(safe_float(row[name]))
        return avg(vals)

    reached = any(
        row.get("phase") == "reached" or row.get("reached") == "True"
        for row in rows
    )
    estop = sum(1 for row in rows if row.get("emergency_stop") == "True")

    print("CSV:", path)
    print("duration: %.3f sec" % dt)
    print("goal: x=%.3f y=%.3f" % (gx, gy))
    print("start: x=%.3f y=%.3f dist=%.3f" % (x0, y0, d0))
    print("end:   x=%.3f y=%.3f dist=%.3f" % (x1, y1, d1))
    print("distance reduction: %.3f m" % (d0 - d1))
    print("travel distance: %.3f m" % travel)
    print("avg raw_vx_axis: %.4f" % (avg_field("raw_vx_axis") or 0.0))
    print("avg vx_axis: %.4f" % (avg_field("vx_axis") or 0.0))
    print("avg raw_yaw_axis: %.4f" % (avg_field("raw_yaw_axis") or 0.0))
    print("avg yaw_axis: %.4f" % (avg_field("yaw_axis") or 0.0))
    print("avg rho_v_mean: %.4f" % (avg_field("rho_v_mean") or 0.0))
    print("avg sigma_v: %.4f" % (avg_field("sigma_v") or 0.0))
    print("avg beta_v: %.4f" % (avg_field("beta_v") or 0.0))
    print("avg beta_s: %.4f" % (avg_field("beta_s") or 0.0))
    print("avg beta_e: %.4f" % (avg_field("beta_e") or 0.0))
    print("modes:", modes)
    print("emergency_stop rows:", estop)
    print("reached:", reached)


if __name__ == "__main__":
    main()
