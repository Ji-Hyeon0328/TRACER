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


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_goal_csv.py <csv>")
        sys.exit(1)

    path = sys.argv[1]

    rows = []
    with open(path, "r") as f:
        r = csv.DictReader(f)
        for row in r:
            if row.get("phase") in ("track", "reached"):
                rows.append(row)

    if len(rows) < 2:
        print("Not enough goal tracking rows.")
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

    avg_vx_axis = sum(safe_float(r["vx_axis"]) for r in rows) / float(len(rows))
    avg_yaw_axis = sum(safe_float(r["yaw_axis"]) for r in rows) / float(len(rows))

    reached = any(r.get("phase") == "reached" or r.get("reached") == "True" for r in rows)

    print("CSV:", path)
    print("duration: %.3f sec" % dt)
    print("goal: x=%.3f y=%.3f" % (gx, gy))
    print("start: x=%.3f y=%.3f dist=%.3f" % (x0, y0, d0))
    print("end:   x=%.3f y=%.3f dist=%.3f" % (x1, y1, d1))
    print("distance reduction: %.3f m" % (d0 - d1))
    print("travel distance: %.3f m" % travel)
    print("avg vx_axis: %.4f" % avg_vx_axis)
    print("avg yaw_axis: %.4f" % avg_yaw_axis)
    print("reached:", reached)


if __name__ == "__main__":
    main()
