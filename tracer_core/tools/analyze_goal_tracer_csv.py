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
        print("Usage: python tracer_core/tools/analyze_goal_tracer_csv.py <log.csv>")
        sys.exit(1)

    path = sys.argv[1]

    rows = []
    with open(path, "r") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)

    if not rows:
        print("No rows.")
        return

    track_rows = []
    for row in rows:
        if row.get("phase", "") in ("track", "reached", ""):
            track_rows.append(row)

    if not track_rows:
        track_rows = rows

    first = track_rows[0]
    last = track_rows[-1]

    t0 = safe_float(first.get("ros_time", 0.0))
    t1 = safe_float(last.get("ros_time", 0.0))
    duration = max(0.0, t1 - t0)

    gx = safe_float(last.get("goal_x", 0.0))
    gy = safe_float(last.get("goal_y", 0.0))

    x0 = safe_float(first.get("x", 0.0))
    y0 = safe_float(first.get("y", 0.0))
    x1 = safe_float(last.get("x", 0.0))
    y1 = safe_float(last.get("y", 0.0))

    start_dist = math.sqrt((gx - x0) ** 2 + (gy - y0) ** 2)
    end_dist = math.sqrt((gx - x1) ** 2 + (gy - y1) ** 2)
    distance_reduction = start_dist - end_dist

    travel = 0.0
    prev = None
    for row in track_rows:
        x = safe_float(row.get("x", 0.0))
        y = safe_float(row.get("y", 0.0))
        if prev is not None:
            dx = x - prev[0]
            dy = y - prev[1]
            travel += math.sqrt(dx * dx + dy * dy)
        prev = (x, y)

    def avg_field(name):
        vals = []
        for row in track_rows:
            if name in row and row[name] not in ("", None):
                vals.append(safe_float(row[name]))
        if not vals:
            return 0.0
        return sum(vals) / float(len(vals))

    modes = {}
    ram_sources = {}
    emergency_stop_rows = 0
    reached = False

    for row in track_rows:
        mode = row.get("mode", "")
        modes[mode] = modes.get(mode, 0) + 1

        ram_source = row.get("ram_source", "")
        if ram_source:
            ram_sources[ram_source] = ram_sources.get(ram_source, 0) + 1

        if row.get("emergency_stop", "") == "True":
            emergency_stop_rows += 1

        if row.get("phase", "") == "reached" or row.get("reached", "") == "True":
            reached = True

    print("CSV:", path)
    print("duration: %.3f sec" % duration)
    print("goal: x=%.3f y=%.3f" % (gx, gy))
    print("start: x=%.3f y=%.3f dist=%.3f" % (x0, y0, start_dist))
    print("end:   x=%.3f y=%.3f dist=%.3f" % (x1, y1, end_dist))
    print("distance reduction: %.3f m" % distance_reduction)
    print("travel distance: %.3f m" % travel)
    print("avg raw_vx_axis: %.4f" % avg_field("raw_vx_axis"))
    print("avg vx_axis: %.4f" % avg_field("vx_axis"))
    print("avg raw_yaw_axis: %.4f" % avg_field("raw_yaw_axis"))
    print("avg yaw_axis: %.4f" % avg_field("yaw_axis"))
    print("avg rho_v_mean: %.4f" % avg_field("rho_v_mean"))
    print("avg sigma_v: %.4f" % avg_field("sigma_v"))
    print("avg beta_v: %.4f" % avg_field("beta_v"))
    print("avg beta_s: %.4f" % avg_field("beta_s"))
    print("avg beta_e: %.4f" % avg_field("beta_e"))
    print("modes:", modes)
    if ram_sources:
        print("ram_sources:", ram_sources)
    print("emergency_stop rows:", emergency_stop_rows)
    print("reached:", reached)


if __name__ == "__main__":
    main()
