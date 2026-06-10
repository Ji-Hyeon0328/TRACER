#!/usr/bin/env python
from __future__ import print_function

import sys
import csv
import math


def unwrap_angle(vals):
    if not vals:
        return []
    out = [vals[0]]
    for i in range(1, len(vals)):
        d = vals[i] - vals[i - 1]
        while d > math.pi:
            d -= 2.0 * math.pi
        while d < -math.pi:
            d += 2.0 * math.pi
        out.append(out[-1] + d)
    return out


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_calib_sync_csv.py <csv>")
        sys.exit(1)

    path = sys.argv[1]
    rows = []

    with open(path, "r") as f:
        r = csv.DictReader(f)
        for row in r:
            if row.get("phase") == "cmd":
                rows.append(row)

    if len(rows) < 2:
        print("Not enough cmd rows.")
        return

    t = [float(r["ros_time"]) for r in rows]
    x = [float(r["x"]) for r in rows]
    y = [float(r["y"]) for r in rows]
    z = [float(r["z"]) for r in rows]
    yaw = unwrap_angle([float(r["yaw"]) for r in rows])

    cmd_vx_axis = float(rows[0].get("cmd_vx_axis", 0.0))
    cmd_yaw_axis = float(rows[0].get("cmd_yaw_axis", 0.0))

    dt = max(1e-6, t[-1] - t[0])
    dx = x[-1] - x[0]
    dy = y[-1] - y[0]
    dz = z[-1] - z[0]
    dyaw = yaw[-1] - yaw[0]

    dist = math.sqrt(dx * dx + dy * dy)
    speed = dist / dt
    yaw_rate = dyaw / dt
    heading = math.atan2(dy, dx) if dist > 1e-6 else 0.0

    max_step = 0.0
    for i in range(1, len(x)):
        jx = x[i] - x[i - 1]
        jy = y[i] - y[i - 1]
        jump = math.sqrt(jx * jx + jy * jy)
        if jump > max_step:
            max_step = jump

    valid = 1
    if max_step > 0.50:
        valid = 0

    print("CSV:", path)
    print("cmd_vx_axis: %.4f" % cmd_vx_axis)
    print("cmd_yaw_axis: %.4f" % cmd_yaw_axis)
    print("duration: %.3f sec" % dt)
    print("start: x=%.4f y=%.4f z=%.4f yaw=%.4f" % (x[0], y[0], z[0], yaw[0]))
    print("end:   x=%.4f y=%.4f z=%.4f yaw=%.4f" % (x[-1], y[-1], z[-1], yaw[-1]))
    print("delta: dx=%.4f dy=%.4f dz=%.4f dyaw=%.4f rad" % (dx, dy, dz, dyaw))
    print("xy displacement: %.4f m" % dist)
    print("avg speed from displacement: %.4f m/s" % speed)
    print("avg yaw rate from yaw difference: %.4f rad/s" % yaw_rate)
    print("world displacement heading: %.4f rad" % heading)
    print("max step jump: %.4f m" % max_step)
    print("valid:", valid)


if __name__ == "__main__":
    main()
