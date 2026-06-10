#!/usr/bin/env python
from __future__ import print_function

import sys
import csv
import math


def unwrap_angle(a):
    out = [a[0]]
    for i in range(1, len(a)):
        d = a[i] - a[i - 1]
        while d > math.pi:
            d -= 2.0 * math.pi
        while d < -math.pi:
            d += 2.0 * math.pi
        out.append(out[-1] + d)
    return out


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_calib_csv.py <csv>")
        sys.exit(1)

    path = sys.argv[1]

    rows = []
    with open(path, "r") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)

    if len(rows) < 2:
        print("Not enough rows.")
        return

    t = [float(r["ros_time"]) for r in rows]
    x = [float(r["x"]) for r in rows]
    y = [float(r["y"]) for r in rows]
    z = [float(r["z"]) for r in rows]
    yaw = unwrap_angle([float(r["yaw"]) for r in rows])
    vx = [float(r["vx"]) for r in rows]
    vy = [float(r["vy"]) for r in rows]
    wz = [float(r["wz"]) for r in rows]

    dt = max(1e-6, t[-1] - t[0])
    dx = x[-1] - x[0]
    dy = y[-1] - y[0]
    dz = z[-1] - z[0]
    dist = math.sqrt(dx * dx + dy * dy)
    dyaw = yaw[-1] - yaw[0]

    avg_speed_from_disp = dist / dt
    avg_vx_world = sum(vx) / float(len(vx))
    avg_vy_world = sum(vy) / float(len(vy))
    avg_wz = sum(wz) / float(len(wz))

    heading_world = math.atan2(dy, dx) if dist > 1e-6 else 0.0

    print("CSV:", path)
    print("duration: %.3f sec" % dt)
    print("start: x=%.4f y=%.4f z=%.4f yaw=%.4f" % (x[0], y[0], z[0], yaw[0]))
    print("end:   x=%.4f y=%.4f z=%.4f yaw=%.4f" % (x[-1], y[-1], z[-1], yaw[-1]))
    print("delta: dx=%.4f dy=%.4f dz=%.4f dyaw=%.4f rad" % (dx, dy, dz, dyaw))
    print("xy displacement: %.4f m" % dist)
    print("avg speed from displacement: %.4f m/s" % avg_speed_from_disp)
    print("avg world velocity: vx=%.4f vy=%.4f m/s" % (avg_vx_world, avg_vy_world))
    print("avg yaw rate wz: %.4f rad/s" % avg_wz)
    print("world displacement heading: %.4f rad" % heading_world)


if __name__ == "__main__":
    main()
