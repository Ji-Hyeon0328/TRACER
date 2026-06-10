#!/usr/bin/env python
from __future__ import print_function

import sys
import csv
import math


def main():
    if len(sys.argv) < 2:
        print("Usage: python tracer_core/tools/analyze_closed_loop_csv.py <csv_path>")
        sys.exit(1)

    path = sys.argv[1]

    rows = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("phase", "") == "walk":
                rows.append(row)

    if len(rows) < 2:
        print("Not enough walk-phase rows.")
        return

    first = rows[0]
    last = rows[-1]

    t0 = float(first["ros_time"])
    t1 = float(last["ros_time"])

    x0 = float(first["base_x"])
    y0 = float(first["base_y"])
    z0 = float(first["base_z"])

    x1 = float(last["base_x"])
    y1 = float(last["base_y"])
    z1 = float(last["base_z"])

    dx = x1 - x0
    dy = y1 - y0
    dz = z1 - z0

    duration = max(1e-6, t1 - t0)
    dist_xy = math.sqrt(dx * dx + dy * dy)
    speed_xy = dist_xy / duration

    vx0 = float(first["vx_axis"])
    vx1 = float(last["vx_axis"])


    def avg_field(name):
        vals = []
        for r in rows:
            if name in r and r[name] not in ("", None):
                try:
                    vals.append(float(r[name]))
                except Exception:
                    pass
        if not vals:
            return None
        return sum(vals) / float(len(vals))

    avg_v_cmd = avg_field("v_cmd")
    avg_v_meas = avg_field("v_meas")
    avg_rho_v = avg_field("rho_v")
    avg_rho_v_inst = avg_field("rho_v_inst")
    avg_rho_v_mean = avg_field("rho_v_mean")
    avg_sigma_v = avg_field("sigma_v")

    estop_count = sum(1 for r in rows if r.get("emergency_stop", "") == "True")
    recovery_count = sum(1 for r in rows if r.get("mode", "") == "recovery")

    print("CSV:", path)
    print("walk rows:", len(rows))
    print("duration: %.3f sec" % duration)
    print("start pos: x=%.4f y=%.4f z=%.4f" % (x0, y0, z0))
    print("end pos:   x=%.4f y=%.4f z=%.4f" % (x1, y1, z1))
    print("delta:     dx=%.4f dy=%.4f dz=%.4f" % (dx, dy, dz))
    print("xy displacement: %.4f m" % dist_xy)
    print("avg xy speed: %.4f m/s" % speed_xy)
    print("vx_axis: %.4f -> %.4f" % (vx0, vx1))
    if avg_v_cmd is not None:
        print("avg v_cmd: %.4f m/s" % avg_v_cmd)
    if avg_v_meas is not None:
        print("avg v_meas: %.4f m/s" % avg_v_meas)
    if avg_rho_v is not None:
        print("avg rho_v: %.4f" % avg_rho_v)
    if avg_rho_v_inst is not None:
        print("avg rho_v_inst: %.4f" % avg_rho_v_inst)
    if avg_rho_v_mean is not None:
        print("avg rho_v_mean: %.4f" % avg_rho_v_mean)
    if avg_sigma_v is not None:
        print("avg sigma_v: %.4f" % avg_sigma_v)
    print("estop rows:", estop_count)
    print("recovery rows:", recovery_count)


if __name__ == "__main__":
    main()
