#!/usr/bin/env python3
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

def ground_z_flat(x: np.ndarray, center_z: float = 0.0) -> np.ndarray:
    return np.zeros_like(x) + center_z

def ground_z_sloped(
    x: np.ndarray,
    deg: float,
    direction: str,
    center_z: float = -0.04981,
    thickness: float = 0.10,
) -> np.ndarray:
    theta = math.radians(deg)
    top_at_origin = center_z + thickness / (2.0 * math.cos(theta))

    if direction == "downslope":
        return top_at_origin - np.tan(theta) * x
    if direction == "upslope":
        return top_at_origin + np.tan(theta) * x

    raise ValueError(f"unknown direction: {direction}")

def infer_geom_from_world_name(world: str) -> str:
    w = world.lower()

    if "downslope_10deg" in w:
        return "downslope_10deg"
    if "downslope_5deg" in w:
        return "downslope_5deg"
    if "slope_10deg" in w:
        return "upslope_10deg"
    if "slope_5deg" in w:
        return "upslope_5deg"

    return "flat"

def compute_ground_z(x: np.ndarray, geom: str) -> np.ndarray:
    if geom == "flat":
        return ground_z_flat(x, 0.0)
    if geom == "downslope_5deg":
        return ground_z_sloped(x, 5.0, "downslope")
    if geom == "downslope_10deg":
        return ground_z_sloped(x, 10.0, "downslope")
    if geom == "upslope_5deg":
        return ground_z_sloped(x, 5.0, "upslope")
    if geom == "upslope_10deg":
        return ground_z_sloped(x, 10.0, "upslope")

    raise ValueError(f"unknown TRACER_TERRAIN_GEOM={geom}")

def main():
    logs = sorted((ROOT / "data/mission_logs").glob("tracer_mission_log_*.npz"))
    sums = sorted((ROOT / "data/mission_logs").glob("tracer_mission_summary_*.json"))

    if not logs:
        raise SystemExit("no mission log found")

    world = os.environ.get("TRACER_WORLD_NAME", "")
    geom = os.environ.get("TRACER_TERRAIN_GEOM", "")
    if not geom:
        geom = infer_geom_from_world_name(world)

    p = logs[-1]
    d = np.load(p, allow_pickle=True)

    pro = d["proprio"]
    mpc = d["mpc_reference"]

    valid_p = np.isfinite(pro).all(axis=1)
    valid_m = np.isfinite(mpc).all(axis=1)

    pv = pro[valid_p]
    mv = mpc[valid_m]

    print("latest npz:", p)
    if sums:
        print("latest summary:", sums[-1])
    print("world:", world if world else "(unknown)")
    print("terrain geom:", geom)

    if len(pv) < 2:
        raise SystemExit("not enough valid proprio samples")

    x = pv[:, 1]
    y = pv[:, 2]
    z = pv[:, 3]
    roll = pv[:, 4]
    pitch = pv[:, 5]

    gz = compute_ground_z(x, geom)
    rel_z = z - gz

    max_roll = float(np.abs(roll).max())
    max_pitch = float(np.abs(pitch).max())

    fallen_abs = (float(z.min()) < 0.12) or (max_roll > 1.2) or (max_pitch > 1.0)
    fallen_rel = (float(rel_z.min()) < 0.12) or (max_roll > 1.2) or (max_pitch > 1.0)

    print("\n========== absolute motion ==========")
    print("start xyz:", float(x[0]), float(y[0]), float(z[0]))
    print("end   xyz:", float(x[-1]), float(y[-1]), float(z[-1]))
    print("delta x:", float(x[-1] - x[0]))
    print("delta y:", float(y[-1] - y[0]))
    print("min abs z:", float(z.min()))
    print("max |roll|:", max_roll)
    print("max |pitch|:", max_pitch)

    print("\n========== terrain-relative height ==========")
    print("start ground z:", float(gz[0]))
    print("end ground z:", float(gz[-1]))
    print("start rel z:", float(rel_z[0]))
    print("end rel z:", float(rel_z[-1]))
    print("min rel z:", float(rel_z.min()))
    print("mean rel z:", float(rel_z.mean()))

    print("\n========== fall heuristic ==========")
    print("fallen_abs_z_based:", bool(fallen_abs))
    print("fallen_relative_z_based:", bool(fallen_rel))

    if len(mv) > 1:
        enable_fraction = float(np.mean(mv[:, 5] > 0.5))
        vx_mean = float(np.mean(mv[:, 1]))

        print("\n========== mpc ==========")
        print("first:", mv[0].tolist())
        print("last: ", mv[-1].tolist())
        print("enable fraction:", enable_fraction)
        print("vx mean:", vx_mean)
        print("body height mean:", float(np.mean(mv[:, 3])))
        print("clearance mean:", float(np.mean(mv[:, 4])))

        valid_locomotion = (
            enable_fraction > 0.5
            and not fallen_rel
            and abs(float(x[-1] - x[0])) > 0.05
        )

        stable_hold = (
            enable_fraction <= 0.5
            and not fallen_rel
            and abs(float(x[-1] - x[0])) < 0.15
            and abs(float(y[-1] - y[0])) < 0.15
        )

        print("\n========== decision helper ==========")
        print("valid_locomotion_candidate:", bool(valid_locomotion))
        print("stable_hold_candidate:", bool(stable_hold))

if __name__ == "__main__":
    main()
