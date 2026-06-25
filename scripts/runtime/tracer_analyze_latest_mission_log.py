#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def latest_file(pattern: str, root: Path) -> Path | None:
    files = sorted(root.glob(pattern))
    return files[-1] if files else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze latest TRACER mission log with NaN-safe metrics.")
    parser.add_argument(
        "--log",
        type=Path,
        default=None,
        help="Path to a tracer_mission_log_*.npz file. If omitted, use latest in data/mission_logs.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Optional path to matching tracer_mission_summary_*.json.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/mission_logs"),
        help="Mission log directory. Default: data/mission_logs",
    )
    parser.add_argument(
        "--success-z-min",
        type=float,
        default=0.18,
        help="Fall-like threshold for minimum base z.",
    )
    parser.add_argument(
        "--success-rpy-max",
        type=float,
        default=0.7,
        help="Fall-like threshold for max absolute roll/pitch.",
    )
    args = parser.parse_args()

    root = args.root
    log_path = args.log or latest_file("tracer_mission_log_*.npz", root)
    summary_path = args.summary or latest_file("tracer_mission_summary_*.json", root)

    print("latest npz:", log_path)
    print("latest summary:", summary_path)

    if summary_path and summary_path.exists():
        print("\n========== summary ==========")
        print(summary_path.read_text())

    if log_path is None or not log_path.exists():
        print("No mission log found.")
        return 1

    d = np.load(log_path, allow_pickle=True)
    print("\nkeys:", list(d.keys()))

    if "proprio" not in d:
        print("No proprio array in log.")
        return 1

    p = d["proprio"]
    m = d["mpc_reference"] if "mpc_reference" in d else np.empty((0, 6))

    print("proprio shape:", p.shape)
    print("mpc_reference shape:", m.shape)

    finite_pose = np.isfinite(p[:, 1:6]).all(axis=1) if p.ndim == 2 and p.shape[1] >= 6 else np.zeros(len(p), dtype=bool)
    finite_mpc = np.isfinite(m).all(axis=1) if m.ndim == 2 and m.shape[1] >= 6 else np.zeros(len(m), dtype=bool)

    print("\n========== validity ==========")
    print("finite proprio rows:", int(finite_pose.sum()), "/", len(p))
    print("finite mpc rows:", int(finite_mpc.sum()), "/", len(m))

    if finite_pose.any():
        pf = p[finite_pose]

        x0, y0, z0 = pf[0, 1], pf[0, 2], pf[0, 3]
        x1, y1, z1 = pf[-1, 1], pf[-1, 2], pf[-1, 3]
        roll = pf[:, 4]
        pitch = pf[:, 5]

        min_z = float(np.nanmin(pf[:, 3]))
        max_z = float(np.nanmax(pf[:, 3]))
        max_abs_roll = float(np.nanmax(np.abs(roll)))
        max_abs_pitch = float(np.nanmax(np.abs(pitch)))

        fall_like = (
            min_z < args.success_z_min
            or max_abs_roll > args.success_rpy_max
            or max_abs_pitch > args.success_rpy_max
        )

        duration_est = None
        if "wall_time" in d:
            wt = d["wall_time"]
            finite_wt = wt[np.isfinite(wt)]
            if len(finite_wt) >= 2:
                duration_est = float(finite_wt[-1] - finite_wt[0])

        print("\n========== NaN-safe motion ==========")
        print("start xyz:", float(x0), float(y0), float(z0))
        print("end   xyz:", float(x1), float(y1), float(z1))
        print("delta x:", float(x1 - x0))
        print("delta y:", float(y1 - y0))
        print("delta z:", float(z1 - z0))
        print("min z:", min_z)
        print("max z:", max_z)
        print("max |roll|:", max_abs_roll)
        print("max |pitch|:", max_abs_pitch)
        print("fall_like:", bool(fall_like))
        if duration_est is not None and duration_est > 1e-6:
            print("duration_est:", duration_est)
            print("observed vx:", float((x1 - x0) / duration_est))
            print("observed vy:", float((y1 - y0) / duration_est))

    print("\n========== mpc reference ==========")
    if finite_mpc.any():
        mf = m[finite_mpc]
        print("first:", mf[0].tolist())
        print("last: ", mf[-1].tolist())
        print("vx mean:", float(np.nanmean(mf[:, 1])))
        print("vx max:", float(np.nanmax(mf[:, 1])))
        print("body height mean:", float(np.nanmean(mf[:, 3])))
        print("clearance mean:", float(np.nanmean(mf[:, 4])))
        print("enable mean:", float(np.nanmean(mf[:, 5])))
    else:
        print("No finite MPC reference rows recorded.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
