#!/usr/bin/env python3

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = (
    ROOT
    / "results"
    / "icra27"
    / "m3_vx_yaw_coupling"
)

OUTPUT = (
    ROOT
    / "results"
    / "icra27"
    / "m3_authority"
    / "vx_yaw_coupling_v0.json"
)


def load_points():
    points = []

    for path in sorted(INPUT_DIR.glob("*.json")):
        data = json.loads(
            path.read_text(encoding="utf-8")
        )

        command = data["command"]
        measured = data["measured"]

        points.append({
            "file": str(path.relative_to(ROOT)),

            "vx_command_mps":
                float(command["vx_mps"]),

            "yaw_command_rad_s":
                float(command["yaw_rate_rad_s"]),

            "vx_measured_mps":
                float(
                    measured[
                        "mean_body_forward_vx_mps"
                    ]
                ),

            "yaw_measured_rad_s":
                float(
                    measured[
                        "mean_yaw_rate_rad_s"
                    ]
                ),

            "base_z_std_m":
                float(
                    measured["std_base_z_m"]
                ),

            "yaw_change_rad":
                float(
                    measured["yaw_change_rad"]
                ),
        })

    if len(points) != 9:
        raise RuntimeError(
            f"Expected 9 coupling points, got {len(points)}"
        )

    return points


def fit_yaw(points):
    x = np.asarray(
        [p["yaw_command_rad_s"] for p in points],
        dtype=float,
    )

    y = np.asarray(
        [p["yaw_measured_rad_s"] for p in points],
        dtype=float,
    )

    slope, intercept = np.polyfit(x, y, 1)

    pred = slope * x + intercept

    ss_res = float(
        np.sum((y - pred) ** 2)
    )

    ss_tot = float(
        np.sum((y - np.mean(y)) ** 2)
    )

    r2 = (
        1.0 - ss_res / ss_tot
        if ss_tot > 1e-15
        else None
    )

    return (
        float(slope),
        float(intercept),
        float(r2) if r2 is not None else None,
    )


def main():
    points = load_points()

    grouped = defaultdict(list)

    for point in points:
        grouped[
            point["vx_command_mps"]
        ].append(point)

    rows = []

    for vx in sorted(grouped):
        group = sorted(
            grouped[vx],
            key=lambda p:
                p["yaw_command_rad_s"],
        )

        slope, intercept, r2 = fit_yaw(group)

        straight = next(
            p for p in group
            if abs(
                p["yaw_command_rad_s"]
            ) < 1e-9
        )

        turns = [
            p for p in group
            if abs(
                p["yaw_command_rad_s"]
            ) > 1e-9
        ]

        mean_turn_vx = float(
            np.mean(
                [
                    p["vx_measured_mps"]
                    for p in turns
                ]
            )
        )

        straight_vx = float(
            straight["vx_measured_mps"]
        )

        speed_retention = (
            mean_turn_vx / straight_vx
            if abs(straight_vx) > 1e-9
            else None
        )

        negative = group[0]
        positive = group[-1]

        yaw_symmetry_error = abs(
            abs(
                negative["yaw_measured_rad_s"]
            )
            - abs(
                positive["yaw_measured_rad_s"]
            )
        )

        row = {
            "vx_command_mps": float(vx),

            "yaw_gain":
                slope,

            "yaw_intercept_rad_s":
                intercept,

            "yaw_fit_r_squared":
                r2,

            "straight_measured_vx_mps":
                straight_vx,

            "mean_turn_measured_vx_mps":
                mean_turn_vx,

            "turn_speed_retention_ratio":
                (
                    float(speed_retention)
                    if speed_retention is not None
                    else None
                ),

            "left_right_yaw_symmetry_error_rad_s":
                float(yaw_symmetry_error),

            "max_base_z_std_m":
                float(
                    max(
                        p["base_z_std_m"]
                        for p in group
                    )
                ),
        }

        rows.append(row)

    output = {
        "grid": {
            "vx_command_mps":
                sorted(grouped.keys()),
            "yaw_command_rad_s":
                [-0.3, 0.0, 0.3],
        },

        "interpretation": {
            "yaw_gain": (
                "slope of measured yaw rate vs commanded "
                "yaw rate at fixed vx"
            ),
            "turn_speed_retention_ratio": (
                "mean forward speed at +/- yaw divided by "
                "straight forward speed at the same vx"
            ),
        },

        "points": points,
        "per_vx_summary": rows,
    }

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT.write_text(
        json.dumps(
            output,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    print("=" * 100)
    print("ICRA27 M3 vx × yaw coupling summary")
    print("=" * 100)

    print(
        f"{'vx cmd':>10s}"
        f"{'yaw gain':>14s}"
        f"{'yaw bias':>14s}"
        f"{'R^2':>12s}"
        f"{'straight vx':>16s}"
        f"{'turn vx':>14s}"
        f"{'retention':>14s}"
        f"{'LR asym':>12s}"
    )

    print("-" * 100)

    for row in rows:
        print(
            f"{row['vx_command_mps']:10.3f}"
            f"{row['yaw_gain']:14.6f}"
            f"{row['yaw_intercept_rad_s']:14.6f}"
            f"{row['yaw_fit_r_squared']:12.6f}"
            f"{row['straight_measured_vx_mps']:16.6f}"
            f"{row['mean_turn_measured_vx_mps']:14.6f}"
            f"{row['turn_speed_retention_ratio']:14.6f}"
            f"{row['left_right_yaw_symmetry_error_rad_s']:12.6f}"
        )

    print("=" * 100)
    print()
    print("Saved:", OUTPUT)


if __name__ == "__main__":
    main()
