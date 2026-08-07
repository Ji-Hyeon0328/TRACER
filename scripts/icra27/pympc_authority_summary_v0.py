#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "icra27"


AXES = {
    "vx": {
        "glob": "m1_vx/*.json",
        "command_key": "requested_vx_mps",
        "measured_key": "mean_measured_vx_mps",
        "std_key": "std_measured_vx_mps",
        "unit": "m/s",
        "response_kind": "physical_body_velocity",
    },

    "yaw_rate": {
        "glob": "m1_yaw_rate/*.json",
        "command_key": "commanded_yaw_rate_rad_s",
        "measured_key": "mean_measured_yaw_rate_rad_s",
        "std_key": "std_measured_yaw_rate_rad_s",
        "unit": "rad/s",
        "response_kind": "physical_body_angular_velocity",
    },

    "body_height": {
        "glob": "m1_body_height/*.json",
        "command_key": "requested_body_height_m",
        "measured_key": "mean_measured_base_z_m",
        "std_key": "std_measured_base_z_m",
        "unit": "m",
        "response_kind": "physical_base_height",
    },

    "swing_clearance": {
        "glob": "m1_swing_clearance/*.json",
        "command_key": "requested_swing_clearance_m",
        "measured_key": "mean_measured_clearance_m",
        "std_key": "std_across_legs_m",
        "unit": "m",
        "response_kind": "physical_foot_clearance",
    },

    "gait_frequency": {
        "glob": "m1_gait_frequency/*.json",
        "command_key": "commanded_frequency_hz",
        "measured_key": "mean_measured_frequency_hz",
        "std_key": "std_across_legs_hz",
        "unit": "Hz",
        "response_kind": "physical_gait_frequency",
    },

    "duty_factor": {
        "glob": "m1_duty_factor/*.json",
        "command_key": "commanded_duty_factor",
        "measured_key": "mean_measured_duty_factor",
        "std_key": "std_across_legs",
        "unit": "ratio",
        "response_kind": "planned_contact_schedule",
    },
}


def load_axis(name, spec):
    paths = sorted(
        RESULTS.glob(spec["glob"])
    )

    if len(paths) < 3:
        raise RuntimeError(
            f"{name}: expected at least 3 JSON files, "
            f"found {len(paths)} for {spec['glob']}"
        )

    points = []

    for path in paths:
        data = json.loads(
            path.read_text(encoding="utf-8")
        )

        for key in (
            spec["command_key"],
            spec["measured_key"],
        ):
            if key not in data:
                raise KeyError(
                    f"{path}: missing key {key!r}"
                )

        point = {
            "file": str(path.relative_to(ROOT)),
            "command": float(
                data[spec["command_key"]]
            ),
            "measured": float(
                data[spec["measured_key"]]
            ),
            "std": (
                float(data[spec["std_key"]])
                if spec["std_key"] in data
                else None
            ),
        }

        points.append(point)

    points.sort(
        key=lambda item: item["command"]
    )

    return points


def fit_axis(points):
    x = np.asarray(
        [p["command"] for p in points],
        dtype=float,
    )

    y = np.asarray(
        [p["measured"] for p in points],
        dtype=float,
    )

    slope, intercept = np.polyfit(
        x,
        y,
        deg=1,
    )

    predicted = slope * x + intercept
    residual = y - predicted

    ss_res = float(
        np.sum(residual ** 2)
    )

    ss_tot = float(
        np.sum(
            (y - np.mean(y)) ** 2
        )
    )

    r2 = (
        1.0 - ss_res / ss_tot
        if ss_tot > 1e-15
        else None
    )

    command_error = y - x

    if abs(x[-1] - x[0]) > 1e-15:
        endpoint_sensitivity = float(
            (y[-1] - y[0])
            / (x[-1] - x[0])
        )
    else:
        endpoint_sensitivity = None

    return {
        "n_points": int(len(x)),

        "command_min": float(x.min()),
        "command_max": float(x.max()),

        "linear_fit_slope": float(slope),
        "linear_fit_intercept": float(intercept),

        "r_squared": (
            float(r2)
            if r2 is not None
            else None
        ),

        "fit_rmse": float(
            np.sqrt(
                np.mean(residual ** 2)
            )
        ),

        "mean_abs_command_response_error": float(
            np.mean(
                np.abs(command_error)
            )
        ),

        "max_abs_command_response_error": float(
            np.max(
                np.abs(command_error)
            )
        ),

        "mean_signed_command_response_bias": float(
            np.mean(command_error)
        ),

        "endpoint_sensitivity": (
            endpoint_sensitivity
        ),
    }


def main():
    summary = {
        "note": (
            "Preliminary M3 sensitivity summary based on "
            "existing M1 three-point sweeps. "
            "Do not interpret high R^2 as proof of global linearity."
        ),
        "axes": {},
    }

    rows = []

    for name, spec in AXES.items():
        points = load_axis(
            name,
            spec,
        )

        fit = fit_axis(points)

        axis_result = {
            "unit": spec["unit"],
            "response_kind":
                spec["response_kind"],
            "points": points,
            **fit,
        }

        summary["axes"][name] = axis_result

        rows.append({
            "axis": name,
            "kind": spec["response_kind"],
            "unit": spec["unit"],
            **fit,
        })

    output_dir = (
        RESULTS / "m3_authority"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_path = (
        output_dir
        / "preliminary_authority_v0.json"
    )

    csv_path = (
        output_dir
        / "preliminary_authority_v0.csv"
    )

    json_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    fieldnames = [
        "axis",
        "kind",
        "unit",
        "n_points",
        "command_min",
        "command_max",
        "linear_fit_slope",
        "linear_fit_intercept",
        "r_squared",
        "fit_rmse",
        "mean_abs_command_response_error",
        "max_abs_command_response_error",
        "mean_signed_command_response_bias",
        "endpoint_sensitivity",
    ]

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    print("=" * 112)
    print(
        "ICRA27 M3 PRELIMINARY "
        "1D ACTION SENSITIVITY"
    )
    print("=" * 112)

    print(
        f"{'axis':18s}"
        f"{'N':>4s}"
        f"{'slope':>12s}"
        f"{'intercept':>14s}"
        f"{'R^2':>12s}"
        f"{'fit RMSE':>14s}"
        f"{'cmd MAE':>14s}"
        f"{'max cmd err':>14s}"
    )

    print("-" * 112)

    for row in rows:
        r2 = row["r_squared"]

        print(
            f"{row['axis']:18s}"
            f"{row['n_points']:4d}"
            f"{row['linear_fit_slope']:12.6f}"
            f"{row['linear_fit_intercept']:14.6f}"
            f"{r2 if r2 is not None else float('nan'):12.6f}"
            f"{row['fit_rmse']:14.6f}"
            f"{row['mean_abs_command_response_error']:14.6f}"
            f"{row['max_abs_command_response_error']:14.6f}"
        )

    print("=" * 112)

    print()
    print("Important:")
    print(
        "  - duty_factor response is the PyMPC planned "
        "contact schedule, not MuJoCo physical contact."
    )
    print(
        "  - body_height compares requested base-height "
        "reference against measured base z; a constant "
        "offset is expected from the CoM/base correction."
    )
    print(
        "  - N=3 is preliminary characterization only."
    )

    print()
    print("JSON:", json_path)
    print("CSV :", csv_path)


if __name__ == "__main__":
    main()
