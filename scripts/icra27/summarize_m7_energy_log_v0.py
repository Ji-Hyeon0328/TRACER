#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


GRAVITY_M_S2 = 9.81


def safe_cot(
    energy_j: float,
    *,
    mass_kg: float,
    distance_m: float,
):
    if distance_m <= 1e-6:
        return None

    return float(
        energy_j
        / (
            mass_kg
            * GRAVITY_M_S2
            * distance_m
        )
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "log",
        type=Path,
    )

    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.log.read_text().splitlines()
        if line.strip()
    ]

    if not rows:
        raise SystemExit(
            "No energy rows found."
        )

    first = rows[0]
    last = rows[-1]

    mass_values = np.asarray(
        [
            row["robot_mass_kg"]
            for row in rows
        ],
        dtype=float,
    )

    if not np.allclose(
        mass_values,
        mass_values[0],
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError(
            "Robot mass changed within rollout."
        )

    mass_kg = float(
        mass_values[0]
    )

    start = np.asarray(
        first["base_position_pre_world"],
        dtype=float,
    )

    end = np.asarray(
        last["base_position_post_world"],
        dtype=float,
    )

    net_xy_m = float(
        np.linalg.norm(
            end[:2] - start[:2]
        )
    )

    forward_x_m = float(
        end[0] - start[0]
    )

    path_xy_m = 0.0

    for row in rows:
        pre = np.asarray(
            row[
                "base_position_pre_world"
            ],
            dtype=float,
        )

        post = np.asarray(
            row[
                "base_position_post_world"
            ],
            dtype=float,
        )

        path_xy_m += float(
            np.linalg.norm(
                post[:2] - pre[:2]
            )
        )

    energy = last[
        "cumulative_energy"
    ]

    elapsed_s = float(
        energy["elapsed_s"]
    )

    applied_abs_j = float(
        energy["applied_abs_j"]
    )

    applied_positive_j = float(
        energy["applied_positive_j"]
    )

    applied_signed_j = float(
        energy["applied_signed_j"]
    )

    commanded_abs_j = float(
        energy["commanded_abs_j"]
    )

    print(
        "rows                 :",
        len(rows),
    )

    print(
        "elapsed_s            :",
        f"{elapsed_s:.6f}",
    )

    print(
        "robot_mass_kg        :",
        f"{mass_kg:.6f}",
    )

    print(
        "forward_x_m          :",
        f"{forward_x_m:.6f}",
    )

    print(
        "net_xy_m             :",
        f"{net_xy_m:.6f}",
    )

    print(
        "path_xy_m            :",
        f"{path_xy_m:.6f}",
    )

    print()

    print(
        "applied_abs_j        :",
        f"{applied_abs_j:.6f}",
    )

    print(
        "applied_positive_j   :",
        f"{applied_positive_j:.6f}",
    )

    print(
        "applied_signed_j     :",
        f"{applied_signed_j:.6f}",
    )

    print(
        "commanded_abs_j      :",
        f"{commanded_abs_j:.6f}",
    )

    print()

    if elapsed_s > 0.0:
        print(
            "mean_abs_power_w      :",
            f"{applied_abs_j / elapsed_s:.6f}",
        )

        print(
            "mean_positive_power_w :",
            f"{applied_positive_j / elapsed_s:.6f}",
        )

    cot_abs_net = safe_cot(
        applied_abs_j,
        mass_kg=mass_kg,
        distance_m=net_xy_m,
    )

    cot_abs_path = safe_cot(
        applied_abs_j,
        mass_kg=mass_kg,
        distance_m=path_xy_m,
    )

    cot_pos_net = safe_cot(
        applied_positive_j,
        mass_kg=mass_kg,
        distance_m=net_xy_m,
    )

    cot_pos_path = safe_cot(
        applied_positive_j,
        mass_kg=mass_kg,
        distance_m=path_xy_m,
    )

    print()

    print(
        "CoT_abs_net_xy       :",
        cot_abs_net,
    )

    print(
        "CoT_abs_path_xy      :",
        cot_abs_path,
    )

    print(
        "CoT_positive_net_xy  :",
        cot_pos_net,
    )

    print(
        "CoT_positive_path_xy :",
        cot_pos_path,
    )

    print()
    print(
        "[ICRA27] mechanical-energy "
        "rollout summary: PASS"
    )


if __name__ == "__main__":
    main()
