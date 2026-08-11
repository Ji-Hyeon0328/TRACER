#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]

RUNNER = (
    ROOT
    / "scripts"
    / "icra27"
    / "run_m7_pympc_state_tap_terrain_seed_v0.py"
)

GRAVITY = 9.81

TERRAINS = (
    "flat",
    "low_friction",
    "rough_perlin",
)


def parse_seeds(
    text: str,
) -> list[int]:
    out = [
        int(x.strip())
        for x in text.split(",")
        if x.strip()
    ]

    if not out:
        raise ValueError(
            "At least one seed is required."
        )

    return out


def cot(
    energy_j: float,
    mass_kg: float,
    distance_m: float,
):
    if distance_m <= 1e-6:
        return None

    return float(
        energy_j
        / (
            mass_kg
            * GRAVITY
            * distance_m
        )
    )


def load_energy(
    path: Path,
):
    rows = [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]

    if not rows:
        raise RuntimeError(
            f"No energy rows in {path}"
        )

    first = rows[0]
    last = rows[-1]

    start = np.asarray(
        first["base_position_pre_world"],
        dtype=float,
    )

    end = np.asarray(
        last["base_position_post_world"],
        dtype=float,
    )

    path_xy = 0.0

    for row in rows:
        pre = np.asarray(
            row["base_position_pre_world"],
            dtype=float,
        )

        post = np.asarray(
            row["base_position_post_world"],
            dtype=float,
        )

        path_xy += float(
            np.linalg.norm(
                post[:2] - pre[:2]
            )
        )

    net_xy = float(
        np.linalg.norm(
            end[:2] - start[:2]
        )
    )

    forward_x = float(
        end[0] - start[0]
    )

    energy = last[
        "cumulative_energy"
    ]

    mass = float(
        last["robot_mass_kg"]
    )

    return {
        "samples":
            int(energy["samples"]),

        "elapsed_s":
            float(energy["elapsed_s"]),

        "mass_kg":
            mass,

        "forward_x_m":
            forward_x,

        "net_xy_m":
            net_xy,

        "path_xy_m":
            path_xy,

        "abs_j":
            float(
                energy["applied_abs_j"]
            ),

        "positive_j":
            float(
                energy["applied_positive_j"]
            ),

        "signed_j":
            float(
                energy["applied_signed_j"]
            ),

        "commanded_abs_j":
            float(
                energy["commanded_abs_j"]
            ),
    }


def parse_m4(
    text: str,
):
    unsafe = (
        "[M4 standalone]"
        in text
        and "-> UNSAFE"
        in text
    )

    override = (
        "[M4 standalone ACTIVE]"
        in text
    )

    native_termination = (
        "Environment terminated"
        in text
    )

    return (
        unsafe,
        override,
        native_termination,
    )


def mean_std(
    values,
):
    values = [
        float(x)
        for x in values
        if x is not None
        and np.isfinite(x)
    ]

    if not values:
        return None, None

    mean = statistics.mean(
        values
    )

    std = (
        statistics.stdev(values)
        if len(values) >= 2
        else 0.0
    )

    return mean, std


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seeds",
        default="0,1,2,3,4",
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
    )

    parser.add_argument(
        "--base-port",
        type=int,
        default=52210,
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=(
            ROOT
            / "results"
            / "icra27"
            / "m7_energy_nominal_v0"
        ),
    )

    args = parser.parse_args()

    seeds = parse_seeds(
        args.seeds
    )

    args.out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = []

    run_index = 0

    for terrain in TERRAINS:
        for seed in seeds:
            run_dir = (
                args.out_dir
                / terrain
                / f"seed_{seed:05d}"
            )

            run_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            energy_log = (
                run_dir
                / "energy.jsonl"
            )

            runner_log = (
                run_dir
                / "runner.log"
            )

            if energy_log.exists():
                energy_log.unlink()

            command_port = (
                args.base_port
                + 10 * run_index
            )

            run_index += 1

            env = os.environ.copy()

            env[
                "TRACER_M7_ENERGY_LOG"
            ] = str(
                energy_log
            )

            env[
                "TRACER_M7_STATE_PORT"
            ] = str(
                command_port + 2
            )

            env[
                "TRACER_M7_STATE_HZ"
            ] = "50"

            command = [
                sys.executable,
                str(RUNNER),

                "--terrain",
                terrain,

                "--rough-height-scale",
                "1.0",

                "--command-port",
                str(command_port),

                "--telemetry-port",
                str(command_port + 1),

                "--telemetry-hz",
                "50",

                "--duration",
                str(args.duration),

                "--seed",
                str(seed),

                "--no-render",
            ]

            print(
                f"[run] terrain={terrain:12s} "
                f"seed={seed:03d}"
            )

            with runner_log.open(
                "w",
                encoding="utf-8",
            ) as handle:
                proc = subprocess.run(
                    command,
                    cwd=ROOT,
                    env=env,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    check=False,
                )

            log_text = (
                runner_log.read_text(
                    encoding="utf-8",
                    errors="replace",
                )
            )

            (
                m4_unsafe,
                m4_override,
                native_termination,
            ) = parse_m4(
                log_text
            )

            record = {
                "terrain":
                    terrain,

                "seed":
                    seed,

                "returncode":
                    int(proc.returncode),

                "m4_unsafe":
                    bool(m4_unsafe),

                "m4_override":
                    bool(m4_override),

                "native_termination":
                    bool(native_termination),
            }

            if (
                proc.returncode == 0
                and energy_log.exists()
            ):
                measured = load_energy(
                    energy_log
                )

                record.update(
                    measured
                )

                record[
                    "mean_abs_power_w"
                ] = (
                    measured["abs_j"]
                    / measured["elapsed_s"]
                )

                record[
                    "mean_positive_power_w"
                ] = (
                    measured["positive_j"]
                    / measured["elapsed_s"]
                )

                record[
                    "cot_abs_net"
                ] = cot(
                    measured["abs_j"],
                    measured["mass_kg"],
                    measured["net_xy_m"],
                )

                record[
                    "cot_abs_path"
                ] = cot(
                    measured["abs_j"],
                    measured["mass_kg"],
                    measured["path_xy_m"],
                )

                record[
                    "cot_positive_net"
                ] = cot(
                    measured["positive_j"],
                    measured["mass_kg"],
                    measured["net_xy_m"],
                )

                record[
                    "command_applied_abs_error_j"
                ] = abs(
                    measured["commanded_abs_j"]
                    - measured["abs_j"]
                )

                print(
                    "      "
                    f"Eabs={measured['abs_j']:.2f} J "
                    f"Pabs={record['mean_abs_power_w']:.2f} W "
                    f"CoT={record['cot_abs_net']:.3f} "
                    f"M4={m4_unsafe}"
                )

            else:
                print(
                    "      RUN FAILURE "
                    f"returncode={proc.returncode}"
                )

            records.append(
                record
            )

    csv_path = (
        args.out_dir
        / "runs.csv"
    )

    fieldnames = sorted(
        {
            key
            for row in records
            for key in row
        }
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            records
        )

    aggregates = {}

    print()
    print("=" * 88)
    print(
        "ICRA27 NOMINAL MECHANICAL-ENERGY CHARACTERIZATION"
    )
    print("=" * 88)

    for terrain in TERRAINS:
        rows = [
            row
            for row in records
            if row["terrain"] == terrain
            and "abs_j" in row
        ]

        if not rows:
            continue

        e_mean, e_std = mean_std(
            [
                row["abs_j"]
                for row in rows
            ]
        )

        p_mean, p_std = mean_std(
            [
                row["mean_abs_power_w"]
                for row in rows
            ]
        )

        cot_mean, cot_std = mean_std(
            [
                row["cot_abs_net"]
                for row in rows
            ]
        )

        path_ratio_mean, path_ratio_std = (
            mean_std(
                [
                    row["path_xy_m"]
                    / max(
                        row["net_xy_m"],
                        1e-9,
                    )
                    for row in rows
                ]
            )
        )

        m4_count = sum(
            bool(row["m4_unsafe"])
            for row in records
            if row["terrain"] == terrain
        )

        total_count = sum(
            1
            for row in records
            if row["terrain"] == terrain
        )

        aggregates[
            terrain
        ] = {
            "num_runs":
                len(rows),

            "energy_abs_j_mean":
                e_mean,

            "energy_abs_j_std":
                e_std,

            "mean_abs_power_w_mean":
                p_mean,

            "mean_abs_power_w_std":
                p_std,

            "cot_abs_net_mean":
                cot_mean,

            "cot_abs_net_std":
                cot_std,

            "path_over_net_mean":
                path_ratio_mean,

            "path_over_net_std":
                path_ratio_std,

            "m4_unsafe_count":
                m4_count,

            "total_attempts":
                total_count,
        }

        print(
            f"{terrain:12s} "
            f"Eabs={e_mean:8.3f}±{e_std:6.3f} J  "
            f"Pabs={p_mean:7.3f}±{p_std:6.3f} W  "
            f"CoT={cot_mean:6.3f}±{cot_std:6.3f}  "
            f"M4={m4_count}/{total_count}"
        )

    summary_path = (
        args.out_dir
        / "summary.json"
    )

    summary_path.write_text(
        json.dumps(
            {
                "schema":
                    "icra27_m7_energy_nominal_v0",

                "duration_s":
                    float(args.duration),

                "seeds":
                    seeds,

                "terrains":
                    list(TERRAINS),

                "aggregates":
                    aggregates,
            },
            indent=2,
        )
        + "\n"
    )

    print("=" * 88)
    print(
        "runs CSV   :",
        csv_path,
    )
    print(
        "summary JSON:",
        summary_path,
    )
    print()
    print(
        "[ICRA27] nominal energy characterization: PASS"
    )


if __name__ == "__main__":
    main()
