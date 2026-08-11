#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(
    0,
    str(ROOT),
)


from tracer_core.highlevel_rl.contracts import (
    normalized_action_to_meta_gait,
)

from tracer_core.highlevel_rl.pympc_env import (
    PyMPCM7Env,
)


P_REF_ABS_W = 21.847058714679683


PROFILES_3D = {
    "nominal":
        (0.0, 0.0, 0.0),

    "vx_zero":
        (-1.0, 0.0, 0.0),

    "vx_slow":
        (-0.5, 0.0, 0.0),

    "vx_fast":
        (+0.5, 0.0, 0.0),

    "vx_max":
        (+1.0, 0.0, 0.0),

    "yaw_left":
        (0.0, +0.5, 0.0),

    "yaw_right":
        (0.0, -0.5, 0.0),

    "height_low":
        (0.0, 0.0, -0.5),

    "height_high":
        (0.0, 0.0, +0.5),
}


def action4(
    action3,
):
    """
    Frozen learned-action contract:
        [vx, yaw, height] + fixed clearance action 0.
    """

    a = np.asarray(
        [
            float(action3[0]),
            float(action3[1]),
            float(action3[2]),
            0.0,
        ],
        dtype=np.float32,
    )

    assert a.shape == (4,)

    return a


def physical_mapping(
    action,
):
    command = (
        normalized_action_to_meta_gait(
            action
        )
    )

    return {
        "vx":
            float(command.vx),

        "yaw_rate":
            float(command.yaw_rate),

        "body_height":
            float(command.body_height),

        "swing_clearance":
            float(command.swing_clearance),

        "gait_period":
            float(command.gait_period),

        "duty_factor":
            float(command.duty_factor),
    }


def seed_bank(
    terrain,
):
    # Flat and low-friction geometry are deterministic
    # in the current characterization setup.
    if terrain in {
        "flat",
        "low_friction",
    }:
        return [0]

    # Rough realization variability matters.
    if terrain == "rough_perlin":
        return [
            0,
            1,
            2,
            3,
            4,
        ]

    raise ValueError(
        f"Unsupported terrain {terrain!r}"
    )


def mean_std(
    values,
):
    values = [
        float(x)
        for x in values
        if np.isfinite(
            float(x)
        )
    ]

    if not values:
        return None, None

    mean = statistics.mean(
        values
    )

    std = (
        statistics.stdev(values)
        if len(values) > 1
        else 0.0
    )

    return mean, std


def run_one(
    *,
    terrain,
    profile_name,
    action3_value,
    seed,
    ports,
    settling_steps,
    measure_steps,
    out_dir,
):
    command_port = ports
    telemetry_port = ports + 1
    state_port = ports + 2

    run_dir = (
        out_dir
        / terrain
        / profile_name
        / f"seed_{seed:05d}"
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    env = PyMPCM7Env(
        terrain=terrain,

        # Goal is intentionally far away.
        # We are characterizing locomotion energy,
        # not goal completion here.
        goal_distance_m=10.0,
        success_radius_m=0.05,

        decision_dt_s=0.20,

        max_episode_steps=(
            settling_steps
            + measure_steps
            + 5
        ),

        terminate_on_m4_unsafe=True,

        command_port=command_port,
        telemetry_port=telemetry_port,
        state_port=state_port,

        telemetry_hz=100.0,
        state_hz=100.0,

        log_dir=run_dir,
    )

    nominal4 = action4(
        PROFILES_3D[
            "nominal"
        ]
    )

    target4 = action4(
        action3_value
    )

    target_physical = (
        physical_mapping(
            target4
        )
    )

    intervals = []

    m4_unsafe = False
    override_seen = False
    error = None

    try:
        env.reset(
            seed=seed
        )

        # ----------------------------------------------------
        # Settling: excluded from measurement.
        # ----------------------------------------------------

        for _ in range(
            settling_steps
        ):
            sample = (
                env.transport.step(
                    nominal4,
                    target_sim_dt=0.20,
                    timeout_s=10.0,
                )
            )

            if (
                sample.safety_state
                == "unsafe"
            ):
                m4_unsafe = True
                break

            if sample.override_active:
                override_seen = True

        # ----------------------------------------------------
        # Measured target interval.
        # ----------------------------------------------------

        if not m4_unsafe:
            for _ in range(
                measure_steps
            ):
                sample = (
                    env.transport.step(
                        target4,
                        target_sim_dt=0.20,
                        timeout_s=10.0,
                    )
                )

                if (
                    sample.safety_state
                    == "unsafe"
                ):
                    m4_unsafe = True

                if sample.override_active:
                    override_seen = True

                energy = (
                    sample.energy_interval
                )

                if energy is None:
                    raise RuntimeError(
                        "Missing mechanical-energy "
                        "interval"
                    )

                if (
                    energy.dt_s <= 0.0
                ):
                    raise RuntimeError(
                        "Invalid energy dt"
                    )

                normalized_abs = (
                    energy.applied_abs_j
                    / (
                        P_REF_ABS_W
                        * energy.dt_s
                    )
                )

                intervals.append(
                    {
                        "dt_s":
                            float(
                                energy.dt_s
                            ),

                        "abs_j":
                            float(
                                energy.applied_abs_j
                            ),

                        "positive_j":
                            float(
                                energy.applied_positive_j
                            ),

                        "signed_j":
                            float(
                                energy.applied_signed_j
                            ),

                        "normalized_abs_rate":
                            float(
                                normalized_abs
                            ),
                    }
                )

                if m4_unsafe:
                    break

    except Exception as exc:
        error = (
            f"{type(exc).__name__}: "
            f"{exc}"
        )

    finally:
        env.close()

    if intervals:
        total_dt = sum(
            row["dt_s"]
            for row in intervals
        )

        total_abs = sum(
            row["abs_j"]
            for row in intervals
        )

        total_positive = sum(
            row["positive_j"]
            for row in intervals
        )

        total_signed = sum(
            row["signed_j"]
            for row in intervals
        )

        normalized_abs_rate = (
            total_abs
            / (
                P_REF_ABS_W
                * total_dt
            )
        )

        mean_abs_power = (
            total_abs
            / total_dt
        )

    else:
        total_dt = 0.0
        total_abs = 0.0
        total_positive = 0.0
        total_signed = 0.0
        normalized_abs_rate = None
        mean_abs_power = None

    record = {
        "terrain":
            terrain,

        "profile":
            profile_name,

        "seed":
            int(seed),

        "action_vx_norm":
            float(target4[0]),

        "action_yaw_norm":
            float(target4[1]),

        "action_height_norm":
            float(target4[2]),

        "action_clearance_norm":
            float(target4[3]),

        "physical_vx":
            target_physical["vx"],

        "physical_yaw_rate":
            target_physical[
                "yaw_rate"
            ],

        "physical_body_height":
            target_physical[
                "body_height"
            ],

        "physical_clearance":
            target_physical[
                "swing_clearance"
            ],

        "num_energy_intervals":
            len(intervals),

        "measured_dt_s":
            float(total_dt),

        "applied_abs_j":
            float(total_abs),

        "applied_positive_j":
            float(total_positive),

        "applied_signed_j":
            float(total_signed),

        "mean_abs_power_w":
            mean_abs_power,

        "normalized_abs_rate":
            normalized_abs_rate,

        "m4_unsafe":
            bool(m4_unsafe),

        "override_seen":
            bool(override_seen),

        "error":
            error,
    }

    (
        run_dir
        / "energy_intervals.json"
    ).write_text(
        json.dumps(
            intervals,
            indent=2,
        )
        + "\n"
    )

    return record


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--settling-steps",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--measure-steps",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--base-port",
        type=int,
        default=52410,
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=(
            ROOT
            / "results"
            / "icra27"
            / "m7_action_energy_v0"
        ),
    )

    args = parser.parse_args()

    if args.settling_steps < 0:
        raise ValueError(
            "--settling-steps must be >= 0"
        )

    if args.measure_steps <= 0:
        raise ValueError(
            "--measure-steps must be > 0"
        )

    args.out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    terrains = (
        "flat",
        "low_friction",
        "rough_perlin",
    )

    records = []

    run_index = 0

    for terrain in terrains:
        for profile_name, action3_value in (
            PROFILES_3D.items()
        ):
            for seed in seed_bank(
                terrain
            ):
                port = (
                    args.base_port
                    + run_index * 10
                )

                run_index += 1

                print(
                    f"[run] "
                    f"terrain={terrain:12s} "
                    f"profile={profile_name:11s} "
                    f"seed={seed:03d}"
                )

                record = run_one(
                    terrain=terrain,
                    profile_name=profile_name,
                    action3_value=action3_value,
                    seed=seed,
                    ports=port,
                    settling_steps=(
                        args.settling_steps
                    ),
                    measure_steps=(
                        args.measure_steps
                    ),
                    out_dir=args.out_dir,
                )

                records.append(
                    record
                )

                print(
                    "      "
                    f"vx={record['physical_vx']:.3f} "
                    f"yaw={record['physical_yaw_rate']:+.3f} "
                    f"h={record['physical_body_height']:.3f} "
                    f"E~={record['normalized_abs_rate']} "
                    f"M4={record['m4_unsafe']} "
                    f"err={record['error']}"
                )

    csv_path = (
        args.out_dir
        / "runs.csv"
    )

    fieldnames = sorted(
        {
            key
            for record in records
            for key in record.keys()
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

    print()
    print("=" * 96)
    print(
        "ICRA27 3D ACTION-SPACE MECHANICAL ENERGY"
    )
    print("=" * 96)

    summary = {}

    for terrain in terrains:
        summary[terrain] = {}

        for profile_name in (
            PROFILES_3D
        ):
            rr = [
                row
                for row in records
                if (
                    row["terrain"]
                    == terrain
                    and row["profile"]
                    == profile_name
                )
            ]

            valid = [
                row
                for row in rr
                if (
                    row[
                        "normalized_abs_rate"
                    ]
                    is not None
                    and not row[
                        "m4_unsafe"
                    ]
                    and row[
                        "error"
                    ] is None
                    and row[
                        "num_energy_intervals"
                    ] == args.measure_steps
                )
            ]

            invalid = [
                row
                for row in rr
                if row not in valid
            ]

            rates = [
                row[
                    "normalized_abs_rate"
                ]
                for row in valid
            ]

            mean, std = mean_std(
                rates
            )

            m4_count = sum(
                bool(
                    row["m4_unsafe"]
                )
                for row in rr
            )

            summary[
                terrain
            ][
                profile_name
            ] = {
                "mean_normalized_abs_rate":
                    mean,

                "std_normalized_abs_rate":
                    std,

                "m4_unsafe":
                    m4_count,

                "attempts":
                    len(rr),

                "valid_runs":
                    len(valid),

                "invalid_runs":
                    len(invalid),
            }

            print(
                f"{terrain:12s} "
                f"{profile_name:11s} "
                f"E~={mean!s:>8} "
                f"std={std!s:>8} "
                f"M4={m4_count}/{len(rr)} "
                f"valid={len(valid)}/{len(rr)}"
            )

    (
        args.out_dir
        / "summary.json"
    ).write_text(
        json.dumps(
            {
                "schema":
                    "icra27_m7_action_energy_v0",

                "p_ref_abs_w":
                    P_REF_ABS_W,

                "settling_steps":
                    args.settling_steps,

                "measure_steps":
                    args.measure_steps,

                "profiles_3d":
                    {
                        key: list(value)
                        for key, value
                        in PROFILES_3D.items()
                    },

                "summary":
                    summary,
            },
            indent=2,
        )
        + "\n"
    )

    print("=" * 96)
    print(
        "runs CSV:",
        csv_path,
    )
    print()
    print(
        "[ICRA27] action-energy "
        "characterization: PASS"
    )


if __name__ == "__main__":
    main()
