#!/usr/bin/env python3

from __future__ import annotations

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


from tracer_core.highlevel_rl.pympc_env import (
    PyMPCM7Env,
)


SETTLING_STEPS = 5
MEASURE_STEPS = 10

# IMPORTANT:
# rough_perlin calibration must use TRAIN-only seeds.
CONDITIONS = (
    (
        "flat",
        (0,),
    ),
    (
        "low_friction",
        (0,),
    ),
    (
        "rough_perlin",
        (13, 7, 15, 0, 5),
    ),
)

METRIC_KEYS = (
    "cost_motion",
    "cost_stability",
    "cost_energy",
    "progress_rate_mps",
    "energy_power_ratio",
    "objective_cost",
)


def mean(values):
    return float(
        statistics.mean(
            float(x)
            for x in values
        )
    )


def run_one(
    *,
    terrain: str,
    seed: int,
    run_index: int,
) -> dict:
    port = (
        52710
        + 10 * run_index
    )

    log_dir = (
        ROOT
        / "results"
        / "icra27"
        / "m7_cost_vector_v2"
        / terrain
        / f"seed_{seed:03d}"
    )

    env = PyMPCM7Env(
        terrain=terrain,

        reward_mode="tracer_cost_v2",

        # Prevent goal completion from terminating
        # this fixed-duration characterization.
        goal_distance_m=10.0,
        success_radius_m=0.05,

        decision_dt_s=0.20,
        max_episode_steps=25,

        terminate_on_m4_unsafe=True,

        command_port=port,
        telemetry_port=port + 1,
        state_port=port + 2,

        telemetry_hz=100.0,
        state_hz=100.0,

        log_dir=log_dir,
    )

    action = np.zeros(
        4,
        dtype=np.float32,
    )

    rows = []

    try:
        obs, _ = env.reset(
            seed=seed
        )

        if obs.shape != (21,):
            raise RuntimeError(
                f"Unexpected obs shape: "
                f"{obs.shape}"
            )

        # ------------------------------------------
        # 1 s settling, excluded from statistics.
        # ------------------------------------------
        for step in range(
            SETTLING_STEPS
        ):
            (
                _,
                _,
                terminated,
                truncated,
                info,
            ) = env.step(
                action
            )

            if (
                terminated
                or truncated
            ):
                return {
                    "terrain":
                        terrain,

                    "seed":
                        seed,

                    "valid":
                        False,

                    "failure_phase":
                        "settling",

                    "failure_step":
                        step,

                    "safety_state":
                        info.get(
                            "safety_state"
                        ),

                    "m4_intervention":
                        bool(
                            info.get(
                                "m4_intervention",
                                False,
                            )
                        ),
                }

        # ------------------------------------------
        # 2 s post-settling measurement.
        # ------------------------------------------
        for step in range(
            MEASURE_STEPS
        ):
            (
                obs,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(
                action
            )

            rc = info[
                "reward_components"
            ]

            if (
                info.get(
                    "reward_schema"
                )
                !=
                "icra27_simplified_tracer_cost_v2"
            ):
                raise RuntimeError(
                    "Unexpected reward schema: "
                    f"{info.get('reward_schema')!r}"
                )

            row = {
                key:
                    float(rc[key])
                for key in METRIC_KEYS
            }

            row[
                "objective_reward"
            ] = float(
                reward
            )

            row[
                "m4_intervention"
            ] = bool(
                info[
                    "m4_intervention"
                ]
            )

            row[
                "safety_state"
            ] = str(
                info[
                    "safety_state"
                ]
            )

            rows.append(
                row
            )

            if abs(
                row[
                    "objective_reward"
                ]
                + row[
                    "objective_cost"
                ]
            ) > 1e-9:
                raise RuntimeError(
                    "R != -J"
                )

            if (
                terminated
                or truncated
            ):
                return {
                    "terrain":
                        terrain,

                    "seed":
                        seed,

                    "valid":
                        False,

                    "failure_phase":
                        "measurement",

                    "failure_step":
                        step,

                    "safety_state":
                        info.get(
                            "safety_state"
                        ),

                    "m4_intervention":
                        bool(
                            info.get(
                                "m4_intervention",
                                False,
                            )
                        ),

                    "partial_rows":
                        rows,
                }

        summary = {
            key:
                mean(
                    row[key]
                    for row in rows
                )
            for key in METRIC_KEYS
        }

        summary[
            "objective_reward"
        ] = mean(
            row[
                "objective_reward"
            ]
            for row in rows
        )

        return {
            "terrain":
                terrain,

            "seed":
                seed,

            "valid":
                True,

            "num_samples":
                len(rows),

            "summary":
                summary,
        }

    finally:
        env.close()


def main():
    out_dir = (
        ROOT
        / "results"
        / "icra27"
        / "m7_cost_vector_v2"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    runs = []

    run_index = 0

    print(
        "=" * 88
    )
    print(
        "ICRA27 SIMPLIFIED TRACER "
        "TERRAIN COST-VECTOR CHARACTERIZATION"
    )
    print(
        "=" * 88
    )

    for (
        terrain,
        seeds,
    ) in CONDITIONS:

        print()
        print(
            f"[{terrain}]"
        )

        for seed in seeds:
            result = run_one(
                terrain=terrain,
                seed=seed,
                run_index=run_index,
            )

            run_index += 1
            runs.append(
                result
            )

            if not result[
                "valid"
            ]:
                print(
                    f"  seed={seed:2d} "
                    "INVALID "
                    f"phase="
                    f"{result.get('failure_phase')} "
                    f"M4="
                    f"{result.get('m4_intervention')} "
                    f"safety="
                    f"{result.get('safety_state')}"
                )

                continue

            s = result[
                "summary"
            ]

            print(
                f"  seed={seed:2d} "
                f"Cm={s['cost_motion']:.4f} "
                f"Cs={s['cost_stability']:.4f} "
                f"CE={s['cost_energy']:.4f} "
                f"prog="
                f"{s['progress_rate_mps']:.4f} "
                f"Eratio="
                f"{s['energy_power_ratio']:.4f} "
                f"J="
                f"{s['objective_cost']:.4f}"
            )

    # --------------------------------------------------
    # Terrain-level aggregation.
    #
    # For rough_perlin, aggregate SEED MEANS rather than
    # flattening all individual HL samples. This keeps
    # terrain realization as the experimental unit.
    # --------------------------------------------------

    terrain_summary = {}

    print()
    print(
        "=" * 88
    )
    print(
        "TERRAIN-LEVEL COST VECTOR"
    )
    print(
        "=" * 88
    )

    for (
        terrain,
        _,
    ) in CONDITIONS:
        valid = [
            run
            for run in runs
            if (
                run[
                    "terrain"
                ]
                == terrain
                and run[
                    "valid"
                ]
            )
        ]

        if not valid:
            terrain_summary[
                terrain
            ] = {
                "valid_runs":
                    0,
            }

            print(
                f"{terrain:15s}: "
                "NO VALID RUN"
            )

            continue

        stats = {
            "valid_runs":
                len(valid),

            "attempts":
                sum(
                    1
                    for run in runs
                    if run[
                        "terrain"
                    ] == terrain
                ),
        }

        for key in METRIC_KEYS:
            values = [
                run[
                    "summary"
                ][key]
                for run in valid
            ]

            stats[
                key + "_mean"
            ] = mean(
                values
            )

            stats[
                key + "_std"
            ] = (
                float(
                    statistics.stdev(
                        values
                    )
                )
                if len(values) > 1
                else 0.0
            )

        terrain_summary[
            terrain
        ] = stats

        print(
            f"{terrain:15s} "
            f"Cm="
            f"{stats['cost_motion_mean']:.4f} "
            f"Cs="
            f"{stats['cost_stability_mean']:.4f} "
            f"CE="
            f"{stats['cost_energy_mean']:.4f} "
            f"prog="
            f"{stats['progress_rate_mps_mean']:.4f} "
            f"Eratio="
            f"{stats['energy_power_ratio_mean']:.4f} "
            f"valid="
            f"{stats['valid_runs']}/"
            f"{stats['attempts']}"
        )

    payload = {
        "schema":
            "icra27_m7_cost_vector_v2",

        "settling_steps":
            SETTLING_STEPS,

        "measure_steps":
            MEASURE_STEPS,

        "rough_perlin_train_only_seeds":
            [13, 7, 15, 0, 5],

        "runs":
            runs,

        "terrain_summary":
            terrain_summary,
    }

    out_file = (
        out_dir
        / "summary.json"
    )

    out_file.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print()
    print(
        "summary:",
        out_file,
    )

    print()
    print(
        "[ICRA27] terrain cost-vector "
        "characterization: PASS"
    )


if __name__ == "__main__":
    main()
