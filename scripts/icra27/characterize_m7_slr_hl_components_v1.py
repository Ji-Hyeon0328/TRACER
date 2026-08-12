#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path
import statistics
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT),
    )


from tracer_core.highlevel_rl.pympc_env import (
    PyMPCM7Env,
)

from tracer_core.highlevel_rl.reward_slr_hl_v1 import (
    REWARD_SCHEMA,
    MAX_POSITIVE_REWARD_RATE,
    SCALE_TRACKING_LIN_VEL,
    SCALE_TRACKING_ANG_VEL,
    SCALE_LIN_VEL_Z,
    SCALE_ANG_VEL_XY,
    SCALE_BASE_HEIGHT,
    SCALE_POWER,
    SCALE_ACTION_RATE,
    SCALE_ACTION_SMOOTHNESS,
    SCALE_ORIENTATION,
)


SETTLING_STEPS = 5
MEASURE_STEPS = 10

# Keep exactly the same calibration conditions as the
# integrated TRACER reward-v2 characterizer.
#
# IMPORTANT:
# rough_perlin uses TRAIN-only seeds.
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
    # Raw SLR-HL reward components.
    "tracking_lin_vel",
    "tracking_ang_vel",
    "cost_lin_vel_z",
    "cost_ang_vel_xy",
    "cost_base_height",
    "power_w",
    "cost_action_rate",
    "cost_action_smoothness",
    "cost_orientation",

    # Reward-rate deficits relative to the mathematically
    # perfect SLR-HL tracking transition.
    "deficit_tracking_lin",
    "deficit_tracking_ang",
    "deficit_lin_vel_z",
    "deficit_ang_vel_xy",
    "deficit_base_height",
    "deficit_power",
    "deficit_action_rate",
    "deficit_action_smoothness",
    "deficit_orientation",

    "component_deficit_rate",
    "centered_deficit_rate",

    "weighted_rate_before_clip",
    "centered_reward",
    "positive_reward",

    "decision_dt_s",
    "energy_dt_s",
)


def mean(values):
    return float(
        statistics.mean(
            float(x)
            for x in values
        )
    )


def build_row(
    *,
    reward: float,
    info: dict,
) -> dict:
    rc = info[
        "reward_components"
    ]

    schema = info.get(
        "reward_schema"
    )

    if schema != REWARD_SCHEMA:
        raise RuntimeError(
            "Unexpected reward schema: "
            f"{schema!r}; "
            f"expected {REWARD_SCHEMA!r}"
        )

    if (
        rc.get("reward_schema")
        != REWARD_SCHEMA
    ):
        raise RuntimeError(
            "reward_components schema mismatch"
        )

    tracking_lin = float(
        rc["slr_tracking_lin_vel"]
    )

    tracking_ang = float(
        rc["slr_tracking_ang_vel"]
    )

    if not (
        -1e-12
        <= tracking_lin
        <= 1.0 + 1e-12
    ):
        raise RuntimeError(
            "linear tracking reward outside [0,1]"
        )

    if not (
        -1e-12
        <= tracking_ang
        <= 1.0 + 1e-12
    ):
        raise RuntimeError(
            "angular tracking reward outside [0,1]"
        )

    cost_vz = float(
        rc["slr_cost_lin_vel_z"]
    )

    cost_wxy = float(
        rc["slr_cost_ang_vel_xy"]
    )

    cost_height = float(
        rc["slr_cost_base_height"]
    )

    power_w = float(
        rc["slr_power_w"]
    )

    cost_action_rate = float(
        rc["slr_cost_action_rate"]
    )

    cost_action_smooth = float(
        rc["slr_cost_action_smoothness"]
    )

    cost_orientation = float(
        rc["slr_cost_orientation"]
    )

    # --------------------------------------------------
    # Weighted reward-rate deficits.
    #
    # Perfect transition:
    #
    #   +1.0 linear tracking
    #   +0.5 angular tracking
    #
    # gives maximum reward rate = 1.5.
    #
    # All quantities below are nonnegative losses from
    # that maximum.
    # --------------------------------------------------

    deficit_tracking_lin = (
        SCALE_TRACKING_LIN_VEL
        * (
            1.0
            - tracking_lin
        )
    )

    deficit_tracking_ang = (
        SCALE_TRACKING_ANG_VEL
        * (
            1.0
            - tracking_ang
        )
    )

    deficit_vz = (
        -SCALE_LIN_VEL_Z
        * cost_vz
    )

    deficit_wxy = (
        -SCALE_ANG_VEL_XY
        * cost_wxy
    )

    deficit_height = (
        -SCALE_BASE_HEIGHT
        * cost_height
    )

    deficit_power = (
        -SCALE_POWER
        * power_w
    )

    deficit_action_rate = (
        -SCALE_ACTION_RATE
        * cost_action_rate
    )

    deficit_action_smooth = (
        -SCALE_ACTION_SMOOTHNESS
        * cost_action_smooth
    )

    deficit_orientation = (
        -SCALE_ORIENTATION
        * cost_orientation
    )

    deficits = (
        deficit_tracking_lin,
        deficit_tracking_ang,
        deficit_vz,
        deficit_wxy,
        deficit_height,
        deficit_power,
        deficit_action_rate,
        deficit_action_smooth,
        deficit_orientation,
    )

    if any(
        value < -1e-12
        for value in deficits
    ):
        raise RuntimeError(
            "SLR-HL deficit became negative"
        )

    component_deficit_rate = float(
        sum(deficits)
    )

    weighted_rate = float(
        rc[
            "slr_hl_weighted_rate_before_clip"
        ]
    )

    expected_weighted_rate = (
        MAX_POSITIVE_REWARD_RATE
        - component_deficit_rate
    )

    if not np.isclose(
        weighted_rate,
        expected_weighted_rate,
        rtol=0.0,
        atol=1e-9,
    ):
        raise RuntimeError(
            "SLR-HL component algebra mismatch: "
            f"weighted={weighted_rate} "
            f"expected={expected_weighted_rate}"
        )

    dt = float(
        rc["decision_dt_s"]
    )

    energy_dt = float(
        rc["energy_dt_s"]
    )

    if dt <= 0.0:
        raise RuntimeError(
            "decision_dt_s must be positive"
        )

    if energy_dt <= 0.0:
        raise RuntimeError(
            "energy_dt_s must be positive"
        )

    centered_reward = float(
        rc["slr_hl_centered_reward"]
    )

    positive_reward = float(
        rc["slr_hl_positive_reward"]
    )

    expected_centered_reward = (
        dt
        * (
            max(
                0.0,
                expected_weighted_rate,
            )
            - MAX_POSITIVE_REWARD_RATE
        )
    )

    if not np.isclose(
        centered_reward,
        expected_centered_reward,
        rtol=0.0,
        atol=1e-9,
    ):
        raise RuntimeError(
            "SLR-HL centered reward "
            "algebra mismatch: "
            f"actual={centered_reward} "
            f"expected={expected_centered_reward}"
        )

    if centered_reward > 1e-10:
        raise RuntimeError(
            "centered SLR-HL reward must be <= 0"
        )

    # This fixed-duration characterization never reaches
    # goal/horizon during the measured valid transitions.
    # Therefore the common task-feasibility tail must be zero.
    task_tail = float(
        rc["task_feasibility_cost"]
    )

    if abs(task_tail) > 1e-12:
        raise RuntimeError(
            "Unexpected task-feasibility cost "
            "during valid characterization: "
            f"{task_tail}"
        )

    if not np.isclose(
        float(reward),
        centered_reward,
        rtol=0.0,
        atol=1e-9,
    ):
        raise RuntimeError(
            "Returned reward != centered SLR-HL reward"
        )

    # Nominal action is constant zero downstream.
    # Five settling steps make a_t, a_t-1, a_t-2 all zero.
    if cost_action_rate > 1e-12:
        raise RuntimeError(
            "Nominal action-rate cost is nonzero: "
            f"{cost_action_rate}"
        )

    if cost_action_smooth > 1e-12:
        raise RuntimeError(
            "Nominal action-smoothness cost is nonzero: "
            f"{cost_action_smooth}"
        )

    centered_deficit_rate = (
        -centered_reward
        / dt
    )

    return {
        "tracking_lin_vel":
            tracking_lin,

        "tracking_ang_vel":
            tracking_ang,

        "cost_lin_vel_z":
            cost_vz,

        "cost_ang_vel_xy":
            cost_wxy,

        "cost_base_height":
            cost_height,

        "power_w":
            power_w,

        "cost_action_rate":
            cost_action_rate,

        "cost_action_smoothness":
            cost_action_smooth,

        "cost_orientation":
            cost_orientation,

        "deficit_tracking_lin":
            deficit_tracking_lin,

        "deficit_tracking_ang":
            deficit_tracking_ang,

        "deficit_lin_vel_z":
            deficit_vz,

        "deficit_ang_vel_xy":
            deficit_wxy,

        "deficit_base_height":
            deficit_height,

        "deficit_power":
            deficit_power,

        "deficit_action_rate":
            deficit_action_rate,

        "deficit_action_smoothness":
            deficit_action_smooth,

        "deficit_orientation":
            deficit_orientation,

        "component_deficit_rate":
            component_deficit_rate,

        "centered_deficit_rate":
            centered_deficit_rate,

        "weighted_rate_before_clip":
            weighted_rate,

        "centered_reward":
            centered_reward,

        "positive_reward":
            positive_reward,

        "decision_dt_s":
            dt,

        "energy_dt_s":
            energy_dt,

        "safety_state":
            str(
                info["safety_state"]
            ),

        "m4_intervention":
            bool(
                info["m4_intervention"]
            ),
    }


def run_one(
    *,
    terrain: str,
    seed: int,
    run_index: int,
) -> dict:
    port = (
        52810
        + 10 * run_index
    )

    log_dir = (
        ROOT
        / "results"
        / "icra27"
        / "m7_slr_hl_components_v1"
        / terrain
        / f"seed_{seed:03d}"
    )

    env = PyMPCM7Env(
        terrain=terrain,

        reward_mode="slr_hl_v1",

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

    # Downstream 4-D nominal action.
    #
    # This is exactly equivalent to the frozen learned-policy
    # zero action [0,0,0] after the fixed-clearance wrapper:
    #
    #   [0,0,0] -> [0,0,0,0].
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

        friction = float(
            env.terrain_friction
        )

        # ------------------------------------------
        # 1 s settling; excluded from statistics.
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

            if terminated or truncated:
                return {
                    "terrain":
                        terrain,

                    "terrain_friction":
                        friction,

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
                _,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(
                action
            )

            # A terminal transition belongs to an invalid
            # characterization run because failure-tail shaping
            # would intentionally alter the reward.
            if terminated or truncated:
                return {
                    "terrain":
                        terrain,

                    "terrain_friction":
                        friction,

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

            row = build_row(
                reward=reward,
                info=info,
            )

            rows.append(
                row
            )

        if len(rows) != MEASURE_STEPS:
            raise RuntimeError(
                "Unexpected number of measured rows: "
                f"{len(rows)}"
            )

        summary = {
            key:
                mean(
                    row[key]
                    for row in rows
                )
            for key in METRIC_KEYS
        }

        return {
            "terrain":
                terrain,

            "terrain_friction":
                friction,

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
        / "m7_slr_hl_components_v1"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    runs = []

    run_index = 0

    print(
        "=" * 112
    )

    print(
        "ICRA27 SLR-HL-ADAPTED-V1 "
        "3-TERRAIN COMPONENT CHARACTERIZATION"
    )

    print(
        "=" * 112
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
                f"mu="
                f"{result['terrain_friction']:.3f} "
                f"Dv="
                f"{s['deficit_tracking_lin']:.5f} "
                f"Dw="
                f"{s['deficit_tracking_ang']:.5f} "
                f"Dvz="
                f"{s['deficit_lin_vel_z']:.5f} "
                f"Dwxy="
                f"{s['deficit_ang_vel_xy']:.5f} "
                f"Dh="
                f"{s['deficit_base_height']:.5f} "
                f"DP="
                f"{s['deficit_power']:.5f} "
                f"Dori="
                f"{s['deficit_orientation']:.5f} "
                f"Dtot="
                f"{s['component_deficit_rate']:.5f} "
                f"Rctr="
                f"{s['centered_reward']:.5f} "
                f"P="
                f"{s['power_w']:.2f}W"
            )

    # --------------------------------------------------
    # Terrain-level aggregation.
    #
    # Exactly like the canonical TRACER-v2 characterizer,
    # rough_perlin is aggregated over SEED MEANS.
    # --------------------------------------------------

    terrain_summary = {}

    print()
    print(
        "=" * 112
    )

    print(
        "TERRAIN-LEVEL SLR-HL COMPONENT DEFICITS"
    )

    print(
        "=" * 112
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

        attempts = sum(
            1
            for run in runs
            if run[
                "terrain"
            ] == terrain
        )

        if not valid:
            terrain_summary[
                terrain
            ] = {
                "valid_runs":
                    0,

                "attempts":
                    attempts,
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
                attempts,

            "terrain_friction_mean":
                mean(
                    run[
                        "terrain_friction"
                    ]
                    for run in valid
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
            f"mu="
            f"{stats['terrain_friction_mean']:.3f} "
            f"Dv="
            f"{stats['deficit_tracking_lin_mean']:.5f} "
            f"Dw="
            f"{stats['deficit_tracking_ang_mean']:.5f} "
            f"Dvz="
            f"{stats['deficit_lin_vel_z_mean']:.5f} "
            f"Dwxy="
            f"{stats['deficit_ang_vel_xy_mean']:.5f} "
            f"Dh="
            f"{stats['deficit_base_height_mean']:.5f} "
            f"DP="
            f"{stats['deficit_power_mean']:.5f} "
            f"Dori="
            f"{stats['deficit_orientation_mean']:.5f} "
            f"Dtot="
            f"{stats['component_deficit_rate_mean']:.5f} "
            f"Rctr="
            f"{stats['centered_reward_mean']:.5f} "
            f"P="
            f"{stats['power_w_mean']:.2f}W "
            f"valid="
            f"{stats['valid_runs']}/"
            f"{stats['attempts']}"
        )

    payload = {
        "schema":
            "icra27_m7_slr_hl_components_v1",

        "reward_schema":
            REWARD_SCHEMA,

        "frozen_slr_hl_baseline_commit":
            "984c121",

        "settling_steps":
            SETTLING_STEPS,

        "measure_steps":
            MEASURE_STEPS,

        "nominal_downstream_action":
            [0.0, 0.0, 0.0, 0.0],

        "rough_perlin_train_only_seeds":
            [13, 7, 15, 0, 5],

        "aggregation":
            "per-run transition mean; "
            "terrain statistics over run/seed means",

        "maximum_positive_reward_rate":
            MAX_POSITIVE_REWARD_RATE,

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
        "[ICRA27] SLR-HL 3-terrain "
        "component characterization: PASS"
    )


if __name__ == "__main__":
    main()
