#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
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


DEFAULT_MUS = (
    0.8,
    0.6,
    0.5,
    0.4,
    0.3,
    0.2,
)

DEFAULT_SEEDS = (
    0,
    1,
    2,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mus",
        type=float,
        nargs="+",
        default=DEFAULT_MUS,
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_SEEDS,
    )

    parser.add_argument(
        "--fixed-vx",
        type=float,
        default=0.20,
        help=(
            "Fixed physical forward command "
            "in m/s. Valid M7 range: [0, 0.4]."
        ),
    )

    parser.add_argument(
        "--goal-distance",
        type=float,
        default=0.50,
    )

    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=25,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            ROOT
            / "results"
            / "icra27"
            / "m7_friction_characterization_v0"
        ),
    )

    return parser.parse_args()


def run_episode(
    *,
    env: PyMPCM7Env,
    seed: int,
    fixed_action,
) -> dict:
    obs, info = env.reset(
        seed=seed
    )

    k = len(
        env.oracle_context
    )

    # Observation contract:
    # context[K]
    # goal dx, dy, distance, heading
    # vx, vy, yaw_rate
    # z, roll, pitch
    # applied[4]
    # previous_action[4]
    vx_index = k + 4
    vy_index = k + 5
    base_z_index = k + 7
    roll_index = k + 8
    pitch_index = k + 9

    max_abs_roll = abs(
        float(obs[roll_index])
    )

    max_abs_pitch = abs(
        float(obs[pitch_index])
    )

    action = np.asarray(
        fixed_action,
        dtype=np.float32,
    ).reshape(4)

    episode_return = 0.0

    measured_vx = []
    measured_vy = []
    measured_base_z = []

    first_intervention_step = None
    intervention_count = 0
    watch_count = 0

    final_info = dict(info)

    terminated = False
    truncated = False

    runtime_failure = None
    runtime_error = None
    runner_exit_code = None

    steps = 0

    for step in range(
        env.max_episode_steps
    ):
        try:
            (
                obs,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(
                action
            )

        except TimeoutError as exc:
            steps = step + 1

            runtime_error = str(exc)

            proc = env.runner_process

            runner_exit_code = (
                None
                if proc is None
                else proc.poll()
            )

            runtime_failure = (
                "state_timeout"
                if runner_exit_code is None
                else "runner_exit"
            )

            break

        except RuntimeError as exc:
            if (
                "runner is not alive"
                not in str(exc).lower()
            ):
                raise

            steps = step + 1
            runtime_error = str(exc)

            proc = env.runner_process

            runner_exit_code = (
                None
                if proc is None
                else proc.poll()
            )

            runtime_failure = (
                "runner_exit"
            )

            break

        steps = step + 1

        episode_return += float(
            reward
        )

        measured_vx.append(
            float(obs[vx_index])
        )

        measured_vy.append(
            float(obs[vy_index])
        )

        measured_base_z.append(
            float(obs[base_z_index])
        )

        roll = abs(
            float(
                obs[
                    roll_index
                ]
            )
        )

        pitch = abs(
            float(
                obs[
                    pitch_index
                ]
            )
        )

        max_abs_roll = max(
            max_abs_roll,
            roll,
        )

        max_abs_pitch = max(
            max_abs_pitch,
            pitch,
        )

        safety_state = str(
            info.get(
                "safety_state",
                "",
            )
        ).strip().lower()

        if safety_state == "watch":
            watch_count += 1

        intervention = bool(
            info.get(
                "m4_intervention",
                False,
            )
        )

        if intervention:
            intervention_count += 1

            if (
                first_intervention_step
                is None
            ):
                first_intervention_step = (
                    step
                )

        final_info = dict(info)

        if terminated or truncated:
            break

    final_goal_distance = float(
        final_info[
            "goal_distance"
        ]
    )

    success = bool(
        terminated
        and final_goal_distance
        <= env.success_radius_m
        and not bool(
            final_info.get(
                "m4_terminal",
                False,
            )
        )
    )

    m4_terminal = bool(
        final_info.get(
            "m4_terminal",
            False,
        )
    )

    if runtime_failure is not None:
        outcome = runtime_failure

    elif success:
        outcome = "success"

    elif m4_terminal:
        outcome = "m4_terminal"

    elif terminated:
        outcome = "native_terminal"

    elif truncated:
        outcome = "time_limit"

    else:
        outcome = "incomplete"

    vx_array = np.asarray(
        measured_vx,
        dtype=float,
    )

    vy_array = np.asarray(
        measured_vy,
        dtype=float,
    )

    z_array = np.asarray(
        measured_base_z,
        dtype=float,
    )

    commanded_vx = (
        0.20
        + 0.20 * float(
            fixed_action[0]
        )
    )

    vx_rmse = float(
        np.sqrt(
            np.mean(
                (
                    vx_array
                    - commanded_vx
                ) ** 2
            )
        )
    )

    return {
        "seed":
            int(seed),

        "mean_measured_vx_mps":
            float(np.mean(vx_array)),

        "vx_tracking_rmse_mps":
            vx_rmse,

        "max_abs_vy_mps":
            float(
                np.max(
                    np.abs(vy_array)
                )
            ),

        "min_base_z_m":
            float(np.min(z_array)),

        "steps":
            int(steps),

        "success":
            bool(success),

        "outcome":
            outcome,

        "runtime_failure":
            runtime_failure,

        "runtime_error":
            runtime_error,

        "runner_exit_code":
            runner_exit_code,

        "episode_return":
            float(episode_return),

        "final_goal_distance_m":
            final_goal_distance,

        "first_m4_intervention_step":
            first_intervention_step,

        "m4_intervention_count":
            int(intervention_count),

        "watch_count":
            int(watch_count),

        "max_abs_roll_rad":
            float(max_abs_roll),

        "max_abs_pitch_rad":
            float(max_abs_pitch),

        "terminated":
            bool(terminated),

        "truncated":
            bool(truncated),
    }


def main():
    args = parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not (
        0.0
        <= float(args.fixed_vx)
        <= 0.40
    ):
        raise ValueError(
            "--fixed-vx must be in [0, 0.4]"
        )

    # M7 frozen vx mapping:
    # normalized -1 -> 0.0 m/s
    # normalized  0 -> 0.2 m/s
    # normalized +1 -> 0.4 m/s
    normalized_vx = (
        float(args.fixed_vx) - 0.20
    ) / 0.20

    fixed_action = np.array(
        [
            normalized_vx,
            0.0,
            0.0,
            0.0,
        ],
        dtype=np.float32,
    )

    rows = []

    command_port = 51010
    telemetry_port = 51011
    state_port = 51012

    print("=" * 78)
    print(
        "ICRA27 M7 NOMINAL FRICTION "
        "CHARACTERIZATION"
    )
    print("=" * 78)

    print(
        "controller assumed mu : "
        "0.5 (frozen PyMPC config)"
    )

    print(
        "fixed vx               : "
        f"{args.fixed_vx:.3f} m/s"
    )

    print(
        "normalized action      : "
        f"{fixed_action.tolist()}"
    )

    print(
        "friction probes        : "
        f"{list(args.mus)}"
    )

    print(
        "seeds                  : "
        f"{list(args.seeds)}"
    )

    print("=" * 78)

    for mu in args.mus:
        env = PyMPCM7Env(
            terrain="low_friction",
            terrain_friction=float(mu),

            goal_distance_m=(
                args.goal_distance
            ),

            max_episode_steps=(
                args.max_episode_steps
            ),

            terminate_on_m4_unsafe=True,

            command_port=command_port,
            telemetry_port=telemetry_port,
            state_port=state_port,

            log_dir=(
                args.output_dir
                / (
                    f"mu_{mu:.3f}"
                )
            ),
        )

        try:
            for seed in args.seeds:
                result = run_episode(
                    env=env,
                    seed=seed,
                    fixed_action=fixed_action,
                )

                result[
                    "fixed_vx_mps"
                ] = float(
                    args.fixed_vx
                )

                result[
                    "ground_friction"
                ] = float(mu)

                result[
                    "controller_mu"
                ] = 0.5

                rows.append(
                    result
                )

                print(
                    f"mu={mu:.3f} "
                    f"seed={seed:02d} "
                    f"outcome="
                    f"{result['outcome']:<12s} "
                    f"steps={result['steps']:02d} "
                    f"goal="
                    f"{result['final_goal_distance_m']:.3f} "
                    f"return="
                    f"{result['episode_return']:+.3f} "
                    f"watch="
                    f"{result['watch_count']:02d} "
                    f"m4="
                    f"{result['m4_intervention_count']:02d} "
                    f"roll="
                    f"{np.degrees(result['max_abs_roll_rad']):.1f}deg "
                    f"pitch="
                    f"{np.degrees(result['max_abs_pitch_rad']):.1f}deg "
                    f"vx_meas="
                    f"{result['mean_measured_vx_mps']:.3f} "
                    f"vx_rmse="
                    f"{result['vx_tracking_rmse_mps']:.3f}"
                )

        finally:
            env.close()

    csv_path = (
        args.output_dir
        / "episodes.csv"
    )

    fieldnames = [
        "ground_friction",
        "controller_mu",
        "fixed_vx_mps",
        "seed",
        "mean_measured_vx_mps",
        "vx_tracking_rmse_mps",
        "max_abs_vy_mps",
        "min_base_z_m",
        "outcome",
        "runtime_failure",
        "runtime_error",
        "runner_exit_code",
        "success",
        "steps",
        "episode_return",
        "final_goal_distance_m",
        "first_m4_intervention_step",
        "m4_intervention_count",
        "watch_count",
        "max_abs_roll_rad",
        "max_abs_pitch_rad",
        "terminated",
        "truncated",
    ]

    with open(
        csv_path,
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(
                row
            )

    summary = {}

    for mu in args.mus:
        group = [
            row
            for row in rows
            if abs(
                row[
                    "ground_friction"
                ]
                - float(mu)
            ) < 1e-12
        ]

        summary[
            f"{mu:.3f}"
        ] = {
            "episodes":
                len(group),

            "success_rate":
                float(
                    np.mean(
                        [
                            row["success"]
                            for row in group
                        ]
                    )
                ),

            "m4_terminal_rate":
                float(
                    np.mean(
                        [
                            row["outcome"]
                            == "m4_terminal"
                            for row in group
                        ]
                    )
                ),

            "mean_final_goal_distance_m":
                float(
                    np.mean(
                        [
                            row[
                                "final_goal_distance_m"
                            ]
                            for row in group
                        ]
                    )
                ),

            "mean_episode_return":
                float(
                    np.mean(
                        [
                            row[
                                "episode_return"
                            ]
                            for row in group
                        ]
                    )
                ),

            "mean_watch_count":
                float(
                    np.mean(
                        [
                            row[
                                "watch_count"
                            ]
                            for row in group
                        ]
                    )
                ),

            "max_roll_deg":
                float(
                    np.degrees(
                        max(
                            row[
                                "max_abs_roll_rad"
                            ]
                            for row in group
                        )
                    )
                ),

            "max_pitch_deg":
                float(
                    np.degrees(
                        max(
                            row[
                                "max_abs_pitch_rad"
                            ]
                            for row in group
                        )
                    )
                ),
        }

    summary_path = (
        args.output_dir
        / "summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)

    for mu in args.mus:
        s = summary[
            f"{mu:.3f}"
        ]

        print(
            f"mu={mu:.3f} "
            f"success="
            f"{s['success_rate']:.2f} "
            f"m4_terminal="
            f"{s['m4_terminal_rate']:.2f} "
            f"goal="
            f"{s['mean_final_goal_distance_m']:.3f} "
            f"return="
            f"{s['mean_episode_return']:+.3f} "
            f"watch="
            f"{s['mean_watch_count']:.2f} "
            f"max_roll="
            f"{s['max_roll_deg']:.1f}deg "
            f"max_pitch="
            f"{s['max_pitch_deg']:.1f}deg"
        )

    print()
    print(
        "episodes :",
        csv_path,
    )

    print(
        "summary  :",
        summary_path,
    )


if __name__ == "__main__":
    main()
