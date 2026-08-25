#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(ROOT))


from tracer_core.highlevel_rl.ppo import (
    PPOActorCritic,
)

from tracer_core.highlevel_rl.pympc_env import (
    PyMPCM7Env,
)


TERRAIN_CONFIGS = {
    "flat": {
        "terrain": "flat",
        "terrain_friction": None,
        "rough_height_scale": 1.0,
    },
    "low_friction": {
        "terrain": "low_friction",
        "terrain_friction": 0.152,
        "rough_height_scale": 1.0,
    },
    "rough_boxes": {
        "terrain": "rough_boxes",
        "terrain_friction": None,
        "rough_height_scale": 0.75,
    },
}


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sigmas",
        nargs="+",
        type=float,
        default=[
            0.02,
            0.05,
            0.08,
            0.10,
            0.15,
        ],
    )

    parser.add_argument(
        "--terrains",
        nargs="+",
        choices=tuple(
            TERRAIN_CONFIGS.keys()
        ),
        default=[
            "flat",
            "low_friction",
            "rough_boxes",
        ],
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[
            0,
            1,
            2,
        ],
    )

    parser.add_argument(
        "--goal-distance",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=60,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/icra27/"
            "m7_exploration_v0"
        ),
    )

    return parser.parse_args()


def _safe_mean(
    values,
) -> float:
    if len(values) == 0:
        return float("nan")

    return float(
        np.mean(
            np.asarray(
                values,
                dtype=float,
            )
        )
    )


def _safe_max(
    values,
) -> float:
    if len(values) == 0:
        return float("nan")

    return float(
        np.max(
            np.asarray(
                values,
                dtype=float,
            )
        )
    )


def run_exploration_episode(
    *,
    env: PyMPCM7Env,
    sigma: float,
    seed: int,
    goal_distance_m: float,
) -> dict:
    obs, reset_info = env.reset(
        seed=seed
    )

    obs = np.asarray(
        obs,
        dtype=np.float32,
    )

    context_dim = (
        int(obs.shape[0]) - 18
    )

    if context_dim <= 0:
        raise RuntimeError(
            "Invalid M7 observation dimension: "
            f"{obs.shape}"
        )

    # Observation contract:
    #
    # context[K]
    # goal dx, dy, distance, heading
    # vx, vy, yaw rate, base_z, roll, pitch
    # applied command[4]
    # previous action[4]
    vx_index = context_dim + 4
    vy_index = context_dim + 5
    base_z_index = context_dim + 7
    roll_index = context_dim + 8
    pitch_index = context_dim + 9

    actor = PPOActorCritic(
        obs_dim=int(obs.shape[0]),
        act_dim=4,
        initial_std=float(sigma),
    )

    actor.eval()

    # Actor construction initializes hidden layers and consumes
    # torch RNG. Re-seed AFTER construction so action-sampling
    # randomness itself is paired across sigma conditions.
    torch.manual_seed(
        int(seed)
    )

    episode_return = 0.0
    steps = 0

    watch_count = 0
    intervention_count = 0
    first_intervention_step = None

    runtime_failure = None
    runtime_error = None
    runner_exit_code = None

    measured_vx = []
    commanded_vx = []
    measured_vy = []
    measured_base_z = []

    sampled_actions = []

    max_abs_roll = 0.0
    max_abs_pitch = 0.0

    final_info = dict(
        reset_info
    )

    initial_goal_distance = float(
        final_info.get(
            "goal_distance",
            goal_distance_m,
        )
    )

    terminated = False
    truncated = False

    for step in range(
        env.max_episode_steps
    ):
        action, _, _ = actor.act(
            obs,
            deterministic=False,
        )

        action = np.asarray(
            action,
            dtype=np.float32,
        )

        sampled_actions.append(
            action.copy()
        )

        # Frozen M7 vx action mapping:
        #
        # -1 -> 0.0 m/s
        #  0 -> 0.2 m/s
        # +1 -> 0.4 m/s
        command_vx = (
            0.20
            + 0.20
            * float(action[0])
        )

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

        obs = np.asarray(
            obs,
            dtype=np.float32,
        )

        steps = step + 1

        episode_return += float(
            reward
        )

        measured_vx.append(
            float(
                obs[vx_index]
            )
        )

        commanded_vx.append(
            command_vx
        )

        measured_vy.append(
            float(
                obs[vy_index]
            )
        )

        measured_base_z.append(
            float(
                obs[base_z_index]
            )
        )

        roll = abs(
            float(
                obs[roll_index]
            )
        )

        pitch = abs(
            float(
                obs[pitch_index]
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

        final_info = dict(
            info
        )

        if terminated or truncated:
            break

    final_goal_distance = float(
        final_info.get(
            "goal_distance",
            initial_goal_distance,
        )
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

    actions = np.asarray(
        sampled_actions,
        dtype=float,
    )

    measured_vx_array = np.asarray(
        measured_vx,
        dtype=float,
    )

    commanded_vx_array = np.asarray(
        commanded_vx,
        dtype=float,
    )

    if (
        measured_vx_array.size > 0
        and commanded_vx_array.size
        == measured_vx_array.size
    ):
        vx_tracking_rmse = float(
            np.sqrt(
                np.mean(
                    (
                        measured_vx_array
                        - commanded_vx_array
                    )
                    ** 2
                )
            )
        )
    else:
        vx_tracking_rmse = (
            float("nan")
        )

    if actions.size > 0:
        mean_abs_action = float(
            np.mean(
                np.abs(actions)
            )
        )

        rms_action = float(
            np.sqrt(
                np.mean(
                    actions ** 2
                )
            )
        )

        max_abs_action = float(
            np.max(
                np.abs(actions)
            )
        )

        per_dim_mean_abs = (
            np.mean(
                np.abs(actions),
                axis=0,
            )
        )
    else:
        mean_abs_action = (
            float("nan")
        )

        rms_action = (
            float("nan")
        )

        max_abs_action = (
            float("nan")
        )

        per_dim_mean_abs = (
            np.full(
                4,
                np.nan,
                dtype=float,
            )
        )

    goal_progress = (
        initial_goal_distance
        - final_goal_distance
    )

    return {
        "seed":
            int(seed),

        "sigma":
            float(sigma),

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

        "initial_goal_distance_m":
            float(
                initial_goal_distance
            ),

        "final_goal_distance_m":
            float(
                final_goal_distance
            ),

        "goal_progress_m":
            float(
                goal_progress
            ),

        "mean_measured_vx_mps":
            _safe_mean(
                measured_vx
            ),

        "vx_tracking_rmse_mps":
            vx_tracking_rmse,

        "max_abs_vy_mps":
            _safe_max(
                np.abs(
                    np.asarray(
                        measured_vy,
                        dtype=float,
                    )
                )
                if measured_vy
                else []
            ),

        "min_base_z_m":
            (
                float(
                    np.min(
                        np.asarray(
                            measured_base_z,
                            dtype=float,
                        )
                    )
                )
                if measured_base_z
                else float("nan")
            ),

        "first_m4_intervention_step":
            first_intervention_step,

        "m4_intervention_count":
            int(
                intervention_count
            ),

        "watch_count":
            int(
                watch_count
            ),

        "max_abs_roll_rad":
            float(
                max_abs_roll
            ),

        "max_abs_pitch_rad":
            float(
                max_abs_pitch
            ),

        "mean_abs_action":
            mean_abs_action,

        "rms_action":
            rms_action,

        "max_abs_action":
            max_abs_action,

        "mean_abs_action_vx":
            float(
                per_dim_mean_abs[0]
            ),

        "mean_abs_action_yaw":
            float(
                per_dim_mean_abs[1]
            ),

        "mean_abs_action_height":
            float(
                per_dim_mean_abs[2]
            ),

        "mean_abs_action_clearance":
            float(
                per_dim_mean_abs[3]
            ),

        "terminated":
            bool(terminated),

        "truncated":
            bool(truncated),
    }


def main():
    args = parse_args()

    for sigma in args.sigmas:
        if sigma <= 0.0:
            raise ValueError(
                "All sigma values must be > 0"
            )

    output_dir = (
        ROOT
        / args.output_dir
    ).resolve()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 78)
    print(
        "ICRA27 M7 PPO EXPLORATION "
        "CHARACTERIZATION"
    )
    print("=" * 78)

    print(
        "sigmas   :",
        args.sigmas,
    )

    print(
        "terrains :",
        args.terrains,
    )

    print(
        "seeds    :",
        args.seeds,
    )

    print(
        "sampling : PPOActorCritic.act("
        "deterministic=False)"
    )

    print("=" * 78)

    rows = []

    for terrain_name in (
        args.terrains
    ):
        cfg = TERRAIN_CONFIGS[
            terrain_name
        ]

        for sigma in args.sigmas:
            for seed in args.seeds:
                kwargs = {
                    "terrain":
                        cfg["terrain"],

                    "rough_height_scale":
                        cfg[
                            "rough_height_scale"
                        ],

                    "goal_distance_m":
                        args.goal_distance,

                    "max_episode_steps":
                        args.max_episode_steps,

                    "terminate_on_m4_unsafe":
                        True,
                }

                if (
                    cfg[
                        "terrain_friction"
                    ]
                    is not None
                ):
                    kwargs[
                        "terrain_friction"
                    ] = cfg[
                        "terrain_friction"
                    ]

                env = PyMPCM7Env(
                    **kwargs
                )

                try:
                    result = (
                        run_exploration_episode(
                            env=env,
                            sigma=sigma,
                            seed=seed,
                            goal_distance_m=(
                                args.goal_distance
                            ),
                        )
                    )

                finally:
                    env.close()

                result[
                    "terrain"
                ] = terrain_name

                result[
                    "terrain_friction"
                ] = (
                    cfg[
                        "terrain_friction"
                    ]
                    if cfg[
                        "terrain_friction"
                    ]
                    is not None
                    else (
                        0.8
                    )
                )

                result[
                    "rough_height_scale"
                ] = cfg[
                    "rough_height_scale"
                ]

                rows.append(
                    result
                )

                print(
                    f"terrain="
                    f"{terrain_name:<12} "
                    f"sigma={sigma:.2f} "
                    f"seed={seed:02d} "
                    f"outcome="
                    f"{result['outcome']:<13} "
                    f"steps="
                    f"{result['steps']:02d} "
                    f"goal="
                    f"{result['final_goal_distance_m']:.3f} "
                    f"m4="
                    f"{result['m4_intervention_count']:02d} "
                    f"watch="
                    f"{result['watch_count']:02d} "
                    f"vx_rmse="
                    f"{result['vx_tracking_rmse_mps']:.3f} "
                    f"|a|="
                    f"{result['mean_abs_action']:.3f}"
                )

    csv_path = (
        output_dir
        / "episodes.csv"
    )

    if rows:
        fieldnames = list(
            rows[0].keys()
        )

        with csv_path.open(
            "w",
            newline="",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
            )

            writer.writeheader()
            writer.writerows(
                rows
            )

    summaries = []

    for terrain_name in (
        args.terrains
    ):
        for sigma in args.sigmas:
            group = [
                row
                for row in rows
                if (
                    row["terrain"]
                    == terrain_name
                    and np.isclose(
                        row["sigma"],
                        sigma,
                    )
                )
            ]

            n = len(group)

            if n == 0:
                continue

            summary = {
                "terrain":
                    terrain_name,

                "sigma":
                    float(sigma),

                "episodes":
                    n,

                "success_rate":
                    float(
                        np.mean(
                            [
                                row[
                                    "success"
                                ]
                                for row
                                in group
                            ]
                        )
                    ),

                "m4_terminal_rate":
                    float(
                        np.mean(
                            [
                                row[
                                    "outcome"
                                ]
                                == "m4_terminal"
                                for row
                                in group
                            ]
                        )
                    ),

                "time_limit_rate":
                    float(
                        np.mean(
                            [
                                row[
                                    "outcome"
                                ]
                                == "time_limit"
                                for row
                                in group
                            ]
                        )
                    ),

                "runtime_failure_rate":
                    float(
                        np.mean(
                            [
                                row[
                                    "runtime_failure"
                                ]
                                is not None
                                for row
                                in group
                            ]
                        )
                    ),

                "mean_steps":
                    _safe_mean(
                        [
                            row["steps"]
                            for row
                            in group
                        ]
                    ),

                "mean_goal_progress_m":
                    _safe_mean(
                        [
                            row[
                                "goal_progress_m"
                            ]
                            for row
                            in group
                        ]
                    ),

                "mean_vx_tracking_rmse_mps":
                    _safe_mean(
                        [
                            row[
                                "vx_tracking_rmse_mps"
                            ]
                            for row
                            in group
                        ]
                    ),

                "mean_abs_action":
                    _safe_mean(
                        [
                            row[
                                "mean_abs_action"
                            ]
                            for row
                            in group
                        ]
                    ),

                "max_abs_action":
                    _safe_max(
                        [
                            row[
                                "max_abs_action"
                            ]
                            for row
                            in group
                        ]
                    ),

                "max_roll_deg":
                    float(
                        np.degrees(
                            max(
                                row[
                                    "max_abs_roll_rad"
                                ]
                                for row
                                in group
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
                                for row
                                in group
                            )
                        )
                    ),
            }

            summaries.append(
                summary
            )

    summary_path = (
        output_dir
        / "summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summaries,
            indent=2,
            sort_keys=True,
            allow_nan=True,
        )
        + "\n"
    )

    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)

    for summary in summaries:
        print(
            f"terrain="
            f"{summary['terrain']:<12} "
            f"sigma="
            f"{summary['sigma']:.2f} "
            f"success="
            f"{summary['success_rate']:.2f} "
            f"m4="
            f"{summary['m4_terminal_rate']:.2f} "
            f"time_limit="
            f"{summary['time_limit_rate']:.2f} "
            f"runtime="
            f"{summary['runtime_failure_rate']:.2f} "
            f"steps="
            f"{summary['mean_steps']:.1f} "
            f"progress="
            f"{summary['mean_goal_progress_m']:.3f} "
            f"vx_rmse="
            f"{summary['mean_vx_tracking_rmse_mps']:.3f} "
            f"|a|="
            f"{summary['mean_abs_action']:.3f}"
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
