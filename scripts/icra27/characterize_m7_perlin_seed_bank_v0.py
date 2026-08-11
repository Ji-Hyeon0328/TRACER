#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


from tracer_core.highlevel_rl.pympc_env import (
    PyMPCM7Env,
)

from tracer_core.highlevel_rl.fixed_clearance_wrapper import (
    M7FixedClearanceActionWrapper,
)


SETTLING_STEPS = 5
POLICY_HORIZON = 50


def classify_transition(
    *,
    info,
    terminated,
    truncated,
):
    if bool(
        info.get(
            "success",
            False,
        )
    ):
        return "success"

    if bool(
        info.get(
            "m4_intervention",
            False,
        )
    ):
        return "m4"

    if truncated:
        return "time_limit"

    if terminated:
        return "terminal"

    return "running"


def run_nominal_episode(
    env,
    *,
    seed,
):
    # --------------------------------------------------------
    # Reset
    # --------------------------------------------------------
    obs, info = env.reset(
        seed=seed
    )

    nominal_action = np.zeros(
        env.action_space.shape,
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # Five policy-excluded nominal settling steps.
    # --------------------------------------------------------
    for settling_step in range(
        SETTLING_STEPS
    ):
        (
            obs,
            _reward,
            terminated,
            truncated,
            info,
        ) = env.step(
            nominal_action
        )

        if terminated or truncated:
            return {
                "status":
                    "settling_fail",

                "settling_steps":
                    settling_step + 1,

                "policy_steps":
                    0,

                "safety_state":
                    info.get(
                        "safety_state"
                    ),

                "failure_reason":
                    info.get(
                        "failure_reason"
                    ),

                "goal_distance":
                    float(
                        info.get(
                            "goal_distance",
                            np.nan,
                        )
                    ),
            }

    # --------------------------------------------------------
    # Frozen nominal high-level command.
    #
    # Wrapper maps normalized:
    #   [0, 0, 0]
    # to downstream:
    #   [0, 0, 0, 0]
    #
    # Physical command therefore remains nominal:
    #   vx         = 0.20 m/s
    #   yaw        = 0
    #   body_h     = 0.30 m
    #   clearance  = 0.060 m
    # --------------------------------------------------------
    status = "running"

    for policy_step in range(
        POLICY_HORIZON
    ):
        (
            obs,
            _reward,
            terminated,
            truncated,
            info,
        ) = env.step(
            nominal_action
        )

        status = classify_transition(
            info=info,
            terminated=terminated,
            truncated=truncated,
        )

        if terminated or truncated:
            break

    return {
        "status":
            status,

        "settling_steps":
            SETTLING_STEPS,

        "policy_steps":
            policy_step + 1,

        "safety_state":
            info.get(
                "safety_state"
            ),

        "failure_reason":
            info.get(
                "failure_reason"
            ),

        "goal_distance":
            float(
                info.get(
                    "goal_distance",
                    np.nan,
                )
            ),
    }


def classify_seed(
    trials,
):
    successes = sum(
        row["status"] == "success"
        for row in trials
    )

    total = len(trials)

    if successes == total:
        return "VIABLE"

    if successes == 0:
        return "HARD"

    return "BOUNDARY"


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--seed-start",
        type=int,
        default=0,
    )

    ap.add_argument(
        "--num-seeds",
        type=int,
        default=30,
    )

    ap.add_argument(
        "--repeats",
        type=int,
        default=3,
    )

    ap.add_argument(
        "--base-port",
        type=int,
        default=51410,
    )

    ap.add_argument(
        "--out-dir",
        default=(
            "results/icra27/"
            "m7_perlin_seed_bank_v0"
        ),
    )

    args = ap.parse_args()

    if args.num_seeds <= 0:
        raise ValueError(
            "--num-seeds must be positive"
        )

    if args.repeats <= 0:
        raise ValueError(
            "--repeats must be positive"
        )

    out_dir = Path(
        args.out_dir
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    base_env = PyMPCM7Env(
        terrain="rough_perlin",
        terminate_on_m4_unsafe=True,

        goal_distance_m=2.0,
        success_radius_m=0.15,

        decision_dt_s=0.20,

        max_episode_steps=(
            POLICY_HORIZON
            + SETTLING_STEPS
        ),

        command_port=(
            args.base_port
        ),
        telemetry_port=(
            args.base_port + 1
        ),
        state_port=(
            args.base_port + 2
        ),

        command_repeat_hz=20.0,
        telemetry_hz=100.0,
        state_hz=100.0,

        log_dir=(
            out_dir
            / "env_logs"
        ),
    )

    env = M7FixedClearanceActionWrapper(
        base_env
    )

    if env.action_space.shape != (3,):
        raise RuntimeError(
            "Expected 3D policy action space; "
            f"got {env.action_space.shape}"
        )

    seeds = range(
        args.seed_start,
        args.seed_start
        + args.num_seeds,
    )

    summary = {
        "schema":
            "icra27_m7_perlin_seed_bank_v0",

        "terrain":
            "rough_perlin",

        "policy":
            "frozen_nominal_zero_action",

        "fixed_clearance_m":
            0.060,

        "settling_steps":
            SETTLING_STEPS,

        "policy_horizon":
            POLICY_HORIZON,

        "seed_start":
            int(args.seed_start),

        "num_seeds":
            int(args.num_seeds),

        "repeats":
            int(args.repeats),

        "seeds":
            {},

        "banks": {
            "VIABLE": [],
            "BOUNDARY": [],
            "HARD": [],
        },
    }

    print("=" * 96)
    print(
        "ICRA27 M7 ROUGH-PERLIN "
        "NOMINAL SEED-BANK CHARACTERIZATION"
    )
    print("=" * 96)

    print(
        "terrain       :",
        base_env.terrain_scene,
    )

    print(
        "friction      :",
        base_env.terrain_friction,
    )

    print(
        "oracle context:",
        base_env.oracle_context,
    )

    print(
        "seed range    :",
        f"{args.seed_start}.."
        f"{args.seed_start + args.num_seeds - 1}",
    )

    print(
        "repeats/seed  :",
        args.repeats,
    )

    print(
        "policy        : nominal zero action"
    )

    print(
        "clearance     : fixed 0.060 m"
    )

    print()

    try:
        for seed in seeds:
            trials = []

            for repeat in range(
                args.repeats
            ):
                row = run_nominal_episode(
                    env,
                    seed=seed,
                )

                row[
                    "repeat"
                ] = int(repeat)

                trials.append(
                    row
                )

            seed_class = classify_seed(
                trials
            )

            counts = {
                status: sum(
                    row["status"] == status
                    for row in trials
                )
                for status in (
                    "success",
                    "m4",
                    "settling_fail",
                    "time_limit",
                    "terminal",
                    "running",
                )
            }

            success_steps = [
                row["policy_steps"]
                for row in trials
                if row["status"]
                == "success"
            ]

            mean_success_steps = (
                float(
                    np.mean(
                        success_steps
                    )
                )
                if success_steps
                else None
            )

            summary[
                "seeds"
            ][
                str(seed)
            ] = {
                "class":
                    seed_class,

                "counts":
                    counts,

                "mean_success_steps":
                    mean_success_steps,

                "trials":
                    trials,
            }

            summary[
                "banks"
            ][
                seed_class
            ].append(
                int(seed)
            )

            pattern = "/".join(
                row["status"]
                for row in trials
            )

            mean_steps_text = (
                "n/a"
                if mean_success_steps is None
                else f"{mean_success_steps:.1f}"
            )

            print(
                f"seed={seed:05d} "
                f"{seed_class:<8} "
                f"success="
                f"{counts['success']}/"
                f"{args.repeats} "
                f"m4="
                f"{counts['m4']}/"
                f"{args.repeats} "
                f"settle_fail="
                f"{counts['settling_fail']}/"
                f"{args.repeats} "
                f"mean_success_steps="
                f"{mean_steps_text:<5} "
                f"[{pattern}]"
            )

    finally:
        env.close()

    # --------------------------------------------------------
    # Aggregate bank summary.
    # --------------------------------------------------------
    print()
    print("=" * 96)
    print("SEED BANKS")
    print("=" * 96)

    for bank in (
        "VIABLE",
        "BOUNDARY",
        "HARD",
    ):
        members = summary[
            "banks"
        ][
            bank
        ]

        print(
            f"{bank:<8}: "
            f"{len(members):2d}/"
            f"{args.num_seeds} "
            f"{members}"
        )

    summary_path = (
        out_dir
        / "seed_bank.json"
    )

    with open(
        summary_path,
        "w",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
            sort_keys=True,
        )

    print()
    print(
        "saved:",
        summary_path,
    )

    print()
    print(
        "[ICRA27] rough-Perlin "
        "seed-bank characterization: PASS"
    )


if __name__ == "__main__":
    main()
