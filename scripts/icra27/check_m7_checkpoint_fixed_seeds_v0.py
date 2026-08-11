#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


from tracer_core.highlevel_rl.ppo import (
    load_ppo_checkpoint,
)

from tracer_core.highlevel_rl.pympc_env import (
    PyMPCM7Env,
)

from tracer_core.highlevel_rl.fixed_clearance_wrapper import (
    M7FixedClearanceActionWrapper,
)


SETTLING_STEPS = 5
POLICY_HORIZON = 50

TERRAINS = (
    "flat",
    "low_friction",
    "rough_perlin",
)


def reset_with_settling(
    env,
    *,
    seed,
):
    obs, info = env.reset(
        seed=seed
    )

    nominal = np.zeros(
        env.action_space.shape,
        dtype=np.float32,
    )

    for k in range(
        SETTLING_STEPS
    ):
        (
            obs,
            _reward,
            terminated,
            truncated,
            info,
        ) = env.step(
            nominal
        )

        if terminated or truncated:
            return (
                obs,
                info,
                False,
                k + 1,
            )

    return (
        obs,
        info,
        True,
        SETTLING_STEPS,
    )


def classify(
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


def run_episode(
    env,
    *,
    model,
    mode,
    seed,
):
    (
        obs,
        info,
        settled,
        settling_steps,
    ) = reset_with_settling(
        env,
        seed=seed,
    )

    if not settled:
        return {
            "status":
                "settling_fail",

            "steps":
                0,

            "max_abs_action":
                0.0,

            "mean_abs_action":
                0.0,
        }

    action_abs = []

    terminated = False
    truncated = False

    status = "running"

    for step in range(
        POLICY_HORIZON
    ):
        if mode == "nominal":
            action = np.zeros(
                env.action_space.shape,
                dtype=np.float32,
            )

        elif mode == "trained":
            (
                action,
                _log_prob,
                _value,
            ) = model.act(
                obs,
                deterministic=True,
            )

            action = np.asarray(
                action,
                dtype=np.float32,
            )

        else:
            raise ValueError(
                mode
            )

        action_abs.append(
            np.abs(
                action
            )
        )

        (
            obs,
            _reward,
            terminated,
            truncated,
            info,
        ) = env.step(
            action
        )

        status = classify(
            info,
            terminated,
            truncated,
        )

        if terminated or truncated:
            break

    if action_abs:
        x = np.stack(
            action_abs
        )

        max_abs_action = float(
            np.max(x)
        )

        mean_abs_action = float(
            np.mean(x)
        )

        mean_abs_axis = np.mean(
            x,
            axis=0,
        )

    else:
        max_abs_action = 0.0
        mean_abs_action = 0.0
        mean_abs_axis = np.zeros(
            3,
            dtype=np.float32,
        )

    return {
        "status":
            status,

        "steps":
            step + 1,

        "max_abs_action":
            max_abs_action,

        "mean_abs_action":
            mean_abs_action,

        "mean_abs_axis":
            mean_abs_axis,
    }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--checkpoint",
        required=True,
    )

    ap.add_argument(
        "--seed-start",
        type=int,
        default=0,
    )

    ap.add_argument(
        "--num-seeds",
        type=int,
        default=10,
    )

    ap.add_argument(
        "--base-port",
        type=int,
        default=51110,
    )

    ap.add_argument(
        "--out-dir",
        default=(
            "results/icra27/"
            "m7_checkpoint_fixed_seeds_v0"
        ),
    )

    args = ap.parse_args()

    (
        model,
        _cfg,
        _extra,
    ) = load_ppo_checkpoint(
        args.checkpoint
    )

    model.eval()

    seeds = range(
        args.seed_start,
        args.seed_start
        + args.num_seeds,
    )

    print("=" * 92)
    print(
        "ICRA27 M7 FIXED-SEED CHECKPOINT EVALUATION"
    )
    print("=" * 92)

    print(
        "checkpoint:",
        args.checkpoint,
    )

    print(
        "seeds     :",
        f"{args.seed_start}.."
        f"{args.seed_start + args.num_seeds - 1}",
    )

    print(
        "comparison: nominal zero action "
        "vs trained deterministic policy"
    )

    print()

    for terrain_index, terrain in enumerate(
        TERRAINS
    ):
        command_port = (
            args.base_port
            + 10 * terrain_index
        )

        base_env = PyMPCM7Env(
            terrain=terrain,
            terminate_on_m4_unsafe=True,

            goal_distance_m=2.0,
            success_radius_m=0.15,

            decision_dt_s=0.20,

            max_episode_steps=(
                POLICY_HORIZON
                + SETTLING_STEPS
            ),

            command_port=command_port,
            telemetry_port=(
                command_port + 1
            ),
            state_port=(
                command_port + 2
            ),

            command_repeat_hz=20.0,
            telemetry_hz=100.0,
            state_hz=100.0,

            log_dir=(
                Path(args.out_dir)
                / terrain
            ),
        )

        env = M7FixedClearanceActionWrapper(
            base_env
        )

        print("-" * 92)
        print(
            f"terrain={terrain} "
            f"friction={base_env.terrain_friction} "
            f"context={base_env.oracle_context}"
        )
        print("-" * 92)

        try:
            aggregates = {}

            for mode in (
                "nominal",
                "trained",
            ):
                rows = []

                print(
                    f"\n[{mode}]"
                )

                for seed in seeds:
                    row = run_episode(
                        env,
                        model=model,
                        mode=mode,
                        seed=seed,
                    )

                    rows.append(
                        row
                    )

                    axis = row.get(
                        "mean_abs_axis",
                        np.zeros(3),
                    )

                    print(
                        f"seed={seed:05d} "
                        f"{row['status']:<14} "
                        f"steps={row['steps']:02d} "
                        f"amax="
                        f"{row['max_abs_action']:.3f} "
                        f"|a|mean="
                        f"{row['mean_abs_action']:.3f} "
                        f"axis="
                        f"[{axis[0]:.3f},"
                        f"{axis[1]:.3f},"
                        f"{axis[2]:.3f}]"
                    )

                aggregates[
                    mode
                ] = rows

            print()

            for mode in (
                "nominal",
                "trained",
            ):
                rows = aggregates[
                    mode
                ]

                success = sum(
                    x["status"]
                    == "success"
                    for x in rows
                )

                m4 = sum(
                    x["status"]
                    == "m4"
                    for x in rows
                )

                settling = sum(
                    x["status"]
                    == "settling_fail"
                    for x in rows
                )

                print(
                    f"SUMMARY {terrain:<14} "
                    f"{mode:<7} "
                    f"success="
                    f"{success}/{len(rows)} "
                    f"m4="
                    f"{m4}/{len(rows)} "
                    f"settle_fail="
                    f"{settling}/{len(rows)}"
                )

            print()

        finally:
            env.close()


if __name__ == "__main__":
    main()
