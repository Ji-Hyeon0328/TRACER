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

SEEDS = (
    9,
    22000,
    22001,
    22002,
    22004,
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
            )

    return (
        obs,
        info,
        True,
    )


def classify(
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


def run_once(
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
    ) = reset_with_settling(
        env,
        seed=seed,
    )

    if not settled:
        return "settling_fail", 0

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
            raise ValueError(mode)

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
            info=info,
            terminated=terminated,
            truncated=truncated,
        )

        if terminated or truncated:
            return (
                status,
                step + 1,
            )

    return "running", POLICY_HORIZON


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--checkpoint",
        required=True,
    )

    ap.add_argument(
        "--repeats",
        type=int,
        default=5,
    )

    ap.add_argument(
        "--base-port",
        type=int,
        default=51310,
    )

    ap.add_argument(
        "--out-dir",
        default=(
            "results/icra27/"
            "m7_rough_repeatability_v0"
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

        command_port=args.base_port,
        telemetry_port=args.base_port + 1,
        state_port=args.base_port + 2,

        command_repeat_hz=20.0,
        telemetry_hz=100.0,
        state_hz=100.0,

        log_dir=Path(
            args.out_dir
        ),
    )

    env = M7FixedClearanceActionWrapper(
        base_env
    )

    print("=" * 92)
    print(
        "ICRA27 M7 ROUGH-PERLIN "
        "REPEATABILITY CHECK"
    )
    print("=" * 92)

    print(
        "seeds   :",
        SEEDS,
    )

    print(
        "repeats :",
        args.repeats,
    )

    print()

    try:
        for seed in SEEDS:
            print(
                "-" * 92
            )
            print(
                f"seed={seed}"
            )

            for mode in (
                "nominal",
                "trained",
            ):
                counts = {
                    "success": 0,
                    "m4": 0,
                    "settling_fail": 0,
                    "time_limit": 0,
                    "terminal": 0,
                    "running": 0,
                }

                steps = []

                for repeat in range(
                    args.repeats
                ):
                    (
                        status,
                        n_steps,
                    ) = run_once(
                        env,
                        model=model,
                        mode=mode,
                        seed=seed,
                    )

                    counts[
                        status
                    ] += 1

                    steps.append(
                        n_steps
                    )

                    print(
                        f"  {mode:<7} "
                        f"rep={repeat:02d} "
                        f"{status:<14} "
                        f"steps={n_steps:02d}"
                    )

                print(
                    f"  SUMMARY {mode:<7} "
                    f"success="
                    f"{counts['success']}/"
                    f"{args.repeats} "
                    f"m4="
                    f"{counts['m4']}/"
                    f"{args.repeats} "
                    f"settle_fail="
                    f"{counts['settling_fail']}/"
                    f"{args.repeats}"
                )

            print()

    finally:
        env.close()


if __name__ == "__main__":
    main()
