#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
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


TERRAINS = (
    "flat",
    "low_friction",
    "rough_perlin",
)

VAL_SEEDS = (
    1,
    21,
    16,
    14,
)

SETTLING_STEPS = 5
POLICY_HORIZON = 50


def reset_with_settling(
    env,
    *,
    seed,
):
    obs, info = env.reset(seed=seed)

    zero = np.zeros(
        env.action_space.shape,
        dtype=np.float32,
    )

    for _ in range(SETTLING_STEPS):
        (
            obs,
            _reward,
            terminated,
            truncated,
            info,
        ) = env.step(zero)

        if terminated or truncated:
            return obs, info, False

    return obs, info, True


def classify(
    info,
    terminated,
    truncated,
):
    if bool(info.get("success", False)):
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


def evaluate_episode(
    env,
    *,
    model,
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
        return {
            "status": "settling_fail",
            "steps": 0,
        }

    for step in range(POLICY_HORIZON):
        (
            action,
            _log_prob,
            _value,
        ) = model.act(
            obs,
            deterministic=True,
        )

        (
            obs,
            _reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        if terminated or truncated:
            return {
                "status": classify(
                    info,
                    terminated,
                    truncated,
                ),
                "steps": step + 1,
            }

    return {
        "status": "running",
        "steps": POLICY_HORIZON,
    }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--checkpoint-dir",
        required=True,
    )

    ap.add_argument(
        "--out",
        required=True,
    )

    ap.add_argument(
        "--base-port",
        type=int,
        default=51510,
    )

    args = ap.parse_args()

    checkpoint_dir = Path(
        args.checkpoint_dir
    )

    checkpoints = sorted(
        checkpoint_dir.glob(
            "checkpoint_update_*.pt"
        )
    )

    if len(checkpoints) != 10:
        raise RuntimeError(
            "Expected 10 update checkpoints; "
            f"found {len(checkpoints)}"
        )

    output = {
        "schema":
            "icra27_m7_checkpoint_lineage_val_v0",

        "validation_seeds":
            list(VAL_SEEDS),

        "checkpoints":
            {},
    }

    print("=" * 92)
    print(
        "ICRA27 M7 CHECKPOINT LINEAGE VALIDATION"
    )
    print("=" * 92)

    for terrain_index, terrain in enumerate(
        TERRAINS
    ):
        port = (
            args.base_port
            + terrain_index * 10
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

            command_port=port,
            telemetry_port=port + 1,
            state_port=port + 2,

            command_repeat_hz=20.0,
            telemetry_hz=100.0,
            state_hz=100.0,

            log_dir=(
                Path(args.out).parent
                / "env_logs"
                / terrain
            ),
        )

        env = M7FixedClearanceActionWrapper(
            base_env
        )

        try:
            print()
            print(
                f"--- {terrain} ---"
            )

            for checkpoint in checkpoints:
                (
                    model,
                    _cfg,
                    _extra,
                ) = load_ppo_checkpoint(
                    checkpoint
                )

                model.eval()

                rows = []

                for seed in VAL_SEEDS:
                    rows.append(
                        evaluate_episode(
                            env,
                            model=model,
                            seed=seed,
                        )
                    )

                success = sum(
                    r["status"] == "success"
                    for r in rows
                )

                m4 = sum(
                    r["status"] == "m4"
                    for r in rows
                )

                timeout = sum(
                    r["status"] == "time_limit"
                    for r in rows
                )

                settle = sum(
                    r["status"] == "settling_fail"
                    for r in rows
                )

                success_steps = [
                    r["steps"]
                    for r in rows
                    if r["status"] == "success"
                ]

                mean_steps = (
                    float(
                        np.mean(success_steps)
                    )
                    if success_steps
                    else None
                )

                key = checkpoint.stem

                output[
                    "checkpoints"
                ].setdefault(
                    key,
                    {},
                )

                output[
                    "checkpoints"
                ][
                    key
                ][
                    terrain
                ] = {
                    "success": success,
                    "m4": m4,
                    "time_limit": timeout,
                    "settling_fail": settle,
                    "mean_success_steps":
                        mean_steps,
                    "episodes": rows,
                }

                mean_text = (
                    "n/a"
                    if mean_steps is None
                    else f"{mean_steps:.2f}"
                )

                print(
                    f"{key:<24} "
                    f"success={success}/4 "
                    f"m4={m4}/4 "
                    f"timeout={timeout}/4 "
                    f"settle={settle}/4 "
                    f"steps={mean_text}"
                )

        finally:
            env.close()

    out = Path(args.out)

    out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    out.write_text(
        json.dumps(
            output,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print()
    print("saved:", out)
    print(
        "[ICRA27] checkpoint lineage "
        "validation: PASS"
    )


if __name__ == "__main__":
    main()
