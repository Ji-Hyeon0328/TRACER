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

SETTLING_STEPS = 5
POLICY_HORIZON = 50

SPLIT_PATH = (
    ROOT
    / "configs"
    / "icra27"
    / "m7_perlin_seed_split_v0.json"
)


def load_validation_seeds():
    data = json.loads(
        SPLIT_PATH.read_text()
    )

    seeds = [
        int(x)
        for x in (
            data[
                "splits"
            ][
                "validation"
            ]
        )
    ]

    if not seeds:
        raise RuntimeError(
            "Validation seed bank is empty."
        )

    return seeds


def reset_with_settling(
    env,
    *,
    seed,
):
    obs, info = env.reset(
        seed=seed
    )

    zero = np.zeros(
        env.action_space.shape,
        dtype=np.float32,
    )

    for _ in range(
        SETTLING_STEPS
    ):
        (
            obs,
            _reward,
            terminated,
            truncated,
            info,
        ) = env.step(
            zero
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


def run_episode(
    env,
    *,
    model,
    nominal,
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
            "status":
                "settling_fail",

            "steps":
                0,

            "mean_action":
                None,

            "mean_applied":
                None,
        }

    actions = []
    applied = []

    for step in range(
        POLICY_HORIZON
    ):
        if nominal:
            action = np.zeros(
                env.action_space.shape,
                dtype=np.float32,
            )

        else:
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

        if action.shape != (3,):
            raise RuntimeError(
                "Expected policy action shape (3,), "
                f"got {action.shape}"
            )

        actions.append(
            action.copy()
        )

        (
            next_obs,
            _reward,
            terminated,
            truncated,
            info,
        ) = env.step(
            action
        )

        # Frozen 21D observation contract:
        #
        # [13] applied_vx
        # [14] applied_yaw
        # [15] applied_height
        # [16] applied_clearance
        #
        # We intentionally record only the learned
        # 3D physical reference dimensions here.
        next_obs = np.asarray(
            next_obs,
            dtype=np.float32,
        )

        if next_obs.shape != (21,):
            raise RuntimeError(
                "Unexpected observation shape: "
                f"{next_obs.shape}"
            )

        applied.append(
            next_obs[
                13:16
            ].copy()
        )

        obs = next_obs

        if terminated or truncated:
            status = classify(
                info=info,
                terminated=terminated,
                truncated=truncated,
            )

            break

    else:
        status = "running"

    mean_action = np.mean(
        np.stack(actions),
        axis=0,
    )

    mean_applied = np.mean(
        np.stack(applied),
        axis=0,
    )

    return {
        "status":
            status,

        "steps":
            step + 1,

        "mean_action":
            mean_action,

        "mean_applied":
            mean_applied,
    }


def aggregate(
    rows,
):
    success = sum(
        row["status"] == "success"
        for row in rows
    )

    m4 = sum(
        row["status"] == "m4"
        for row in rows
    )

    timeout = sum(
        row["status"] == "time_limit"
        for row in rows
    )

    settling = sum(
        row["status"] == "settling_fail"
        for row in rows
    )

    success_steps = [
        row["steps"]
        for row in rows
        if row["status"] == "success"
    ]

    valid_actions = [
        row["mean_action"]
        for row in rows
        if row["mean_action"] is not None
    ]

    valid_applied = [
        row["mean_applied"]
        for row in rows
        if row["mean_applied"] is not None
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

    mean_action = (
        np.mean(
            np.stack(
                valid_actions
            ),
            axis=0,
        )
        if valid_actions
        else None
    )

    mean_applied = (
        np.mean(
            np.stack(
                valid_applied
            ),
            axis=0,
        )
        if valid_applied
        else None
    )

    return {
        "success":
            success,

        "m4":
            m4,

        "time_limit":
            timeout,

        "settling_fail":
            settling,

        "mean_success_steps":
            mean_success_steps,

        "mean_action":
            mean_action,

        "mean_applied":
            mean_applied,
    }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--checkpoint-dir",
        required=True,
    )

    ap.add_argument(
        "--updates",
        type=int,
        nargs="+",
        default=[
            1,
            6,
            10,
        ],
    )

    ap.add_argument(
        "--base-port",
        type=int,
        default=51610,
    )

    ap.add_argument(
        "--out",
        required=True,
    )

    args = ap.parse_args()

    validation_seeds = (
        load_validation_seeds()
    )

    policies = [
        (
            "nominal",
            None,
            True,
        )
    ]

    checkpoint_dir = Path(
        args.checkpoint_dir
    )

    for update in args.updates:
        path = (
            checkpoint_dir
            / (
                "checkpoint_update_"
                f"{update:04d}.pt"
            )
        )

        if not path.exists():
            raise FileNotFoundError(
                path
            )

        (
            model,
            _cfg,
            _extra,
        ) = load_ppo_checkpoint(
            path
        )

        model.eval()

        policies.append(
            (
                f"update_{update:04d}",
                model,
                False,
            )
        )

    output = {
        "schema":
            "icra27_m7_policy_profile_val_v0",

        "validation_seeds":
            validation_seeds,

        "policies":
            {},
    }

    print("=" * 106)
    print(
        "ICRA27 M7 POLICY PROFILE "
        "ON FROZEN VALIDATION BANK"
    )
    print("=" * 106)

    print(
        "validation seeds:",
        validation_seeds,
    )

    print()

    for terrain_index, terrain in enumerate(
        TERRAINS
    ):
        port = (
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

        env = (
            M7FixedClearanceActionWrapper(
                base_env
            )
        )

        print(
            "-" * 106
        )
        print(
            f"terrain={terrain}"
        )
        print(
            "-" * 106
        )

        try:
            for (
                policy_name,
                model,
                nominal,
            ) in policies:
                rows = []

                for seed in (
                    validation_seeds
                ):
                    row = run_episode(
                        env,
                        model=model,
                        nominal=nominal,
                        seed=seed,
                    )

                    rows.append(
                        row
                    )

                a = aggregate(
                    rows
                )

                output[
                    "policies"
                ].setdefault(
                    policy_name,
                    {},
                )

                serializable = {
                    k: (
                        v.tolist()
                        if isinstance(
                            v,
                            np.ndarray,
                        )
                        else v
                    )
                    for k, v
                    in a.items()
                }

                output[
                    "policies"
                ][
                    policy_name
                ][
                    terrain
                ] = serializable

                step_text = (
                    "n/a"
                    if (
                        a[
                            "mean_success_steps"
                        ]
                        is None
                    )
                    else (
                        f"{a['mean_success_steps']:.2f}"
                    )
                )

                action = (
                    a[
                        "mean_action"
                    ]
                )

                applied = (
                    a[
                        "mean_applied"
                    ]
                )

                action_text = (
                    "n/a"
                    if action is None
                    else (
                        "["
                        f"{action[0]:+.4f}, "
                        f"{action[1]:+.4f}, "
                        f"{action[2]:+.4f}"
                        "]"
                    )
                )

                applied_text = (
                    "n/a"
                    if applied is None
                    else (
                        "["
                        f"{applied[0]:.4f}, "
                        f"{applied[1]:+.4f}, "
                        f"{applied[2]:.4f}"
                        "]"
                    )
                )

                print(
                    f"{policy_name:<12} "
                    f"success="
                    f"{a['success']}/"
                    f"{len(validation_seeds)} "
                    f"m4="
                    f"{a['m4']} "
                    f"timeout="
                    f"{a['time_limit']} "
                    f"steps="
                    f"{step_text:<5} "
                    f"a_norm="
                    f"{action_text} "
                    f"applied="
                    f"{applied_text}"
                )

        finally:
            env.close()

        print()

    out = Path(
        args.out
    )

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

    print(
        "saved:",
        out,
    )

    print(
        "[ICRA27] validation policy profile: PASS"
    )


if __name__ == "__main__":
    main()
