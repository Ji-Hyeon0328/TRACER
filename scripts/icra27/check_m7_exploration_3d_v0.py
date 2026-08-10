#!/usr/bin/env python3

from __future__ import annotations

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

from tracer_core.highlevel_rl.fixed_clearance_wrapper import (
    M7FixedClearanceActionWrapper,
)


SETTLING_STEPS = 5
POLICY_HORIZON = 50
INITIAL_STD = 0.15

SEEDS = tuple(range(5))

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
        return "m4_terminal"

    if truncated:
        return "time_limit"

    if terminated:
        return "terminated"

    return "running"


def main():
    torch.manual_seed(0)

    actor = PPOActorCritic(
        obs_dim=21,
        act_dim=3,
        hidden_sizes=(
            128,
            128,
        ),
        initial_std=INITIAL_STD,
    )

    actor.eval()

    print("=" * 88)
    print(
        "ICRA27 M7 3D INITIAL EXPLORATION SANITY"
    )
    print("=" * 88)

    print(
        "policy action : "
        "[vx, yaw_rate, body_height]"
    )
    print(
        "initial std   :",
        INITIAL_STD,
    )
    print(
        "clearance     : fixed 0.060 m"
    )
    print(
        "settling      :",
        SETTLING_STEPS,
    )
    print(
        "policy horizon:",
        POLICY_HORIZON,
    )
    print()

    base_port = 50910

    all_results = {}

    for terrain_index, terrain in enumerate(
        TERRAINS
    ):
        command_port = (
            base_port
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
                ROOT
                / "results"
                / "icra27"
                / "m7_exploration_3d_v0"
                / terrain
            ),
        )

        env = M7FixedClearanceActionWrapper(
            base_env
        )

        results = []

        try:
            print(
                "-" * 88
            )
            print(
                f"terrain={terrain} "
                f"scene={base_env.terrain_scene} "
                f"friction={base_env.terrain_friction} "
                f"context={base_env.oracle_context}"
            )
            print(
                "-" * 88
            )

            assert (
                env.action_space.shape
                == (3,)
            )

            assert (
                base_env.action_space.shape
                == (4,)
            )

            for seed in SEEDS:
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
                    status = (
                        "settling_terminal"
                    )

                    print(
                        f"seed={seed:02d} "
                        f"{status:<18} "
                        f"settle="
                        f"{settling_steps}/"
                        f"{SETTLING_STEPS} "
                        f"safety="
                        f"{info.get('safety_state')}"
                    )

                    results.append(
                        {
                            "seed": seed,
                            "status": status,
                            "m4": bool(
                                info.get(
                                    "m4_intervention",
                                    False,
                                )
                            ),
                            "steps": 0,
                        }
                    )

                    continue

                # Match stochastic action-noise
                # sequence across terrains for a
                # given seed.
                torch.manual_seed(
                    10000 + seed
                )

                interventions = 0
                max_abs_action = 0.0

                terminated = False
                truncated = False

                status = "running"
                policy_steps = 0

                for policy_step in range(
                    POLICY_HORIZON
                ):
                    (
                        action,
                        _log_prob,
                        _value,
                    ) = actor.act(
                        obs,
                        deterministic=False,
                    )

                    action = np.asarray(
                        action,
                        dtype=np.float32,
                    )

                    if action.shape != (3,):
                        raise RuntimeError(
                            "Unexpected policy "
                            f"action shape: "
                            f"{action.shape}"
                        )

                    max_abs_action = max(
                        max_abs_action,
                        float(
                            np.max(
                                np.abs(
                                    action
                                )
                            )
                        ),
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

                    policy_steps += 1

                    interventions += int(
                        bool(
                            info.get(
                                "m4_intervention",
                                False,
                            )
                        )
                    )

                    # K=3 oracle context.
                    context_dim = 3

                    applied_clearance = float(
                        obs[
                            context_dim
                            + 13
                        ]
                    )

                    previous_clearance = float(
                        obs[
                            context_dim
                            + 17
                        ]
                    )

                    if not np.isclose(
                        applied_clearance,
                        0.060,
                        atol=1e-6,
                    ):
                        raise RuntimeError(
                            "Applied clearance "
                            "contract violated: "
                            f"{applied_clearance}"
                        )

                    if not np.isclose(
                        previous_clearance,
                        0.0,
                        atol=1e-6,
                    ):
                        raise RuntimeError(
                            "Previous normalized "
                            "clearance contract "
                            "violated: "
                            f"{previous_clearance}"
                        )

                    status = classify(
                        info=info,
                        terminated=terminated,
                        truncated=truncated,
                    )

                    if (
                        terminated
                        or truncated
                    ):
                        break

                goal_distance = float(
                    info.get(
                        "goal_distance",
                        np.nan,
                    )
                )

                m4 = (
                    interventions > 0
                    or status
                    == "m4_terminal"
                )

                results.append(
                    {
                        "seed": seed,
                        "status": status,
                        "m4": m4,
                        "steps": policy_steps,
                    }
                )

                print(
                    f"seed={seed:02d} "
                    f"{status:<18} "
                    f"steps={policy_steps:02d} "
                    f"m4={int(m4)} "
                    f"goal={goal_distance:.3f} "
                    f"amax={max_abs_action:.3f}"
                )

        finally:
            env.close()

        success_count = sum(
            r["status"] == "success"
            for r in results
        )

        m4_count = sum(
            bool(r["m4"])
            for r in results
        )

        settling_fail_count = sum(
            r["status"]
            == "settling_terminal"
            for r in results
        )

        all_results[
            terrain
        ] = {
            "success":
                success_count,
            "m4":
                m4_count,
            "settling_fail":
                settling_fail_count,
            "total":
                len(results),
        }

        print()
        print(
            f"SUMMARY {terrain}: "
            f"success="
            f"{success_count}/"
            f"{len(results)} "
            f"m4="
            f"{m4_count}/"
            f"{len(results)} "
            f"settling_fail="
            f"{settling_fail_count}/"
            f"{len(results)}"
        )
        print()

    print("=" * 88)
    print("FINAL")
    print("=" * 88)

    for terrain in TERRAINS:
        x = all_results[
            terrain
        ]

        print(
            f"{terrain:<14} "
            f"success="
            f"{x['success']}/{x['total']} "
            f"m4="
            f"{x['m4']}/{x['total']} "
            f"settling_fail="
            f"{x['settling_fail']}/"
            f"{x['total']}"
        )


if __name__ == "__main__":
    main()
