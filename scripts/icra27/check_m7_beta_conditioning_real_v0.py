#!/usr/bin/env python3

from __future__ import annotations

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

from tracer_core.highlevel_rl.beta_conditioning_wrapper import (
    M7BetaConditioningWrapper,
)


SETTLING_STEPS = 5

BETAS = {
    "balanced": (
        1.0 / 3.0,
        1.0 / 3.0,
        1.0 / 3.0,
    ),

    "motion": (
        0.70,
        0.15,
        0.15,
    ),

    "stability": (
        0.15,
        0.70,
        0.15,
    ),

    "energy": (
        0.15,
        0.15,
        0.70,
    ),
}


def check_transition(
    *,
    name,
    beta,
    obs,
    reward,
    info,
):
    obs = np.asarray(
        obs,
        dtype=np.float32,
    )

    if obs.shape != (24,):
        raise RuntimeError(
            f"{name}: expected obs24, "
            f"got {obs.shape}"
        )

    np.testing.assert_allclose(
        obs[21:24],
        beta,
        rtol=0.0,
        atol=1e-7,
    )

    rc = info[
        "reward_components"
    ]

    reward_beta = np.array(
        [
            rc["beta_motion"],
            rc["beta_stability"],
            rc["beta_energy"],
        ],
        dtype=np.float64,
    )

    np.testing.assert_allclose(
        reward_beta,
        beta,
        rtol=0.0,
        atol=1e-10,
    )

    costs = np.array(
        [
            rc["cost_motion"],
            rc["cost_stability"],
            rc["cost_energy"],
        ],
        dtype=np.float64,
    )

    beta_arr = np.asarray(
        beta,
        dtype=np.float64,
    )

    manual_objective_reward = -float(
        np.dot(
            beta_arr,
            costs,
        )
    )

    task_feasibility_cost = float(
        rc["task_feasibility_cost"]
    )

    manual_total_reward = (
        manual_objective_reward
        - task_feasibility_cost
    )

    np.testing.assert_allclose(
        float(
            rc["objective_reward"]
        ),
        manual_objective_reward,
        rtol=0.0,
        atol=1e-10,
    )

    np.testing.assert_allclose(
        float(reward),
        manual_total_reward,
        rtol=0.0,
        atol=1e-10,
    )

    np.testing.assert_allclose(
        float(
            rc["total_reward"]
        ),
        manual_total_reward,
        rtol=0.0,
        atol=1e-10,
    )

    print(
        f"{name:<10} "
        f"beta="
        f"[{beta[0]:.3f}, "
        f"{beta[1]:.3f}, "
        f"{beta[2]:.3f}] "
        f"C="
        f"[{costs[0]:.3f}, "
        f"{costs[1]:.3f}, "
        f"{costs[2]:.3f}] "
        f"Robj="
        f"{manual_objective_reward:+.4f} "
        f"CF="
        f"{task_feasibility_cost:.3f} "
        f"R="
        f"{manual_total_reward:+.4f} "
        "PASS"
    )


def run_one(
    *,
    name,
    beta,
    command_port,
):
    base_env = PyMPCM7Env(
        terrain="flat",
        reward_mode="tracer_cost_v2",

        terminate_on_m4_unsafe=True,

        goal_distance_m=2.0,
        success_radius_m=0.15,

        decision_dt_s=0.20,

        max_episode_steps=(
            50
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
            / "beta_conditioning_real_smoke_v0"
            / name
        ),
    )

    action_env = (
        M7FixedClearanceActionWrapper(
            base_env
        )
    )

    env = M7BetaConditioningWrapper(
        action_env,
        beta=beta,
    )

    if env.observation_space.shape != (
        24,
    ):
        raise RuntimeError(
            "Expected policy obs24, got "
            f"{env.observation_space.shape}"
        )

    if env.action_space.shape != (
        3,
    ):
        raise RuntimeError(
            "Expected policy action3, got "
            f"{env.action_space.shape}"
        )

    zero_action = np.zeros(
        (3,),
        dtype=np.float32,
    )

    try:
        obs, _info = env.reset(
            seed=27027
        )

        np.testing.assert_allclose(
            np.asarray(obs)[21:24],
            beta,
            rtol=0.0,
            atol=1e-7,
        )

        # Five characterized nominal settling decisions.
        for settling_i in range(
            SETTLING_STEPS
        ):
            (
                obs,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(
                zero_action
            )

            env.assert_reward_beta(
                info
            )

            if terminated or truncated:
                raise RuntimeError(
                    f"{name}: settling terminated "
                    f"at step {settling_i + 1}"
                )

        # One policy-like transition after settling.
        (
            obs,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(
            zero_action
        )

        env.assert_reward_beta(
            info
        )

        if terminated or truncated:
            raise RuntimeError(
                f"{name}: unexpected terminal "
                "on smoke transition"
            )

        check_transition(
            name=name,
            beta=beta,
            obs=obs,
            reward=reward,
            info=info,
        )

    finally:
        env.close()


def main():
    print("=" * 92)
    print(
        "ICRA27 PHASE-1A REAL BETA-CONDITIONING SMOKE"
    )
    print("=" * 92)

    print(
        "terrain       : flat"
    )

    print(
        "terrain seed  : 27027"
    )

    print(
        "reward        : tracer_cost_v2"
    )

    print(
        "policy obs    : 24D = obs21 + beta3"
    )

    print(
        "policy action : 3D"
    )

    print()

    base_port = 55110

    for i, (
        name,
        beta,
    ) in enumerate(
        BETAS.items()
    ):
        run_one(
            name=name,
            beta=beta,
            command_port=(
                base_port
                + 10 * i
            ),
        )

    print()
    print("=" * 92)
    print(
        "[ICRA27] real beta-conditioned "
        "reward/observation contract: PASS"
    )
    print("=" * 92)


if __name__ == "__main__":
    main()
