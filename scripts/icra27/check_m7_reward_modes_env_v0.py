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


MODES = (
    "fixed_additive",
    "tracer_uniform",
)

SETTLING_STEPS = 5
POLICY_STEPS = 8


def run_mode(
    *,
    mode,
    port,
):
    base_env = PyMPCM7Env(
        terrain="flat",
        reward_mode=mode,

        terminate_on_m4_unsafe=True,

        goal_distance_m=2.0,
        success_radius_m=0.15,

        decision_dt_s=0.20,

        max_episode_steps=(
            SETTLING_STEPS
            + POLICY_STEPS
            + 10
        ),

        command_port=port,
        telemetry_port=port + 1,
        state_port=port + 2,

        command_repeat_hz=20.0,
        telemetry_hz=100.0,
        state_hz=100.0,

        log_dir=(
            "results/icra27/"
            "m7_reward_modes_env_v0/"
            f"{mode}"
        ),
    )

    env = M7FixedClearanceActionWrapper(
        base_env
    )

    zero = np.zeros(
        env.action_space.shape,
        dtype=np.float32,
    )

    rows = []

    try:
        obs, info = env.reset(
            seed=0
        )

        for k in range(
            SETTLING_STEPS
        ):
            (
                obs,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(zero)

            if terminated or truncated:
                raise RuntimeError(
                    f"{mode}: settling failed "
                    f"at step {k + 1}"
                )

        for k in range(
            POLICY_STEPS
        ):
            (
                obs,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(zero)

            components = info[
                "reward_components"
            ]

            rows.append(
                (
                    float(reward),
                    info["reward_mode"],
                    info["reward_schema"],
                    components,
                )
            )

            if terminated or truncated:
                break

    finally:
        env.close()

    return rows


def main():
    all_rows = {}

    for i, mode in enumerate(
        MODES
    ):
        rows = run_mode(
            mode=mode,
            port=51810 + 10 * i,
        )

        if not rows:
            raise RuntimeError(
                f"{mode}: no policy steps"
            )

        all_rows[
            mode
        ] = rows

        reward_values = [
            row[0]
            for row in rows
        ]

        schemas = {
            row[2]
            for row in rows
        }

        print(
            f"{mode:<16} "
            f"steps={len(rows)} "
            f"mean_reward="
            f"{np.mean(reward_values):+.6f} "
            f"schemas={schemas}"
        )

        for (
            _reward,
            row_mode,
            _schema,
            components,
        ) in rows:
            assert row_mode == mode

            assert bool(
                components[
                    "m4_intervention"
                ]
                >= 0.0
            )

    fixed_schema = {
        row[2]
        for row in (
            all_rows[
                "fixed_additive"
            ]
        )
    }

    tracer_schema = {
        row[2]
        for row in (
            all_rows[
                "tracer_uniform"
            ]
        )
    }

    assert fixed_schema == {
        "icra27_fixed_additive_v0"
    }

    assert tracer_schema == {
        "icra27_tracer_structured_v1"
    }

    tracer_components = (
        all_rows[
            "tracer_uniform"
        ][0][3]
    )

    assert np.allclose(
        tracer_components[
            "beta"
        ],
        [
            1.0 / 3.0,
            1.0 / 3.0,
            1.0 / 3.0,
        ],
    )

    assert (
        "objective_motion"
        in tracer_components
    )

    assert (
        "objective_stability"
        in tracer_components
    )

    assert (
        "objective_command_economy_proxy"
        in tracer_components
    )

    print()
    print(
        "TRACER beta:",
        tracer_components[
            "beta"
        ],
    )

    print(
        "TRACER objective keys: PASS"
    )

    print()
    print(
        "[ICRA27] A/B environment "
        "reward-mode smoke: PASS"
    )


if __name__ == "__main__":
    main()
