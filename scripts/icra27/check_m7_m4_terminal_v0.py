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


# Deliberately aggressive high-level commands.
#
# Normalized dimensions:
#   [vx, yaw, body_height, clearance]
#
# These remain inside the valid M7 action contract.
STRESS_ACTIONS = (
    np.array(
        [1.0, 1.0, -1.0, -1.0],
        dtype=np.float32,
    ),
    np.array(
        [1.0, -1.0, -1.0, -1.0],
        dtype=np.float32,
    ),
    np.array(
        [1.0, 1.0, 1.0, -1.0],
        dtype=np.float32,
    ),
    np.array(
        [1.0, -1.0, 1.0, -1.0],
        dtype=np.float32,
    ),
)


def main():
    print("=" * 72)
    print(
        "ICRA27 M7 M4 TERMINAL CHECK"
    )
    print("=" * 72)

    env = PyMPCM7Env(
        terrain="flat",

        # Keep success well away from this safety test.
        goal_distance_m=5.0,

        terminate_on_m4_unsafe=True,

        command_port=50910,
        telemetry_port=50911,
        state_port=50912,

        max_episode_steps=30,

        log_dir=(
            ROOT
            / "results"
            / "icra27"
            / "m7_m4_terminal"
        ),
    )

    try:
        obs, info = env.reset(
            seed=0
        )

        terminal_info = None
        terminal_reward = None
        terminal_step = None

        for step in range(
            env.max_episode_steps
        ):
            action = STRESS_ACTIONS[
                step % len(STRESS_ACTIONS)
            ]

            (
                obs,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(
                action
            )

            print(
                f"step {step:02d}: "
                f"a={action.tolist()} "
                f"reward={reward:+.4f} "
                f"goal={info['goal_distance']:.4f} "
                f"safety={info['safety_state']} "
                f"override={info['override_active']} "
                f"m4_terminal={info['m4_terminal']} "
                f"terminated={terminated} "
                f"truncated={truncated}"
            )

            if terminated or truncated:
                terminal_info = info
                terminal_reward = float(
                    reward
                )
                terminal_step = step
                break

        if terminal_info is None:
            raise RuntimeError(
                "No terminal transition observed."
            )

        # This test specifically requires M4,
        # not success or time-limit termination.
        if not bool(
            terminal_info[
                "m4_intervention"
            ]
        ):
            raise RuntimeError(
                "Stress rollout terminated "
                "without M4 intervention."
            )

        if not bool(
            terminal_info[
                "m4_terminal"
            ]
        ):
            raise RuntimeError(
                "M4 intervention did not "
                "produce m4_terminal=True."
            )

        if not (
            terminal_info[
                "safety_state"
            ].strip().lower()
            == "unsafe"
            or bool(
                terminal_info[
                    "override_active"
                ]
            )
        ):
            raise RuntimeError(
                "M4 terminal did not contain "
                "unsafe/override evidence."
            )

        print()
        print(
            "terminal step   :",
            terminal_step,
        )

        print(
            "terminal reward :",
            f"{terminal_reward:+.4f}",
        )

        print(
            "safety          :",
            terminal_info[
                "safety_state"
            ],
        )

        print(
            "override        :",
            terminal_info[
                "override_active"
            ],
        )

        print(
            "m4 terminal     :",
            terminal_info[
                "m4_terminal"
            ],
        )

        # episode_done should now block any
        # post-fallback samples.
        try:
            env.step(
                STRESS_ACTIONS[0]
            )

        except RuntimeError:
            print(
                "post-terminal step rejected : PASS"
            )

        else:
            raise RuntimeError(
                "Environment accepted a step "
                "after M4 terminal."
            )

    finally:
        env.close()

    print()
    print(
        "[ICRA27] M7 M4 terminal "
        "semantics: PASS"
    )


if __name__ == "__main__":
    main()
