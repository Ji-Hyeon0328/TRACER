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


def _reward_view(info):
    """
    Resolve reward-component diagnostics without assuming
    whether the environment exposes them top-level or under
    a nested reward-components mapping.

    This checker must not change the environment schema.
    """

    if (
        isinstance(info, dict)
        and "cost_motion" in info
    ):
        return info

    for key in (
        "reward_components",
        "reward_info",
        "reward_terms",
    ):
        candidate = info.get(key)

        if (
            isinstance(candidate, dict)
            and "cost_motion" in candidate
        ):
            print(
                f"[checker] reward diagnostics "
                f"resolved via info[{key!r}]"
            )

            return candidate

    print()
    print(
        "[checker] top-level info keys:"
    )

    print(
        sorted(info.keys())
    )

    print()

    for key, value in info.items():
        if isinstance(value, dict):
            print(
                f"[checker] nested dict {key!r}: "
                f"{sorted(value.keys())}"
            )

    raise RuntimeError(
        "Could not locate reward-component diagnostics "
        "in Gym info."
    )


def main():
    env = PyMPCM7Env(
        terrain="flat",

        reward_mode="tracer_cost_v4",

        tracer_beta=(
            1.0 / 3.0,
            1.0 / 3.0,
            1.0 / 3.0,
        ),

        goal_distance_m=0.50,
        success_radius_m=0.15,

        decision_dt_s=0.20,
        max_episode_steps=5,

        terminate_on_m4_unsafe=True,

        command_port=64100,
        telemetry_port=64101,
        state_port=64102,

        log_dir=(
            ROOT
            / "results"
            / "icra27"
            / "os_t4p3_tracer_cost_v4_smoke_v0"
        ),
    )

    try:
        obs, info = env.reset(
            seed=27100
        )

        print("=" * 96)
        print(
            "ICRA27 OS-T4.7b TRACER COST V4 SMOKE"
        )
        print("=" * 96)

        print(
            "reset obs shape:",
            obs.shape,
        )

        # Mid-range normalized command.
        action = np.zeros(
            4,
            dtype=np.float32,
        )

        for step in range(5):
            (
                obs,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(action)

            reward_info = (
                _reward_view(info)
            )

            # Checker-local flattened view only.
            #
            # Keep the production Gym info schema unchanged,
            # while allowing the remaining diagnostic assertions
            # to use historical top-level key access.
            info = {
                **info,
                **reward_info,
            }

            print(
                f"step={step} "
                f"R={reward:+.6f} "
                f"Cm={reward_info['cost_motion']:.6f} "
                f"Cpost={reward_info['cost_posture']:.6f} "
                f"Ctr={reward_info['cost_traction']:.6f} "
                f"Cs={reward_info['cost_stability']:.6f} "
                f"Ce={reward_info['cost_energy']:.6f} "
                f"Tcontact="
                f"{reward_info['traction_interval_contact_dt_s']:.6f} "
                f"safety={info['safety_state']}"
            )

            expected = max(
                float(
                    reward_info[
                        "cost_posture"
                    ]
                ),
                float(
                    reward_info[
                        "cost_traction"
                    ]
                ),
            )

            if not np.isclose(
                float(
                    reward_info[
                        "cost_stability"
                    ]
                ),
                expected,
                rtol=0.0,
                atol=1e-12,
            ):
                raise RuntimeError(
                    "C_stability != "
                    "max(C_posture, C_traction)"
                )

            expected_objective = (
                (
                    float(
                        reward_info[
                            "cost_motion"
                        ]
                    )
                    + float(
                        info[
                            "cost_stability"
                        ]
                    )
                    + float(
                        reward_info[
                            "cost_energy"
                        ]
                    )
                )
                / 3.0
            )

            if not np.isclose(
                float(
                    reward_info[
                        "objective_cost"
                    ]
                ),
                expected_objective,
                rtol=0.0,
                atol=1e-10,
            ):
                raise RuntimeError(
                    "Uniform-beta objective decomposition "
                    "does not match."
                )

            if (
                terminated
                or truncated
            ):
                break

        print()
        print(
            "reward schema:",
            reward_info[
                "reward_schema"
            ],
        )

        print(
            "stability composition:",
            reward_info[
                "stability_composition"
            ],
        )

        print()
        print(
            "[ICRA27] tracer_cost_v4 smoke: PASS"
        )

    finally:
        env.close()


if __name__ == "__main__":
    main()
