#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(
    0,
    str(ROOT),
)


from tracer_core.highlevel_rl.pympc_env import (
    PyMPCM7Env,
)


TERRAINS = (
    "flat",
    "low_friction",
    "rough_boxes",
)


def main():
    print("=" * 72)
    print(
        "ICRA27 M7 MULTI-TERRAIN "
        "GYM SMOKE"
    )
    print("=" * 72)

    base_port = 50810

    for index, terrain in enumerate(
        TERRAINS
    ):
        command_port = (
            base_port
            + 10 * index
        )

        log_dir = (
            ROOT
            / "results"
            / "icra27"
            / "m7_multiterrain_gym_smoke"
            / terrain
        )

        env = PyMPCM7Env(
            terrain=terrain,
            command_port=command_port,
            telemetry_port=(
                command_port + 1
            ),
            state_port=(
                command_port + 2
            ),
            log_dir=log_dir,
            max_episode_steps=5,
        )

        try:
            obs, info = env.reset(
                seed=0
            )

            expected_context = np.asarray(
                env.oracle_context,
                dtype=np.float32,
            )

            observed_context = np.asarray(
                obs[
                    :len(expected_context)
                ],
                dtype=np.float32,
            )

            if not np.allclose(
                observed_context,
                expected_context,
                rtol=0.0,
                atol=1e-6,
            ):
                raise RuntimeError(
                    f"{terrain}: "
                    "oracle context mismatch: "
                    f"obs={observed_context} "
                    f"expected={expected_context}"
                )

            action = np.zeros(
                4,
                dtype=np.float32,
            )

            (
                obs2,
                reward,
                terminated,
                truncated,
                info2,
            ) = env.step(
                action
            )

            print()
            print(
                f"terrain    : {terrain}"
            )

            print(
                "scene      : "
                f"{env.terrain_scene}"
            )

            print(
                "friction   : "
                f"{env.terrain_friction}"
            )

            print(
                "context    : "
                f"{observed_context.tolist()}"
            )

            print(
                "obs shape  : "
                f"{obs.shape}"
            )

            print(
                "reward     : "
                f"{float(reward):+.4f}"
            )

            print(
                "terminated : "
                f"{terminated}"
            )

            print(
                "truncated  : "
                f"{truncated}"
            )

            print(
                "safety     : "
                f"{info2.get('safety_state')}"
            )

            print(
                "override   : "
                f"{info2.get('override_active')}"
            )

        finally:
            env.close()

    print()
    print(
        "[ICRA27] M7 multi-terrain "
        "Gym smoke: PASS"
    )


if __name__ == "__main__":
    main()
