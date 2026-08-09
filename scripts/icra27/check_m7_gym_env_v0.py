#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


from tracer_core.highlevel_rl.pympc_env import (
    PyMPCM7Env,
)


ACTIONS = [
    [0.0, 0.0, 0.0, 0.0],
    [0.2, 0.0, 0.0, 0.0],
    [0.0, 0.1, 0.0, 0.0],
    [0.0, 0.0, 0.1, -0.1],
]


def check_finite(name, x):
    arr = np.asarray(
        x,
        dtype=float,
    )

    if not np.all(
        np.isfinite(arr)
    ):
        raise AssertionError(
            f"{name} contains non-finite values"
        )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--log-dir",
        default=(
            "results/icra27/"
            "m7_gym_env_smoke"
        ),
    )

    args = ap.parse_args()

    log_dir = Path(
        args.log_dir
    )

    log_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    env = PyMPCM7Env(
        oracle_context=(
            1.0,
            0.0,
            0.0,
        ),

        goal_distance_m=0.50,
        success_radius_m=0.15,

        decision_dt_s=0.20,
        max_episode_steps=10,

        command_port=50610,
        telemetry_port=50611,
        state_port=50612,

        command_repeat_hz=20.0,
        telemetry_hz=100.0,
        state_hz=100.0,

        log_dir=log_dir,
    )

    try:
        print("=" * 72)
        print(
            "ICRA27 M7 GYMNASIUM ENV CHECK"
        )
        print("=" * 72)

        # --------------------------------------------------------
        # Reset A
        # --------------------------------------------------------
        obs_a, info_a = env.reset(
            seed=0
        )

        print(
            "reset A obs shape :",
            obs_a.shape,
        )

        print(
            "reset A goal      :",
            info_a["goal_distance"],
        )

        print(
            "reset A sim time  :",
            info_a[
                "initial_state_sim_time_s"
            ],
        )

        check_finite(
            "reset A observation",
            obs_a,
        )

        if obs_a.shape != (21,):
            raise AssertionError(
                "K=3 observation must "
                f"have shape (21,), got "
                f"{obs_a.shape}"
            )

        if not np.allclose(
            obs_a[:3],
            np.asarray(
                [1.0, 0.0, 0.0],
                dtype=np.float32,
            ),
        ):
            raise AssertionError(
                "oracle context not present "
                "at observation head"
            )

        # --------------------------------------------------------
        # A few actual Gym steps
        # --------------------------------------------------------
        rewards = []

        for i, action in enumerate(
            ACTIONS
        ):
            (
                obs,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(action)

            check_finite(
                f"step {i} observation",
                obs,
            )

            if not math.isfinite(
                reward
            ):
                raise AssertionError(
                    "reward is not finite"
                )

            required_info = {
                "requested_normalized",
                "requested_physical",
                "applied_physical",
                "safety_state",
                "override_active",
                "m4_intervention",
                "reward_components",
                "goal_distance",
                "decision_dt_s",
            }

            missing = (
                required_info
                - set(info)
            )

            if missing:
                raise AssertionError(
                    f"missing info fields: "
                    f"{sorted(missing)}"
                )

            rewards.append(
                float(reward)
            )

            print(
                f"step {i}: "
                f"reward={reward:+.4f} "
                f"goal={info['goal_distance']:.4f} "
                f"dt={info['decision_dt_s']:.4f} "
                f"safety={info['safety_state']} "
                f"override={info['override_active']}"
            )

            if (
                terminated
                or truncated
            ):
                break

        # --------------------------------------------------------
        # Reset B with same seed:
        # this kills A runner and launches a genuinely new episode.
        # --------------------------------------------------------
        obs_b, info_b = env.reset(
            seed=0
        )

        check_finite(
            "reset B observation",
            obs_b,
        )

        initial_diff = float(
            np.max(
                np.abs(
                    obs_a - obs_b
                )
            )
        )

        print()
        print(
            "same-seed reset "
            "max |obsA-obsB|:",
            f"{initial_diff:.8f}",
        )

        # Initial PyMPC state should be deterministic.
        # Small tolerance allows harmless numerical/runtime
        # initialization noise without hiding real reset drift.
        if initial_diff > 1e-3:
            raise AssertionError(
                "same-seed reset is not "
                "sufficiently deterministic: "
                f"max diff={initial_diff}"
            )

        if (
            info_a["goal_distance"]
            != info_b["goal_distance"]
        ):
            if not math.isclose(
                info_a["goal_distance"],
                info_b["goal_distance"],
                abs_tol=1e-6,
            ):
                raise AssertionError(
                    "same-seed goal reset differs"
                )

        # --------------------------------------------------------
        # Logging
        # --------------------------------------------------------
        episode_logs = sorted(
            log_dir.glob(
                "episode_*.jsonl"
            )
        )

        if len(episode_logs) < 1:
            raise AssertionError(
                "episode JSONL log not created"
            )

        print()
        print(
            "Gym reset semantics       : PASS"
        )

        print(
            "Gym step semantics        : PASS"
        )

        print(
            "oracle context input      : PASS"
        )

        print(
            "M4 reward/info exposure   : PASS"
        )

        print(
            "same-seed reset           : PASS"
        )

        print(
            "episode logging           : PASS"
        )

        print()
        print(
            "[ICRA27] M7 minimal "
            "Gymnasium environment: PASS"
        )

    finally:
        env.close()

    if env.runner_process is not None:
        raise AssertionError(
            "runner process not cleaned up"
        )

    if env.transport is not None:
        raise AssertionError(
            "transport not cleaned up"
        )

    print(
        "runtime cleanup           : PASS"
    )


if __name__ == "__main__":
    main()
