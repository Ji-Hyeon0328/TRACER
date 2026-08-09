#!/usr/bin/env python3

from __future__ import annotations

import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

from tracer_core.highlevel_rl.contracts import (
    M7Observation,
    build_observation,
    normalized_action_to_meta_gait,
    observation_names,
    physical_action_to_normalized,
)

from tracer_core.highlevel_rl.reward import (
    compute_m7_reward,
)


def close(a, b, tol=1e-8):
    if abs(float(a) - float(b)) > tol:
        raise AssertionError(
            f"{a!r} != {b!r}"
        )


def main():
    print("=" * 72)
    print("ICRA27 M7 HIGH-LEVEL RL CONTRACT CHECK")
    print("=" * 72)

    # ------------------------------------------------------------
    # Action mapping: zero must be characterized nominal.
    # ------------------------------------------------------------
    nominal = normalized_action_to_meta_gait(
        [0.0, 0.0, 0.0, 0.0]
    )

    close(nominal.vx, 0.20)
    close(nominal.yaw_rate, 0.0)
    close(nominal.body_height, 0.30)
    close(nominal.swing_clearance, 0.06)
    close(nominal.gait_period, 1.0 / 1.4)
    close(nominal.duty_factor, 0.65)

    print("zero action -> nominal      : PASS")

    lo = normalized_action_to_meta_gait(
        [-1.0, -1.0, -1.0, -1.0]
    )

    close(lo.vx, 0.00)
    close(lo.yaw_rate, -0.40)
    close(lo.body_height, 0.24)
    close(lo.swing_clearance, 0.03)

    hi = normalized_action_to_meta_gait(
        [1.0, 1.0, 1.0, 1.0]
    )

    close(hi.vx, 0.40)
    close(hi.yaw_rate, 0.40)
    close(hi.body_height, 0.32)
    close(hi.swing_clearance, 0.09)

    print("normalized bounds           : PASS")

    roundtrip = physical_action_to_normalized(
        [
            nominal.vx,
            nominal.yaw_rate,
            nominal.body_height,
            nominal.swing_clearance,
        ]
    )

    if not np.allclose(
        roundtrip,
        np.zeros(4),
        atol=1e-7,
    ):
        raise AssertionError(
            f"nominal roundtrip failed: {roundtrip}"
        )

    print("physical/action roundtrip   : PASS")

    # Structural authority must remain fixed even at action extremes.
    close(lo.gait_period, hi.gait_period)
    close(lo.duty_factor, hi.duty_factor)

    print("structural fixed nominal    : PASS")

    # ------------------------------------------------------------
    # Observation contract.
    # ------------------------------------------------------------
    obs = M7Observation(
        oracle_context=(1.0, 0.0, 0.0),

        goal_dx_body=0.50,
        goal_dy_body=0.10,
        goal_distance=math.hypot(0.50, 0.10),
        heading_error=math.atan2(0.10, 0.50),

        base_vx_body=0.18,
        base_vy_body=0.01,
        yaw_rate=0.02,
        base_z=0.30,
        roll=0.01,
        pitch=-0.02,

        applied_vx=0.20,
        applied_yaw=0.0,
        applied_height=0.30,
        applied_clearance=0.06,

        previous_action=(
            0.0,
            0.0,
            0.0,
            0.0,
        ),
    )

    vec = build_observation(obs)
    names = observation_names(3)

    if vec.shape != (21,):
        raise AssertionError(
            f"expected obs dim 21, got {vec.shape}"
        )

    if len(names) != len(vec):
        raise AssertionError(
            "observation name/vector mismatch"
        )

    print("observation K+18 layout     : PASS")

    # ------------------------------------------------------------
    # Reward semantics.
    # ------------------------------------------------------------
    reward_forward, c_forward = compute_m7_reward(
        previous_goal_distance=0.50,
        goal_distance=0.46,
        heading_error=0.0,
        roll=0.0,
        pitch=0.0,
        normalized_action=[0, 0, 0, 0],
        previous_normalized_action=[0, 0, 0, 0],
        decision_dt=0.20,
    )

    reward_backward, _ = compute_m7_reward(
        previous_goal_distance=0.50,
        goal_distance=0.54,
        heading_error=0.0,
        roll=0.0,
        pitch=0.0,
        normalized_action=[0, 0, 0, 0],
        previous_normalized_action=[0, 0, 0, 0],
        decision_dt=0.20,
    )

    if not reward_forward > reward_backward:
        raise AssertionError(
            "forward progress must score above backward motion"
        )

    reward_override, c_override = compute_m7_reward(
        previous_goal_distance=0.50,
        goal_distance=0.46,
        heading_error=0.0,
        roll=0.0,
        pitch=0.0,
        normalized_action=[0, 0, 0, 0],
        previous_normalized_action=[0, 0, 0, 0],
        decision_dt=0.20,
        override_active=True,
        safety_state="unsafe",
    )

    if not reward_override < reward_forward:
        raise AssertionError(
            "M4 intervention must reduce reward"
        )

    reward_success, _ = compute_m7_reward(
        previous_goal_distance=0.17,
        goal_distance=0.14,
        heading_error=0.0,
        roll=0.0,
        pitch=0.0,
        normalized_action=[0, 0, 0, 0],
        previous_normalized_action=[0, 0, 0, 0],
        decision_dt=0.20,
        success=True,
    )

    if not reward_success > reward_forward:
        raise AssertionError(
            "success bonus must increase reward"
        )

    print("progress reward semantics   : PASS")
    print("M4 intervention penalty     : PASS")
    print("success reward semantics    : PASS")

    print()
    print("sample nominal:")
    print(
        [
            nominal.vx,
            nominal.yaw_rate,
            nominal.body_height,
            nominal.swing_clearance,
            nominal.gait_period,
            nominal.duty_factor,
        ]
    )

    print()
    print("sample observation dim:", len(vec))
    print("sample progress reward:", reward_forward)
    print(
        "sample intervention reward:",
        reward_override,
    )
    print(
        "intervention component:",
        c_override["m4_intervention"],
    )

    print()
    print("[ICRA27] M7 high-level contracts: PASS")


if __name__ == "__main__":
    main()
