#!/usr/bin/env python3

from __future__ import annotations

import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tracer_core.highlevel_rl.reward_slr_hl_v1 import (
    MAX_POSITIVE_REWARD_RATE,
    goal_tracking_reference,
    projected_gravity_xy_cost,
    compute_slr_hl_reward,
)


def close(
    actual,
    expected,
    *,
    atol=1e-10,
):
    if abs(
        float(actual)
        - float(expected)
    ) > atol:
        raise AssertionError(
            f"{actual} != {expected}"
        )


def nominal_kwargs():
    return {
        "heading_error": 0.0,

        "base_vx_body": 0.20,
        "base_vy_body": 0.0,
        "base_vz_world": 0.0,

        "base_wx": 0.0,
        "base_wy": 0.0,
        "base_wz": 0.0,

        "base_z": 0.32,
        "roll": 0.0,
        "pitch": 0.0,

        "applied_abs_energy_j": 0.0,
        "energy_dt_s": 0.20,

        "normalized_action":
            [0.0, 0.0, 0.0, 0.0],

        "previous_normalized_action":
            [0.0, 0.0, 0.0, 0.0],

        "previous_previous_normalized_action":
            [0.0, 0.0, 0.0, 0.0],

        "decision_dt_s": 0.20,
    }


def main():
    # ------------------------------------------------------
    # Goal-derived reference.
    # ------------------------------------------------------

    vx, vy, wz = (
        goal_tracking_reference(
            heading_error=0.4
        )
    )

    close(vx, 0.20)
    close(vy, 0.0)
    close(wz, 0.20)

    _, _, wz_sat = (
        goal_tracking_reference(
            heading_error=2.0
        )
    )

    close(wz_sat, 0.40)

    # ------------------------------------------------------
    # Projected gravity.
    # ------------------------------------------------------

    close(
        projected_gravity_xy_cost(
            roll=0.0,
            pitch=0.0,
        ),
        0.0,
    )

    close(
        projected_gravity_xy_cost(
            roll=math.pi / 2.0,
            pitch=0.0,
        ),
        1.0,
    )

    # ------------------------------------------------------
    # Mathematically perfect adapted SLR transition.
    #
    # Positive SLR reward:
    #
    #   dt * (1.0 + 0.5)
    #   = 0.2 * 1.5
    #   = 0.3
    #
    # Centered reward = 0.
    # ------------------------------------------------------

    perfect = compute_slr_hl_reward(
        **nominal_kwargs()
    )

    close(
        MAX_POSITIVE_REWARD_RATE,
        1.5,
    )

    close(
        perfect.tracking_lin_vel_reward,
        1.0,
    )

    close(
        perfect.tracking_ang_vel_reward,
        1.0,
    )

    close(
        perfect.positive_reward,
        0.30,
    )

    close(
        perfect.maximum_positive_reward,
        0.30,
    )

    close(
        perfect.centered_reward,
        0.0,
    )

    # ------------------------------------------------------
    # Zero forward velocity.
    # ------------------------------------------------------

    kw = nominal_kwargs()
    kw["base_vx_body"] = 0.0

    stopped = compute_slr_hl_reward(
        **kw
    )

    expected_lin = math.exp(
        -(0.20 ** 2) / 0.25
    )

    close(
        stopped.tracking_lin_vel_reward,
        expected_lin,
    )

    expected_positive = (
        0.20
        * (
            expected_lin
            + 0.5
        )
    )

    close(
        stopped.positive_reward,
        expected_positive,
    )

    close(
        stopped.centered_reward,
        expected_positive - 0.30,
    )

    # ------------------------------------------------------
    # Action-rate and second-difference adaptation.
    #
    # a_t=[1,0,0,0]
    # a_t-1=0
    # a_t-2=0
    #
    # rate = 1
    # second difference = 1
    # total weighted rate reduction = 0.02.
    # ------------------------------------------------------

    kw = nominal_kwargs()

    kw[
        "normalized_action"
    ] = [1.0, 0.0, 0.0, 0.0]

    changed = compute_slr_hl_reward(
        **kw
    )

    close(
        changed.action_rate_cost,
        1.0,
    )

    close(
        changed.action_smoothness_cost,
        1.0,
    )

    close(
        changed.positive_reward,
        0.20 * 1.48,
    )

    close(
        changed.centered_reward,
        -0.004,
    )

    # ------------------------------------------------------
    # ACK-interval average mechanical power adaptation.
    #
    # 4 J / 0.2 s = 20 W.
    #
    # SLR power reduction:
    #   -2e-5 * 20 = -0.0004 reward-rate.
    # ------------------------------------------------------

    kw = nominal_kwargs()
    kw["applied_abs_energy_j"] = 4.0

    powered = compute_slr_hl_reward(
        **kw
    )

    close(
        powered.power_w,
        20.0,
    )

    close(
        powered.centered_reward,
        -0.00008,
    )

    # ------------------------------------------------------
    # only_positive_rewards contract.
    #
    # Huge base-height error makes raw dense rate negative.
    # SLR positive reward clips to zero, then centering gives
    # exactly -0.30 for a 0.20 s decision.
    # ------------------------------------------------------

    kw = nominal_kwargs()
    kw["base_z"] = 1.30

    clipped = compute_slr_hl_reward(
        **kw
    )

    assert (
        clipped.weighted_reward_rate_before_clip
        < 0.0
    )

    close(
        clipped.positive_reward,
        0.0,
    )

    close(
        clipped.centered_reward,
        -0.30,
    )

    # ------------------------------------------------------
    # Metadata/provenance.
    # ------------------------------------------------------

    d = perfect.as_dict()

    assert (
        d["reward_schema"]
        == "icra27_slr_hl_adapted_v1"
    )

    assert (
        d["slr_omitted_dof_acc"]
        is True
    )

    assert (
        d["slr_omitted_foot_clearance"]
        is True
    )

    assert (
        d["slr_adaptation_tracking_frame"]
        == "m7_body_yaw"
    )

    print(
        "perfect positive reward :",
        perfect.positive_reward,
    )

    print(
        "perfect centered reward :",
        perfect.centered_reward,
    )

    print(
        "stopped centered reward :",
        stopped.centered_reward,
    )

    print(
        "changed centered reward :",
        changed.centered_reward,
    )

    print(
        "powered centered reward :",
        powered.centered_reward,
    )

    print(
        "clipped centered reward :",
        clipped.centered_reward,
    )

    print()
    print(
        "[ICRA27] SLR-HL adapted v1 "
        "pure reward contract: PASS"
    )


if __name__ == "__main__":
    main()
