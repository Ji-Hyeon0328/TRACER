#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT),
    )


from tracer_core.highlevel_rl.reward_slr_hl_v2 import (
    REWARD_SCHEMA,
    compute_slr_hl_reward_v2,
)


def close(
    actual,
    expected,
    *,
    atol=1e-12,
):
    if not np.isclose(
        float(actual),
        float(expected),
        rtol=0.0,
        atol=atol,
    ):
        raise AssertionError(
            f"actual={actual!r} "
            f"expected={expected!r}"
        )


def evaluate(
    *,
    robot_height=0.30,
    applied_height=0.30,
    energy_j=0.0,
):
    return compute_slr_hl_reward_v2(
        heading_error=0.0,

        base_vx_body=0.20,
        base_vy_body=0.0,
        base_vz_world=0.0,

        base_wx=0.0,
        base_wy=0.0,
        base_wz=0.0,

        pympc_robot_height_estimate=(
            robot_height
        ),

        applied_body_height=(
            applied_height
        ),

        roll=0.0,
        pitch=0.0,

        applied_abs_energy_j=energy_j,
        energy_dt_s=0.20,

        normalized_action=(
            0.0,
            0.0,
            0.0,
            0.0,
        ),

        previous_normalized_action=(
            0.0,
            0.0,
            0.0,
            0.0,
        ),

        previous_previous_normalized_action=(
            0.0,
            0.0,
            0.0,
            0.0,
        ),

        decision_dt_s=0.20,
    )


# ------------------------------------------------------------
# 1. Perfect tracking at nominal height.
# ------------------------------------------------------------

perfect = evaluate()

close(
    perfect.positive_reward,
    0.30,
)

close(
    perfect.centered_reward,
    0.0,
)

close(
    perfect.base_height_cost,
    0.0,
)


# ------------------------------------------------------------
# 2. Variable target must also be perfect.
#
# This is the key v2 contract:
# 0.24 m is NOT penalized simply for being different from
# the original SLR-Go2 fixed target of 0.32 m.
# ------------------------------------------------------------

low_height_perfect = evaluate(
    robot_height=0.24,
    applied_height=0.24,
)

close(
    low_height_perfect.base_height_cost,
    0.0,
)

close(
    low_height_perfect.centered_reward,
    0.0,
)


# ------------------------------------------------------------
# 3. 1 cm height tracking error.
#
# Cost = 0.01^2 = 1e-4
# weighted rate loss = 10 * 1e-4 = 0.001
# reward loss over dt=.2 = 0.0002
# ------------------------------------------------------------

height_error = evaluate(
    robot_height=0.31,
    applied_height=0.30,
)

close(
    height_error.base_height_cost,
    1.0e-4,
)

close(
    height_error.centered_reward,
    -2.0e-4,
)


# ------------------------------------------------------------
# 4. Power contract preserved from v1.
#
# 4 J / .2 s = 20 W
# weight = -2e-5
# rate loss = .0004
# transition loss = .00008
# ------------------------------------------------------------

powered = evaluate(
    energy_j=4.0,
)

close(
    powered.centered_reward,
    -8.0e-5,
)


# ------------------------------------------------------------
# 5. Large height error activates SLR only-positive clipping.
# ------------------------------------------------------------

clipped = evaluate(
    robot_height=1.00,
    applied_height=0.24,
)

close(
    clipped.positive_reward,
    0.0,
)

close(
    clipped.centered_reward,
    -0.30,
)


metadata = perfect.as_dict()

if (
    metadata["reward_schema"]
    != REWARD_SCHEMA
):
    raise AssertionError(
        "reward schema mismatch"
    )

if (
    metadata[
        "slr_height_target_adaptation"
    ]
    !=
    "current_applied_high_level_body_height"
):
    raise AssertionError(
        "unexpected height-target adaptation"
    )

if (
    metadata[
        "slr_height_measurement"
    ]
    !=
    "pympc_robot_height_estimate"
):
    raise AssertionError(
        "unexpected height measurement"
    )


print(
    "perfect centered reward       :",
    perfect.centered_reward,
)

print(
    "low-height centered reward    :",
    low_height_perfect.centered_reward,
)

print(
    "1cm height-error reward       :",
    height_error.centered_reward,
)

print(
    "powered centered reward       :",
    powered.centered_reward,
)

print(
    "clipped centered reward       :",
    clipped.centered_reward,
)

print()

print(
    "[ICRA27] SLR-HL adapted v2 "
    "pure reward contract: PASS"
)
