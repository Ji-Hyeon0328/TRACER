#!/usr/bin/env python3

from __future__ import annotations

import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(
    0,
    str(ROOT),
)


from tracer_core.highlevel_rl.reward_v2 import (
    ENERGY_REFERENCE_POWER_W,
    ENERGY_SELECTOR_SCALE,
    NOMINAL_HIGH_LEVEL_DT_S,
    _find_m4_unsafe_limit,
    compute_simplified_tracer_costs,
    energy_cost,
    m4_unsafe_attitude_limits,
    motion_cost,
    objective_cost,
    objective_reward,
    stability_cost,
    TASK_FAILURE_ABSORBING_COST,
    task_feasibility_tail_cost,
)


def close(
    actual,
    expected,
    *,
    atol=1e-10,
):
    if not math.isclose(
        float(actual),
        float(expected),
        rel_tol=0.0,
        abs_tol=atol,
    ):
        raise AssertionError(
            f"{actual} != {expected}"
        )


def main():
    (
        roll_name,
        roll_limit,
    ) = _find_m4_unsafe_limit(
        axis="roll"
    )

    (
        pitch_name,
        pitch_limit,
    ) = _find_m4_unsafe_limit(
        axis="pitch"
    )

    print(
        "M4 roll unsafe field :",
        roll_name,
    )

    print(
        "M4 roll unsafe rad   :",
        roll_limit,
    )

    print(
        "M4 roll unsafe deg   :",
        math.degrees(
            roll_limit
        ),
    )

    print(
        "M4 pitch unsafe field:",
        pitch_name,
    )

    print(
        "M4 pitch unsafe rad  :",
        pitch_limit,
    )

    print(
        "M4 pitch unsafe deg  :",
        math.degrees(
            pitch_limit
        ),
    )

    # ------------------------------------------------------
    # Motion semantics
    # ------------------------------------------------------

    # 0.4 m/s toward goal.
    cm, rate, normalized = motion_cost(
        previous_goal_distance=1.0,
        goal_distance=0.92,
        decision_dt=0.20,
    )

    close(rate, 0.4)
    close(normalized, 1.0)
    close(cm, 0.0)

    # No progress.
    cm_stop, _, norm_stop = (
        motion_cost(
            previous_goal_distance=1.0,
            goal_distance=1.0,
            decision_dt=0.20,
        )
    )

    close(norm_stop, 0.0)
    close(cm_stop, 0.5)

    # Maximum reverse progress.
    cm_reverse, _, norm_reverse = (
        motion_cost(
            previous_goal_distance=1.0,
            goal_distance=1.08,
            decision_dt=0.20,
        )
    )

    close(norm_reverse, -1.0)
    close(cm_reverse, 1.0)

    # ------------------------------------------------------
    # Stability semantics
    # ------------------------------------------------------

    cs_zero, _, _ = stability_cost(
        roll=0.0,
        pitch=0.0,
        roll_unsafe_rad=roll_limit,
        pitch_unsafe_rad=pitch_limit,
    )

    close(cs_zero, 0.0)

    cs_half_roll, _, _ = stability_cost(
        roll=0.5 * roll_limit,
        pitch=0.0,
        roll_unsafe_rad=roll_limit,
        pitch_unsafe_rad=pitch_limit,
    )

    close(
        cs_half_roll,
        0.25,
    )

    cs_pitch_limit, _, _ = stability_cost(
        roll=0.0,
        pitch=pitch_limit,
        roll_unsafe_rad=roll_limit,
        pitch_unsafe_rad=pitch_limit,
    )

    close(
        cs_pitch_limit,
        1.0,
    )

    # ------------------------------------------------------
    # Energy semantics
    # ------------------------------------------------------

    # Zero mechanical power.
    ce_zero, e_zero = energy_cost(
        applied_abs_energy_j=0.0,
        energy_dt_s=0.20,
    )

    close(e_zero, 0.0)
    close(ce_zero, 0.0)

    # Flat nominal mechanical power:
    # physical ratio = 1, selector cost = 0.5.
    nominal_energy = (
        ENERGY_REFERENCE_POWER_W
        * 0.20
    )

    ce_nominal, e_nominal = (
        energy_cost(
            applied_abs_energy_j=(
                nominal_energy
            ),
            energy_dt_s=0.20,
        )
    )

    close(e_nominal, 1.0)

    close(
        ce_nominal,
        1.0
        / ENERGY_SELECTOR_SCALE,
    )

    close(
        ce_nominal,
        0.5,
    )

    # Twice nominal:
    # physical ratio 2 -> selector cost 1.
    ce_double, e_double = (
        energy_cost(
            applied_abs_energy_j=(
                2.0
                * nominal_energy
            ),
            energy_dt_s=0.20,
        )
    )

    close(e_double, 2.0)
    close(ce_double, 1.0)

    # Same mechanical work delivered over half the
    # measurement interval:
    #
    #   power ratio doubles,
    #   integrated energy cost stays unchanged.
    ce_same_work_fast, e_same_work_fast = (
        energy_cost(
            applied_abs_energy_j=(
                nominal_energy
            ),
            energy_dt_s=0.10,
        )
    )

    close(
        e_same_work_fast,
        2.0,
    )

    close(
        ce_same_work_fast,
        0.5,
    )

    # Same nominal physical power over only half the
    # interval means half the mechanical work.
    half_energy = (
        ENERGY_REFERENCE_POWER_W
        * 0.10
    )

    ce_half_work, e_same_power = (
        energy_cost(
            applied_abs_energy_j=(
                half_energy
            ),
            energy_dt_s=0.10,
        )
    )

    close(
        e_same_power,
        1.0,
    )

    close(
        ce_half_work,
        0.25,
    )

    close(
        NOMINAL_HIGH_LEVEL_DT_S,
        0.20,
    )

    # ------------------------------------------------------
    # Task-feasibility tail semantics
    # ------------------------------------------------------

    close(
        TASK_FAILURE_ABSORBING_COST,
        1.0,
    )

    # PPO env example:
    # 5 settling steps + 50 policy steps = horizon 55.
    #
    # Failure after 5 policy steps:
    # episode_step = 10 -> 45 policy decisions remain.
    tail, failure, remaining = (
        task_feasibility_tail_cost(
            episode_step=10,
            max_episode_steps=55,
            success=False,
            m4_terminal=True,
            native_terminated=False,
            native_truncated=False,
        )
    )

    close(tail, 45.0)
    assert failure is True
    assert remaining == 45

    # Failure after 25 policy steps.
    tail_mid, failure_mid, remaining_mid = (
        task_feasibility_tail_cost(
            episode_step=30,
            max_episode_steps=55,
            success=False,
            m4_terminal=False,
            native_terminated=True,
            native_truncated=False,
        )
    )

    close(tail_mid, 25.0)
    assert failure_mid is True
    assert remaining_mid == 25

    # Premature native truncation is also failure.
    tail_trunc, failure_trunc, remaining_trunc = (
        task_feasibility_tail_cost(
            episode_step=50,
            max_episode_steps=55,
            success=False,
            m4_terminal=False,
            native_terminated=False,
            native_truncated=True,
        )
    )

    close(tail_trunc, 5.0)
    assert failure_trunc is True
    assert remaining_trunc == 5

    # Success never receives absorbing failure cost.
    tail_success, failure_success, remaining_success = (
        task_feasibility_tail_cost(
            episode_step=45,
            max_episode_steps=55,
            success=True,
            m4_terminal=False,
            native_terminated=False,
            native_truncated=False,
        )
    )

    close(tail_success, 0.0)
    assert failure_success is False
    assert remaining_success == 10

    # Ordinary horizon exhaustion is not failure.
    tail_timeout, failure_timeout, remaining_timeout = (
        task_feasibility_tail_cost(
            episode_step=55,
            max_episode_steps=55,
            success=False,
            m4_terminal=False,
            native_terminated=False,
            native_truncated=False,
        )
    )

    close(tail_timeout, 0.0)
    assert failure_timeout is False
    assert remaining_timeout == 0

    # WATCH-only behavior enters this helper with all
    # failure flags false and therefore has no tail cost.
    tail_watch, failure_watch, remaining_watch = (
        task_feasibility_tail_cost(
            episode_step=20,
            max_episode_steps=55,
            success=False,
            m4_terminal=False,
            native_terminated=False,
            native_truncated=False,
        )
    )

    close(tail_watch, 0.0)
    assert failure_watch is False
    assert remaining_watch == 35

    # ------------------------------------------------------
    # Full objective / beta semantics
    # ------------------------------------------------------

    roll_limit_2, pitch_limit_2 = (
        m4_unsafe_attitude_limits()
    )

    close(
        roll_limit_2,
        roll_limit,
    )

    close(
        pitch_limit_2,
        pitch_limit,
    )

    costs = compute_simplified_tracer_costs(
        previous_goal_distance=1.0,

        # 0.2 m/s goal-directed progress:
        # normalized progress = 0.5,
        # C_m = 0.25.
        goal_distance=0.96,

        decision_dt=0.20,

        # Half roll safety fraction:
        # C_s = 0.25.
        roll=0.5 * roll_limit,
        pitch=0.0,

        roll_unsafe_rad=roll_limit,
        pitch_unsafe_rad=pitch_limit,

        # Flat nominal energy:
        # C_E = 0.5.
        applied_abs_energy_j=(
            nominal_energy
        ),

        energy_dt_s=0.20,
    )

    close(
        costs.motion,
        0.25,
    )

    close(
        costs.stability,
        0.25,
    )

    close(
        costs.energy,
        0.5,
    )

    uniform = (
        1.0 / 3.0,
        1.0 / 3.0,
        1.0 / 3.0,
    )

    expected_uniform_cost = (
        (
            0.25
            + 0.25
            + 0.5
        )
        / 3.0
    )

    j = objective_cost(
        costs=costs,
        beta=uniform,
    )

    r = objective_reward(
        costs=costs,
        beta=uniform,
    )

    close(
        j,
        expected_uniform_cost,
    )

    close(
        r,
        -expected_uniform_cost,
    )

    print()
    print(
        "example costs:",
        costs.as_dict(),
    )

    print(
        "uniform objective cost:",
        j,
    )

    print(
        "uniform objective reward:",
        r,
    )

    print()
    print(
        "[ICRA27] simplified TRACER "
        "cost/reward v2 contract: PASS"
    )


if __name__ == "__main__":
    main()
