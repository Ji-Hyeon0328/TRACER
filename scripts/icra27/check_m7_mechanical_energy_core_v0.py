#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(
    0,
    str(ROOT),
)


from tracer_core.highlevel_rl.mechanical_energy import (
    MechanicalEnergyAccumulator,
    applied_generalized_power,
    commanded_joint_power,
)


def main():
    # --------------------------------------------------------
    # Fake 12-actuator quadruped.
    #
    # action indices: 0..11
    # qvel indices : 6..17
    #
    # Base generalized coordinates intentionally contain
    # large values to verify that applied power excludes them.
    # --------------------------------------------------------

    legs_tau_idx = SimpleNamespace(
        FL=np.array([0, 1, 2]),
        FR=np.array([3, 4, 5]),
        RL=np.array([6, 7, 8]),
        RR=np.array([9, 10, 11]),
    )

    legs_qvel_idx = SimpleNamespace(
        FL=np.array([6, 7, 8]),
        FR=np.array([9, 10, 11]),
        RL=np.array([12, 13, 14]),
        RR=np.array([15, 16, 17]),
    )

    action = np.arange(
        1.0,
        13.0,
    )

    qvel = np.zeros(
        18,
        dtype=float,
    )

    qvel[:6] = 1000.0
    qvel[6:] = 2.0

    qfrc = np.zeros(
        18,
        dtype=float,
    )

    # If floating-base entries were accidentally used,
    # this would make the test fail dramatically.
    qfrc[:6] = 1000.0
    qfrc[6:] = action

    env = SimpleNamespace(
        legs_tau_idx=legs_tau_idx,
        legs_qvel_idx=legs_qvel_idx,
        mjData=SimpleNamespace(
            qvel=qvel,
            qfrc_actuator=qfrc,
        ),
    )

    commanded = (
        commanded_joint_power(
            env,
            action,
        )
    )

    applied = (
        applied_generalized_power(
            env
        )
    )

    # 2 * sum(1..12) = 156 W.
    expected_w = 156.0

    assert np.isclose(
        commanded["signed_w"],
        expected_w,
    )

    assert np.isclose(
        commanded["abs_w"],
        expected_w,
    )

    assert np.isclose(
        commanded["positive_w"],
        expected_w,
    )

    assert np.isclose(
        applied["signed_w"],
        expected_w,
    )

    assert np.isclose(
        applied["abs_w"],
        expected_w,
    )

    assert np.isclose(
        applied["positive_w"],
        expected_w,
    )

    acc = MechanicalEnergyAccumulator()

    acc.add(
        dt=0.01,
        commanded=commanded,
        applied=applied,
    )

    summary = acc.as_dict()

    # 156 W * 0.01 s = 1.56 J.
    assert np.isclose(
        summary[
            "commanded_abs_j"
        ],
        1.56,
    )

    assert np.isclose(
        summary[
            "applied_abs_j"
        ],
        1.56,
    )

    print(
        "commanded:",
        commanded,
    )

    print(
        "applied  :",
        applied,
    )

    print(
        "energy   :",
        summary,
    )

    print()
    print(
        "[ICRA27] mechanical-energy "
        "core contract: PASS"
    )


if __name__ == "__main__":
    main()
