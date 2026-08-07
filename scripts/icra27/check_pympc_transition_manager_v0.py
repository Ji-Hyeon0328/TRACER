#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


from tracer_core.lowlevel.pympc_meta_gait_adapter import (
    PyMPCMetaGaitReference,
)
from tracer_core.lowlevel.pympc_transition_manager import (
    PyMPCMetaGaitTransitionManager,
)


DT = 0.002


def ref(
    *,
    vx,
    yaw,
    height,
    clearance,
    frequency,
    duty,
):
    return PyMPCMetaGaitReference(
        vx=float(vx),
        yaw_rate=float(yaw),
        body_height=float(height),
        swing_clearance=float(clearance),
        gait_period=1.0 / float(frequency),
        duty_factor=float(duty),
    )


def assert_close(a, b, eps=1e-10):
    if abs(float(a) - float(b)) > eps:
        raise AssertionError(
            f"{a} != {b}"
        )


def main():
    manager = PyMPCMetaGaitTransitionManager()

    slow = ref(
        vx=0.12,
        yaw=0.00,
        height=0.28,
        clearance=0.04,
        frequency=1.0,
        duty=0.75,
    )

    nominal = ref(
        vx=0.20,
        yaw=0.20,
        height=0.30,
        clearance=0.06,
        frequency=1.4,
        duty=0.65,
    )

    fast = ref(
        vx=0.30,
        yaw=-0.20,
        height=0.32,
        clearance=0.08,
        frequency=2.0,
        duty=0.55,
    )

    # ------------------------------------------------------------
    # Initial command: applied directly.
    # ------------------------------------------------------------
    applied = manager.update(
        slow,
        current_contact=[1, 1, 1, 1],
        dt=DT,
    )

    assert_close(applied.vx, 0.12)
    assert_close(applied.body_height, 0.28)
    assert_close(applied.swing_clearance, 0.04)
    assert_close(applied.gait_frequency, 1.0)
    assert_close(applied.duty_factor, 0.75)

    # ------------------------------------------------------------
    # New target during non-full-stance.
    #
    # Continuous:
    # vx rate = 0.60 m/s^2 -> +0.0012 per tick
    # yaw rate = 1.00 rad/s^2 -> +0.002 per tick
    # h rate = 0.05 m/s -> +0.0001 per tick
    #
    # Structural values must remain old.
    # ------------------------------------------------------------
    applied = manager.update(
        nominal,
        current_contact=[1, 0, 0, 1],
        dt=DT,
    )

    assert_close(applied.vx, 0.1212)
    assert_close(applied.yaw_rate, 0.002)
    assert_close(applied.body_height, 0.2801)

    assert_close(applied.swing_clearance, 0.04)
    assert_close(applied.gait_frequency, 1.0)
    assert_close(applied.duty_factor, 0.75)

    if not manager.has_pending_structural_update:
        raise AssertionError(
            "Structural update was not queued"
        )

    if manager.last_structural_commit:
        raise AssertionError(
            "Structural update committed outside full stance"
        )

    # ------------------------------------------------------------
    # Latest pending target wins.
    # ------------------------------------------------------------
    applied = manager.update(
        fast,
        current_contact=[0, 1, 1, 0],
        dt=DT,
    )

    assert_close(applied.swing_clearance, 0.04)
    assert_close(applied.gait_frequency, 1.0)
    assert_close(applied.duty_factor, 0.75)

    # ------------------------------------------------------------
    # Full stance: commit structural part atomically.
    # Continuous values are still ramping.
    # ------------------------------------------------------------
    applied = manager.update(
        fast,
        current_contact=[1, 1, 1, 1],
        dt=DT,
    )

    assert_close(applied.swing_clearance, 0.08)
    assert_close(applied.gait_frequency, 2.0)
    assert_close(applied.duty_factor, 0.55)

    if not manager.last_structural_commit:
        raise AssertionError(
            "Structural update did not commit at full stance"
        )

    if manager.structural_commit_count != 1:
        raise AssertionError(
            "Unexpected structural commit count"
        )

    # ------------------------------------------------------------
    # Continue ticks: continuous channels must converge.
    # Keep structural target identical so no extra commit occurs.
    # ------------------------------------------------------------
    for _ in range(1000):
        applied = manager.update(
            fast,
            current_contact=[1, 0, 0, 1],
            dt=DT,
        )

    assert_close(applied.vx, 0.30)
    assert_close(applied.yaw_rate, -0.20)
    assert_close(applied.body_height, 0.32)

    if manager.structural_commit_count != 1:
        raise AssertionError(
            "Redundant structural commit occurred"
        )

    print("PyMPC transition manager unit check")
    print("-----------------------------------")
    print("initial reference                 : PASS")
    print("vx slew-rate limiting             : PASS")
    print("yaw-rate slew limiting            : PASS")
    print("body-height slew limiting         : PASS")
    print("structural hold outside stance    : PASS")
    print("latest pending target wins        : PASS")
    print("full-stance atomic commit         : PASS")
    print("continuous convergence            : PASS")
    print("no redundant structural commit   : PASS")
    print()
    print("M2 transition-manager unit check: PASS")


if __name__ == "__main__":
    main()
