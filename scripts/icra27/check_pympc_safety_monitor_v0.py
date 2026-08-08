#!/usr/bin/env python3

from __future__ import annotations

import sys
from math import radians
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


from tracer_core.lowlevel.pympc_safety_monitor import (
    PyMPCSafetyMonitor,
    PyMPCSafetyState,
)


DT = 0.002

PLANNED = [1, 1, 1, 1]
PHYSICAL_OK = [1, 1, 1, 1]
PHYSICAL_DEFICIT = [1, 1, 1, 0]


def update(
    monitor,
    physical,
    *,
    roll_deg=0.0,
    pitch_deg=0.0,
    z=0.30,
):
    return monitor.update(
        planned_contact=PLANNED,
        physical_contact=physical,
        roll_rad=radians(roll_deg),
        pitch_rad=radians(pitch_deg),
        base_height_m=z,
        dt=DT,
    )


def main():
    monitor = PyMPCSafetyMonitor()

    # ----------------------------------------------------------
    # 1. Nominal contact: NORMAL.
    # ----------------------------------------------------------
    monitor.notify_structural_commit()

    for _ in range(120):
        status = update(
            monitor,
            PHYSICAL_OK,
        )

    assert (
        status.state
        == PyMPCSafetyState.NORMAL
    )

    # ----------------------------------------------------------
    # Partial-window regression:
    # one mismatch immediately after commit must NOT appear as
    # a 100% sustained deficit.
    # ----------------------------------------------------------
    monitor.reset()
    monitor.notify_structural_commit()

    status = update(
        monitor,
        PHYSICAL_DEFICIT,
    )

    assert (
        status.state
        == PyMPCSafetyState.NORMAL
    )

    assert (
        0.009
        <= status.support_deficit_fraction
        <= 0.011
    )

    # ----------------------------------------------------------
    # 2. 20% support-deficit occupancy in 200 ms:
    #    WATCH, but not UNSAFE.
    # ----------------------------------------------------------
    monitor.reset()
    monitor.notify_structural_commit()

    for i in range(100):
        physical = (
            PHYSICAL_DEFICIT
            if i >= 80
            else PHYSICAL_OK
        )

        status = update(
            monitor,
            physical,
        )

    assert (
        status.state
        == PyMPCSafetyState.WATCH
    )

    assert (
        0.19
        <= status.support_deficit_fraction
        <= 0.21
    )

    # ----------------------------------------------------------
    # 3. 40% support-deficit occupancy:
    #    UNSAFE and latched.
    # ----------------------------------------------------------
    monitor.reset()
    monitor.notify_structural_commit()

    for i in range(100):
        physical = (
            PHYSICAL_DEFICIT
            if i >= 60
            else PHYSICAL_OK
        )

        status = update(
            monitor,
            physical,
        )

    assert (
        status.state
        == PyMPCSafetyState.UNSAFE
    )

    assert monitor.unsafe_latched

    # Latch must remain even after contact recovers.
    for _ in range(20):
        status = update(
            monitor,
            PHYSICAL_OK,
        )

    assert (
        status.state
        == PyMPCSafetyState.UNSAFE
    )

    # ----------------------------------------------------------
    # 4. Hard attitude limit works even without a structural
    #    commit.
    # ----------------------------------------------------------
    monitor.reset()

    status = update(
        monitor,
        PHYSICAL_OK,
        roll_deg=16.0,
    )

    assert (
        status.state
        == PyMPCSafetyState.UNSAFE
    )

    # ----------------------------------------------------------
    # 5. Reset clears the latch.
    # ----------------------------------------------------------
    monitor.reset()

    assert (
        monitor.last_status.state
        == PyMPCSafetyState.NORMAL
    )

    assert not monitor.unsafe_latched

    print("=" * 72)
    print("ICRA27 M4 PyMPC safety monitor checker")
    print("=" * 72)
    print("nominal       : PASS")
    print("WATCH         : PASS")
    print("UNSAFE        : PASS")
    print("UNSAFE latch  : PASS")
    print("attitude hard : PASS")
    print("reset         : PASS")
    print("=" * 72)
    print("[ICRA27] safety monitor core PASS")


if __name__ == "__main__":
    main()
