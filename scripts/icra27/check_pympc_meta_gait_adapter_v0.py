#!/usr/bin/env python3

from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


from tracer_core.lowlevel.pympc_meta_gait_adapter import (
    PyMPCMetaGaitAdapter,
)


class FakePGG:
    def __init__(self):
        self.step_freq = 1.4
        self.duty_factor = 0.65


class FakeFRG:
    def __init__(self):
        self.stance_time = 0.65 / 1.4


class FakeSTC:
    def __init__(self):
        self.calls = []

    def regenerate_swing_trajectory_generator(
        self,
        *,
        step_height,
        swing_period,
    ):
        self.calls.append(
            {
                "step_height": float(step_height),
                "swing_period": float(swing_period),
            }
        )


class FakeWB:
    def __init__(self):
        self.pgg = FakePGG()
        self.frg = FakeFRG()
        self.stc = FakeSTC()
        self.step_height = 0.06


class FakeCfg:
    simulation_params = {
        "ref_z": 0.28,
    }


def command(**kwargs):
    values = dict(
        vx=0.20,
        yaw_rate=0.0,
        body_height=0.28,
        swing_clearance=0.06,
        gait_period=1.0 / 1.4,
        duty_factor=0.65,
    )
    values.update(kwargs)
    return SimpleNamespace(**values)


def assert_close(a, b, eps=1e-12):
    if abs(float(a) - float(b)) > eps:
        raise AssertionError(f"{a} != {b}")


def main():
    wb = FakeWB()
    cfg = FakeCfg()
    adapter = PyMPCMetaGaitAdapter()

    # ------------------------------------------------------------
    # Case 1: deliberately outside validated bounds.
    # Verify physical safety projection.
    # ------------------------------------------------------------
    projected = adapter.apply_runtime(
        wb,
        cfg,
        command(
            vx=0.80,
            yaw_rate=-0.80,
            body_height=0.20,
            swing_clearance=0.15,
            gait_period=0.30,
            duty_factor=0.90,
        ),
    )

    assert_close(projected.vx, 0.40)
    assert_close(projected.yaw_rate, -0.40)
    assert_close(projected.body_height, 0.24)
    assert_close(projected.swing_clearance, 0.09)
    assert_close(projected.gait_period, 0.50)
    assert_close(projected.gait_frequency, 2.0)
    assert_close(projected.duty_factor, 0.80)

    assert_close(wb.pgg.step_freq, 2.0)
    assert_close(wb.pgg.duty_factor, 0.80)

    # stance = D/f = 0.8/2 = 0.4 s
    assert_close(wb.frg.stance_time, 0.40)

    # swing = (1-D)/f = 0.2/2 = 0.1 s
    assert_close(
        wb.stc.calls[-1]["swing_period"],
        0.10,
    )
    assert_close(
        wb.stc.calls[-1]["step_height"],
        0.09,
    )
    assert_close(cfg.simulation_params["ref_z"], 0.24)

    calls_after_first = len(wb.stc.calls)

    # ------------------------------------------------------------
    # Case 2: exact same structural command.
    # Swing trajectory must NOT regenerate again.
    # ------------------------------------------------------------
    adapter.apply_runtime(
        wb,
        cfg,
        command(
            vx=0.40,
            yaw_rate=-0.40,
            body_height=0.24,
            swing_clearance=0.09,
            gait_period=0.50,
            duty_factor=0.80,
        ),
    )

    if len(wb.stc.calls) != calls_after_first:
        raise AssertionError(
            "Unchanged command regenerated swing trajectory"
        )

    # ------------------------------------------------------------
    # Case 3: body-height only.
    # Must update ref_z without regenerating swing trajectory.
    # ------------------------------------------------------------
    adapter.apply_runtime(
        wb,
        cfg,
        command(
            vx=0.40,
            yaw_rate=-0.40,
            body_height=0.30,
            swing_clearance=0.09,
            gait_period=0.50,
            duty_factor=0.80,
        ),
    )

    assert_close(cfg.simulation_params["ref_z"], 0.30)

    if len(wb.stc.calls) != calls_after_first:
        raise AssertionError(
            "Body-height-only change regenerated swing trajectory"
        )

    # ------------------------------------------------------------
    # Case 4: timing change.
    # T = 0.8 -> f = 1.25 Hz
    # D = 0.6
    # stance = 0.48 s
    # swing = 0.32 s
    # ------------------------------------------------------------
    ref = adapter.apply_runtime(
        wb,
        cfg,
        command(
            vx=0.20,
            yaw_rate=0.20,
            body_height=0.30,
            swing_clearance=0.06,
            gait_period=0.80,
            duty_factor=0.60,
        ),
    )

    assert_close(ref.gait_frequency, 1.25)
    assert_close(wb.pgg.step_freq, 1.25)
    assert_close(wb.pgg.duty_factor, 0.60)
    assert_close(wb.frg.stance_time, 0.48)

    assert_close(
        wb.stc.calls[-1]["swing_period"],
        0.32,
    )
    assert_close(
        wb.stc.calls[-1]["step_height"],
        0.06,
    )

    vx, yaw = adapter.velocity_command(ref)
    assert_close(vx, 0.20)
    assert_close(yaw, 0.20)

    print("PyMPC MetaGait adapter unit check")
    print("--------------------------------")
    print("physical safety projection : PASS")
    print("period -> frequency         : PASS")
    print("PGG timing update           : PASS")
    print("FRG stance-time sync        : PASS")
    print("STC swing-period sync       : PASS")
    print("body-height runtime ref     : PASS")
    print("no redundant regeneration   : PASS")
    print("velocity/yaw extraction     : PASS")
    print()
    print("M2 adapter unit check: PASS")


if __name__ == "__main__":
    main()
