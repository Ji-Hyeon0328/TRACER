#!/usr/bin/env python3

from __future__ import annotations

from math import radians
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


from tracer_core.highlevel.meta_gait import (
    MetaGaitCommand,
)
from tracer_core.lowlevel.pympc_lowlevel_runtime import (
    PyMPCLowLevelRuntime,
)


NOMINAL = MetaGaitCommand(
    vx=0.20,
    yaw_rate=0.0,
    body_height=0.30,
    swing_clearance=0.06,
    gait_period=1.0 / 1.4,
    duty_factor=0.65,
    source_mode="icra27_runtime_check",
    source_reason="known nominal",
)


TARGET = MetaGaitCommand(
    vx=0.20,
    yaw_rate=0.0,
    body_height=0.30,
    swing_clearance=0.06,
    gait_period=1.0 / 1.0,
    duty_factor=0.55,
    source_mode="icra27_runtime_check",
    source_reason="structural target",
)


class DummyPGG:
    def __init__(self):
        self.step_freq = None
        self.duty_factor = None


class DummyFRG:
    def __init__(self):
        self.stance_time = None


class DummySTC:
    def __init__(self):
        self.calls = []

    def regenerate_swing_trajectory_generator(
        self,
        *,
        step_height,
        swing_period,
    ):
        self.calls.append({
            "step_height": float(step_height),
            "swing_period": float(swing_period),
        })


class DummyWB:
    def __init__(self):
        self.pgg = DummyPGG()
        self.frg = DummyFRG()
        self.stc = DummySTC()
        self.step_height = None


class DummyCfg:
    def __init__(self):
        self.simulation_params = {}


def close(a, b, tol=1e-12):
    return abs(float(a) - float(b)) <= tol


def main():
    runtime = PyMPCLowLevelRuntime(
        fallback_command=NOMINAL,
    )

    full_stance = np.ones(4)
    not_full_stance = np.array(
        [1, 0, 0, 1]
    )

    dt = 0.002

    # --------------------------------------------------
    # 1. First nominal command passes directly.
    # --------------------------------------------------
    r0 = runtime.resolve_command(
        requested_label="nominal",
        requested_command=NOMINAL,
        current_contact=full_stance,
        dt=dt,
    )

    assert r0.requested_label == "nominal"
    assert r0.selected_label == "nominal"
    assert r0.override_active is False
    assert r0.structural_commit is False

    assert close(
        r0.applied.gait_frequency,
        1.4,
    )

    assert close(
        r0.applied.duty_factor,
        0.65,
    )

    assert runtime.command is r0.applied
    assert runtime.target_label == "nominal"

    # --------------------------------------------------
    # 2. Structural target waits outside full stance.
    # --------------------------------------------------
    r1 = runtime.resolve_command(
        requested_label="target",
        requested_command=TARGET,
        current_contact=not_full_stance,
        dt=dt,
    )

    assert r1.selected_label == "target"
    assert r1.override_active is False

    assert (
        runtime.transition_manager
        .has_pending_structural_update
    )

    assert close(
        r1.selected_projected.gait_frequency,
        1.0,
    )

    # Structural timing must still be nominal.
    assert close(
        r1.applied.gait_frequency,
        1.4,
    )

    assert close(
        r1.applied.duty_factor,
        0.65,
    )

    # --------------------------------------------------
    # 3. Full stance atomically commits structural target.
    # --------------------------------------------------
    r2 = runtime.resolve_command(
        requested_label="target",
        requested_command=TARGET,
        current_contact=full_stance,
        dt=dt,
    )

    assert r2.structural_commit is True

    assert close(
        r2.applied.gait_frequency,
        1.0,
    )

    assert close(
        r2.applied.duty_factor,
        0.55,
    )

    assert not (
        runtime.transition_manager
        .has_pending_structural_update
    )

    # --------------------------------------------------
    # 4. Applied reference reaches existing PyMPC adapter.
    # --------------------------------------------------
    wb = DummyWB()
    cfg = DummyCfg()

    applied_runtime = runtime.apply_to_pympc(
        wb_interface=wb,
        pympc_cfg=cfg,
    )

    assert close(
        applied_runtime.gait_frequency,
        1.0,
    )

    assert close(
        wb.pgg.step_freq,
        1.0,
    )

    assert close(
        wb.pgg.duty_factor,
        0.55,
    )

    assert close(
        wb.step_height,
        0.06,
    )

    assert close(
        cfg.simulation_params["ref_z"],
        0.30,
    )

    assert len(wb.stc.calls) == 1

    # Reapplying the exact same reference must not
    # regenerate the swing trajectory again.
    runtime.apply_to_pympc(
        wb_interface=wb,
        pympc_cfg=cfg,
    )

    assert len(wb.stc.calls) == 1

    # --------------------------------------------------
    # 5. Physical UNSAFE observation latches supervisor.
    # --------------------------------------------------
    obs = runtime.observe_physical_state(
        planned_contact=full_stance,
        physical_contact=full_stance,
        roll_rad=radians(20.0),
        pitch_rad=0.0,
        base_height_m=0.30,
        dt=dt,
        time_s=1.0,
    )

    assert obs.status.state.value == "unsafe"
    assert obs.override_activated is True

    assert runtime.safety_monitor.unsafe_latched
    assert runtime.supervisor.override_active

    # --------------------------------------------------
    # 6. Same requested target is now replaced by
    #    the known nominal M4 fallback.
    # --------------------------------------------------
    r3 = runtime.resolve_command(
        requested_label="target",
        requested_command=TARGET,
        current_contact=full_stance,
        dt=dt,
    )

    assert r3.requested_label == "target"

    assert (
        r3.selected_label
        == "m4_backoff_nominal"
    )

    assert r3.override_active is True

    assert close(
        r3.requested_projected.gait_frequency,
        1.0,
    )

    assert close(
        r3.selected_projected.gait_frequency,
        1.4,
    )

    assert r3.structural_commit is True

    assert close(
        r3.applied.gait_frequency,
        1.4,
    )

    assert close(
        r3.applied.duty_factor,
        0.65,
    )

    # --------------------------------------------------
    # 7. Episode reset clears execution state only.
    # --------------------------------------------------
    runtime.reset_episode()

    assert runtime.command is None
    assert runtime.target_label is None

    assert (
        runtime.transition_manager
        .current_reference
        is None
    )

    assert not (
        runtime.transition_manager
        .has_pending_structural_update
    )

    assert (
        runtime.safety_monitor
        .last_status.state.value
        == "normal"
    )

    assert not runtime.safety_monitor.unsafe_latched
    assert not runtime.supervisor.override_active

    assert runtime.lifecycle.reset_count == 1

    print(
        "PyMPC low-level runtime checker: PASS"
    )

    print(
        "command path: "
        "requested -> selected -> projected "
        "-> guarded applied -> PyMPC"
    )

    print(
        "M4 path: "
        "physical UNSAFE -> override latch "
        "-> nominal selected -> guarded commit"
    )

    print(
        "reset: execution state -> clean"
    )


if __name__ == "__main__":
    main()
