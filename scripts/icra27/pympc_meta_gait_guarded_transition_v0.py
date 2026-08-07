#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ICRA27_SCRIPTS = ROOT / "scripts" / "icra27"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ICRA27_SCRIPTS))

import pympc_meta_gait_integration_v0 as base

from tracer_core.highlevel.meta_gait import MetaGaitCommand
from tracer_core.lowlevel.pympc_transition_manager import (
    PyMPCMetaGaitTransitionManager,
)


DT = float(base.cfg.simulation_params["dt"])
DURATION = 12.0

MANAGER = PyMPCMetaGaitTransitionManager()

CURRENT_TARGET_LABEL = None
STRUCTURAL_EVENTS = []


SCHEDULE = [
    (
        "slow",
        0.0,
        4.0,
        MetaGaitCommand(
            vx=0.12,
            yaw_rate=0.0,
            body_height=0.28,
            swing_clearance=0.04,
            gait_period=1.0,
            duty_factor=0.75,
            source_mode="icra27_guarded_transition",
            source_reason="slow",
        ),
    ),
    (
        "nominal",
        4.0,
        8.0,
        MetaGaitCommand(
            vx=0.20,
            yaw_rate=0.0,
            body_height=0.30,
            swing_clearance=0.06,
            gait_period=1.0 / 1.4,
            duty_factor=0.65,
            source_mode="icra27_guarded_transition",
            source_reason="nominal",
        ),
    ),
    (
        "fast",
        8.0,
        12.0,
        MetaGaitCommand(
            vx=0.30,
            yaw_rate=0.0,
            body_height=0.32,
            swing_clearance=0.08,
            gait_period=0.50,
            duty_factor=0.55,
            source_mode="icra27_guarded_transition",
            source_reason="fast",
        ),
    ),
]


def target_for_time(t):
    for label, t0, t1, command in SCHEDULE:
        if t0 <= t < t1:
            return label, command

    return (
        SCHEDULE[-1][0],
        SCHEDULE[-1][3],
    )


def clear_logs():
    global CURRENT_TARGET_LABEL

    for key, value in base.LOG.items():
        if key == "feet_z":
            for leg in base.LEGS:
                value[leg].clear()
        else:
            value.clear()

    base.ADAPTER.reset()
    MANAGER.reset()

    base.COMMAND = None

    CURRENT_TARGET_LABEL = None
    STRUCTURAL_EVENTS.clear()


def guarded_compute_actions(
    self,
    *args,
    **kwargs,
):
    global CURRENT_TARGET_LABEL

    if "step_num" in kwargs:
        step_num = int(kwargs["step_num"])
    else:
        step_num = int(args[13])

    if "simulation_dt" in kwargs:
        simulation_dt = float(
            kwargs["simulation_dt"]
        )
    else:
        simulation_dt = float(args[10])

    t = step_num * simulation_dt

    label, command = target_for_time(t)

    target = base.ADAPTER.project(command)

    if label != CURRENT_TARGET_LABEL:
        print()
        print("=" * 72)
        print(
            f"[ICRA27 target] "
            f"t={t:.3f}s -> {label}"
        )
        print(
            f"target: "
            f"vx={target.vx:.3f}, "
            f"h={target.body_height:.3f}, "
            f"clr={target.swing_clearance:.3f}, "
            f"f={target.gait_frequency:.3f}, "
            f"D={target.duty_factor:.3f}"
        )
        print("=" * 72)

        CURRENT_TARGET_LABEL = label

    applied = MANAGER.update(
        target=target,
        current_contact=(
            self.wb_interface.current_contact
        ),
        dt=simulation_dt,
    )

    if MANAGER.last_structural_commit:
        contact = np.asarray(
            self.wb_interface.current_contact,
            dtype=int,
        ).reshape(-1)

        event = {
            "time_s": float(t),
            "target_label": label,
            "planned_contact": [
                int(x) for x in contact
            ],
            "applied_clearance_m":
                float(applied.swing_clearance),
            "applied_frequency_hz":
                float(applied.gait_frequency),
            "applied_duty_factor":
                float(applied.duty_factor),
        }

        STRUCTURAL_EVENTS.append(event)

        print()
        print(
            "[ICRA27 structural commit] "
            f"t={t:.3f}s "
            f"contact={event['planned_contact']} "
            f"clr={applied.swing_clearance:.3f} "
            f"f={applied.gait_frequency:.3f} "
            f"D={applied.duty_factor:.3f}"
        )

    # base integration hook accepts any attribute-compatible
    # command object, including PyMPCMetaGaitReference.
    base.COMMAND = applied

    return base.tracer_compute_actions(
        self,
        *args,
        **kwargs,
    )


def mean_slice(name, begin, end):
    x = np.asarray(
        base.LOG[name][begin:end],
        dtype=float,
    )

    if len(x) == 0:
        return None

    return float(np.mean(x))


def estimate_frequency(
    z,
    expected_frequency,
):
    return base.estimate_frequency(
        z,
        DT,
        expected_frequency,
    )


def segment_summary(
    label,
    t0,
    t1,
    command,
):
    # Ignore first 1 s after target switch.
    begin = int((t0 + 1.0) / DT)
    end = int(t1 / DT)

    target = base.ADAPTER.project(command)

    contacts = np.asarray(
        base.LOG["contacts"][begin:end],
        dtype=float,
    )

    clearances = []
    frequencies = []

    for leg in base.LEGS:
        z = np.asarray(
            base.LOG["feet_z"][leg][begin:end],
            dtype=float,
        )

        clearance = float(
            np.percentile(z, 95)
            - np.percentile(z, 5)
        )

        clearances.append(clearance)

        frequency = estimate_frequency(
            z,
            target.gait_frequency,
        )

        if frequency is not None:
            frequencies.append(frequency)

    return {
        "label": label,

        "target": {
            "vx_mps": float(target.vx),
            "body_height_m":
                float(target.body_height),
            "swing_clearance_m":
                float(target.swing_clearance),
            "gait_frequency_hz":
                float(target.gait_frequency),
            "duty_factor":
                float(target.duty_factor),
        },

        "applied_internal_mean": {
            "vx_mps":
                mean_slice(
                    "ref_speed",
                    begin,
                    end,
                ),
            "body_height_m":
                mean_slice(
                    "ref_height",
                    begin,
                    end,
                ),
            "swing_clearance_m":
                mean_slice(
                    "step_height",
                    begin,
                    end,
                ),
            "gait_frequency_hz":
                mean_slice(
                    "step_freq",
                    begin,
                    end,
                ),
            "duty_factor":
                mean_slice(
                    "duty_factor",
                    begin,
                    end,
                ),
        },

        "measured": {
            "body_forward_vx_mps":
                mean_slice(
                    "body_forward_vx",
                    begin,
                    end,
                ),
            "base_z_m":
                mean_slice(
                    "base_z",
                    begin,
                    end,
                ),
            "foot_clearance_m":
                float(np.mean(clearances)),
            "gait_frequency_hz":
                (
                    float(np.mean(frequencies))
                    if frequencies
                    else None
                ),
            "planned_duty_factor":
                (
                    float(np.mean(contacts))
                    if len(contacts)
                    else None
                ),
        },
    }


def sample_internal(t):
    idx = int(round(t / DT))

    max_idx = len(base.LOG["ref_speed"]) - 1

    idx = max(
        0,
        min(idx, max_idx),
    )

    return {
        "time_s": float(t),
        "ref_vx_mps":
            float(
                base.LOG["ref_speed"][idx]
            ),
        "ref_body_height_m":
            float(
                base.LOG["ref_height"][idx]
            ),
        "step_height_m":
            float(
                base.LOG["step_height"][idx]
            ),
        "step_frequency_hz":
            float(
                base.LOG["step_freq"][idx]
            ),
        "duty_factor":
            float(
                base.LOG["duty_factor"][idx]
            ),
    }


def main():
    clear_logs()

    base.cfg.simulation_params["gait"] = "trot"
    base.cfg.mpc_params["optimize_step_freq"] = False

    base.sim.QuadrupedPyMPC_Wrapper.compute_actions = (
        guarded_compute_actions
    )

    print("=" * 72)
    print(
        "ICRA27 M2 guarded MetaGaitCommand transition"
    )
    print("=" * 72)

    print(
        "continuous slew rates:"
    )
    print(
        f"  vx          : "
        f"{MANAGER.rates.vx_rate_mps2:.3f} m/s^2"
    )
    print(
        f"  yaw         : "
        f"{MANAGER.rates.yaw_rate_rate_rad_s2:.3f} rad/s^2"
    )
    print(
        f"  body height : "
        f"{MANAGER.rates.body_height_rate_mps:.3f} m/s"
    )

    print()
    print(
        "structural policy:"
    )
    print(
        "  clearance / frequency / duty "
        "commit only in planned full stance"
    )
    print("=" * 72)

    base.sim.run_simulation(
        qpympc_cfg=base.cfg,
        num_episodes=1,
        num_seconds_per_episode=DURATION,
        ref_base_lin_vel=(
            0.20 / float(base.cfg.hip_height)
        ),
        ref_base_ang_vel=0.0,
        friction_coeff=0.8,
        base_vel_command_type="forward",
        seed=0,
        render=False,
        recording_path=None,
    )

    segments = [
        segment_summary(
            label,
            t0,
            t1,
            command,
        )
        for label, t0, t1, command
        in SCHEDULE
    ]

    # Samples around both target switches.
    sample_times = [
        3.998,
        4.000,
        4.100,
        4.500,
        7.998,
        8.000,
        8.100,
        8.500,
    ]

    samples = [
        sample_internal(t)
        for t in sample_times
    ]

    result = {
        "duration_s": DURATION,

        "transition_type":
            "continuous_slew_plus_full_stance_structural_commit",

        "rates": {
            "vx_rate_mps2":
                MANAGER.rates.vx_rate_mps2,
            "yaw_rate_rate_rad_s2":
                MANAGER.rates.yaw_rate_rate_rad_s2,
            "body_height_rate_mps":
                MANAGER.rates.body_height_rate_mps,
        },

        "structural_commit_events":
            STRUCTURAL_EVENTS,

        "transition_samples":
            samples,

        "segments":
            segments,
    }

    output = (
        ROOT
        / "results"
        / "icra27"
        / "m2_meta_gait"
        / "guarded_transition_v0.json"
    )

    output.write_text(
        json.dumps(
            result,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("GUARDED TRANSITION RESULT")
    print("=" * 72)

    print()
    print("structural commits:")
    for event in STRUCTURAL_EVENTS:
        print(
            f"  t={event['time_s']:.3f}s "
            f"target={event['target_label']} "
            f"contact={event['planned_contact']} "
            f"f={event['applied_frequency_hz']:.3f} "
            f"D={event['applied_duty_factor']:.3f}"
        )

    print()
    print("transition samples:")
    for sample in samples:
        print(
            f"  t={sample['time_s']:5.3f} "
            f"vx={sample['ref_vx_mps']:.4f} "
            f"h={sample['ref_body_height_m']:.4f} "
            f"clr={sample['step_height_m']:.4f} "
            f"f={sample['step_frequency_hz']:.3f} "
            f"D={sample['duty_factor']:.3f}"
        )

    print()
    print("steady segments:")

    for segment in segments:
        target = segment["target"]
        internal = (
            segment["applied_internal_mean"]
        )
        measured = segment["measured"]

        print()
        print(f"[{segment['label']}]")

        print(
            "  target  : "
            f"vx={target['vx_mps']:.3f}, "
            f"h={target['body_height_m']:.3f}, "
            f"clr={target['swing_clearance_m']:.3f}, "
            f"f={target['gait_frequency_hz']:.3f}, "
            f"D={target['duty_factor']:.3f}"
        )

        print(
            "  internal: "
            f"vx={internal['vx_mps']:.3f}, "
            f"h={internal['body_height_m']:.3f}, "
            f"clr={internal['swing_clearance_m']:.3f}, "
            f"f={internal['gait_frequency_hz']:.3f}, "
            f"D={internal['duty_factor']:.3f}"
        )

        print(
            "  measured: "
            f"vx={measured['body_forward_vx_mps']:.3f}, "
            f"z={measured['base_z_m']:.3f}, "
            f"clr={measured['foot_clearance_m']:.3f}, "
            f"f={measured['gait_frequency_hz']}, "
            f"D={measured['planned_duty_factor']:.3f}"
        )

    print()
    print("=" * 72)
    print(
        "[ICRA27] guarded runtime transition completed."
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
