#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts" / "icra27"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPTS))

import pympc_meta_gait_integration_v0 as base

from tracer_core.highlevel.meta_gait import MetaGaitCommand


DT = float(base.cfg.simulation_params["dt"])
SEGMENT_DURATION = 4.0
DURATION = 12.0


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
            source_mode="icra27_transition",
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
            source_mode="icra27_transition",
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
            source_mode="icra27_transition",
            source_reason="fast",
        ),
    ),
]


CURRENT_LABEL = None


def command_for_time(t: float):
    for label, t0, t1, command in SCHEDULE:
        if t0 <= t < t1:
            return label, command

    return SCHEDULE[-1][0], SCHEDULE[-1][3]


def scheduled_compute_actions(self, *args, **kwargs):
    global CURRENT_LABEL

    if "step_num" in kwargs:
        step_num = int(kwargs["step_num"])
    else:
        if len(args) <= 13:
            raise RuntimeError(
                "Could not locate step_num in compute_actions arguments"
            )
        step_num = int(args[13])

    t = step_num * DT

    label, command = command_for_time(t)

    if label != CURRENT_LABEL:
        projected = base.ADAPTER.project(command)

        print()
        print("=" * 72)
        print(
            f"[ICRA27 transition] t={t:.3f}s -> {label}"
        )
        print(
            f"vx={projected.vx:.3f}, "
            f"h={projected.body_height:.3f}, "
            f"clr={projected.swing_clearance:.3f}, "
            f"f={projected.gait_frequency:.3f}, "
            f"D={projected.duty_factor:.3f}"
        )
        print("=" * 72)

        CURRENT_LABEL = label

    base.COMMAND = command

    return base.tracer_compute_actions(
        self,
        *args,
        **kwargs,
    )


def estimate_frequency(z, expected_frequency):
    z = np.asarray(z, dtype=float)

    if len(z) < 10:
        return None

    amplitude = (
        np.percentile(z, 95)
        - np.percentile(z, 5)
    )

    prominence = max(
        0.005,
        0.20 * amplitude,
    )

    min_distance = max(
        1,
        int(
            0.45
            / (expected_frequency * DT)
        ),
    )

    peaks, _ = find_peaks(
        z,
        distance=min_distance,
        prominence=prominence,
    )

    if len(peaks) < 2:
        return None

    periods = np.diff(peaks) * DT

    return float(
        1.0 / np.mean(periods)
    )


def clear_logs():
    for key, value in base.LOG.items():
        if key == "feet_z":
            for leg in base.LEGS:
                value[leg].clear()
        else:
            value.clear()

    base.ADAPTER.reset()
    base.COMMAND = None


def segment_summary(
    label,
    t0,
    t1,
    command,
):
    # Ignore first 1 s after each abrupt transition.
    begin = int((t0 + 1.0) / DT)
    end = int(t1 / DT)

    projected = base.ADAPTER.project(command)

    def mean_log(name):
        x = np.asarray(
            base.LOG[name][begin:end],
            dtype=float,
        )

        if len(x) == 0:
            return None

        return float(np.mean(x))

    contacts = np.asarray(
        base.LOG["contacts"][begin:end],
        dtype=float,
    )

    per_leg_clearance = {}
    per_leg_frequency = {}

    for leg in base.LEGS:
        z = np.asarray(
            base.LOG["feet_z"][leg][begin:end],
            dtype=float,
        )

        per_leg_clearance[leg] = float(
            np.percentile(z, 95)
            - np.percentile(z, 5)
        )

        per_leg_frequency[leg] = (
            estimate_frequency(
                z,
                projected.gait_frequency,
            )
        )

    valid_frequency = [
        x
        for x in per_leg_frequency.values()
        if x is not None
    ]

    measured_duty = (
        float(np.mean(contacts))
        if len(contacts)
        else None
    )

    return {
        "label": label,

        "command": {
            "vx_mps": command.vx,
            "body_height_m": command.body_height,
            "swing_clearance_m":
                command.swing_clearance,
            "gait_period_s":
                command.gait_period,
            "gait_frequency_hz":
                projected.gait_frequency,
            "duty_factor":
                command.duty_factor,
        },

        "internal": {
            "ref_speed_mps":
                mean_log("ref_speed"),
            "ref_height_m":
                mean_log("ref_height"),
            "step_height_m":
                mean_log("step_height"),
            "step_frequency_hz":
                mean_log("step_freq"),
            "duty_factor":
                mean_log("duty_factor"),
            "stance_time_s":
                mean_log("stance_time"),
            "swing_period_s":
                mean_log("swing_period"),
        },

        "measured": {
            "body_forward_vx_mps":
                mean_log("body_forward_vx"),
            "base_z_m":
                mean_log("base_z"),
            "foot_clearance_m":
                float(
                    np.mean(
                        list(
                            per_leg_clearance.values()
                        )
                    )
                ),
            "gait_frequency_hz":
                (
                    float(np.mean(valid_frequency))
                    if valid_frequency
                    else None
                ),
            "planned_duty_factor":
                measured_duty,
        },

        "per_leg": {
            leg: {
                "clearance_m":
                    per_leg_clearance[leg],
                "frequency_hz":
                    per_leg_frequency[leg],
            }
            for leg in base.LEGS
        },
    }


def main():
    clear_logs()

    base.cfg.simulation_params["gait"] = "trot"
    base.cfg.mpc_params["optimize_step_freq"] = False

    base.sim.QuadrupedPyMPC_Wrapper.compute_actions = (
        scheduled_compute_actions
    )

    print("=" * 72)
    print("ICRA27 M2 runtime MetaGaitCommand transition")
    print("=" * 72)

    for label, t0, t1, command in SCHEDULE:
        ref = base.ADAPTER.project(command)

        print(
            f"{label:8s}: "
            f"{t0:4.1f}-{t1:4.1f}s | "
            f"vx={ref.vx:.2f} "
            f"h={ref.body_height:.2f} "
            f"clr={ref.swing_clearance:.2f} "
            f"f={ref.gait_frequency:.2f} "
            f"D={ref.duty_factor:.2f}"
        )

    print("=" * 72)

    # Upstream generator is only providing a heading direction here.
    # The TRACER adapter overwrites its magnitude every control step.
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

    summaries = [
        segment_summary(
            label,
            t0,
            t1,
            command,
        )
        for label, t0, t1, command in SCHEDULE
    ]

    result = {
        "duration_s": DURATION,
        "transition_type": "abrupt_no_rate_limit",
        "segments": summaries,
    }

    output = (
        ROOT
        / "results"
        / "icra27"
        / "m2_meta_gait"
        / "dynamic_transition_v0.json"
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
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
    print("TRANSITION RESULT")
    print("=" * 72)

    for segment in summaries:
        c = segment["command"]
        i = segment["internal"]
        m = segment["measured"]

        print()
        print(f"[{segment['label']}]")
        print(
            "  command : "
            f"vx={c['vx_mps']:.3f}, "
            f"h={c['body_height_m']:.3f}, "
            f"clr={c['swing_clearance_m']:.3f}, "
            f"f={c['gait_frequency_hz']:.3f}, "
            f"D={c['duty_factor']:.3f}"
        )
        print(
            "  internal: "
            f"vx={i['ref_speed_mps']:.3f}, "
            f"h={i['ref_height_m']:.3f}, "
            f"clr={i['step_height_m']:.3f}, "
            f"f={i['step_frequency_hz']:.3f}, "
            f"D={i['duty_factor']:.3f}"
        )
        print(
            "  measured: "
            f"vx={m['body_forward_vx_mps']:.3f}, "
            f"z={m['base_z_m']:.3f}, "
            f"clr={m['foot_clearance_m']:.3f}, "
            f"f={m['gait_frequency_hz']}, "
            f"D={m['planned_duty_factor']:.3f}"
        )

    print()
    print("=" * 72)
    print(
        "[ICRA27] abrupt runtime transition completed."
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
