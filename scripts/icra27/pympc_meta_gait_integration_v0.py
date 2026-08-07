#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks


ROOT = Path(__file__).resolve().parents[2]
PYMPC_ROOT = ROOT / "external_baselines" / "Quadruped-PyMPC"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PYMPC_ROOT))

from tracer_core.highlevel.meta_gait import MetaGaitCommand
from tracer_core.lowlevel.pympc_meta_gait_adapter import (
    PyMPCMetaGaitAdapter,
)

from quadruped_pympc import config as cfg
import simulation.simulation as sim


LEGS = ("FL", "FR", "RL", "RR")

ADAPTER = PyMPCMetaGaitAdapter()

COMMAND: MetaGaitCommand | None = None

LOG = {
    "base_z": [],
    "yaw_rate": [],
    "body_forward_vx": [],
    "yaw": [],
    "ref_speed": [],
    "ref_yaw_rate": [],
    "ref_height": [],
    "step_height": [],
    "step_freq": [],
    "duty_factor": [],
    "stance_time": [],
    "swing_period": [],
    "contacts": [],
    "feet_z": {leg: [] for leg in LEGS},
}

OriginalComputeActions = (
    sim.QuadrupedPyMPC_Wrapper.compute_actions
)


def yaw_component(value) -> float:
    arr = np.asarray(value, dtype=float).reshape(-1)

    if arr.size >= 3:
        return float(arr[2])

    if arr.size:
        return float(arr[-1])

    raise RuntimeError("Empty angular-velocity value")


def tracer_compute_actions(
    self,
    com_pos,
    base_pos,
    base_lin_vel,
    base_ori_euler_xyz,
    base_ang_vel,
    feet_pos,
    hip_pos,
    joints_pos,
    heightmaps,
    legs_order,
    simulation_dt,
    ref_base_lin_vel,
    ref_base_ang_vel,
    step_num,
    qpos,
    qvel,
    feet_jac,
    feet_jac_dot,
    feet_vel,
    legs_qfrc_passive,
    legs_qfrc_bias,
    legs_mass_matrix,
    legs_qpos_idx,
    legs_qvel_idx,
    tau,
    inertia,
    mujoco_contact,
):
    if COMMAND is None:
        raise RuntimeError("MetaGaitCommand has not been initialized")

    # ------------------------------------------------------------
    # TRACER MetaGaitCommand -> physical safety projection
    # -> PyMPC runtime gait references.
    # ------------------------------------------------------------
    ref = ADAPTER.apply_runtime(
        self.wb_interface,
        cfg,
        COMMAND,
    )

    # run_simulation's forward+rotate command generator already
    # provides the linear reference in the appropriate world-frame
    # heading. Normalize its XY magnitude to the projected TRACER vx.
    ref_lin = np.asarray(
        ref_base_lin_vel,
        dtype=float,
    ).copy().reshape(-1)

    if ref_lin.size < 3:
        padded = np.zeros(3, dtype=float)
        padded[:ref_lin.size] = ref_lin
        ref_lin = padded

    xy_norm = float(np.linalg.norm(ref_lin[:2]))

    if xy_norm > 1e-9:
        ref_lin[:2] *= ref.vx / xy_norm
    else:
        yaw = float(
            np.asarray(
                base_ori_euler_xyz,
                dtype=float,
            ).reshape(-1)[2]
        )
        ref_lin[0] = ref.vx * np.cos(yaw)
        ref_lin[1] = ref.vx * np.sin(yaw)

    ref_lin[2] = 0.0

    ref_ang = np.zeros(3, dtype=float)
    ref_ang[2] = ref.yaw_rate

    tau_out = OriginalComputeActions(
        self,
        com_pos,
        base_pos,
        base_lin_vel,
        base_ori_euler_xyz,
        base_ang_vel,
        feet_pos,
        hip_pos,
        joints_pos,
        heightmaps,
        legs_order,
        simulation_dt,
        ref_lin,
        ref_ang,
        step_num,
        qpos,
        qvel,
        feet_jac,
        feet_jac_dot,
        feet_vel,
        legs_qfrc_passive,
        legs_qfrc_bias,
        legs_mass_matrix,
        legs_qpos_idx,
        legs_qvel_idx,
        tau,
        inertia,
        mujoco_contact,
    )

    # ------------------------------------------------------------
    # Runtime internal-reference logging
    # ------------------------------------------------------------
    wb = self.wb_interface

    LOG["ref_speed"].append(
        float(np.linalg.norm(ref_lin[:2]))
    )
    LOG["ref_yaw_rate"].append(float(ref_ang[2]))
    LOG["ref_height"].append(
        float(cfg.simulation_params["ref_z"])
    )
    LOG["step_height"].append(
        float(wb.step_height)
    )
    LOG["step_freq"].append(
        float(wb.pgg.step_freq)
    )
    LOG["duty_factor"].append(
        float(wb.pgg.duty_factor)
    )
    LOG["stance_time"].append(
        float(wb.frg.stance_time)
    )
    LOG["swing_period"].append(
        float(wb.stc.swing_period)
    )

    LOG["contacts"].append(
        np.asarray(
            wb.current_contact,
            dtype=float,
        ).copy()
    )

    # ------------------------------------------------------------
    # Physical-response logging
    # ------------------------------------------------------------
    base_pos_arr = np.asarray(
        base_pos,
        dtype=float,
    ).reshape(-1)

    base_vel_arr = np.asarray(
        base_lin_vel,
        dtype=float,
    ).reshape(-1)

    base_ori_arr = np.asarray(
        base_ori_euler_xyz,
        dtype=float,
    ).reshape(-1)

    yaw = float(base_ori_arr[2])

    # World velocity projected onto robot heading.
    forward_vx = (
        np.cos(yaw) * float(base_vel_arr[0])
        + np.sin(yaw) * float(base_vel_arr[1])
    )

    LOG["base_z"].append(float(base_pos_arr[2]))
    LOG["yaw_rate"].append(
        yaw_component(base_ang_vel)
    )
    LOG["body_forward_vx"].append(
        float(forward_vx)
    )
    LOG["yaw"].append(yaw)

    for leg in LEGS:
        p = np.asarray(
            feet_pos[leg],
            dtype=float,
        ).reshape(-1)

        LOG["feet_z"][leg].append(
            float(p[2])
        )

    return tau_out


def estimate_frequency(z, dt, expected_frequency):
    z = np.asarray(z, dtype=float)

    warmup = int(2.0 / dt)
    z = z[warmup:]

    min_distance = max(
        1,
        int(
            0.45
            / (expected_frequency * dt)
        ),
    )

    amplitude = (
        np.percentile(z, 95)
        - np.percentile(z, 5)
    )

    prominence = max(
        0.005,
        0.20 * amplitude,
    )

    peaks, _ = find_peaks(
        z,
        distance=min_distance,
        prominence=prominence,
    )

    if len(peaks) < 2:
        return None

    periods = np.diff(peaks) * dt
    return float(1.0 / np.mean(periods))


def summarize(
    command: MetaGaitCommand,
    projected,
    output: Path,
):
    dt = float(cfg.simulation_params["dt"])
    warmup = int(2.0 / dt)

    def steady(name):
        return np.asarray(
            LOG[name],
            dtype=float,
        )[warmup:]

    base_z = steady("base_z")
    yaw_rate = steady("yaw_rate")
    body_vx = steady("body_forward_vx")

    ref_speed = steady("ref_speed")
    ref_yaw = steady("ref_yaw_rate")
    ref_height = steady("ref_height")
    step_height = steady("step_height")
    step_freq = steady("step_freq")
    duty = steady("duty_factor")
    stance_time = steady("stance_time")
    swing_period = steady("swing_period")

    contacts = np.asarray(
        LOG["contacts"],
        dtype=float,
    )[warmup:]

    yaw = np.unwrap(
        np.asarray(
            LOG["yaw"],
            dtype=float,
        )
    )

    per_leg_clearance = {}
    per_leg_frequency = {}

    for leg in LEGS:
        z = np.asarray(
            LOG["feet_z"][leg],
            dtype=float,
        )[warmup:]

        stance_z = float(np.percentile(z, 5))
        apex_z = float(np.percentile(z, 95))

        per_leg_clearance[leg] = float(
            apex_z - stance_z
        )

        per_leg_frequency[leg] = (
            estimate_frequency(
                LOG["feet_z"][leg],
                dt,
                projected.gait_frequency,
            )
        )

    valid_freq = [
        f
        for f in per_leg_frequency.values()
        if f is not None
    ]

    measured_duty = (
        np.mean(contacts, axis=0)
        if len(contacts)
        else np.full(4, np.nan)
    )

    result = {
        "command": {
            "vx_mps": float(command.vx),
            "yaw_rate_rad_s": float(command.yaw_rate),
            "body_height_m": float(command.body_height),
            "swing_clearance_m":
                float(command.swing_clearance),
            "gait_period_s": float(command.gait_period),
            "duty_factor": float(command.duty_factor),
        },

        "projected": {
            "vx_mps": float(projected.vx),
            "yaw_rate_rad_s": float(projected.yaw_rate),
            "body_height_m":
                float(projected.body_height),
            "swing_clearance_m":
                float(projected.swing_clearance),
            "gait_period_s":
                float(projected.gait_period),
            "gait_frequency_hz":
                float(projected.gait_frequency),
            "duty_factor":
                float(projected.duty_factor),
            "stance_time_s":
                float(projected.stance_time),
            "swing_period_s":
                float(projected.swing_period),
        },

        "internal_mean": {
            "ref_forward_speed_mps":
                float(np.mean(ref_speed)),
            "ref_yaw_rate_rad_s":
                float(np.mean(ref_yaw)),
            "ref_body_height_m":
                float(np.mean(ref_height)),
            "step_height_m":
                float(np.mean(step_height)),
            "step_frequency_hz":
                float(np.mean(step_freq)),
            "duty_factor":
                float(np.mean(duty)),
            "stance_time_s":
                float(np.mean(stance_time)),
            "swing_period_s":
                float(np.mean(swing_period)),
        },

        "measured": {
            "mean_body_forward_vx_mps":
                float(np.mean(body_vx)),
            "mean_yaw_rate_rad_s":
                float(np.mean(yaw_rate)),
            "mean_base_z_m":
                float(np.mean(base_z)),
            "std_base_z_m":
                float(np.std(base_z)),
            "mean_foot_clearance_m":
                float(
                    np.mean(
                        list(
                            per_leg_clearance.values()
                        )
                    )
                ),
            "mean_gait_frequency_hz":
                (
                    float(np.mean(valid_freq))
                    if valid_freq
                    else None
                ),
            "mean_planned_duty_factor":
                float(np.mean(measured_duty)),
            "yaw_change_rad":
                (
                    float(yaw[-1] - yaw[0])
                    if len(yaw) >= 2
                    else None
                ),
        },

        "per_leg": {
            leg: {
                "clearance_m":
                    per_leg_clearance[leg],
                "frequency_hz":
                    per_leg_frequency[leg],
                "planned_duty_factor":
                    float(measured_duty[i]),
            }
            for i, leg in enumerate(LEGS)
        },
    }

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )

    return result


def main():
    global COMMAND

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
    )

    parser.add_argument(
        "--no-render",
        action="store_true",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/icra27/m2_meta_gait/"
            "fixed_meta_gait_v0.json"
        ),
    )

    args = parser.parse_args()

    COMMAND = MetaGaitCommand(
        vx=0.20,
        yaw_rate=0.20,
        body_height=0.30,
        swing_clearance=0.06,
        gait_period=1.0 / 1.4,
        duty_factor=0.65,
        source_mode="icra27_fixed_integration",
        source_reason="M2 fixed MetaGaitCommand smoke",
    )

    projected = ADAPTER.project(COMMAND)

    # Fixed baseline settings not controlled by M2 command.
    cfg.simulation_params["gait"] = "trot"
    cfg.mpc_params["optimize_step_freq"] = False

    # Patch only this process.
    sim.QuadrupedPyMPC_Wrapper.compute_actions = (
        tracer_compute_actions
    )

    print("=" * 72)
    print("ICRA27 M2 MetaGaitCommand -> PyMPC integration")
    print("=" * 72)
    print("COMMAND")
    print(f"  vx              = {COMMAND.vx:.6f}")
    print(f"  yaw_rate        = {COMMAND.yaw_rate:.6f}")
    print(f"  body_height     = {COMMAND.body_height:.6f}")
    print(
        f"  swing_clearance = "
        f"{COMMAND.swing_clearance:.6f}"
    )
    print(
        f"  gait_period     = "
        f"{COMMAND.gait_period:.6f}"
    )
    print(
        f"  duty_factor     = "
        f"{COMMAND.duty_factor:.6f}"
    )

    print()
    print("PROJECTED")
    print(f"  vx              = {projected.vx:.6f}")
    print(
        f"  yaw_rate        = "
        f"{projected.yaw_rate:.6f}"
    )
    print(
        f"  body_height     = "
        f"{projected.body_height:.6f}"
    )
    print(
        f"  swing_clearance = "
        f"{projected.swing_clearance:.6f}"
    )
    print(
        f"  gait_frequency  = "
        f"{projected.gait_frequency:.6f}"
    )
    print(
        f"  duty_factor     = "
        f"{projected.duty_factor:.6f}"
    )
    print("=" * 72)

    # run_simulation scales fixed forward input by hip_height.
    sim.run_simulation(
        qpympc_cfg=cfg,
        num_episodes=1,
        num_seconds_per_episode=args.duration,
        ref_base_lin_vel=(
            projected.vx / float(cfg.hip_height)
        ),
        ref_base_ang_vel=projected.yaw_rate,
        friction_coeff=0.8,
        base_vel_command_type="forward+rotate",
        seed=0,
        render=not args.no_render,
        recording_path=None,
    )

    result = summarize(
        COMMAND,
        projected,
        args.output,
    )

    print()
    print("=" * 72)
    print("M2 INTEGRATION RESULT")
    print("=" * 72)

    print("internal:")
    for key, value in result["internal_mean"].items():
        print(f"  {key:32s}: {value}")

    print()
    print("measured:")
    for key, value in result["measured"].items():
        print(f"  {key:32s}: {value}")

    print("=" * 72)
    print(
        "[ICRA27] fixed MetaGaitCommand integration completed."
    )


if __name__ == "__main__":
    main()
