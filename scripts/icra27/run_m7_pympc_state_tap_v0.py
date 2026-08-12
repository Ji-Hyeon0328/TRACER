#!/usr/bin/env python3

from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ICRA27 = ROOT / "scripts" / "icra27"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ICRA27))


import run_pympc_lowlevel_udp as m5

from tracer_core.highlevel_rl.state_protocol import (
    SCHEMA,
    encode_state_payload,
)

from tracer_core.highlevel_rl.mechanical_energy import (
    MechanicalEnergyAccumulator,
    applied_generalized_power,
    commanded_joint_power,
)


standalone = m5.standalone


ORIGINAL_STANDALONE_COMPUTE_ACTIONS = (
    standalone.standalone_compute_actions
)

ORIGINAL_STANDALONE_WRAPPER_RESET = (
    standalone.standalone_wrapper_reset
)

ORIGINAL_M5_UDP_ENV_STEP = (
    m5.udp_env_step
)


LATEST_STATE: dict | None = None
EPISODE_INDEX = -1
STATE_SENDER = None


ENERGY_ACCUMULATOR = (
    MechanicalEnergyAccumulator()
)

LATEST_ENERGY_SAMPLE: dict | None = None

ENERGY_LOG_PATH = (
    os.environ.get(
        "TRACER_M7_ENERGY_LOG",
        "",
    ).strip()
)


def _append_energy_log(
    row: dict,
) -> None:
    """
    Optional logging-only mechanical-energy side channel.

    This does not alter M5 telemetry, M7 observation,
    task reward, or controller behavior.
    """

    if not ENERGY_LOG_PATH:
        return

    path = Path(
        ENERGY_LOG_PATH
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "a",
        encoding="utf-8",
    ) as handle:
        handle.write(
            json.dumps(
                row,
                separators=(",", ":"),
            )
            + "\n"
        )


class M7StateUdpSender:
    def __init__(
        self,
        *,
        host: str,
        port: int,
        state_hz: float,
    ) -> None:
        self.addr = (
            str(host),
            int(port),
        )

        self.period_s = (
            1.0
            / max(float(state_hz), 1e-6)
        )

        self.sock = socket.socket(
            socket.AF_INET,
            socket.SOCK_DGRAM,
        )

        self.seq = 0
        self.last_send_time = None

    def ready(self) -> bool:
        now = time.monotonic()

        if self.last_send_time is None:
            return True

        return (
            now - self.last_send_time
            >= self.period_s
        )

    def send(
        self,
        payload: dict,
    ) -> None:
        payload = dict(payload)
        payload["seq"] = self.seq

        raw = encode_state_payload(
            payload
        )

        self.sock.sendto(
            raw,
            self.addr,
        )

        self.seq += 1
        self.last_send_time = (
            time.monotonic()
        )

    def close(self) -> None:
        self.sock.close()


def _arg(
    args,
    kwargs,
    *,
    name: str,
    index: int,
):
    if name in kwargs:
        return kwargs[name]

    if len(args) <= index:
        raise RuntimeError(
            f"Missing compute_actions argument "
            f"{name!r} at index {index}"
        )

    return args[index]


def m7_compute_actions(
    self,
    *args,
    **kwargs,
):
    """
    Read-only wrapper around the already validated
    standalone M4/M5 compute-actions hook.

    The captured physical state is the controller-input
    state immediately before env.step().
    """

    global LATEST_STATE

    base_pos = _arg(
        args,
        kwargs,
        name="base_pos",
        index=1,
    )

    base_lin_vel = _arg(
        args,
        kwargs,
        name="base_lin_vel",
        index=2,
    )

    base_ori = _arg(
        args,
        kwargs,
        name="base_ori_euler_xyz",
        index=3,
    )

    base_ang_vel = _arg(
        args,
        kwargs,
        name="base_ang_vel",
        index=4,
    )

    simulation_dt = _arg(
        args,
        kwargs,
        name="simulation_dt",
        index=10,
    )

    step_num = _arg(
        args,
        kwargs,
        name="step_num",
        index=13,
    )

    pos = np.asarray(
        base_pos,
        dtype=float,
    ).reshape(-1)

    vel_world = np.asarray(
        base_lin_vel,
        dtype=float,
    ).reshape(-1)

    rpy = np.asarray(
        base_ori,
        dtype=float,
    ).reshape(-1)

    ang_base = np.asarray(
        base_ang_vel,
        dtype=float,
    ).reshape(-1)

    if (
        len(pos) < 3
        or len(vel_world) < 3
        or len(rpy) < 3
        or len(ang_base) < 3
    ):
        raise RuntimeError(
            "Unexpected base-state vector size"
        )

    yaw = float(rpy[2])

    c = float(np.cos(yaw))
    s = float(np.sin(yaw))

    vx_world = float(
        vel_world[0]
    )

    vy_world = float(
        vel_world[1]
    )

    # Same yaw-projection convention already used
    # by the validated M3 integration instrumentation.
    vx_body = (
        c * vx_world
        + s * vy_world
    )

    vy_body = (
        -s * vx_world
        + c * vy_world
    )

    dt = float(simulation_dt)

    LATEST_STATE = {
        "schema": SCHEMA,
        "episode_index":
            int(EPISODE_INDEX),

        "sample_time_s":
            float(step_num) * dt,

        "lowlevel_dt_s":
            dt,

        "sample_phase":
            "controller_input_pre_env_step",

        "base_position_world": [
            float(pos[0]),
            float(pos[1]),
            float(pos[2]),
        ],

        "base_rpy": [
            float(rpy[0]),
            float(rpy[1]),
            float(rpy[2]),
        ],

        "base_linear_velocity_world": [
            float(vel_world[0]),
            float(vel_world[1]),
            float(vel_world[2]),
        ],

        "base_linear_velocity_body_yaw": [
            float(vx_body),
            float(vy_body),
            float(vel_world[2]),
        ],

        "base_angular_velocity_base": [
            float(ang_base[0]),
            float(ang_base[1]),
            float(ang_base[2]),
        ],
    }

    # Execute the exact frozen controller first.
    #
    # WBInterface.update_state_and_reference() updates the
    # existing PyMPC TerrainEstimator from this same
    # controller-input state.
    result = ORIGINAL_STANDALONE_COMPUTE_ACTIONS(
        self,
        *args,
        **kwargs,
    )

    # --------------------------------------------------------
    # Read-only reward/diagnostic telemetry.
    #
    # Do NOT feed these scalars into the frozen 21-D M7
    # observation. They expose estimates already computed
    # internally by the frozen PyMPC controller.
    # --------------------------------------------------------
    terrain_computation = (
        self.wb_interface.terrain_computation
    )

    terrain_height = float(
        terrain_computation.terrain_height
    )

    robot_height = float(
        terrain_computation.robot_height
    )

    if not np.isfinite(terrain_height):
        raise RuntimeError(
            "Non-finite PyMPC terrain-height estimate: "
            f"{terrain_height!r}"
        )

    if not np.isfinite(robot_height):
        raise RuntimeError(
            "Non-finite PyMPC robot-height estimate: "
            f"{robot_height!r}"
        )

    if LATEST_STATE is None:
        raise RuntimeError(
            "PyMPC estimator updated before "
            "LATEST_STATE was captured"
        )

    LATEST_STATE[
        "pympc_terrain_height_estimate_world"
    ] = terrain_height

    LATEST_STATE[
        "pympc_robot_height_estimate"
    ] = robot_height

    LATEST_STATE[
        "pympc_height_estimate_phase"
    ] = "post_controller_compute_pre_env_step"

    return result


def _native_reward(
    result,
):
    try:
        return float(result[1])
    except Exception:
        return None


def m7_udp_env_step(
    self,
    action,
):
    """
    Preserve the exact frozen M5 env-step semantics,
    while adding a read-only mechanical-energy tap.

    Timing:
      1. snapshot qvel/time before mj_step
      2. compute commanded torque power
      3. execute frozen M5/MuJoCo step
      4. read qfrc_actuator generated during that step
      5. pair it with the pre-step qvel snapshot
    """

    global ENERGY_ACCUMULATOR
    global LATEST_ENERGY_SAMPLE

    qvel_pre = np.asarray(
        self.mjData.qvel,
        dtype=float,
    ).copy()

    base_position_pre = np.asarray(
        self.mjData.qpos[:3],
        dtype=float,
    ).copy()

    time_pre = float(
        self.mjData.time
    )

    commanded_power = (
        commanded_joint_power(
            self,
            action,
        )
    )

    # Exact frozen M5/MuJoCo behavior.
    result = ORIGINAL_M5_UDP_ENV_STEP(
        self,
        action,
    )

    time_post = float(
        self.mjData.time
    )

    base_position_post = np.asarray(
        self.mjData.qpos[:3],
        dtype=float,
    ).copy()

    robot_mass_kg = float(
        np.sum(
            np.asarray(
                self.mjModel.body_mass,
                dtype=float,
            )
        )
    )

    if (
        not np.isfinite(robot_mass_kg)
        or robot_mass_kg <= 0.0
    ):
        raise RuntimeError(
            "Invalid MuJoCo robot mass: "
            f"{robot_mass_kg!r}"
        )

    dt = (
        time_post
        - time_pre
    )

    if not np.isfinite(dt) or dt <= 0.0:
        raise RuntimeError(
            "Unexpected MuJoCo step dt: "
            f"{dt!r}"
        )

    applied_power = (
        applied_generalized_power(
            self,
            qvel_snapshot=qvel_pre,
        )
    )

    ENERGY_ACCUMULATOR.add(
        dt=dt,
        commanded=commanded_power,
        applied=applied_power,
    )

    LATEST_ENERGY_SAMPLE = {
        "episode_index":
            int(EPISODE_INDEX),

        "time_pre_s":
            float(time_pre),

        "time_post_s":
            float(time_post),

        "dt_s":
            float(dt),

        "robot_mass_kg":
            float(robot_mass_kg),

        "base_position_pre_world":
            [
                float(x)
                for x in base_position_pre
            ],

        "base_position_post_world":
            [
                float(x)
                for x in base_position_post
            ],

        "commanded_power_w":
            dict(commanded_power),

        "applied_power_w":
            dict(applied_power),

        "cumulative_energy":
            ENERGY_ACCUMULATOR.as_dict(),
    }

    # Keep energy as a separate local diagnostic channel.
    # Do not modify the validated M7 state protocol.
    _append_energy_log(
        LATEST_ENERGY_SAMPLE
    )

    if (
        STATE_SENDER is None
        or LATEST_STATE is None
        or not STATE_SENDER.ready()
    ):
        return result

    payload = dict(
        LATEST_STATE
    )

    payload["terminated"] = bool(
        result[2]
    )

    payload["truncated"] = bool(
        result[3]
    )

    payload["native_reward"] = (
        _native_reward(result)
    )

    if LATEST_ENERGY_SAMPLE is None:
        raise RuntimeError(
            "State packet is ready but "
            "mechanical-energy sample is missing"
        )

    payload["energy_sample_time_s"] = float(
        LATEST_ENERGY_SAMPLE[
            "time_post_s"
        ]
    )

    payload["mechanical_energy"] = dict(
        LATEST_ENERGY_SAMPLE[
            "cumulative_energy"
        ]
    )

    STATE_SENDER.send(
        payload
    )

    return result


def m7_wrapper_reset(
    self,
    initial_feet_pos,
):
    global LATEST_STATE
    global EPISODE_INDEX
    global ENERGY_ACCUMULATOR
    global LATEST_ENERGY_SAMPLE

    result = (
        ORIGINAL_STANDALONE_WRAPPER_RESET(
            self,
            initial_feet_pos,
        )
    )

    EPISODE_INDEX += 1
    LATEST_STATE = None

    ENERGY_ACCUMULATOR = (
        MechanicalEnergyAccumulator()
    )

    LATEST_ENERGY_SAMPLE = None

    print(
        "[M7 state tap] reset "
        f"episode={EPISODE_INDEX}"
    )

    return result


def main():
    global STATE_SENDER

    host = os.environ.get(
        "TRACER_M7_STATE_HOST",
        "127.0.0.1",
    )

    port = int(
        os.environ.get(
            "TRACER_M7_STATE_PORT",
            "50512",
        )
    )

    state_hz = float(
        os.environ.get(
            "TRACER_M7_STATE_HZ",
            "20.0",
        )
    )

    STATE_SENDER = M7StateUdpSender(
        host=host,
        port=port,
        state_hz=state_hz,
    )

    # M5 main() installs these functions into the
    # simulator. Replace only module-level hook targets;
    # no frozen source file is changed.
    standalone.standalone_compute_actions = (
        m7_compute_actions
    )

    standalone.standalone_wrapper_reset = (
        m7_wrapper_reset
    )

    m5.udp_env_step = (
        m7_udp_env_step
    )

    print("=" * 72)
    print(
        "ICRA27 M7 READ-ONLY PYMPC STATE TAP"
    )
    print("=" * 72)
    print(
        f"state UDP       : {host}:{port}"
    )
    print(
        f"state publish Hz: {state_hz:.3f}"
    )
    print(
        "sample phase    : "
        "controller_input_pre_env_step"
    )
    print(
        "frozen M5 source: unchanged"
    )
    print("=" * 72)

    try:
        m5.main()

    finally:
        standalone.standalone_compute_actions = (
            ORIGINAL_STANDALONE_COMPUTE_ACTIONS
        )

        standalone.standalone_wrapper_reset = (
            ORIGINAL_STANDALONE_WRAPPER_RESET
        )

        m5.udp_env_step = (
            ORIGINAL_M5_UDP_ENV_STEP
        )

        if STATE_SENDER is not None:
            STATE_SENDER.close()


if __name__ == "__main__":
    main()
