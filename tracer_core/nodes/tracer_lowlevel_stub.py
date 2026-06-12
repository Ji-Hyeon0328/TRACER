#!/usr/bin/env python3
from __future__ import annotations

import math
import time
from typing import List, Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray


ROBOT_STATE_LEN = 42
HIGH_LEVEL_CMD_LEN = 32
LOW_LEVEL_CMD_LEN = 61

# IsaacLab A1 joint order:
# [FL_hip, FR_hip, RL_hip, RR_hip,
#  FL_thigh, FR_thigh, RL_thigh, RR_thigh,
#  FL_calf, FR_calf, RL_calf, RR_calf]
DEFAULT_STAND_Q = [
    0.1, -0.1, 0.1, -0.1,
    0.8, 0.8, 1.0, 1.0,
    -1.5, -1.5, -1.5, -1.5,
]

# /tracer/robot_state layout
IDX_JOINT_POS_START = 14
IDX_JOINT_POS_END = 26

# /tracer/high_level_cmd v0 layout
# Last field is mode_id.
HIGH_LEVEL_MODE_INDEX = 31

# /tracer/low_level_cmd layout
# data[0]      mode
# data[1:13]   joint_pos_target[12]
# data[13:25]  joint_vel_target[12]
# data[25:37]  kp[12]
# data[37:49]  kd[12]
# data[49:61]  tau_ff[12]


class TracerLowLevelStub(Node):
    """TRACER low-level ROS2 stub.

    v0 purpose:
      - verify /tracer/robot_state -> low-level node -> /tracer/low_level_cmd
      - publish safe joint-position hold command
      - do NOT implement MPC/WBC yet

    The node holds the latest joint position from /tracer/robot_state.
    If robot_state has not arrived yet, it publishes zeros for topic-loop testing only.
    """

    def __init__(self):
        super().__init__("tracer_lowlevel_stub")

        self.declare_parameter("publish_hz", 50.0)
        self.declare_parameter("kp", 35.0)
        self.declare_parameter("kd", 2.0)

        # Optional torque-path test.
        # torque_test_joint_index:
        #   -1 disables torque test
        #   0..11 applies torque feedforward to that joint.
        self.declare_parameter("torque_test_joint_index", -1)
        self.declare_parameter("torque_test_tau", 0.0)
        self.declare_parameter("torque_test_mode", 3.0)
        self.declare_parameter("mode", 1.0)

        # hold_policy:
        #   "default": always command DEFAULT_STAND_Q
        #   "initial": latch first received robot_state joint_pos
        #   "live": command latest robot_state joint_pos, old behavior
        self.declare_parameter("hold_policy", "default")

        # Optional small command test.
        # test_joint_index:
        #   -1 disables test offset
        #   0..11 applies constant offset to that joint target
        self.declare_parameter("test_joint_index", -1)
        self.declare_parameter("test_joint_offset", 0.0)

        # Optional sine test.
        self.declare_parameter("sine_joint_index", -1)
        self.declare_parameter("sine_amp", 0.0)
        self.declare_parameter("sine_period", 4.0)

        # If true, high_level_cmd.data[31] overrides the local test params.
        self.declare_parameter("use_high_level_mode", True)
        self.declare_parameter("high_level_timeout_s", 0.5)

        publish_hz = float(self.get_parameter("publish_hz").value)
        self.dt = 1.0 / max(publish_hz, 1.0)

        self.latest_robot_state: Optional[List[float]] = None
        self.latest_high_level_cmd: Optional[List[float]] = None
        self.latest_high_level_cmd_stamp: float = 0.0
        self.initial_joint_pos: Optional[List[float]] = None
        self.start_time = time.time()

        self.robot_state_sub = self.create_subscription(
            Float64MultiArray,
            "/tracer/robot_state",
            self.on_robot_state,
            10,
        )
        self.high_level_cmd_sub = self.create_subscription(
            Float64MultiArray,
            "/tracer/high_level_cmd",
            self.on_high_level_cmd,
            10,
        )
        self.low_level_pub = self.create_publisher(
            Float64MultiArray,
            "/tracer/low_level_cmd",
            10,
        )

        self.timer = self.create_timer(self.dt, self.on_timer)

        self.get_logger().info(
            "TracerLowLevelStub started. "
            "Subscribing: /tracer/robot_state, /tracer/high_level_cmd. "
            "Publishing: /tracer/low_level_cmd."
        )

    def on_robot_state(self, msg: Float64MultiArray):
        data = list(msg.data)
        if len(data) < ROBOT_STATE_LEN:
            self.get_logger().warn(
                f"Ignore short /tracer/robot_state: len={len(data)}, expected>={ROBOT_STATE_LEN}",
                throttle_duration_sec=1.0,
            )
            return
        self.latest_robot_state = data
        if self.initial_joint_pos is None:
            q0 = data[IDX_JOINT_POS_START:IDX_JOINT_POS_END]
            if len(q0) == 12 and all(math.isfinite(x) for x in q0):
                self.initial_joint_pos = q0
                self.get_logger().info(f"Latched initial joint position: {self.initial_joint_pos}")

    def on_high_level_cmd(self, msg: Float64MultiArray):
        data = list(msg.data)
        if len(data) < HIGH_LEVEL_CMD_LEN:
            self.get_logger().warn(
                f"Short /tracer/high_level_cmd: len={len(data)}, expected>={HIGH_LEVEL_CMD_LEN}. "
                "Keeping it anyway for debug.",
                throttle_duration_sec=1.0,
            )
        self.latest_high_level_cmd = data
        self.latest_high_level_cmd_stamp = time.time()

    def build_hold_command(self) -> Float64MultiArray:
        mode = float(self.get_parameter("mode").value)
        kp_val = float(self.get_parameter("kp").value)
        kd_val = float(self.get_parameter("kd").value)

        out = [0.0] * LOW_LEVEL_CMD_LEN
        out[0] = mode

        hold_policy = str(self.get_parameter("hold_policy").value)

        if hold_policy == "default":
            q_hold = list(DEFAULT_STAND_Q)
        elif hold_policy == "initial":
            if self.initial_joint_pos is not None:
                q_hold = list(self.initial_joint_pos)
            else:
                q_hold = list(DEFAULT_STAND_Q)
        elif hold_policy == "live":
            if self.latest_robot_state is not None:
                q_hold = self.latest_robot_state[IDX_JOINT_POS_START:IDX_JOINT_POS_END]
            else:
                q_hold = list(DEFAULT_STAND_Q)
        else:
            self.get_logger().warn(
                f"Unknown hold_policy={hold_policy}. Falling back to default.",
                throttle_duration_sec=1.0,
            )
            q_hold = list(DEFAULT_STAND_Q)

        use_high_level_mode = bool(self.get_parameter("use_high_level_mode").value)

        high_level_timeout_s = float(self.get_parameter("high_level_timeout_s").value)
        high_level_is_fresh = (
            self.latest_high_level_cmd is not None
            and len(self.latest_high_level_cmd) > HIGH_LEVEL_MODE_INDEX
            and (time.time() - self.latest_high_level_cmd_stamp) <= high_level_timeout_s
        )

        if use_high_level_mode and high_level_is_fresh:
            mode_id = int(round(float(self.latest_high_level_cmd[HIGH_LEVEL_MODE_INDEX])))

            if mode_id == 0:
                # Default stand.
                pass
            elif mode_id == 1:
                # Constant command test: FL hip +0.05 rad.
                q_hold[0] += 0.05
            elif mode_id == 2:
                # Sine command test: FL hip sinusoid.
                phase = 2.0 * math.pi * (time.time() - self.start_time) / 4.0
                q_hold[0] += 0.05 * math.sin(phase)
            else:
                self.get_logger().warn(
                    f"Unknown high_level mode_id={mode_id}. Holding default stand.",
                    throttle_duration_sec=1.0,
                )
        else:
            # Apply constant test offset if requested.
            test_joint_index = int(self.get_parameter("test_joint_index").value)
            test_joint_offset = float(self.get_parameter("test_joint_offset").value)
            if 0 <= test_joint_index < 12:
                q_hold[test_joint_index] += test_joint_offset

            # Apply sine test if requested.
            sine_joint_index = int(self.get_parameter("sine_joint_index").value)
            sine_amp = float(self.get_parameter("sine_amp").value)
            sine_period = float(self.get_parameter("sine_period").value)
            if 0 <= sine_joint_index < 12 and abs(sine_amp) > 0.0 and sine_period > 1.0e-6:
                phase = 2.0 * math.pi * (time.time() - self.start_time) / sine_period
                q_hold[sine_joint_index] += sine_amp * math.sin(phase)

        if len(q_hold) != 12 or any((not math.isfinite(x)) for x in q_hold):
            self.get_logger().warn("Invalid q_hold. Falling back to DEFAULT_STAND_Q.", throttle_duration_sec=1.0)
            q_hold = list(DEFAULT_STAND_Q)

        out[1:13] = q_hold
        out[13:25] = [0.0] * 12
        out[25:37] = [kp_val] * 12
        out[37:49] = [kd_val] * 12
        out[49:61] = [0.0] * 12

        # Apply optional torque-path test.
        torque_test_joint_index = int(self.get_parameter("torque_test_joint_index").value)
        torque_test_tau = float(self.get_parameter("torque_test_tau").value)
        torque_test_mode = float(self.get_parameter("torque_test_mode").value)
        if 0 <= torque_test_joint_index < 12 and abs(torque_test_tau) > 0.0:
            out[0] = torque_test_mode
            out[49 + torque_test_joint_index] = torque_test_tau

        msg = Float64MultiArray()
        msg.data = out
        return msg

    def on_timer(self):
        self.low_level_pub.publish(self.build_hold_command())


def main():
    rclpy.init()
    node = TracerLowLevelStub()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
