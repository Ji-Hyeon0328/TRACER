#!/usr/bin/env python3
from __future__ import annotations

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray


class TracerHighLevelCmdPublisher(Node):
    """Simple high-level command publisher for TRACER low-level bridge tests.

    /tracer/high_level_cmd layout v0:
      0: cmd_vx
      1: cmd_vy
      2: cmd_yaw_rate
      3: body_height
      4: beta_v
      5: beta_s
      6: beta_e
      7:23 rho[16]
      23: sigma
      24: gait_period
      25: duty_factor
      26: clearance
      27: step_length
      28: controller_weight_motion
      29: controller_weight_stability
      30: controller_weight_energy
      31: mode_id
    """

    def __init__(self):
        super().__init__("tracer_highlevel_cmd_pub")

        self.declare_parameter("publish_hz", 10.0)

        self.declare_parameter("cmd_vx", 0.0)
        self.declare_parameter("cmd_vy", 0.0)
        self.declare_parameter("cmd_yaw_rate", 0.0)
        self.declare_parameter("body_height", 0.32)

        self.declare_parameter("beta_v", 0.33)
        self.declare_parameter("beta_s", 0.33)
        self.declare_parameter("beta_e", 0.34)

        self.declare_parameter("sigma", 0.0)
        self.declare_parameter("gait_period", 0.6)
        self.declare_parameter("duty_factor", 0.75)
        self.declare_parameter("clearance", 0.08)
        self.declare_parameter("step_length", 0.10)

        self.declare_parameter("w_motion", 1.0)
        self.declare_parameter("w_stability", 1.0)
        self.declare_parameter("w_energy", 1.0)

        # 0: default stand
        # 1: FL hip +0.05
        # 2: FL hip sine
        self.declare_parameter("mode_id", 0.0)

        self.pub = self.create_publisher(Float64MultiArray, "/tracer/high_level_cmd", 10)

        hz = float(self.get_parameter("publish_hz").value)
        self.timer = self.create_timer(1.0 / max(hz, 1.0), self.on_timer)

        self.get_logger().info("TracerHighLevelCmdPublisher started: /tracer/high_level_cmd")

    def p(self, name: str) -> float:
        return float(self.get_parameter(name).value)

    def build_msg(self) -> Float64MultiArray:
        rho = [0.0] * 16

        data = [
            self.p("cmd_vx"),
            self.p("cmd_vy"),
            self.p("cmd_yaw_rate"),
            self.p("body_height"),
            self.p("beta_v"),
            self.p("beta_s"),
            self.p("beta_e"),
            *rho,
            self.p("sigma"),
            self.p("gait_period"),
            self.p("duty_factor"),
            self.p("clearance"),
            self.p("step_length"),
            self.p("w_motion"),
            self.p("w_stability"),
            self.p("w_energy"),
            self.p("mode_id"),
        ]

        msg = Float64MultiArray()
        msg.data = data
        return msg

    def on_timer(self):
        self.pub.publish(self.build_msg())


def main():
    rclpy.init()
    node = TracerHighLevelCmdPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
