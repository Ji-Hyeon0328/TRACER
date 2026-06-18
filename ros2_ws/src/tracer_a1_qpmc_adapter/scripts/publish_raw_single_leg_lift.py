#!/usr/bin/env python3

import argparse
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray


JOINT_ORDER = [
    "FL_hip", "FR_hip", "RL_hip", "RR_hip",
    "FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh",
    "FL_calf", "FR_calf", "RL_calf", "RR_calf",
]

LEG_INDEX = {
    "FL": 0,
    "FR": 1,
    "RL": 2,
    "RR": 3,
}


DEFAULT_Q = [
    0.1, -0.1, 0.1, -0.1,
    0.8, 0.8, 1.0, 1.0,
    -1.5, -1.5, -1.5, -1.5,
]


def build_cmd(q, kp=35.0, kd=3.0):
    data = []
    data.append(1.0)                 # mode = q target only
    data.extend(q)                   # q_target, 12
    data.extend([0.0] * 12)          # qd_target
    data.extend([kp] * 12)           # kp
    data.extend([kd] * 12)           # kd
    data.extend([0.0] * 12)          # tau_ff
    assert len(data) == 61
    msg = Float64MultiArray()
    msg.data = data
    return msg


class RawSingleLegLift(Node):
    def __init__(self, args):
        super().__init__("raw_single_leg_lift_publisher")
        self.args = args
        self.pub = self.create_publisher(Float64MultiArray, "/tracer/low_level_cmd", 10)
        self.start_time = time.time()

        self.leg = LEG_INDEX[args.leg]
        self.thigh_i = 4 + self.leg
        self.calf_i = 8 + self.leg
        self.hip_i = self.leg

        self.timer = self.create_timer(1.0 / args.rate, self.on_timer)

        self.get_logger().info(
            f"Raw single-leg lift: leg={args.leg}, "
            f"hip_delta={args.hip_delta}, thigh_delta={args.thigh_delta}, calf_delta={args.calf_delta}, "
            f"hold_s={args.hold_s}, lift_s={args.lift_s}"
        )

    def on_timer(self):
        t = time.time() - self.start_time
        q = list(DEFAULT_Q)

        if self.args.hold_s <= t < self.args.hold_s + self.args.lift_s:
            q[self.hip_i] += self.args.hip_delta
            q[self.thigh_i] += self.args.thigh_delta
            q[self.calf_i] += self.args.calf_delta

        # After lift window, return to default.
        self.pub.publish(build_cmd(q, self.args.kp, self.args.kd))

        if int(t * 10) % 10 == 0:
            self.get_logger().info(
                f"t={t:.2f} q_{self.args.leg}="
                f"[hip={q[self.hip_i]:.3f}, thigh={q[self.thigh_i]:.3f}, calf={q[self.calf_i]:.3f}]"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--leg", type=str, default="FR", choices=["FL", "FR", "RL", "RR"])
    parser.add_argument("--hip_delta", type=float, default=0.0)
    parser.add_argument("--thigh_delta", type=float, default=-0.25)
    parser.add_argument("--calf_delta", type=float, default=0.50)
    parser.add_argument("--kp", type=float, default=35.0)
    parser.add_argument("--kd", type=float, default=3.0)
    parser.add_argument("--rate", type=float, default=50.0)
    parser.add_argument("--hold_s", type=float, default=1.0)
    parser.add_argument("--lift_s", type=float, default=3.0)
    args = parser.parse_args()

    rclpy.init()
    node = RawSingleLegLift(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
