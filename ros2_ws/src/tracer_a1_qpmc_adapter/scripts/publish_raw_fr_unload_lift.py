#!/usr/bin/env python3

import argparse
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray


DEFAULT_Q = [
    0.1, -0.1, 0.1, -0.1,
    0.8, 0.8, 1.0, 1.0,
    -1.5, -1.5, -1.5, -1.5,
]


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def smoothstep(a):
    a = clamp(a, 0.0, 1.0)
    return a * a * (3.0 - 2.0 * a)


def build_cmd(q, kp=30.0, kd=4.0):
    data = [1.0]
    data.extend(q)
    data.extend([0.0] * 12)
    data.extend([kp] * 12)
    data.extend([kd] * 12)
    data.extend([0.0] * 12)
    msg = Float64MultiArray()
    msg.data = data
    return msg


class FRUnloadLift(Node):
    def __init__(self, args):
        super().__init__("raw_fr_unload_lift_publisher")
        self.args = args
        self.pub = self.create_publisher(Float64MultiArray, "/tracer/low_level_cmd", 10)
        self.t0 = time.time()
        self.timer = self.create_timer(1.0 / args.rate, self.tick)
        self.get_logger().info("FR unload-lift ramp publisher started")

    def tick(self):
        t = time.time() - self.t0
        q = list(DEFAULT_Q)

        # phase 1: settle default
        # phase 2: stance support: slightly extend the other 3 legs
        # phase 3: ramp FR lift
        settle_s = self.args.settle_s
        support_s = self.args.support_s
        lift_s = self.args.lift_s

        support_alpha = smoothstep((t - settle_s) / support_s)
        lift_alpha = smoothstep((t - settle_s - support_s) / lift_s)

        # FR indices
        FR_thigh = 5
        FR_calf = 9

        # stance legs: FL, RL, RR
        stance_thigh = [4, 6, 7]
        stance_calf = [8, 10, 11]

        # Slightly extend/support stance legs.
        for i in stance_thigh:
            q[i] += support_alpha * self.args.stance_thigh_delta
        for i in stance_calf:
            q[i] += support_alpha * self.args.stance_calf_delta

        # Lift FR gradually.
        q[FR_thigh] += lift_alpha * self.args.fr_thigh_delta
        q[FR_calf] += lift_alpha * self.args.fr_calf_delta

        self.pub.publish(build_cmd(q, self.args.kp, self.args.kd))

        if int(t * 5) % 5 == 0:
            self.get_logger().info(
                f"t={t:.2f} support={support_alpha:.2f} lift={lift_alpha:.2f} "
                f"FR=[{q[1]:.3f}, {q[5]:.3f}, {q[9]:.3f}]"
            )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kp", type=float, default=30.0)
    p.add_argument("--kd", type=float, default=4.0)
    p.add_argument("--rate", type=float, default=50.0)

    p.add_argument("--settle_s", type=float, default=1.0)
    p.add_argument("--support_s", type=float, default=1.5)
    p.add_argument("--lift_s", type=float, default=2.0)

    # FR lift, B sign
    p.add_argument("--fr_thigh_delta", type=float, default=-0.35)
    p.add_argument("--fr_calf_delta", type=float, default=0.70)

    # stance support: small extension. Keep conservative.
    p.add_argument("--stance_thigh_delta", type=float, default=0.05)
    p.add_argument("--stance_calf_delta", type=float, default=-0.08)

    args = p.parse_args()

    rclpy.init()
    node = FRUnloadLift(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
