#!/usr/bin/env python3
from pathlib import Path
import math
import os
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray


def finite(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


class DirectSignedYawSelector(Node):
    """
    Live /tracer/robot_odom_flat layout:
      [x_rel, y_rel, z_rel, yaw_rel, vx_world, vy_world]

    The topic intentionally has no timestamp. Command phases therefore use
    odom-fresh active wall time. Final acceptance uses the ROS1 logger's Gazebo
    simulation timestamps and explicit duration gates.
    """

    def __init__(self):
        super().__init__("tracer_direct_signed_yaw_selector_v0")

        self.ref_topic = os.environ.get(
            "TRACER_REF_TOPIC",
            "/tracer/mpc_reference",
        )
        self.odom_topic = os.environ.get(
            "TRACER_ODOM_TOPIC",
            "/tracer/robot_odom_flat",
        )
        self.arm_file = Path(
            os.environ.get(
                "TRACER_YAW_ARM_FILE",
                "/tmp/tracer_direct_signed_yaw_arm",
            )
        )
        self.done_file = Path(
            os.environ.get(
                "TRACER_YAW_DONE_FILE",
                "/tmp/tracer_direct_signed_yaw_done",
            )
        )

        self.vx = 0.090
        self.body_h = 0.320
        self.clearance = 0.045

        # Targets approximately 4/6/4/6/4 Gazebo seconds at the recently
        # observed real-time factor. The analyzer checks actual sim durations.
        self.z0_wall_s = 16.0
        self.p_wall_s = 24.0
        self.z1_wall_s = 16.0
        self.n_wall_s = 24.0
        self.z2_wall_s = 16.0

        self.t_z0_end = self.z0_wall_s
        self.t_p_end = self.t_z0_end + self.p_wall_s
        self.t_z1_end = self.t_p_end + self.z1_wall_s
        self.t_n_end = self.t_z1_end + self.n_wall_s
        self.t_z2_end = self.t_n_end + self.z2_wall_s

        self.counter = 0
        self.latest_odom = None
        self.last_odom_wall = None
        self.last_bad_odom_log_wall = 0.0

        self.active_elapsed_wall = 0.0
        self.last_timer_wall = time.monotonic()
        self.sequence_armed = False
        self.done_written = False
        self.last_status_log_wall = 0.0

        self.pub = self.create_publisher(
            Float64MultiArray,
            self.ref_topic,
            10,
        )
        self.sub_odom = self.create_subscription(
            Float64MultiArray,
            self.odom_topic,
            self.on_odom,
            10,
        )
        self.timer = self.create_timer(
            0.05,
            self.on_timer,
        )

        self.get_logger().info(
            "Direct signed yaw selector started: "
            f"ref={self.ref_topic}, odom={self.odom_topic}, "
            "odom_layout=[x_rel,y_rel,z_rel,yaw_rel,vx_world,vy_world], "
            f"active_wall_schedule="
            f"[{self.z0_wall_s:.1f},{self.p_wall_s:.1f},"
            f"{self.z1_wall_s:.1f},{self.n_wall_s:.1f},"
            f"{self.z2_wall_s:.1f}]"
        )

    def on_odom(self, msg):
        data = list(msg.data)
        now = time.monotonic()

        if len(data) != 6:
            if now - self.last_bad_odom_log_wall >= 1.0:
                self.last_bad_odom_log_wall = now
                self.get_logger().error(
                    "robot_odom_flat layout mismatch: "
                    f"expected 6 values, got {len(data)}"
                )
            return

        values = [finite(value) for value in data]

        if any(value is None for value in values):
            if now - self.last_bad_odom_log_wall >= 1.0:
                self.last_bad_odom_log_wall = now
                self.get_logger().error(
                    "robot_odom_flat contains non-finite values"
                )
            return

        self.latest_odom = tuple(values)
        self.last_odom_wall = now

    def odom_is_fresh(self, now):
        return (
            self.latest_odom is not None
            and self.last_odom_wall is not None
            and now - self.last_odom_wall <= 0.50
        )

    def phase_for_elapsed(self, elapsed):
        if elapsed < self.t_z0_end:
            return "Z0", 0.0
        if elapsed < self.t_p_end:
            return "P", +0.050
        if elapsed < self.t_z1_end:
            return "Z1", 0.0
        if elapsed < self.t_n_end:
            return "N", -0.050
        if elapsed < self.t_z2_end:
            return "Z2", 0.0
        return "DONE", 0.0

    def schedule(self, now, dt):
        if not self.arm_file.exists():
            self.sequence_armed = False
            self.active_elapsed_wall = 0.0
            return "WAIT_ARM", 0.0, self.odom_is_fresh(now)

        if not self.odom_is_fresh(now):
            return "WAIT_ODOM", 0.0, False

        if not self.sequence_armed:
            self.sequence_armed = True
            self.active_elapsed_wall = 0.0
            self.get_logger().info(
                "Yaw sequence armed with fresh six-value odom."
            )

        self.active_elapsed_wall += dt
        phase, yaw_rate = self.phase_for_elapsed(
            self.active_elapsed_wall
        )

        if phase == "DONE" and not self.done_written:
            self.done_file.write_text(
                "direct_signed_yaw_sequence_complete\n",
                encoding="utf-8",
            )
            self.done_written = True
            self.get_logger().info(
                "Direct signed yaw sequence complete."
            )

        return phase, yaw_rate, True

    def on_timer(self):
        now = time.monotonic()
        dt = max(
            0.0,
            min(
                0.20,
                now - self.last_timer_wall,
            ),
        )
        self.last_timer_wall = now

        phase, yaw_rate, odom_fresh = self.schedule(
            now,
            dt,
        )

        msg = Float64MultiArray()
        msg.data = [
            float(self.counter),
            float(self.vx),
            float(yaw_rate),
            float(self.body_h),
            float(self.clearance),
            1.0,
        ]
        self.pub.publish(msg)
        self.counter += 1

        if now - self.last_status_log_wall >= 1.0:
            self.last_status_log_wall = now

            if self.latest_odom is None:
                odom_text = "odom=NA"
            else:
                x, y, z, yaw, vx, vy = self.latest_odom
                odom_age = (
                    float("nan")
                    if self.last_odom_wall is None
                    else now - self.last_odom_wall
                )
                odom_text = (
                    f"odom=[x={x:+.3f},y={y:+.3f},z={z:+.3f},"
                    f"yaw={yaw:+.3f},vx={vx:+.3f},vy={vy:+.3f},"
                    f"age_wall={odom_age:.3f}]"
                )

            self.get_logger().info(
                f"phase={phase} "
                f"elapsed_active_wall={self.active_elapsed_wall:.3f} "
                f"odom_fresh={int(odom_fresh)} "
                f"{odom_text} "
                f"cmd=[vx={self.vx:.3f},yaw={yaw_rate:+.3f},"
                f"h={self.body_h:.3f},clr={self.clearance:.3f}]"
            )


def main():
    rclpy.init()
    node = DirectSignedYawSelector()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
