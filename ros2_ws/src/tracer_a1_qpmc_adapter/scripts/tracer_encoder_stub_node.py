#!/usr/bin/env python3

import math
from typing import List

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


class TracerEncoderStubNode(Node):
    """
    TRACER encoder stub.

    Input:
      /tracer/proprio_vector, length 45

    Proprio layout:
      [0] stamp_wall

      [1:4]   base position x,y,z
      [4:7]   roll,pitch,yaw
      [7:10]  base linear velocity vx,vy,vz
      [10:13] base angular velocity wx,wy,wz

      [13:25] joint_pos[12]
      [25:37] joint_vel[12]

      [37:41] contact_binary[4]     FL,FR,RL,RR
      [41:45] contact_force_z[4]    FL,FR,RL,RR

    Outputs:
      /tracer/context_vector
      /tracer/latent_vector

    This is not the final learned encoder.
    It defines the ROS2 interface for future proprio/RGB/depth/elevation encoders.
    """

    def __init__(self):
        super().__init__("tracer_encoder_stub_node")

        self.declare_parameter("proprio_topic", "/tracer/proprio_vector")
        self.declare_parameter("context_topic", "/tracer/context_vector")
        self.declare_parameter("latent_topic", "/tracer/latent_vector")
        self.declare_parameter("proprio_timeout_sec", 1.0)
        self.declare_parameter("publish_hz", 20.0)

        self.proprio_topic = self.get_parameter("proprio_topic").value
        self.context_topic = self.get_parameter("context_topic").value
        self.latent_topic = self.get_parameter("latent_topic").value
        self.proprio_timeout_sec = float(self.get_parameter("proprio_timeout_sec").value)
        self.publish_hz = float(self.get_parameter("publish_hz").value)

        self.last_proprio = None
        self.last_proprio_time = None

        self.pub_context = self.create_publisher(Float64MultiArray, self.context_topic, 10)
        self.pub_latent = self.create_publisher(Float64MultiArray, self.latent_topic, 10)

        self.create_subscription(
            Float64MultiArray,
            self.proprio_topic,
            self.proprio_callback,
            10,
        )

        self.timer = self.create_timer(
            1.0 / max(1.0, self.publish_hz),
            self.timer_callback,
        )

        self.prev_vx = None
        self.prev_vy = None
        self.prev_time = None

        self.get_logger().info(
            f"TRACER encoder stub: {self.proprio_topic} -> {self.context_topic}, {self.latent_topic}"
        )

    def proprio_callback(self, msg: Float64MultiArray):
        data = list(msg.data)
        if len(data) < 45:
            self.get_logger().warn(f"proprio_vector expects 45 values, got {len(data)}")
            return

        self.last_proprio = data
        self.last_proprio_time = self.get_clock().now()

    def compute_features(self, p: List[float]):
        stamp_wall = p[0]

        base_z = p[3]
        roll = p[4]
        pitch = p[5]
        yaw = p[6]

        vx = p[7]
        vy = p[8]
        vz = p[9]

        wx = p[10]
        wy = p[11]
        wz = p[12]

        joint_pos = p[13:25]
        joint_vel = p[25:37]

        contacts = p[37:41]
        force_z = p[41:45]

        speed_xy = math.sqrt(vx * vx + vy * vy)
        angular_speed = math.sqrt(wx * wx + wy * wy + wz * wz)

        contact_ratio = sum(contacts) / 4.0
        mean_abs_joint_vel = sum(abs(x) for x in joint_vel) / max(1, len(joint_vel))
        mean_abs_force_z = sum(abs(x) for x in force_z) / 4.0

        # Simple stability proxies.
        attitude_norm = math.sqrt(roll * roll + pitch * pitch)
        roll_pitch_risk = clamp(attitude_norm / 0.6, 0.0, 1.0)

        low_contact_risk = clamp((2.0 - sum(contacts)) / 2.0, 0.0, 1.0)

        # Slip proxy:
        # High body speed while few contacts or alternating contacts can indicate instability.
        slip_proxy = clamp(speed_xy * (1.0 - 0.5 * contact_ratio) / 0.5, 0.0, 1.0)

        # Future sensor availability placeholders.
        vision_available = 0.0
        depth_available = 0.0
        elevation_available = 0.0

        # Context vector c_t.
        # This should later be produced by proprio + visual + elevation encoders.
        context = [
            base_z,                    # 0 body height
            roll,                      # 1 roll
            pitch,                     # 2 pitch
            yaw,                       # 3 yaw
            vx,                        # 4 body/world vx proxy
            vy,                        # 5 body/world vy proxy
            speed_xy,                  # 6 planar speed
            wz,                        # 7 yaw rate
            contact_ratio,             # 8 contact ratio
            mean_abs_joint_vel,        # 9 joint activity
            mean_abs_force_z,          # 10 contact load proxy
            attitude_norm,             # 11 attitude deviation
            vision_available,          # 12 RGB available placeholder
            depth_available,           # 13 depth available placeholder
            elevation_available,       # 14 elevation/lidar map available placeholder
            1.0,                       # 15 proprio available
        ]

        # Latent vector z_t / rho-sigma style placeholder.
        # This should later be learned from history mismatch + terrain observation.
        latent = [
            roll_pitch_risk,           # 0 stability risk proxy
            low_contact_risk,          # 1 insufficient contact proxy
            slip_proxy,                # 2 slip/mismatch proxy
            clamp(abs(vz) / 0.5, 0.0, 1.0),       # 3 vertical motion risk
            clamp(abs(wz) / 1.0, 0.0, 1.0),       # 4 yaw activity
            clamp(mean_abs_joint_vel / 5.0, 0.0, 1.0),  # 5 actuation activity
            0.0,                       # 6 terrain slope latent placeholder
            0.0,                       # 7 terrain roughness latent placeholder
            0.0,                       # 8 deformability/mismatch placeholder
            0.0,                       # 9 epistemic uncertainty placeholder
            1.0 if contact_ratio >= 0.5 else 0.0, # 10 gait feasibility hint
            0.0,                       # 11 reserved
            0.0,                       # 12 reserved
            0.0,                       # 13 reserved
            0.0,                       # 14 reserved
            0.0,                       # 15 reserved
        ]

        return context, latent

    def publish_zero(self, reason: str):
        context = [0.0] * 16
        latent = [0.0] * 16

        context_msg = Float64MultiArray()
        latent_msg = Float64MultiArray()

        context_msg.data = context
        latent_msg.data = latent

        self.pub_context.publish(context_msg)
        self.pub_latent.publish(latent_msg)

        self.get_logger().info(f"publish zero context/latent: {reason}", throttle_duration_sec=1.0)

    def timer_callback(self):
        if self.last_proprio is None or self.last_proprio_time is None:
            self.publish_zero("waiting for proprio")
            return

        age = (self.get_clock().now() - self.last_proprio_time).nanoseconds * 1e-9
        if age > self.proprio_timeout_sec:
            self.publish_zero(f"proprio timeout age={age:.2f}s")
            return

        context, latent = self.compute_features(self.last_proprio)

        context_msg = Float64MultiArray()
        latent_msg = Float64MultiArray()

        context_msg.data = context
        latent_msg.data = latent

        self.pub_context.publish(context_msg)
        self.pub_latent.publish(latent_msg)

        self.get_logger().info(
            f"context h={context[0]:.2f} roll={context[1]:.2f} pitch={context[2]:.2f} "
            f"speed={context[6]:.2f} contact={context[8]:.2f} | "
            f"latent risk={latent[0]:.2f} low_contact={latent[1]:.2f} slip={latent[2]:.2f}",
            throttle_duration_sec=1.0,
        )


def main():
    rclpy.init()
    node = TracerEncoderStubNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
