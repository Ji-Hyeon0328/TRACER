#!/usr/bin/env python3

import math
from typing import List, Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def normalize_beta(beta):
    beta = [max(0.0, float(x)) for x in beta[:3]]
    s = sum(beta)
    if s <= 1e-9:
        return [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
    return [x / s for x in beta]


class TracerObjectiveConditionedHighLevelControllerStubNode(Node):
    """
    Objective-conditioned high-level controller stub.

    Inputs:
      /tracer/relative_goal:
        [x_rel, y_rel, yaw_rel_optional, enable_optional]

      /tracer/context_vector:
        16-dim context from encoder

      /tracer/latent_vector:
        16-dim latent/risk from encoder

      /tracer/objective_weights:
        [beta_motion, beta_stability, beta_energy]

    Output:
      /tracer/mpc_reference:
        [counter, vx, yaw_rate, body_height, clearance, enable]

    This node is a label generator for objective-conditioned learned policy v1.
    """

    def __init__(self):
        super().__init__("tracer_objective_conditioned_high_level_controller_stub_node")

        self.declare_parameter("relative_goal_topic", "/tracer/relative_goal")
        self.declare_parameter("context_topic", "/tracer/context_vector")
        self.declare_parameter("latent_topic", "/tracer/latent_vector")
        self.declare_parameter("objective_topic", "/tracer/objective_weights")
        self.declare_parameter("mpc_ref_topic", "/tracer/mpc_reference")

        self.declare_parameter("publish_hz", 20.0)

        self.declare_parameter("input_timeout_sec", 1.0)

        self.declare_parameter("kx", 0.35)
        self.declare_parameter("kyaw", 1.2)

        self.declare_parameter("vx_max_nominal", 0.22)
        self.declare_parameter("yaw_rate_max_nominal", 0.45)

        self.declare_parameter("goal_stop_distance", 0.18)
        self.declare_parameter("yaw_slowdown_angle", 0.9)

        self.declare_parameter("body_height_nominal", 0.30)
        self.declare_parameter("clearance_nominal", 0.03)

        self.declare_parameter("risk_stop_threshold", 0.90)
        self.declare_parameter("risk_safe_threshold", 0.35)

        self.relative_goal_topic = self.get_parameter("relative_goal_topic").value
        self.context_topic = self.get_parameter("context_topic").value
        self.latent_topic = self.get_parameter("latent_topic").value
        self.objective_topic = self.get_parameter("objective_topic").value
        self.mpc_ref_topic = self.get_parameter("mpc_ref_topic").value

        self.publish_hz = float(self.get_parameter("publish_hz").value)
        self.input_timeout_sec = float(self.get_parameter("input_timeout_sec").value)

        self.kx = float(self.get_parameter("kx").value)
        self.kyaw = float(self.get_parameter("kyaw").value)

        self.vx_max_nominal = float(self.get_parameter("vx_max_nominal").value)
        self.yaw_rate_max_nominal = float(self.get_parameter("yaw_rate_max_nominal").value)

        self.goal_stop_distance = float(self.get_parameter("goal_stop_distance").value)
        self.yaw_slowdown_angle = float(self.get_parameter("yaw_slowdown_angle").value)

        self.body_height_nominal = float(self.get_parameter("body_height_nominal").value)
        self.clearance_nominal = float(self.get_parameter("clearance_nominal").value)

        self.risk_stop_threshold = float(self.get_parameter("risk_stop_threshold").value)
        self.risk_safe_threshold = float(self.get_parameter("risk_safe_threshold").value)

        self.counter = 0.0

        self.last_goal: Optional[List[float]] = None
        self.last_goal_time = None

        self.last_context: Optional[List[float]] = None
        self.last_context_time = None

        self.last_latent: Optional[List[float]] = None
        self.last_latent_time = None

        self.last_beta: Optional[List[float]] = None
        self.last_beta_time = None

        self.pub = self.create_publisher(Float64MultiArray, self.mpc_ref_topic, 10)

        self.create_subscription(Float64MultiArray, self.relative_goal_topic, self.goal_callback, 10)
        self.create_subscription(Float64MultiArray, self.context_topic, self.context_callback, 10)
        self.create_subscription(Float64MultiArray, self.latent_topic, self.latent_callback, 10)
        self.create_subscription(Float64MultiArray, self.objective_topic, self.objective_callback, 10)

        self.timer = self.create_timer(
            1.0 / max(1.0, self.publish_hz),
            self.timer_callback,
        )

        self.get_logger().info(
            f"objective-conditioned HL stub: "
            f"{self.relative_goal_topic} + {self.context_topic} + {self.latent_topic} + {self.objective_topic} "
            f"-> {self.mpc_ref_topic}"
        )

    def goal_callback(self, msg):
        data = list(msg.data)
        if len(data) < 2:
            self.get_logger().warn(f"relative_goal expects [x_rel,y_rel,...], got {len(data)}")
            return

        out = [0.0, 0.0, 0.0, 1.0]
        for i in range(min(4, len(data))):
            out[i] = float(data[i])

        self.last_goal = out
        self.last_goal_time = self.get_clock().now()

    def context_callback(self, msg):
        data = list(msg.data)
        if len(data) < 16:
            self.get_logger().warn(f"context_vector expects >=16 values, got {len(data)}")
            return

        self.last_context = [float(x) for x in data[:16]]
        self.last_context_time = self.get_clock().now()

    def latent_callback(self, msg):
        data = list(msg.data)
        if len(data) < 16:
            self.get_logger().warn(f"latent_vector expects >=16 values, got {len(data)}")
            return

        self.last_latent = [float(x) for x in data[:16]]
        self.last_latent_time = self.get_clock().now()

    def objective_callback(self, msg):
        data = list(msg.data)
        if len(data) < 3:
            self.get_logger().warn(f"objective_weights expects 3 values, got {len(data)}")
            return

        self.last_beta = normalize_beta(data[:3])
        self.last_beta_time = self.get_clock().now()

    def age_sec(self, stamp):
        if stamp is None:
            return 1e9
        return (self.get_clock().now() - stamp).nanoseconds * 1e-9

    def compute_risk(self, context: List[float], latent: List[float]) -> float:
        contact_ratio = context[8]
        attitude_norm = context[11]

        attitude_risk = clamp(attitude_norm / 0.6, 0.0, 1.0)
        low_contact_risk_from_context = clamp((0.5 - contact_ratio) / 0.5, 0.0, 1.0)

        risk_terms = [
            latent[0],
            latent[1],
            latent[2],
            latent[3],
            latent[9],
            attitude_risk,
            low_contact_risk_from_context,
        ]

        return clamp(max(risk_terms), 0.0, 1.0)

    def objective_modulation(self, beta, risk):
        beta_m, beta_s, beta_e = beta

        # Baseline around balanced beta = 1/3 each.
        dm = beta_m - 1.0 / 3.0
        ds = beta_s - 1.0 / 3.0
        de = beta_e - 1.0 / 3.0

        # Risk alpha: independent safety modulation.
        safe_alpha = clamp(
            (risk - self.risk_safe_threshold)
            / max(1e-6, self.risk_stop_threshold - self.risk_safe_threshold),
            0.0,
            1.0,
        )

        # Objective modulation:
        # motion ↑    -> faster vx, slightly faster yaw
        # stability ↑ -> slower vx/yaw, higher body, higher clearance
        # energy ↑    -> lower clearance/body, smoother/slower motion
        vx_scale = 1.0 + 0.75 * dm - 0.55 * ds - 0.25 * de
        yaw_scale = 1.0 + 0.35 * dm - 0.35 * ds - 0.20 * de

        vx_scale = clamp(vx_scale, 0.35, 1.35)
        yaw_scale = clamp(yaw_scale, 0.40, 1.25)

        # Safety risk still slows everything down.
        vx_scale *= (1.0 - 0.55 * safe_alpha)
        yaw_scale *= (1.0 - 0.40 * safe_alpha)

        vx_max = clamp(self.vx_max_nominal * vx_scale, 0.05, 0.30)
        yaw_rate_max = clamp(self.yaw_rate_max_nominal * yaw_scale, 0.10, 0.60)

        body_height = (
            self.body_height_nominal
            + 0.035 * ds
            - 0.015 * de
            + 0.020 * safe_alpha
        )
        body_height = clamp(body_height, 0.27, 0.35)

        clearance = (
            self.clearance_nominal
            + 0.045 * ds
            - 0.012 * de
            + 0.025 * safe_alpha
        )
        clearance = clamp(clearance, 0.015, 0.09)

        return vx_max, yaw_rate_max, body_height, clearance, safe_alpha

    def publish_ref(self, vx, yaw_rate, body_height, clearance, enable):
        msg = Float64MultiArray()
        msg.data = [
            self.counter,
            float(vx),
            float(yaw_rate),
            float(body_height),
            float(clearance),
            float(enable),
        ]
        self.pub.publish(msg)

    def publish_stop(self, reason):
        self.counter += 1.0
        self.publish_ref(0.0, 0.0, self.body_height_nominal, self.clearance_nominal, 0.0)
        self.get_logger().info(f"publish stop: {reason}", throttle_duration_sec=1.0)

    def timer_callback(self):
        self.counter += 1.0

        if self.last_goal is None:
            self.publish_stop("waiting for relative_goal")
            return
        if self.last_context is None:
            self.publish_stop("waiting for context_vector")
            return
        if self.last_latent is None:
            self.publish_stop("waiting for latent_vector")
            return
        if self.last_beta is None:
            self.publish_stop("waiting for objective_weights")
            return

        if self.age_sec(self.last_goal_time) > self.input_timeout_sec:
            self.publish_stop("relative_goal timeout")
            return
        if self.age_sec(self.last_context_time) > self.input_timeout_sec:
            self.publish_stop("context timeout")
            return
        if self.age_sec(self.last_latent_time) > self.input_timeout_sec:
            self.publish_stop("latent timeout")
            return
        if self.age_sec(self.last_beta_time) > self.input_timeout_sec:
            self.publish_stop("objective timeout")
            return

        goal = self.last_goal
        context = self.last_context
        latent = self.last_latent
        beta = self.last_beta

        x_rel = float(goal[0])
        y_rel = float(goal[1])

        dist = math.sqrt(x_rel * x_rel + y_rel * y_rel)
        yaw_error = math.atan2(y_rel, max(1e-6, x_rel))

        enable_override = None
        if len(goal) >= 4:
            enable_override = float(goal[3]) > 0.5

        risk = self.compute_risk(context, latent)

        if risk >= self.risk_stop_threshold:
            self.publish_stop(f"risk stop risk={risk:.2f}")
            return

        vx_max, yaw_rate_max, body_height, clearance, safe_alpha = self.objective_modulation(beta, risk)

        if dist < self.goal_stop_distance:
            vx = 0.0
            yaw_rate = 0.0
            enable = 0.0
        else:
            yaw_alignment = max(
                0.0,
                1.0 - abs(yaw_error) / max(1e-6, self.yaw_slowdown_angle),
            )

            vx_raw = self.kx * x_rel
            vx = clamp(vx_raw, 0.0, vx_max) * yaw_alignment

            yaw_rate = clamp(
                self.kyaw * yaw_error,
                -yaw_rate_max,
                yaw_rate_max,
            )

            enable = 1.0

        if enable_override is not None:
            enable = 1.0 if enable_override else 0.0
            if not enable_override:
                vx = 0.0
                yaw_rate = 0.0

        self.publish_ref(vx, yaw_rate, body_height, clearance, enable)

        self.get_logger().info(
            f"ObjHL beta=({beta[0]:.2f},{beta[1]:.2f},{beta[2]:.2f}) "
            f"dist={dist:.2f} yaw_err={yaw_error:.2f} risk={risk:.2f} safe={safe_alpha:.2f} "
            f"-> vx={vx:.3f} yaw={yaw_rate:.3f} h={body_height:.3f} clr={clearance:.3f} enable={enable:.1f}",
            throttle_duration_sec=1.0,
        )


def main():
    rclpy.init()
    node = TracerObjectiveConditionedHighLevelControllerStubNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
