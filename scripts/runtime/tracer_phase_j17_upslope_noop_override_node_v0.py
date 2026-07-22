#!/usr/bin/env python3
import csv
import os
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String


def env_float(name, default):
    raw = os.environ.get(name, "")
    if raw == "":
        return default
    try:
        return float(raw)
    except Exception:
        return default


def context_key(label):
    label = (label or "unknown").strip()
    known = {"flat", "start_flat", "rough", "upslope", "downslope", "goal_flat", "unknown"}
    return label if label in known else "unknown"


class J17UpslopeNoopOverride(Node):
    def __init__(self):
        super().__init__("tracer_phase_j17_upslope_noop_override_node_v0")

        self.input_theta_topic = os.environ.get("TRACER_PHASE_J17_INPUT_THETA_TOPIC", "/tracer/meta_action_theta_shadow")
        self.output_theta_topic = os.environ.get("TRACER_PHASE_J17_OUTPUT_THETA_TOPIC", "/tracer/meta_action_theta_j17_upslope_noop_shadow")
        self.context_topic = os.environ.get("TRACER_PHASE_J17_CONTEXT_TOPIC", "/tracer/terrain_context_label")
        self.pub_hz = env_float("TRACER_PHASE_J17_PUB_HZ", 10.0)

        self.ctx = "unknown"
        self.last_theta = None
        self.last_theta_t = None

        # theta = [
        #   action_id, vx_scale, vx_delta, yaw_gain_scale,
        #   body_h_delta, clearance_delta,
        #   stability_bias, energy_bias, hold_override
        # ]
        self.upslope_noop_theta = [
            5.0,
            env_float("TRACER_PHASE_J17_UPSLOPE_VX_SCALE", 1.0),
            env_float("TRACER_PHASE_J17_UPSLOPE_VX_DELTA", 0.0),
            env_float("TRACER_PHASE_J17_UPSLOPE_YAW_GAIN_SCALE", 1.0),
            env_float("TRACER_PHASE_J17_UPSLOPE_BODY_H_DELTA", 0.0),
            env_float("TRACER_PHASE_J17_UPSLOPE_CLEARANCE_DELTA", 0.0),
            env_float("TRACER_PHASE_J17_UPSLOPE_STABILITY_BIAS", 0.0),
            env_float("TRACER_PHASE_J17_UPSLOPE_ENERGY_BIAS", 0.0),
            0.0,
        ]

        self.pub = self.create_publisher(Float64MultiArray, self.output_theta_topic, 10)
        self.create_subscription(Float64MultiArray, self.input_theta_topic, self.on_theta, 10)
        self.create_subscription(String, self.context_topic, self.on_context, 10)

        log_dir = Path(os.environ.get("TRACER_PHASE_J17_LOG_DIR", "logs/phase_j/j17_upslope_noop_override_latest"))
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_csv = log_dir / "upslope_noop_override_j17_v0.csv"
        self.log_f = open(self.log_csv, "w", newline="")
        self.writer = csv.DictWriter(
            self.log_f,
            fieldnames=[
                "t_wall", "context",
                "input_action_id", "output_action_id",
                "input_age_s", "override_applied",
                "out_vx_scale", "out_vx_delta", "out_yaw_gain_scale",
                "out_body_h_delta", "out_clearance_delta",
                "out_stability_bias", "out_energy_bias", "out_hold_override",
            ],
        )
        self.writer.writeheader()
        self.log_f.flush()

        self.timer = self.create_timer(1.0 / max(self.pub_hz, 1e-6), self.on_timer)

        self.get_logger().info(f"[TRACER:J17] input_theta={self.input_theta_topic}")
        self.get_logger().info(f"[TRACER:J17] output_theta={self.output_theta_topic}")
        self.get_logger().info(f"[TRACER:J17] upslope_noop_theta={self.upslope_noop_theta}")
        self.get_logger().info(f"[TRACER:J17] log_csv={self.log_csv}")

    def on_context(self, msg):
        self.ctx = context_key(msg.data)

    def on_theta(self, msg):
        data = list(msg.data)
        if len(data) >= 9:
            self.last_theta = [float(v) for v in data[:9]]
            self.last_theta_t = time.time()

    def on_timer(self):
        now = time.time()
        ctx = context_key(self.ctx)

        if ctx == "upslope":
            out = list(self.upslope_noop_theta)
            override = True
        elif self.last_theta is not None:
            out = list(self.last_theta)
            override = False
        else:
            return

        msg = Float64MultiArray()
        msg.data = [float(v) for v in out]
        self.pub.publish(msg)

        input_action_id = ""
        input_age = ""
        if self.last_theta is not None:
            input_action_id = f"{self.last_theta[0]:.0f}"
        if self.last_theta_t is not None:
            input_age = f"{now - self.last_theta_t:.6f}"

        self.writer.writerow({
            "t_wall": f"{now:.6f}",
            "context": ctx,
            "input_action_id": input_action_id,
            "output_action_id": f"{out[0]:.0f}",
            "input_age_s": input_age,
            "override_applied": "1" if override else "0",
            "out_vx_scale": f"{out[1]:.8f}",
            "out_vx_delta": f"{out[2]:.8f}",
            "out_yaw_gain_scale": f"{out[3]:.8f}",
            "out_body_h_delta": f"{out[4]:.8f}",
            "out_clearance_delta": f"{out[5]:.8f}",
            "out_stability_bias": f"{out[6]:.8f}",
            "out_energy_bias": f"{out[7]:.8f}",
            "out_hold_override": f"{out[8]:.8f}",
        })
        self.log_f.flush()

    def destroy_node(self):
        try:
            self.log_f.flush()
            self.log_f.close()
        except Exception:
            pass
        super().destroy_node()


def main():
    rclpy.init()
    node = J17UpslopeNoopOverride()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        if e.__class__.__name__ != "ExternalShutdownException":
            raise
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
