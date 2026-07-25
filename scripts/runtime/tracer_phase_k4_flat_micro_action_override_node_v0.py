#!/usr/bin/env python3
import csv
import os
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String


def parse_theta(s):
    vals = [
        float(x.strip())
        for x in s.split(",")
        if x.strip() != ""
    ]

    if len(vals) != 9:
        raise ValueError(
            f"Expected 9 theta values, "
            f"got {len(vals)} from {s!r}"
        )

    return vals


def context_matches(target, current):
    target = (target or "flat").strip()
    current = (current or "unknown").strip()

    # Preserve historical flat behavior.
    if target == "flat":
        return current in {"flat", "start_flat"}

    return current == target


class FlatMicroActionOverride(Node):
    def __init__(self):
        super().__init__("tracer_phase_k4_flat_micro_action_override_v0")

        self.input_topic = os.environ.get(
            "TRACER_PHASE_K4_INPUT_THETA_TOPIC",
            "/tracer/meta_action_theta_shadow",
        )
        self.output_topic = os.environ.get(
            "TRACER_PHASE_K4_OUTPUT_THETA_TOPIC",
            "/tracer/meta_action_theta_k4_flat_micro_shadow",
        )
        self.context_topic = os.environ.get(
            "TRACER_PHASE_K4_CONTEXT_TOPIC",
            "/tracer/terrain_context_label",
        )

        self.profile = os.environ.get(
            "TRACER_PHASE_K4_PROFILE",
            "flat_noop",
        )

        self.target_context = os.environ.get(
            "TRACER_PHASE_K4_TARGET_CONTEXT",
            "flat",
        ).strip()

        # TARGET_THETA is the generalized interface.
        # FLAT_THETA remains as a backward-compatible fallback.
        theta_env = os.environ.get(
            "TRACER_PHASE_K4_TARGET_THETA",
            os.environ.get(
                "TRACER_PHASE_K4_FLAT_THETA",
                (
                    "1.0,1.0,0.0,1.0,"
                    "0.0,0.0,0.0,0.0,0.0"
                ),
            ),
        )

        self.target_theta = parse_theta(theta_env)

        self.terrain_context = "unknown"
        self.latest_input = None
        self.seq = 0

        log_dir = Path(os.environ.get(
            "TRACER_PHASE_K4_LOG_DIR",
            f"logs/phase_k/k4_flat_micro_override_{int(time.time())}",
        ))
        log_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = log_dir / "flat_micro_override_k4_v0.csv"
        self.csv_f = open(self.csv_path, "w", newline="")
        self.writer = csv.DictWriter(
            self.csv_f,
            fieldnames=[
                "t", "seq", "profile", "context", "override",
                "in_action_id", "out_action_id",
                "in_theta", "out_theta",
            ],
        )
        self.writer.writeheader()

        self.pub = self.create_publisher(Float64MultiArray, self.output_topic, 10)
        self.create_subscription(Float64MultiArray, self.input_topic, self.on_theta, 10)
        self.create_subscription(String, self.context_topic, self.on_context, 10)

        self.timer = self.create_timer(0.02, self.on_timer)

        self.get_logger().info(
            f"K4 flat micro override started: profile={self.profile}, "
            f"input={self.input_topic}, output={self.output_topic}, "
            f"context_topic={self.context_topic}, "
            f"target_context={self.target_context}, "
            f"target_theta={self.target_theta}, "
            f"log={self.csv_path}"
        )

    def on_context(self, msg):
        self.terrain_context = (msg.data or "unknown").strip()

    def on_theta(self, msg):
        self.latest_input = list(msg.data)

    def on_timer(self):
        if self.latest_input is None:
            return

        in_theta = list(self.latest_input)
        out_theta = list(in_theta)
        override = 0

        if context_matches(
            self.target_context,
            self.terrain_context,
        ):
            out_theta = list(self.target_theta)
            override = 1

        msg = Float64MultiArray()
        msg.data = out_theta
        self.pub.publish(msg)

        self.seq += 1
        self.writer.writerow({
            "t": f"{time.time():.6f}",
            "seq": self.seq,
            "profile": self.profile,
            "context": self.terrain_context,
            "override": override,
            "in_action_id": in_theta[0] if len(in_theta) > 0 else "",
            "out_action_id": out_theta[0] if len(out_theta) > 0 else "",
            "in_theta": " ".join(f"{x:.6f}" for x in in_theta),
            "out_theta": " ".join(f"{x:.6f}" for x in out_theta),
        })
        if self.seq % 50 == 0:
            self.csv_f.flush()

    def destroy_node(self):
        try:
            self.csv_f.flush()
            self.csv_f.close()
        except Exception:
            pass
        super().destroy_node()


def main():
    rclpy.init()
    node = FlatMicroActionOverride()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        # ROS2 launch/cleanup can raise ExternalShutdownException during normal teardown.
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
