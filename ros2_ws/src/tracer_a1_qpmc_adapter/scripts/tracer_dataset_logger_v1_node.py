#!/usr/bin/env python3

import os
import time
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray


class TracerDatasetLoggerV1Node(Node):
    """
    TRACER dataset logger v1.

    Adds:
      /tracer/objective_weights

    Required streams:
      /tracer/proprio_vector
      /tracer/context_vector
      /tracer/latent_vector
      /tracer/relative_goal
      /tracer/objective_weights
      /tracer/mpc_reference
    """

    def __init__(self):
        super().__init__("tracer_dataset_logger_v1_node")

        self.declare_parameter("save_dir", os.path.expanduser("~/Tracer/TRACER/data"))
        self.declare_parameter("log_hz", 20.0)
        self.declare_parameter("max_duration_sec", 60.0)
        self.declare_parameter("flush_on_exit", True)

        self.topics = {
            "proprio": "/tracer/proprio_vector",
            "context": "/tracer/context_vector",
            "latent": "/tracer/latent_vector",
            "relative_goal": "/tracer/relative_goal",
            "objective_weights": "/tracer/objective_weights",
            "mpc_reference": "/tracer/mpc_reference",
            "robot_odom": "/tracer/robot_odom_flat",
            "global_goal": "/tracer/global_goal",
        }

        for key, topic in self.topics.items():
            self.declare_parameter(f"{key}_topic", topic)

        self.save_dir = self.get_parameter("save_dir").value
        self.log_hz = float(self.get_parameter("log_hz").value)
        self.max_duration_sec = float(self.get_parameter("max_duration_sec").value)
        self.flush_on_exit = bool(self.get_parameter("flush_on_exit").value)

        os.makedirs(self.save_dir, exist_ok=True)

        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_path = os.path.join(self.save_dir, f"tracer_dataset_v1_{now_str}.npz")

        self.latest: Dict[str, Optional[List[float]]] = {
            key: None for key in self.topics.keys()
        }

        self.latest_time: Dict[str, Optional[float]] = {
            key: None for key in self.topics.keys()
        }

        self.buffers = {
            "stamp_wall": [],
            "proprio": [],
            "context": [],
            "latent": [],
            "relative_goal": [],
            "objective_weights": [],
            "mpc_reference": [],
            "robot_odom": [],
            "global_goal": [],
            "valid_mask": [],
        }

        self.fixed_lengths = {
            "proprio": 45,
            "context": 16,
            "latent": 16,
            "relative_goal": 4,
            "objective_weights": 3,
            "mpc_reference": 6,
            "robot_odom": 6,
            "global_goal": 3,
        }

        self.start_wall = time.time()
        self.saved = False

        for key in self.topics.keys():
            topic = self.get_parameter(f"{key}_topic").value
            self.create_subscription(
                Float64MultiArray,
                topic,
                self.make_callback(key),
                10,
            )

        self.timer = self.create_timer(
            1.0 / max(1.0, self.log_hz),
            self.timer_callback,
        )

        self.get_logger().info("TRACER dataset logger v1 started")
        self.get_logger().info(f"save_path: {self.save_path}")
        self.get_logger().info("logging objective_weights for objective-conditioned policy v1")

    def make_callback(self, key):
        def callback(msg):
            self.latest[key] = list(msg.data)
            self.latest_time[key] = time.time()
        return callback

    def pad_or_empty(self, key):
        target_len = self.fixed_lengths[key]
        value = self.latest[key]
        if value is None:
            return [0.0] * target_len

        out = list(value[:target_len])
        if len(out) < target_len:
            out.extend([0.0] * (target_len - len(out)))
        return out

    def timer_callback(self):
        elapsed = time.time() - self.start_wall

        required = [
            "proprio",
            "context",
            "latent",
            "relative_goal",
            "objective_weights",
            "mpc_reference",
        ]

        valid = 1.0
        for key in required:
            if self.latest[key] is None:
                valid = 0.0
                break

        self.buffers["stamp_wall"].append(time.time())
        for key in self.fixed_lengths.keys():
            self.buffers[key].append(self.pad_or_empty(key))
        self.buffers["valid_mask"].append(valid)

        n = len(self.buffers["stamp_wall"])

        self.get_logger().info(
            f"logging v1 n={n} elapsed={elapsed:.1f}s valid={valid:.0f}",
            throttle_duration_sec=1.0,
        )

        if elapsed >= self.max_duration_sec:
            self.get_logger().info("max duration reached, saving dataset v1")
            self.save()
            rclpy.shutdown()

    def save(self):
        if self.saved:
            return

        arrays = {}
        for key, values in self.buffers.items():
            arrays[key] = np.asarray(values, dtype=np.float64)

        np.savez_compressed(self.save_path, **arrays)
        self.saved = True

        self.get_logger().info(f"saved dataset v1: {self.save_path}")
        for key, arr in arrays.items():
            self.get_logger().info(f"  {key}: shape={arr.shape}")

    def destroy_node(self):
        if self.flush_on_exit and not self.saved and len(self.buffers["stamp_wall"]) > 0:
            self.get_logger().info("node shutdown: saving dataset v1")
            self.save()
        super().destroy_node()


def main():
    rclpy.init()
    node = TracerDatasetLoggerV1Node()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
