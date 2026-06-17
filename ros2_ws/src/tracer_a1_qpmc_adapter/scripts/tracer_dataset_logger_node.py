#!/usr/bin/env python3

import os
import time
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray


class TracerDatasetLoggerNode(Node):
    """
    TRACER dataset logger v0.

    Logs synchronized latest snapshots at fixed Hz.

    Inputs:
      /tracer/proprio_vector
      /tracer/context_vector
      /tracer/latent_vector
      /tracer/relative_goal
      /tracer/mpc_reference

    Optional:
      /tracer/robot_odom_flat
      /tracer/global_goal

    Output:
      .npz file with arrays:
        stamp_wall
        proprio
        context
        latent
        relative_goal
        mpc_reference
        robot_odom
        global_goal
        valid_mask
    """

    def __init__(self):
        super().__init__("tracer_dataset_logger_node")

        self.declare_parameter("save_dir", os.path.expanduser("~/Tracer/TRACER/data"))
        self.declare_parameter("log_hz", 20.0)
        self.declare_parameter("max_duration_sec", 60.0)
        self.declare_parameter("flush_on_exit", True)

        self.declare_parameter("proprio_topic", "/tracer/proprio_vector")
        self.declare_parameter("context_topic", "/tracer/context_vector")
        self.declare_parameter("latent_topic", "/tracer/latent_vector")
        self.declare_parameter("relative_goal_topic", "/tracer/relative_goal")
        self.declare_parameter("mpc_ref_topic", "/tracer/mpc_reference")
        self.declare_parameter("robot_odom_topic", "/tracer/robot_odom_flat")
        self.declare_parameter("global_goal_topic", "/tracer/global_goal")

        self.save_dir = self.get_parameter("save_dir").value
        self.log_hz = float(self.get_parameter("log_hz").value)
        self.max_duration_sec = float(self.get_parameter("max_duration_sec").value)
        self.flush_on_exit = bool(self.get_parameter("flush_on_exit").value)

        os.makedirs(self.save_dir, exist_ok=True)

        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_path = os.path.join(self.save_dir, f"tracer_dataset_{now_str}.npz")

        self.latest: Dict[str, Optional[List[float]]] = {
            "proprio": None,
            "context": None,
            "latent": None,
            "relative_goal": None,
            "mpc_reference": None,
            "robot_odom": None,
            "global_goal": None,
        }

        self.latest_time: Dict[str, Optional[float]] = {
            key: None for key in self.latest.keys()
        }

        self.buffers = {
            "stamp_wall": [],
            "proprio": [],
            "context": [],
            "latent": [],
            "relative_goal": [],
            "mpc_reference": [],
            "robot_odom": [],
            "global_goal": [],
            "valid_mask": [],
        }

        self.start_wall = time.time()
        self.saved = False

        self.create_subscription(
            Float64MultiArray,
            self.get_parameter("proprio_topic").value,
            self.make_callback("proprio"),
            10,
        )
        self.create_subscription(
            Float64MultiArray,
            self.get_parameter("context_topic").value,
            self.make_callback("context"),
            10,
        )
        self.create_subscription(
            Float64MultiArray,
            self.get_parameter("latent_topic").value,
            self.make_callback("latent"),
            10,
        )
        self.create_subscription(
            Float64MultiArray,
            self.get_parameter("relative_goal_topic").value,
            self.make_callback("relative_goal"),
            10,
        )
        self.create_subscription(
            Float64MultiArray,
            self.get_parameter("mpc_ref_topic").value,
            self.make_callback("mpc_reference"),
            10,
        )
        self.create_subscription(
            Float64MultiArray,
            self.get_parameter("robot_odom_topic").value,
            self.make_callback("robot_odom"),
            10,
        )
        self.create_subscription(
            Float64MultiArray,
            self.get_parameter("global_goal_topic").value,
            self.make_callback("global_goal"),
            10,
        )

        self.timer = self.create_timer(
            1.0 / max(1.0, self.log_hz),
            self.timer_callback,
        )

        self.get_logger().info(f"TRACER dataset logger started")
        self.get_logger().info(f"save_dir: {self.save_dir}")
        self.get_logger().info(f"save_path: {self.save_path}")
        self.get_logger().info(f"log_hz={self.log_hz}, max_duration_sec={self.max_duration_sec}")

    def make_callback(self, key: str):
        def callback(msg: Float64MultiArray):
            self.latest[key] = list(msg.data)
            self.latest_time[key] = time.time()
        return callback

    def pad_or_empty(self, key: str, target_len: int):
        value = self.latest[key]
        if value is None:
            return [0.0] * target_len

        out = list(value[:target_len])
        if len(out) < target_len:
            out.extend([0.0] * (target_len - len(out)))
        return out

    def timer_callback(self):
        elapsed = time.time() - self.start_wall

        # Required streams for learning v0
        required_keys = ["proprio", "context", "latent", "relative_goal", "mpc_reference"]
        valid = 1.0
        for key in required_keys:
            if self.latest[key] is None:
                valid = 0.0
                break

        self.buffers["stamp_wall"].append(time.time())

        self.buffers["proprio"].append(self.pad_or_empty("proprio", 45))
        self.buffers["context"].append(self.pad_or_empty("context", 16))
        self.buffers["latent"].append(self.pad_or_empty("latent", 16))
        self.buffers["relative_goal"].append(self.pad_or_empty("relative_goal", 4))
        self.buffers["mpc_reference"].append(self.pad_or_empty("mpc_reference", 6))
        self.buffers["robot_odom"].append(self.pad_or_empty("robot_odom", 6))
        self.buffers["global_goal"].append(self.pad_or_empty("global_goal", 3))
        self.buffers["valid_mask"].append(valid)

        n = len(self.buffers["stamp_wall"])

        self.get_logger().info(
            f"logging n={n} elapsed={elapsed:.1f}s valid={valid:.0f}",
            throttle_duration_sec=1.0,
        )

        if elapsed >= self.max_duration_sec:
            self.get_logger().info("max duration reached, saving dataset")
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

        self.get_logger().info(f"saved dataset: {self.save_path}")
        for key, arr in arrays.items():
            self.get_logger().info(f"  {key}: shape={arr.shape}")

    def destroy_node(self):
        if self.flush_on_exit and not self.saved and len(self.buffers["stamp_wall"]) > 0:
            self.get_logger().info("node shutdown: saving dataset")
            self.save()
        super().destroy_node()


def main():
    rclpy.init()
    node = TracerDatasetLoggerNode()
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
