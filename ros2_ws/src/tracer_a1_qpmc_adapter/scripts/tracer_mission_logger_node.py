#!/usr/bin/env python3
import json
import math
import os
import time
from datetime import datetime

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray


class TracerMissionLogger(Node):
    def __init__(self):
        super().__init__("tracer_mission_logger_node")

        self.declare_parameter("output_dir", os.path.expanduser("~/Tracer/TRACER/data/mission_logs"))
        self.declare_parameter("log_hz", 20.0)
        self.declare_parameter("success_radius", 0.60)

        self.output_dir = self.get_parameter("output_dir").value
        self.log_hz = float(self.get_parameter("log_hz").value)
        self.success_radius = float(self.get_parameter("success_radius").value)

        os.makedirs(self.output_dir, exist_ok=True)

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_npz = os.path.join(self.output_dir, f"tracer_mission_log_{stamp}.npz")
        self.output_json = os.path.join(self.output_dir, f"tracer_mission_summary_{stamp}.json")

        self.topics = {
            "proprio": ("/tracer/proprio_vector", 45),
            "context": ("/tracer/context_vector", 16),
            "latent": ("/tracer/latent_vector", 16),
            "objective": ("/tracer/objective_weights", 3),
            "global_goal": ("/tracer/global_goal", 3),
            "relative_goal": ("/tracer/relative_goal", 4),
            "mpc_reference": ("/tracer/mpc_reference", 6),
            "odom": ("/tracer/robot_odom_flat", 6),
        }

        self.latest = {}
        self.latest_wall = {}

        self.wall_time = []
        self.data = {k: [] for k in self.topics}
        self.valid = {k: [] for k in self.topics}
        self.age = {k: [] for k in self.topics}

        for key, (topic, _dim) in self.topics.items():
            self.create_subscription(
                Float64MultiArray,
                topic,
                lambda msg, key=key: self._callback(key, msg),
                10,
            )

        period = 1.0 / max(self.log_hz, 1e-6)
        self.timer = self.create_timer(period, self._tick)
        self._saved = False

        self.get_logger().info(
            f"mission logger started: hz={self.log_hz}, output={self.output_npz}"
        )

    def _callback(self, key, msg):
        self.latest[key] = np.asarray(msg.data, dtype=np.float64)
        self.latest_wall[key] = time.time()

    def _tick(self):
        now = time.time()
        self.wall_time.append(now)

        for key, (_topic, dim) in self.topics.items():
            if key in self.latest:
                arr = self.latest[key]
                if arr.shape[0] < dim:
                    padded = np.full((dim,), np.nan, dtype=np.float64)
                    padded[: arr.shape[0]] = arr
                    arr = padded
                elif arr.shape[0] > dim:
                    arr = arr[:dim]

                self.data[key].append(arr.astype(np.float64))
                self.valid[key].append(1.0)
                self.age[key].append(now - self.latest_wall.get(key, now))
            else:
                self.data[key].append(np.full((dim,), np.nan, dtype=np.float64))
                self.valid[key].append(0.0)
                self.age[key].append(np.nan)

    def _stack_or_empty(self, key):
        dim = self.topics[key][1]
        if len(self.data[key]) == 0:
            return np.empty((0, dim), dtype=np.float64)
        return np.vstack(self.data[key]).astype(np.float64)

    def _summary(self, arrays):
        summary = {
            "output_npz": self.output_npz,
            "num_samples": int(len(self.wall_time)),
            "log_hz": self.log_hz,
            "success_radius": self.success_radius,
        }

        wall = arrays["wall_time"]
        if wall.size >= 2:
            summary["duration_sec"] = float(wall[-1] - wall[0])
        else:
            summary["duration_sec"] = 0.0

        odom = arrays["odom"]
        odom_valid = arrays["odom_valid"] > 0.5
        if np.count_nonzero(odom_valid) >= 2:
            o = odom[odom_valid]
            xy = o[:, 1:3]
            dxy = np.diff(xy, axis=0)
            path_len = np.linalg.norm(dxy, axis=1).sum()
            summary["odom_start_xy"] = xy[0].tolist()
            summary["odom_end_xy"] = xy[-1].tolist()
            summary["odom_path_length"] = float(path_len)
            summary["odom_net_displacement"] = float(np.linalg.norm(xy[-1] - xy[0]))

        mpc = arrays["mpc_reference"]
        mpc_valid = arrays["mpc_reference_valid"] > 0.5
        if np.count_nonzero(mpc_valid) > 0:
            m = mpc[mpc_valid]
            summary["mpc_enable_fraction"] = float(np.mean(m[:, 5] > 0.5))
            summary["mpc_vx_mean"] = float(np.nanmean(m[:, 1]))
            summary["mpc_vx_max"] = float(np.nanmax(m[:, 1]))
            summary["mpc_body_height_mean"] = float(np.nanmean(m[:, 3]))
            summary["mpc_clearance_mean"] = float(np.nanmean(m[:, 4]))

        rel = arrays["relative_goal"]
        rel_valid = arrays["relative_goal_valid"] > 0.5
        if np.count_nonzero(rel_valid) > 0:
            r = rel[rel_valid]
            dist = np.linalg.norm(r[:, 0:2], axis=1)
            summary["relative_goal_min_dist"] = float(np.nanmin(dist))
            summary["relative_goal_final"] = r[-1].tolist()
            summary["success_inferred_by_radius"] = bool(np.nanmin(dist) <= self.success_radius)

        obj = arrays["objective"]
        obj_valid = arrays["objective_valid"] > 0.5
        if np.count_nonzero(obj_valid) > 0:
            summary["objective_mean"] = np.nanmean(obj[obj_valid], axis=0).tolist()

        return summary

    def save(self):
        if self._saved:
            return
        self._saved = True

        arrays = {
            "wall_time": np.asarray(self.wall_time, dtype=np.float64),
        }

        for key in self.topics:
            arrays[key] = self._stack_or_empty(key)
            arrays[f"{key}_valid"] = np.asarray(self.valid[key], dtype=np.float64)
            arrays[f"{key}_age"] = np.asarray(self.age[key], dtype=np.float64)

        np.savez_compressed(self.output_npz, **arrays)

        summary = self._summary(arrays)
        with open(self.output_json, "w") as f:
            json.dump(summary, f, indent=2)

        self.get_logger().info(f"saved mission log: {self.output_npz}")
        self.get_logger().info(f"saved mission summary: {self.output_json}")


def main():
    rclpy.init()
    node = TracerMissionLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        node.get_logger().error(f"logger exception: {exc}")
    finally:
        node.save()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
