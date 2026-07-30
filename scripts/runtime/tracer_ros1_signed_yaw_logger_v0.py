#!/usr/bin/env python
from __future__ import print_function

import csv
import math
import signal
import sys
import threading

import rospy
from gazebo_msgs.msg import ModelStates
from std_msgs.msg import Float64MultiArray

MODEL_NAME = "a1_gazebo"


def quaternion_to_rpy(q):
    x = float(q.x)
    y = float(q.y)
    z = float(q.z)
    w = float(q.w)

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


class Logger(object):
    def __init__(self, output_path):
        self.lock = threading.Lock()
        self.latest_ref = None
        self.latest_ref_time = None
        self.model_index = None
        self.last_log_sim_time = None
        self.row_count = 0
        self.closed = False

        self.stream = open(output_path, "wb")
        self.writer = csv.writer(self.stream)
        self.writer.writerow([
            "sim_time",
            "x",
            "y",
            "z",
            "roll_deg",
            "pitch_deg",
            "yaw_rad",
            "yaw_deg",
            "wz_world_rad_s",
            "ref_counter",
            "ref_vx",
            "ref_yaw_rate",
            "ref_body_h",
            "ref_clearance",
            "ref_enable",
            "ref_age_sim_s",
        ])
        self.stream.flush()

        self.ref_sub = rospy.Subscriber(
            "/tracer/mpc_reference",
            Float64MultiArray,
            self.on_ref,
            queue_size=20,
        )
        self.model_sub = rospy.Subscriber(
            "/gazebo/model_states",
            ModelStates,
            self.on_models,
            queue_size=20,
        )
        rospy.on_shutdown(self.close)

    def on_ref(self, msg):
        data = list(msg.data)
        if len(data) < 5:
            return

        now = rospy.Time.now().to_sec()
        with self.lock:
            self.latest_ref = data
            self.latest_ref_time = now

    def resolve_model_index(self, msg):
        if (
            self.model_index is not None
            and self.model_index < len(msg.name)
            and msg.name[self.model_index] == MODEL_NAME
        ):
            return self.model_index

        try:
            self.model_index = msg.name.index(MODEL_NAME)
        except ValueError:
            self.model_index = None

        return self.model_index

    def on_models(self, msg):
        sim_time = rospy.Time.now().to_sec()
        if sim_time <= 0.0:
            return

        if (
            self.last_log_sim_time is not None
            and sim_time < self.last_log_sim_time - 0.1
        ):
            self.last_log_sim_time = None

        if (
            self.last_log_sim_time is not None
            and sim_time - self.last_log_sim_time < 0.018
        ):
            return

        index = self.resolve_model_index(msg)
        if index is None:
            return

        with self.lock:
            if self.latest_ref is None:
                return
            ref = list(self.latest_ref)
            ref_time = self.latest_ref_time

        pose = msg.pose[index]
        twist = msg.twist[index]
        roll, pitch, yaw = quaternion_to_rpy(pose.orientation)

        ref_counter = float(ref[0]) if len(ref) > 0 else float("nan")
        ref_vx = float(ref[1]) if len(ref) > 1 else 0.0
        ref_yaw = float(ref[2]) if len(ref) > 2 else 0.0
        ref_body_h = float(ref[3]) if len(ref) > 3 else 0.0
        ref_clearance = float(ref[4]) if len(ref) > 4 else 0.0
        ref_enable = float(ref[5]) if len(ref) > 5 else 1.0
        ref_age = (
            sim_time - ref_time
            if ref_time is not None
            else float("nan")
        )

        self.writer.writerow([
            "%.9f" % sim_time,
            "%.9f" % float(pose.position.x),
            "%.9f" % float(pose.position.y),
            "%.9f" % float(pose.position.z),
            "%.9f" % math.degrees(roll),
            "%.9f" % math.degrees(pitch),
            "%.9f" % yaw,
            "%.9f" % math.degrees(yaw),
            "%.9f" % float(twist.angular.z),
            "%.9f" % ref_counter,
            "%.9f" % ref_vx,
            "%.9f" % ref_yaw,
            "%.9f" % ref_body_h,
            "%.9f" % ref_clearance,
            "%.9f" % ref_enable,
            "%.9f" % ref_age,
        ])

        self.row_count += 1
        self.last_log_sim_time = sim_time
        if self.row_count % 25 == 0:
            self.stream.flush()

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            self.stream.flush()
            self.stream.close()
        except Exception:
            pass


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: logger.py OUTPUT.csv")

    rospy.init_node(
        "tracer_ros1_signed_yaw_logger_v0",
        anonymous=False,
        disable_signals=True,
    )
    logger = Logger(sys.argv[1])

    def shutdown_handler(signum, frame):
        rospy.signal_shutdown("signal {}".format(signum))

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    rospy.loginfo("Signed yaw logger started: %s", sys.argv[1])
    rospy.spin()
    logger.close()


if __name__ == "__main__":
    main()
