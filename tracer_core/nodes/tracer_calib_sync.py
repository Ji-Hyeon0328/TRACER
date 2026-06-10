#!/usr/bin/env python
from __future__ import print_function

import os
import csv
import math
import time

import rospy
from sensor_msgs.msg import Joy
from nav_msgs.msg import Odometry


def quat_to_yaw(x, y, z, w):
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def make_joy(vx_axis, yaw_axis, toggle=False):
    msg = Joy()
    msg.header.stamp = rospy.Time.now()
    msg.axes = [0.0] * 8
    msg.axes[5] = vx_axis
    msg.axes[0] = yaw_axis
    msg.buttons = [0, 0, 0, 0, 0]
    if toggle:
        msg.buttons[0] = 1
    return msg


class SyncCalibration(object):
    def __init__(self):
        self.vx_axis = rospy.get_param("~vx_axis", 0.12)
        self.yaw_axis = rospy.get_param("~yaw_axis", 0.0)
        self.duration = rospy.get_param("~duration", 15.0)
        self.warmup = rospy.get_param("~warmup", 2.0)
        self.rate_hz = rospy.get_param("~rate", 20.0)
        self.toggle_walk = rospy.get_param("~toggle_walk", True)
        self.toggle_stand_on_exit = rospy.get_param("~toggle_stand_on_exit", True)
        self.log_dir = rospy.get_param("~log_dir", "/root/TRACER/logs")
        self.tag = rospy.get_param("~tag", "calib_sync")

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(
            self.log_dir,
            "tracer_calib_sync_%s_%s.csv" % (self.tag, stamp)
        )

        self.latest_odom = None

        self.pub = rospy.Publisher("/joy", Joy, queue_size=10)
        self.sub = rospy.Subscriber(
            "/body_pose_ground_truth",
            Odometry,
            self.odom_cb,
            queue_size=50,
        )

    def odom_cb(self, msg):
        self.latest_odom = msg

    def write_row(self, writer, phase):
        if self.latest_odom is None:
            return

        msg = self.latest_odom
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = quat_to_yaw(q.x, q.y, q.z, q.w)

        writer.writerow([
            rospy.Time.now().to_sec(),
            phase,
            self.vx_axis,
            self.yaw_axis,
            p.x,
            p.y,
            p.z,
            yaw,
        ])

    def toggle_button0(self):
        self.pub.publish(make_joy(0.0, 0.0, toggle=True))
        rospy.sleep(0.25)
        self.pub.publish(make_joy(0.0, 0.0, toggle=False))
        rospy.sleep(0.25)

    def zero_for(self, sec, writer=None, phase="zero"):
        rate = rospy.Rate(self.rate_hz)
        t0 = rospy.Time.now().to_sec()
        while not rospy.is_shutdown():
            if rospy.Time.now().to_sec() - t0 > sec:
                break
            self.pub.publish(make_joy(0.0, 0.0, toggle=False))
            if writer is not None:
                self.write_row(writer, phase)
            rate.sleep()

    def run(self):
        rospy.loginfo("Waiting for odometry...")
        while not rospy.is_shutdown() and self.latest_odom is None:
            rospy.sleep(0.05)

        f = open(self.path, "w")
        writer = csv.writer(f)
        writer.writerow([
            "ros_time",
            "phase",
            "cmd_vx_axis",
            "cmd_yaw_axis",
            "x",
            "y",
            "z",
            "yaw",
        ])

        rospy.loginfo("Calibration sync log: %s", self.path)

        # Start from zero command.
        self.zero_for(1.0, writer, "pre_zero")

        if self.toggle_walk:
            rospy.loginfo("Toggling walking mode on.")
            self.toggle_button0()

        # Let gait settle before measuring.
        self.zero_for(self.warmup, writer, "warmup")

        rospy.loginfo(
            "Command phase: vx_axis=%.3f yaw_axis=%.3f duration=%.2f",
            self.vx_axis,
            self.yaw_axis,
            self.duration,
        )

        rate = rospy.Rate(self.rate_hz)
        t0 = rospy.Time.now().to_sec()
        while not rospy.is_shutdown():
            if rospy.Time.now().to_sec() - t0 > self.duration:
                break

            self.pub.publish(make_joy(self.vx_axis, self.yaw_axis, toggle=False))
            self.write_row(writer, "cmd")
            rate.sleep()

        self.zero_for(1.0, writer, "post_zero")

        if self.toggle_stand_on_exit:
            rospy.loginfo("Toggling walking mode off.")
            self.toggle_button0()

        self.zero_for(1.0, writer, "final_zero")

        f.close()
        rospy.loginfo("Calibration finished: %s", self.path)
        print(self.path)


if __name__ == "__main__":
    rospy.init_node("tracer_calib_sync")
    SyncCalibration().run()
