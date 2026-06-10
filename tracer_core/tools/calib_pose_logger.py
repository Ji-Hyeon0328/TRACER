#!/usr/bin/env python
from __future__ import print_function

import os
import csv
import math
import time
import rospy
from nav_msgs.msg import Odometry


def quat_to_yaw(x, y, z, w):
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


class PoseLogger(object):
    def __init__(self):
        self.log_dir = rospy.get_param("~log_dir", "/root/TRACER/logs")
        self.duration = rospy.get_param("~duration", 20.0)
        self.tag = rospy.get_param("~tag", "calib")

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(
            self.log_dir,
            "tracer_calib_%s_%s.csv" % (self.tag, stamp)
        )

        self.f = open(self.path, "w")
        self.w = csv.writer(self.f)
        self.w.writerow([
            "ros_time",
            "x", "y", "z",
            "yaw",
            "vx", "vy", "vz",
            "wz",
        ])

        self.sub = rospy.Subscriber(
            "/body_pose_ground_truth",
            Odometry,
            self.cb,
            queue_size=50
        )

        self.got = False

    def cb(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        v = msg.twist.twist.linear
        w = msg.twist.twist.angular

        yaw = quat_to_yaw(q.x, q.y, q.z, q.w)

        self.w.writerow([
            rospy.Time.now().to_sec(),
            p.x, p.y, p.z,
            yaw,
            v.x, v.y, v.z,
            w.z,
        ])
        self.got = True

    def run(self):
        rospy.loginfo("Calibration logger writing: %s", self.path)
        t0 = rospy.Time.now().to_sec()
        rate = rospy.Rate(20)

        while not rospy.is_shutdown():
            t = rospy.Time.now().to_sec()
            if t - t0 > self.duration:
                break
            rate.sleep()

        self.f.close()
        rospy.loginfo("Calibration logger done: %s", self.path)
        print(self.path)


if __name__ == "__main__":
    rospy.init_node("tracer_calib_pose_logger")
    PoseLogger().run()
