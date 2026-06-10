#!/usr/bin/env python
from __future__ import print_function

import os
import csv
import math
import time

import rospy
from sensor_msgs.msg import Joy
from nav_msgs.msg import Odometry


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def wrap_to_pi(a):
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


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


class GoalTracker(object):
    def __init__(self):
        self.goal_x = rospy.get_param("~goal_x", 1.0)
        self.goal_y = rospy.get_param("~goal_y", 0.0)
        self.relative_goal = rospy.get_param("~relative_goal", False)
        self.relative_goal_x = rospy.get_param("~relative_goal_x", 1.0)
        self.relative_goal_y = rospy.get_param("~relative_goal_y", 0.0)
        self.goal_initialized = False

        self.max_vx_axis = rospy.get_param("~max_vx_axis", 0.12)
        self.max_yaw_axis = rospy.get_param("~max_yaw_axis", 0.25)

        self.k_v = rospy.get_param("~k_v", 0.20)
        self.k_yaw = rospy.get_param("~k_yaw", 0.35)

        self.goal_tolerance = rospy.get_param("~goal_tolerance", 0.15)
        self.yaw_slowdown_angle = rospy.get_param("~yaw_slowdown_angle", 0.80)

        self.duration_limit = rospy.get_param("~duration_limit", 60.0)
        self.rate_hz = rospy.get_param("~rate", 20.0)

        self.toggle_walk = rospy.get_param("~toggle_walk", True)
        self.toggle_stand_on_exit = rospy.get_param("~toggle_stand_on_exit", True)

        self.log_enable = rospy.get_param("~log_enable", True)
        self.log_dir = rospy.get_param("~log_dir", "/root/TRACER/logs")
        self.tag = rospy.get_param("~tag", "goal")

        self.latest_odom = None
        self.reached = False

        self.pub = rospy.Publisher("/joy", Joy, queue_size=10)
        self.sub = rospy.Subscriber(
            "/body_pose_ground_truth",
            Odometry,
            self.odom_cb,
            queue_size=50,
        )

        self.file = None
        self.writer = None

        if self.log_enable:
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)

            stamp = time.strftime("%Y%m%d_%H%M%S")
            self.log_path = os.path.join(
                self.log_dir,
                "tracer_goal_%s_%s.csv" % (self.tag, stamp)
            )
            self.file = open(self.log_path, "w")
            self.writer = csv.writer(self.file)
            self.writer.writerow([
                "ros_time",
                "phase",
                "goal_x",
                "goal_y",
                "x",
                "y",
                "z",
                "yaw",
                "distance",
                "heading_error",
                "vx_axis",
                "yaw_axis",
                "reached",
            ])
        else:
            self.log_path = ""

    def odom_cb(self, msg):
        self.latest_odom = msg

    def toggle_button0(self):
        self.pub.publish(make_joy(0.0, 0.0, toggle=True))
        rospy.sleep(0.25)
        self.pub.publish(make_joy(0.0, 0.0, toggle=False))
        rospy.sleep(0.25)

    def zero_for(self, sec, phase="zero"):
        rate = rospy.Rate(self.rate_hz)
        t0 = rospy.Time.now().to_sec()
        while not rospy.is_shutdown():
            if rospy.Time.now().to_sec() - t0 > sec:
                break
            self.pub.publish(make_joy(0.0, 0.0, toggle=False))
            self.log_row(phase, 0.0, 0.0, 0.0, 0.0)
            rate.sleep()

    def get_pose(self):
        if self.latest_odom is None:
            return None

        msg = self.latest_odom
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = quat_to_yaw(q.x, q.y, q.z, q.w)

        return p.x, p.y, p.z, yaw

    def compute_cmd(self, x, y, yaw):
        dx = self.goal_x - x
        dy = self.goal_y - y
        dist = math.sqrt(dx * dx + dy * dy)

        desired_heading = math.atan2(dy, dx)
        heading_error = wrap_to_pi(desired_heading - yaw)

        if dist < self.goal_tolerance:
            return dist, heading_error, 0.0, 0.0, True

        yaw_axis = clamp(self.k_yaw * heading_error,
                         -self.max_yaw_axis,
                         self.max_yaw_axis)

        # If heading error is large, reduce forward motion and rotate first.
        yaw_factor = max(0.0, 1.0 - abs(heading_error) / max(1e-6, self.yaw_slowdown_angle))

        vx_axis = clamp(self.k_v * dist, 0.0, self.max_vx_axis)
        vx_axis = vx_axis * yaw_factor

        return dist, heading_error, vx_axis, yaw_axis, False

    def log_row(self, phase, dist, heading_error, vx_axis, yaw_axis):
        if self.writer is None or self.latest_odom is None:
            return

        pose = self.get_pose()
        if pose is None:
            return

        x, y, z, yaw = pose

        self.writer.writerow([
            rospy.Time.now().to_sec(),
            phase,
            self.goal_x,
            self.goal_y,
            x,
            y,
            z,
            yaw,
            dist,
            heading_error,
            vx_axis,
            yaw_axis,
            str(self.reached),
        ])

    def run(self):
        rospy.loginfo("Waiting for odometry...")
        while not rospy.is_shutdown() and self.latest_odom is None:
            rospy.sleep(0.05)

        if self.relative_goal:
            pose = self.get_pose()
            if pose is not None:
                x0, y0, z0, yaw0 = pose
                # Relative goal is expressed in the robot's initial body heading frame.
                self.goal_x = x0 + self.relative_goal_x * math.cos(yaw0) - self.relative_goal_y * math.sin(yaw0)
                self.goal_y = y0 + self.relative_goal_x * math.sin(yaw0) + self.relative_goal_y * math.cos(yaw0)
                self.goal_initialized = True

        rospy.loginfo(
            "GoalTracker start: goal=(%.3f, %.3f), relative=%s",
            self.goal_x,
            self.goal_y,
            str(self.relative_goal),
        )

        self.zero_for(1.0, "pre_zero")

        if self.toggle_walk:
            rospy.loginfo("GoalTracker: toggling walking mode on.")
            self.toggle_button0()

        self.zero_for(2.0, "warmup")

        rate = rospy.Rate(self.rate_hz)
        t0 = rospy.Time.now().to_sec()

        while not rospy.is_shutdown():
            if rospy.Time.now().to_sec() - t0 > self.duration_limit:
                rospy.loginfo("GoalTracker: duration limit reached.")
                break

            pose = self.get_pose()
            if pose is None:
                rate.sleep()
                continue

            x, y, z, yaw = pose
            dist, heading_error, vx_axis, yaw_axis, reached = self.compute_cmd(x, y, yaw)
            self.reached = reached

            if reached:
                rospy.loginfo("GoalTracker: goal reached. dist=%.3f", dist)
                self.pub.publish(make_joy(0.0, 0.0, toggle=False))
                self.log_row("reached", dist, heading_error, 0.0, 0.0)
                break

            self.pub.publish(make_joy(vx_axis, yaw_axis, toggle=False))
            self.log_row("track", dist, heading_error, vx_axis, yaw_axis)

            rospy.loginfo_throttle(
                1.0,
                "GoalTracker | dist=%.3f heading_err=%.3f vx=%.3f yaw=%.3f",
                dist,
                heading_error,
                vx_axis,
                yaw_axis,
            )

            rate.sleep()

        self.zero_for(1.0, "post_zero")

        if self.toggle_stand_on_exit:
            rospy.loginfo("GoalTracker: toggling walking mode off.")
            self.toggle_button0()

        self.zero_for(1.0, "final_zero")

        if self.file is not None:
            self.file.close()
            rospy.loginfo("GoalTracker log: %s", self.log_path)
            print(self.log_path)


if __name__ == "__main__":
    rospy.init_node("tracer_goal_tracker")
    GoalTracker().run()
