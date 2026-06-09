#!/usr/bin/env python
from __future__ import print_function

import os
import sys

_THIS = os.path.abspath(os.path.dirname(__file__))
_REPO = os.path.abspath(os.path.join(_THIS, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import rospy
from sensor_msgs.msg import Joy
from nav_msgs.msg import Odometry

from tracer_core.mapper.theta_to_ref import ThetaToRefMapper
from tracer_core.planner.rule_based_high_level import RuleBasedHighLevelPlanner


class GazeboJoyAdapter(object):
    """
    Converts RefCommand to the A1-QP-MPC /joy message.
    """

    def to_joy(self, ref, force_toggle=False):
        msg = Joy()
        msg.header.stamp = rospy.Time.now()

        msg.axes = [
            float(ref.yaw_rate),     # axes[0]: yaw command
            float(ref.height_rate),  # axes[1]: body height velocity
            float(ref.vy),           # axes[2]: lateral velocity
            0.0,                     # axes[3]
            0.0,                     # axes[4]
            float(ref.vx),           # axes[5]: forward velocity
            0.0,                     # axes[6]: roll rate
            0.0,                     # axes[7]: pitch rate
        ]

        button0 = 1 if force_toggle else 0
        msg.buttons = [button0, 0, 0, 0, 0]
        return msg


class TracerCmdBridge(object):
    def __init__(self):
        self.state = {}
        self.received_odom = False

        self.rate_hz = rospy.get_param("~rate", 20.0)
        self.nominal_vx_axis = rospy.get_param("~nominal_vx_axis", 0.03)
        self.conservative_vx_axis = rospy.get_param("~conservative_vx_axis", 0.01)
        self.vx_max = rospy.get_param("~vx_max", 0.05)
        self.start_in_walk_mode = rospy.get_param("~start_in_walk_mode", True)

        self.planner = RuleBasedHighLevelPlanner(
            nominal_vx_axis=self.nominal_vx_axis,
            conservative_vx_axis=self.conservative_vx_axis,
        )
        self.mapper = ThetaToRefMapper(vx_max=self.vx_max)
        self.adapter = GazeboJoyAdapter()

        self.pub = rospy.Publisher("/joy", Joy, queue_size=10)
        self.sub = rospy.Subscriber(
            "/body_pose_ground_truth",
            Odometry,
            self.odom_callback,
            queue_size=10,
        )

    def odom_callback(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        v = msg.twist.twist.linear
        w = msg.twist.twist.angular

        self.state = {
            "base_height": p.z,
            "position": (p.x, p.y, p.z),
            "orientation": (q.x, q.y, q.z, q.w),
            "linear_velocity": (v.x, v.y, v.z),
            "angular_velocity": (w.x, w.y, w.z),
        }
        self.received_odom = True

    def publish_zero(self):
        msg = Joy()
        msg.header.stamp = rospy.Time.now()
        msg.axes = [0.0] * 8
        msg.buttons = [0, 0, 0, 0, 0]
        self.pub.publish(msg)

    def toggle_walk_once(self):
        rospy.loginfo("TRACER bridge: toggling walking mode once.")

        ref_zero = self.mapper.map({
            "walk": True,
            "target_vx_axis": 0.0,
            "target_yaw_axis": 0.0,
            "emergency_stop": False,
        })

        self.pub.publish(self.adapter.to_joy(ref_zero, force_toggle=True))
        rospy.sleep(0.2)
        self.pub.publish(self.adapter.to_joy(ref_zero, force_toggle=False))
        rospy.sleep(0.2)

    def run(self):
        rate = rospy.Rate(self.rate_hz)

        rospy.loginfo("TRACER bridge started. Waiting for odometry...")
        while not rospy.is_shutdown() and not self.received_odom:
            self.publish_zero()
            rate.sleep()

        rospy.loginfo("TRACER bridge received odometry.")

        if self.start_in_walk_mode:
            self.toggle_walk_once()

        while not rospy.is_shutdown():
            theta = self.planner.compute_theta(self.state)
            ref = self.mapper.map(theta)
            msg = self.adapter.to_joy(ref, force_toggle=False)
            self.pub.publish(msg)

            rospy.loginfo_throttle(
                1.0,
                "TRACER closed-loop cmd | z=%.3f vx_axis=%.3f walk=%s estop=%s",
                self.state.get("base_height", -1.0),
                ref.vx,
                str(ref.walk),
                str(ref.emergency_stop),
            )

            rate.sleep()


if __name__ == "__main__":
    rospy.init_node("tracer_cmd_bridge")
    TracerCmdBridge().run()
