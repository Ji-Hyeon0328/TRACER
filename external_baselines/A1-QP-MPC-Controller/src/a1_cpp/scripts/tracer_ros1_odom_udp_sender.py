#!/usr/bin/env python
from __future__ import print_function

import math
import socket
import struct
import time

import rospy
from nav_msgs.msg import Odometry


def quat_to_yaw(q):
    # q: geometry_msgs/Quaternion, xyzw
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class Ros1OdomUdpSender(object):
    def __init__(self):
        rospy.init_node("tracer_ros1_odom_udp_sender")

        self.odom_topic = rospy.get_param("~odom_topic", "/gazebo_a1/estimation_body_pose")
        self.udp_ip = rospy.get_param("~udp_ip", "127.0.0.1")
        self.udp_port = int(rospy.get_param("~udp_port", 50120))

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.packet_fmt = "<6d"

        self.last_log_wall = 0.0

        self.sub = rospy.Subscriber(self.odom_topic, Odometry, self.odom_callback, queue_size=10)

        rospy.loginfo(
            "[TRACER] ROS1 odom UDP sender: %s -> %s:%d",
            self.odom_topic, self.udp_ip, self.udp_port
        )

    def odom_callback(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        v = msg.twist.twist.linear

        yaw = quat_to_yaw(q)

        # Layout:
        # [stamp_wall, x_world, y_world, yaw_world, vx_world, vy_world]
        values = [
            time.time(),
            float(p.x),
            float(p.y),
            float(yaw),
            float(v.x),
            float(v.y),
        ]

        packet = struct.pack(self.packet_fmt, *values)
        self.sock.sendto(packet, (self.udp_ip, self.udp_port))

        now = time.time()
        if now - self.last_log_wall > 1.0:
            rospy.loginfo(
                "[TRACER] ROS1 odom->UDP x=%.3f y=%.3f yaw=%.3f vx=%.3f vy=%.3f",
                values[1], values[2], values[3], values[4], values[5]
            )
            self.last_log_wall = now


def main():
    node = Ros1OdomUdpSender()
    rospy.spin()


if __name__ == "__main__":
    main()
