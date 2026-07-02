#!/usr/bin/env python
from __future__ import print_function

import math
import socket
import struct
import time

import rospy
from gazebo_msgs.msg import ModelStates


def quat_to_yaw(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class TracerRos1ModelStatesOdomUdpSender(object):
    def __init__(self):
        rospy.init_node("tracer_ros1_model_states_odom_udp_sender")

        self.model_name = rospy.get_param("~model_name", "a1_gazebo")
        self.topic = rospy.get_param("~model_states_topic", "/gazebo/model_states")
        self.udp_ip = rospy.get_param("~udp_ip", "127.0.0.1")
        self.udp_port = int(rospy.get_param("~udp_port", 50120))

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.packet_fmt = "<6d"
        self.last_log_wall = 0.0

        self.sub = rospy.Subscriber(self.topic, ModelStates, self.cb, queue_size=10)

        rospy.loginfo(
            "[TRACER] model_states odom UDP sender: topic=%s model=%s -> %s:%d",
            self.topic, self.model_name, self.udp_ip, self.udp_port
        )

    def cb(self, msg):
        try:
            idx = list(msg.name).index(self.model_name)
        except ValueError:
            rospy.logwarn_throttle(
                1.0,
                "[TRACER] model %s not found in /gazebo/model_states names=%s",
                self.model_name,
                list(msg.name)[:10],
            )
            return

        pose = msg.pose[idx]
        twist = msg.twist[idx]

        p = pose.position
        q = pose.orientation
        v = twist.linear

        yaw = quat_to_yaw(q)

        # UDP payload expected by tracer_udp_odom_to_ros2_node.py:
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
                "[TRACER] model_states odom->UDP model=%s x=%.3f y=%.3f yaw=%.3f vx=%.3f vy=%.3f",
                self.model_name,
                values[1],
                values[2],
                values[3],
                values[4],
                values[5],
            )
            self.last_log_wall = now


def main():
    TracerRos1ModelStatesOdomUdpSender()
    rospy.spin()


if __name__ == "__main__":
    main()
