#!/usr/bin/env python
from __future__ import print_function

import socket
import struct
import time

import rospy
from std_msgs.msg import Float64MultiArray


def main():
    rospy.init_node("tracer_udp_to_ros1_mpc_ref")

    udp_ip = rospy.get_param("~udp_ip", "0.0.0.0")
    udp_port = int(rospy.get_param("~udp_port", 50110))
    topic = rospy.get_param("~topic", "/tracer/mpc_reference")

    pub = rospy.Publisher(topic, Float64MultiArray, queue_size=10)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((udp_ip, udp_port))
    sock.settimeout(0.05)

    packet_fmt = "<6d"
    packet_size = struct.calcsize(packet_fmt)

    rospy.loginfo("[TRACER] UDP receiver listening on %s:%d -> %s", udp_ip, udp_port, topic)
    rospy.loginfo("[TRACER] Using wall-clock loop, independent of /clock and /use_sim_time")

    last_log_wall = 0.0

    while not rospy.is_shutdown():
        try:
            data, addr = sock.recvfrom(1024)
        except socket.timeout:
            time.sleep(0.001)
            continue

        if len(data) < packet_size:
            now = time.time()
            if now - last_log_wall > 1.0:
                rospy.logwarn("[TRACER] short UDP packet: %d bytes", len(data))
                last_log_wall = now
            continue

        values = struct.unpack(packet_fmt, data[:packet_size])

        msg = Float64MultiArray()
        msg.data = list(values)
        pub.publish(msg)

        now = time.time()
        if now - last_log_wall > 1.0:
            rospy.loginfo(
                "[TRACER] UDP->ROS1 from %s:%d counter=%.0f vx=%.3f yaw=%.3f h=%.3f clr=%.3f enable=%.1f",
                addr[0], addr[1],
                values[0], values[1], values[2], values[3], values[4], values[5]
            )
            last_log_wall = now


if __name__ == "__main__":
    main()
