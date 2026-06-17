#!/usr/bin/env python
from __future__ import print_function

import math
import socket
import struct
import time
import threading

import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, JointState
from geometry_msgs.msg import WrenchStamped


def quat_to_rpy(q):
    x = q.x
    y = q.y
    z = q.z
    w = q.w

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


class TracerRos1ProprioUdpSender(object):
    def __init__(self):
        rospy.init_node("tracer_ros1_proprio_udp_sender")

        self.odom_topic = rospy.get_param("~odom_topic", "/torso_odom")
        self.imu_topic = rospy.get_param("~imu_topic", "/trunk_imu")
        self.joint_topic = rospy.get_param("~joint_topic", "/a1_gazebo/joint_states")

        self.udp_ip = rospy.get_param("~udp_ip", "127.0.0.1")
        self.udp_port = int(rospy.get_param("~udp_port", 50130))
        self.publish_hz = float(rospy.get_param("~publish_hz", 50.0))
        self.contact_threshold = float(rospy.get_param("~contact_threshold", 5.0))

        self.contact_topics = [
            rospy.get_param("~contact_fl_topic", "/visual/FL_foot_contact/the_force"),
            rospy.get_param("~contact_fr_topic", "/visual/FR_foot_contact/the_force"),
            rospy.get_param("~contact_rl_topic", "/visual/RL_foot_contact/the_force"),
            rospy.get_param("~contact_rr_topic", "/visual/RR_foot_contact/the_force"),
        ]

        self.joint_order = [
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        ]

        self.lock = threading.Lock()

        self.odom = None
        self.imu = None
        self.joint = None
        self.contact_force_z = [0.0, 0.0, 0.0, 0.0]
        self.contact_binary = [0.0, 0.0, 0.0, 0.0]

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # 45 doubles
        self.packet_fmt = "<45d"

        rospy.Subscriber(self.odom_topic, Odometry, self.odom_callback, queue_size=10)
        rospy.Subscriber(self.imu_topic, Imu, self.imu_callback, queue_size=10)
        rospy.Subscriber(self.joint_topic, JointState, self.joint_callback, queue_size=10)

        for i, topic in enumerate(self.contact_topics):
            rospy.Subscriber(topic, WrenchStamped, self.make_contact_callback(i), queue_size=10)

        rospy.loginfo("[TRACER] ROS1 proprio UDP sender")
        rospy.loginfo("[TRACER] odom:   %s", self.odom_topic)
        rospy.loginfo("[TRACER] imu:    %s", self.imu_topic)
        rospy.loginfo("[TRACER] joint:  %s", self.joint_topic)
        rospy.loginfo("[TRACER] contact topics: %s", str(self.contact_topics))
        rospy.loginfo("[TRACER] UDP -> %s:%d", self.udp_ip, self.udp_port)

    def odom_callback(self, msg):
        with self.lock:
            self.odom = msg

    def imu_callback(self, msg):
        with self.lock:
            self.imu = msg

    def joint_callback(self, msg):
        with self.lock:
            self.joint = msg

    def make_contact_callback(self, idx):
        def callback(msg):
            fz = float(msg.wrench.force.z)
            contact = 1.0 if abs(fz) > self.contact_threshold else 0.0
            with self.lock:
                self.contact_force_z[idx] = fz
                self.contact_binary[idx] = contact
        return callback

    def extract_joints(self, joint_msg):
        q = [0.0] * 12
        dq = [0.0] * 12

        if joint_msg is None:
            return q, dq

        name_to_idx = {}
        for i, name in enumerate(joint_msg.name):
            name_to_idx[name] = i

        # Preferred: name-based mapping
        all_names_exist = True
        for name in self.joint_order:
            if name not in name_to_idx:
                all_names_exist = False
                break

        if all_names_exist:
            for out_i, name in enumerate(self.joint_order):
                src_i = name_to_idx[name]
                if src_i < len(joint_msg.position):
                    q[out_i] = float(joint_msg.position[src_i])
                if src_i < len(joint_msg.velocity):
                    dq[out_i] = float(joint_msg.velocity[src_i])
            return q, dq

        # Fallback: first 12 joints in incoming order
        for i in range(min(12, len(joint_msg.position))):
            q[i] = float(joint_msg.position[i])
        for i in range(min(12, len(joint_msg.velocity))):
            dq[i] = float(joint_msg.velocity[i])

        return q, dq

    def build_vector(self):
        with self.lock:
            odom = self.odom
            imu = self.imu
            joint = self.joint
            contact_binary = list(self.contact_binary)
            contact_force_z = list(self.contact_force_z)

        stamp_wall = time.time()

        base_x = base_y = base_z = 0.0
        roll = pitch = yaw = 0.0
        vx = vy = vz = 0.0
        wx = wy = wz = 0.0

        if odom is not None:
            p = odom.pose.pose.position
            q = odom.pose.pose.orientation
            v = odom.twist.twist.linear
            w = odom.twist.twist.angular

            base_x = float(p.x)
            base_y = float(p.y)
            base_z = float(p.z)

            roll, pitch, yaw = quat_to_rpy(q)

            vx = float(v.x)
            vy = float(v.y)
            vz = float(v.z)

            wx = float(w.x)
            wy = float(w.y)
            wz = float(w.z)

        # Prefer IMU angular velocity when available
        if imu is not None:
            wx = float(imu.angular_velocity.x)
            wy = float(imu.angular_velocity.y)
            wz = float(imu.angular_velocity.z)

        joint_pos, joint_vel = self.extract_joints(joint)

        values = [
            stamp_wall,
            base_x, base_y, base_z,
            roll, pitch, yaw,
            vx, vy, vz,
            wx, wy, wz,
        ]

        values.extend(joint_pos)
        values.extend(joint_vel)
        values.extend(contact_binary)
        values.extend(contact_force_z)

        # Safety check
        if len(values) != 45:
            rospy.logerr("[TRACER] proprio vector length mismatch: %d", len(values))
            return None

        return values

    def run(self):
        rate = rospy.Rate(self.publish_hz)
        last_log_wall = 0.0

        while not rospy.is_shutdown():
            values = self.build_vector()

            if values is not None:
                packet = struct.pack(self.packet_fmt, *values)
                self.sock.sendto(packet, (self.udp_ip, self.udp_port))

                now = time.time()
                if now - last_log_wall > 1.0:
                    rospy.loginfo(
                        "[TRACER] proprio->UDP pos=(%.2f,%.2f,%.2f) rpy=(%.2f,%.2f,%.2f) "
                        "v=(%.2f,%.2f,%.2f) contact=%s",
                        values[1], values[2], values[3],
                        values[4], values[5], values[6],
                        values[7], values[8], values[9],
                        str([int(x) for x in values[37:41]])
                    )
                    last_log_wall = now

            rate.sleep()


def main():
    node = TracerRos1ProprioUdpSender()
    node.run()


if __name__ == "__main__":
    main()
