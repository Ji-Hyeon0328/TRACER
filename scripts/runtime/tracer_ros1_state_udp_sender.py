#!/usr/bin/env python3
import json
import math
import socket
import threading
import time

import rospy
from sensor_msgs.msg import JointState, Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import WrenchStamped


UDP_IP = rospy.get_param("~udp_ip", "127.0.0.1")
UDP_PORT = int(rospy.get_param("~udp_port", 50130))
PUB_HZ = float(rospy.get_param("~pub_hz", 50.0))

JOINT_TOPIC = rospy.get_param("~joint_topic", "/a1_gazebo/joint_states")
IMU_TOPIC = rospy.get_param("~imu_topic", "/trunk_imu")
ODOM_TOPIC = rospy.get_param("~odom_topic", "/torso_odom")

FOOT_TOPICS = {
    "FL": rospy.get_param("~fl_contact_topic", "/visual/FL_foot_contact/the_force"),
    "FR": rospy.get_param("~fr_contact_topic", "/visual/FR_foot_contact/the_force"),
    "RL": rospy.get_param("~rl_contact_topic", "/visual/RL_foot_contact/the_force"),
    "RR": rospy.get_param("~rr_contact_topic", "/visual/RR_foot_contact/the_force"),
}

CONTACT_FORCE_THRESHOLD = float(rospy.get_param("~contact_force_threshold", 5.0))

# Canonical order used by TRACER logs.
CANONICAL_JOINTS = [
    "FL_hip", "FL_thigh", "FL_calf",
    "FR_hip", "FR_thigh", "FR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
    "RR_hip", "RR_thigh", "RR_calf",
]

lock = threading.Lock()
latest = {
    "joint_names": [],
    "q_by_name": {},
    "dq_by_name": {},
    "base_z": 0.0,
    "base_roll": 0.0,
    "base_pitch": 0.0,
    "base_yaw": 0.0,
    "contacts": {k: 0.0 for k in FOOT_TOPICS},
    "contact_forces": {k: 0.0 for k in FOOT_TOPICS},
    "stamp": 0.0,
}


def quat_to_rpy(x, y, z, w):
    # ROS quaternion -> roll, pitch, yaw.
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


def canonicalize_joint_name(name):
    # Accept common Gazebo joint names like FL_hip_joint or FL_hip.
    n = name
    n = n.replace("_joint", "")
    n = n.replace("a1_gazebo::", "")
    return n


def on_joint(msg):
    with lock:
        latest["joint_names"] = [canonicalize_joint_name(n) for n in msg.name]
        for i, name in enumerate(latest["joint_names"]):
            if i < len(msg.position):
                latest["q_by_name"][name] = float(msg.position[i])
            if i < len(msg.velocity):
                latest["dq_by_name"][name] = float(msg.velocity[i])
        latest["stamp"] = time.time()


def on_imu(msg):
    q = msg.orientation
    roll, pitch, yaw = quat_to_rpy(q.x, q.y, q.z, q.w)
    with lock:
        latest["base_roll"] = float(roll)
        latest["base_pitch"] = float(pitch)
        latest["base_yaw"] = float(yaw)
        latest["stamp"] = time.time()


def on_odom(msg):
    with lock:
        latest["base_z"] = float(msg.pose.pose.position.z)
        latest["stamp"] = time.time()


def make_contact_cb(leg):
    def cb(msg):
        f = msg.wrench.force
        mag = math.sqrt(f.x * f.x + f.y * f.y + f.z * f.z)
        with lock:
            latest["contact_forces"][leg] = float(mag)
            latest["contacts"][leg] = 1.0 if mag >= CONTACT_FORCE_THRESHOLD else 0.0
            latest["stamp"] = time.time()
    return cb


def get_joint_value(d, key):
    # Try canonical first, then tolerate names ending with canonical key.
    if key in d:
        return float(d[key])
    for k, v in d.items():
        if k.endswith(key):
            return float(v)
    return 0.0


def main():
    rospy.init_node("tracer_ros1_state_udp_sender", anonymous=False)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    rospy.Subscriber(JOINT_TOPIC, JointState, on_joint, queue_size=1)
    rospy.Subscriber(IMU_TOPIC, Imu, on_imu, queue_size=1)
    rospy.Subscriber(ODOM_TOPIC, Odometry, on_odom, queue_size=1)

    for leg, topic in FOOT_TOPICS.items():
        rospy.Subscriber(topic, WrenchStamped, make_contact_cb(leg), queue_size=1)

    sleep_dt = 1.0 / max(PUB_HZ, 1e-6)
    seq = 0

    rospy.loginfo("TRACER ROS1 state UDP sender -> %s:%d", UDP_IP, UDP_PORT)
    rospy.loginfo("joint_topic=%s imu_topic=%s odom_topic=%s", JOINT_TOPIC, IMU_TOPIC, ODOM_TOPIC)

    while not rospy.is_shutdown():
        with lock:
            q = [get_joint_value(latest["q_by_name"], name) for name in CANONICAL_JOINTS]
            dq = [get_joint_value(latest["dq_by_name"], name) for name in CANONICAL_JOINTS]
            contacts = [latest["contacts"][leg] for leg in ["FL", "FR", "RL", "RR"]]
            contact_forces = [latest["contact_forces"][leg] for leg in ["FL", "FR", "RL", "RR"]]

            packet = {
                "schema": "tracer_robot_state_flat_v0",
                "seq": seq,
                "stamp": time.time(),
                "joint_order": CANONICAL_JOINTS,
                "contact_order": ["FL", "FR", "RL", "RR"],
                "base_z": latest["base_z"],
                "base_roll": latest["base_roll"],
                "base_pitch": latest["base_pitch"],
                "base_yaw": latest["base_yaw"],
                "q": q,
                "dq": dq,
                "contacts": contacts,
                "contact_forces": contact_forces,
                "raw_joint_names": latest["joint_names"],
            }

        data = json.dumps(packet, separators=(",", ":")).encode("utf-8")
        sock.sendto(data, (UDP_IP, UDP_PORT))
        seq += 1
        time.sleep(sleep_dt)


if __name__ == "__main__":
    main()
