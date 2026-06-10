#!/usr/bin/env python
from __future__ import print_function

import rospy
from sensor_msgs.msg import Joy


def make_msg(button0=0):
    msg = Joy()
    msg.header.stamp = rospy.Time.now()
    msg.axes = [0.0] * 8
    msg.buttons = [button0, 0, 0, 0, 0]
    return msg


if __name__ == "__main__":
    rospy.init_node("tracer_stop")

    pub = rospy.Publisher("/joy", Joy, queue_size=10)
    rospy.sleep(0.5)

    toggle = rospy.get_param("~toggle_stand", True)

    rate = rospy.Rate(20)

    # Zero command first.
    for _ in range(10):
        pub.publish(make_msg(button0=0))
        rate.sleep()

    # Toggle walking mode once if requested.
    # Use this only when the robot is known to be in walking mode.
    if toggle:
        rospy.loginfo("TRACER stop: toggling button[0] once.")
        pub.publish(make_msg(button0=1))
        rospy.sleep(0.25)
        pub.publish(make_msg(button0=0))
        rospy.sleep(0.25)

    # Zero command again.
    for _ in range(20):
        pub.publish(make_msg(button0=0))
        rate.sleep()

    rospy.loginfo("TRACER stop: done.")
