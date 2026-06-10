#!/usr/bin/env python
from __future__ import print_function

import rospy
from sensor_msgs.msg import Joy


def make_msg(vx_axis, yaw_axis, toggle=False):
    msg = Joy()
    msg.header.stamp = rospy.Time.now()

    # A1-QP-MPC Gazebo joy mapping:
    # axes[5] -> forward velocity command
    # axes[0] -> yaw command
    msg.axes = [0.0] * 8
    msg.axes[5] = vx_axis
    msg.axes[0] = yaw_axis

    msg.buttons = [0, 0, 0, 0, 0]
    if toggle:
        msg.buttons[0] = 1

    return msg


if __name__ == "__main__":
    rospy.init_node("tracer_calib_cmd")

    vx_axis = rospy.get_param("~vx_axis", 0.0)
    yaw_axis = rospy.get_param("~yaw_axis", 0.0)
    duration = rospy.get_param("~duration", 15.0)
    rate_hz = rospy.get_param("~rate", 20.0)
    toggle_walk = rospy.get_param("~toggle_walk", False)
    toggle_stand_on_exit = rospy.get_param("~toggle_stand_on_exit", True)

    pub = rospy.Publisher("/joy", Joy, queue_size=10)
    rospy.sleep(0.5)

    rate = rospy.Rate(rate_hz)

    if toggle_walk:
        rospy.loginfo("Calibration: toggling walking mode on.")
        pub.publish(make_msg(0.0, 0.0, toggle=True))
        rospy.sleep(0.25)
        pub.publish(make_msg(0.0, 0.0, toggle=False))
        rospy.sleep(1.0)

    rospy.loginfo(
        "Calibration command start: vx_axis=%.3f yaw_axis=%.3f duration=%.2f",
        vx_axis, yaw_axis, duration
    )

    t0 = rospy.Time.now().to_sec()
    while not rospy.is_shutdown():
        t = rospy.Time.now().to_sec()
        if t - t0 > duration:
            break

        pub.publish(make_msg(vx_axis, yaw_axis, toggle=False))
        rate.sleep()

    rospy.loginfo("Calibration command done. Sending zero command.")
    for _ in range(20):
        pub.publish(make_msg(0.0, 0.0, toggle=False))
        rate.sleep()

    if toggle_stand_on_exit:
        rospy.loginfo("Calibration: toggling walking mode off.")
        pub.publish(make_msg(0.0, 0.0, toggle=True))
        rospy.sleep(0.25)
        pub.publish(make_msg(0.0, 0.0, toggle=False))
        rospy.sleep(0.5)

    for _ in range(20):
        pub.publish(make_msg(0.0, 0.0, toggle=False))
        rate.sleep()

    rospy.loginfo("Calibration command finished.")
