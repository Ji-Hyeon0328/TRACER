#!/usr/bin/env python
from __future__ import print_function

import csv
import math
import sys
import time

import rospy

from gazebo_msgs.msg import ModelStates
from std_msgs.msg import Float64MultiArray
from unitree_legged_msgs.msg import MotorCmd, MotorState


JOINT_NAMES = [
    "FL_hip",
    "FL_thigh",
    "FL_calf",
    "FR_hip",
    "FR_thigh",
    "FR_calf",
    "RL_hip",
    "RL_thigh",
    "RL_calf",
    "RR_hip",
    "RR_thigh",
    "RR_calf",
]


def finite(value):
    try:
        value = float(value)

        if math.isnan(value) or math.isinf(value):
            return float("nan")

        return value

    except Exception:
        return float("nan")


def fmt(value):
    value = finite(value)

    if math.isnan(value):
        return "nan"

    return "%.10g" % value


def quaternion_to_euler(quaternion):
    sinr_cosp = 2.0 * (
        quaternion.w * quaternion.x
        + quaternion.y * quaternion.z
    )

    cosr_cosp = 1.0 - 2.0 * (
        quaternion.x * quaternion.x
        + quaternion.y * quaternion.y
    )

    roll = math.atan2(
        sinr_cosp,
        cosr_cosp,
    )

    sinp = 2.0 * (
        quaternion.w * quaternion.y
        - quaternion.z * quaternion.x
    )

    if abs(sinp) >= 1.0:
        pitch = math.copysign(
            math.pi / 2.0,
            sinp,
        )
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (
        quaternion.w * quaternion.z
        + quaternion.x * quaternion.y
    )

    cosy_cosp = 1.0 - 2.0 * (
        quaternion.y * quaternion.y
        + quaternion.z * quaternion.z
    )

    yaw = math.atan2(
        siny_cosp,
        cosy_cosp,
    )

    return roll, pitch, yaw


class JointEnergyVXLogger(object):
    def __init__(self):
        self.publish_hz = float(
            rospy.get_param("~publish_hz", 50.0)
        )

        self.max_age_s = float(
            rospy.get_param("~max_age_s", 0.20)
        )

        self.ref_max_age_s = float(
            rospy.get_param("~ref_max_age_s", 0.75)
        )

        self.model_max_age_s = float(
            rospy.get_param("~model_max_age_s", 0.25)
        )

        self.model_name = str(
            rospy.get_param("~model_name", "a1_gazebo")
        )

        self.tau = dict(
            (name, None)
            for name in JOINT_NAMES
        )

        self.dq = dict(
            (name, None)
            for name in JOINT_NAMES
        )

        self.tau_stamp = dict(
            (name, None)
            for name in JOINT_NAMES
        )

        self.dq_stamp = dict(
            (name, None)
            for name in JOINT_NAMES
        )

        self.ref = None
        self.ref_stamp = None

        self.x = float("nan")
        self.y = float("nan")
        self.z = float("nan")

        self.roll_deg = float("nan")
        self.pitch_deg = float("nan")
        self.yaw_deg = float("nan")

        self.vx_world = float("nan")
        self.vy_world = float("nan")
        self.vx_body = float("nan")
        self.vy_body = float("nan")

        self.model_stamp = None
        self.model_index = None

        self.previous_time = None
        self.previous_valid = False

        self.work_signed = 0.0
        self.work_positive = 0.0
        self.work_abs = 0.0
        self.tau_sq_integral = 0.0
        self.dq_sq_integral = 0.0

        self.row_count = 0

        self.writer = csv.writer(
            sys.stdout,
            lineterminator="\n",
        )

        header = [
            "sim_time",
            "wall_time",
            "valid",
            "max_joint_age_s",
            "ref_age_s",
            "model_age_s",
            "sequence",
            "x",
            "y",
            "z",
            "roll_deg",
            "pitch_deg",
            "yaw_deg",
            "vx_world",
            "vy_world",
            "vx_body",
            "vy_body",
            "context",
            "vx_cmd",
            "yaw_cmd",
            "body_height_cmd",
            "clearance_cmd",
            "enable_cmd",
            "power_signed_W_cmd",
            "power_positive_W_cmd",
            "power_abs_W_cmd",
            "tau_sq_sum",
            "dq_sq_sum",
            "work_signed_J_cmd",
            "work_positive_J_cmd",
            "work_abs_J_cmd",
            "tau_sq_integral",
            "dq_sq_integral",
        ]

        for name in JOINT_NAMES:
            header.extend([
                name + "_tau_cmd",
                name + "_dq",
                name + "_power_cmd",
            ])

        self.writer.writerow(header)
        sys.stdout.flush()

        for name in JOINT_NAMES:
            base = (
                "/a1_gazebo/%s_controller"
                % name
            )

            rospy.Subscriber(
                base + "/command",
                MotorCmd,
                self.make_tau_callback(name),
                queue_size=5,
            )

            rospy.Subscriber(
                base + "/state",
                MotorState,
                self.make_dq_callback(name),
                queue_size=5,
            )

        rospy.Subscriber(
            "/tracer/mpc_reference",
            Float64MultiArray,
            self.on_reference,
            queue_size=10,
        )

        rospy.Subscriber(
            "/gazebo/model_states",
            ModelStates,
            self.on_model_states,
            queue_size=5,
        )

        rospy.Timer(
            rospy.Duration(
                1.0 / max(self.publish_hz, 1.0)
            ),
            self.on_timer,
        )

        rospy.on_shutdown(self.on_shutdown)

    def make_tau_callback(self, name):
        def callback(message):
            self.tau[name] = finite(message.tau)
            self.tau_stamp[name] = rospy.get_time()

        return callback

    def make_dq_callback(self, name):
        def callback(message):
            self.dq[name] = finite(message.dq)
            self.dq_stamp[name] = rospy.get_time()

        return callback

    def on_reference(self, message):
        data = list(message.data)

        if len(data) >= 6:
            self.ref = [
                finite(value)
                for value in data[:6]
            ]

            self.ref_stamp = rospy.get_time()

    def on_model_states(self, message):
        if (
            self.model_index is None
            or self.model_index >= len(message.name)
            or message.name[self.model_index]
            != self.model_name
        ):
            try:
                self.model_index = (
                    message.name.index(
                        self.model_name
                    )
                )
            except ValueError:
                self.model_index = None
                return

        pose = message.pose[self.model_index]
        twist = message.twist[self.model_index]

        roll, pitch, yaw = (
            quaternion_to_euler(
                pose.orientation
            )
        )

        vx_world = finite(twist.linear.x)
        vy_world = finite(twist.linear.y)

        self.x = finite(pose.position.x)
        self.y = finite(pose.position.y)
        self.z = finite(pose.position.z)

        self.roll_deg = math.degrees(roll)
        self.pitch_deg = math.degrees(pitch)
        self.yaw_deg = math.degrees(yaw)

        self.vx_world = vx_world
        self.vy_world = vy_world

        self.vx_body = (
            math.cos(yaw) * vx_world
            + math.sin(yaw) * vy_world
        )

        self.vy_body = (
            -math.sin(yaw) * vx_world
            + math.cos(yaw) * vy_world
        )

        self.model_stamp = rospy.get_time()

    def context_from_x(self):
        if math.isnan(self.x):
            return "unknown"

        if self.x < 2.0:
            return "flat"

        if self.x < 4.0:
            return "upslope"

        if self.x < 6.0:
            return "rough"

        if self.x < 8.0:
            return "downslope"

        return "goal_flat"

    def on_timer(self, _event):
        now = rospy.get_time()

        joint_stamps = []

        for name in JOINT_NAMES:
            if (
                self.tau[name] is None
                or self.dq[name] is None
                or self.tau_stamp[name] is None
                or self.dq_stamp[name] is None
            ):
                joint_stamps = []
                break

            joint_stamps.append(
                self.tau_stamp[name]
            )

            joint_stamps.append(
                self.dq_stamp[name]
            )

        if joint_stamps:
            joint_ages = [
                max(0.0, now - stamp)
                for stamp in joint_stamps
            ]

            max_joint_age = max(joint_ages)

            joints_valid = (
                max_joint_age
                <= self.max_age_s
            )
        else:
            max_joint_age = float("nan")
            joints_valid = False

        if (
            self.ref is not None
            and self.ref_stamp is not None
        ):
            ref_age = max(
                0.0,
                now - self.ref_stamp,
            )

            ref_valid = (
                ref_age
                <= self.ref_max_age_s
            )
        else:
            ref_age = float("nan")
            ref_valid = False

        if self.model_stamp is not None:
            model_age = max(
                0.0,
                now - self.model_stamp,
            )
        else:
            model_age = float("nan")

        model_values = [
            self.x,
            self.y,
            self.z,
            self.roll_deg,
            self.pitch_deg,
            self.yaw_deg,
            self.vx_world,
            self.vy_world,
            self.vx_body,
            self.vy_body,
        ]

        model_values_valid = all(
            not math.isnan(finite(value))
            for value in model_values
        )

        model_valid = (
            self.model_stamp is not None
            and model_values_valid
            and model_age
            <= self.model_max_age_s
        )

        valid = (
            joints_valid
            and ref_valid
            and model_valid
        )

        tau_values = []
        dq_values = []
        powers = []

        if joints_valid:
            for name in JOINT_NAMES:
                tau_value = finite(
                    self.tau[name]
                )

                dq_value = finite(
                    self.dq[name]
                )

                tau_values.append(tau_value)
                dq_values.append(dq_value)

                powers.append(
                    tau_value * dq_value
                )
        else:
            tau_values = [
                float("nan")
            ] * len(JOINT_NAMES)

            dq_values = [
                float("nan")
            ] * len(JOINT_NAMES)

            powers = [
                float("nan")
            ] * len(JOINT_NAMES)

        if joints_valid:
            power_signed = sum(powers)

            power_positive = sum(
                max(value, 0.0)
                for value in powers
            )

            power_abs = sum(
                abs(value)
                for value in powers
            )

            tau_sq_sum = sum(
                value * value
                for value in tau_values
            )

            dq_sq_sum = sum(
                value * value
                for value in dq_values
            )
        else:
            power_signed = float("nan")
            power_positive = float("nan")
            power_abs = float("nan")
            tau_sq_sum = float("nan")
            dq_sq_sum = float("nan")

        if (
            valid
            and self.previous_valid
            and self.previous_time is not None
        ):
            dt = now - self.previous_time

            if 0.0 < dt <= 0.20:
                self.work_signed += (
                    power_signed * dt
                )

                self.work_positive += (
                    power_positive * dt
                )

                self.work_abs += (
                    power_abs * dt
                )

                self.tau_sq_integral += (
                    tau_sq_sum * dt
                )

                self.dq_sq_integral += (
                    dq_sq_sum * dt
                )

        self.previous_time = now
        self.previous_valid = valid

        if self.ref is None:
            ref = [
                float("nan")
            ] * 6
        else:
            ref = self.ref

        row = [
            "%.9f" % now,
            "%.9f" % time.time(),
            1 if valid else 0,
            fmt(max_joint_age),
            fmt(ref_age),
            fmt(model_age),
            fmt(ref[0]),
            fmt(self.x),
            fmt(self.y),
            fmt(self.z),
            fmt(self.roll_deg),
            fmt(self.pitch_deg),
            fmt(self.yaw_deg),
            fmt(self.vx_world),
            fmt(self.vy_world),
            fmt(self.vx_body),
            fmt(self.vy_body),
            self.context_from_x(),
            fmt(ref[1]),
            fmt(ref[2]),
            fmt(ref[3]),
            fmt(ref[4]),
            fmt(ref[5]),
            fmt(power_signed),
            fmt(power_positive),
            fmt(power_abs),
            fmt(tau_sq_sum),
            fmt(dq_sq_sum),
            fmt(self.work_signed),
            fmt(self.work_positive),
            fmt(self.work_abs),
            fmt(self.tau_sq_integral),
            fmt(self.dq_sq_integral),
        ]

        for (
            tau_value,
            dq_value,
            power,
        ) in zip(
            tau_values,
            dq_values,
            powers,
        ):
            row.extend([
                fmt(tau_value),
                fmt(dq_value),
                fmt(power),
            ])

        self.writer.writerow(row)

        self.row_count += 1

        if self.row_count % 10 == 0:
            sys.stdout.flush()

    def on_shutdown(self):
        try:
            sys.stdout.flush()
        except Exception:
            pass

        print(
            "[TRACER] joint-energy-vx logger rows=%d"
            % self.row_count,
            file=sys.stderr,
        )


def main():
    rospy.init_node(
        "tracer_ros1_joint_energy_vx_logger_v0",
        anonymous=False,
        disable_signals=False,
    )

    JointEnergyVXLogger()

    rospy.spin()


if __name__ == "__main__":
    main()
