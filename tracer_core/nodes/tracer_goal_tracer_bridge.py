#!/usr/bin/env python
from __future__ import print_function

import os
import csv
import math
import time

import rospy
from sensor_msgs.msg import Joy
from nav_msgs.msg import Odometry

from tracer_core.adaptation.robust_adaptation_module import RobustAdaptationModule
from tracer_core.objective.objective_selector import ObjectiveSelector


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def wrap_to_pi(a):
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def quat_to_roll_pitch_yaw(x, y, z, w):
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


def make_joy(vx_axis, yaw_axis, toggle=False):
    msg = Joy()
    msg.header.stamp = rospy.Time.now()

    # A1-QP-MPC Gazebo mapping:
    # axes[5] -> forward velocity
    # axes[0] -> yaw command
    msg.axes = [0.0] * 8
    msg.axes[5] = vx_axis
    msg.axes[0] = yaw_axis

    msg.buttons = [0, 0, 0, 0, 0]
    if toggle:
        msg.buttons[0] = 1

    return msg


class GoalTracerBridge(object):
    """
    M11: GoalTracker + TRACER modulation.

    GoalTracker:
        current pose + goal -> raw_vx_axis, raw_yaw_axis

    TRACER:
        RAM -> rho/sigma
        ObjectiveSelector -> beta/mode
        beta -> modulated vx_axis

    Output:
        /joy command to A1-QP-MPC
    """

    def __init__(self):
        # Goal params
        self.goal_x = rospy.get_param("~goal_x", 1.0)
        self.goal_y = rospy.get_param("~goal_y", 0.0)
        self.relative_goal = rospy.get_param("~relative_goal", True)
        self.relative_goal_x = rospy.get_param("~relative_goal_x", 1.0)
        self.relative_goal_y = rospy.get_param("~relative_goal_y", 0.0)
        self.goal_initialized = False

        # Goal tracking gains
        self.max_vx_axis = rospy.get_param("~max_vx_axis", 0.12)
        self.conservative_vx_axis = rospy.get_param("~conservative_vx_axis", 0.05)
        self.max_yaw_axis = rospy.get_param("~max_yaw_axis", 0.20)
        self.k_v = rospy.get_param("~k_v", 0.20)
        self.k_yaw = rospy.get_param("~k_yaw", 0.35)
        self.goal_tolerance = rospy.get_param("~goal_tolerance", 0.20)
        self.yaw_slowdown_angle = rospy.get_param("~yaw_slowdown_angle", 0.80)

        # Safety params
        self.max_roll_pitch_soft = rospy.get_param("~max_roll_pitch_soft", 0.25)
        self.max_roll_pitch_hard = rospy.get_param("~max_roll_pitch_hard", 0.45)
        self.min_base_height = rospy.get_param("~min_base_height", 0.18)

        # TRACER ObjectiveSelector thresholds
        self.mismatch_soft = rospy.get_param("~mismatch_soft", 0.50)
        self.mismatch_hard = rospy.get_param("~mismatch_hard", 0.80)
        self.sigma_soft = rospy.get_param("~sigma_soft", 0.18)
        self.sigma_hard = rospy.get_param("~sigma_hard", 0.35)
        self.mismatch_history_len = rospy.get_param("~mismatch_history_len", 40)

        # Runtime params
        self.duration_limit = rospy.get_param("~duration_limit", 60.0)
        self.rate_hz = rospy.get_param("~rate", 20.0)
        self.toggle_walk = rospy.get_param("~toggle_walk", True)
        self.toggle_stand_on_exit = rospy.get_param("~toggle_stand_on_exit", True)
        self.shutdown_zero = rospy.get_param("~shutdown_zero", True)
        self.shutdown_toggle_stand = rospy.get_param("~shutdown_toggle_stand", True)

        # Logging
        self.log_enable = rospy.get_param("~log_enable", True)
        self.log_dir = rospy.get_param("~log_dir", "/root/TRACER/logs")
        self.tag = rospy.get_param("~tag", "goal_tracer")

        self.latest_odom = None
        self.reached = False
        self.last_target_vx_axis = self.max_vx_axis

        self.ram = RobustAdaptationModule(history_len=self.mismatch_history_len)
        self.objective_selector = ObjectiveSelector(
            mismatch_soft=self.mismatch_soft,
            mismatch_hard=self.mismatch_hard,
            sigma_soft=self.sigma_soft,
            sigma_hard=self.sigma_hard,
        )

        self.pub = rospy.Publisher("/joy", Joy, queue_size=10)
        self.sub = rospy.Subscriber(
            "/body_pose_ground_truth",
            Odometry,
            self.odom_cb,
            queue_size=50,
        )

        self.file = None
        self.writer = None
        self.log_path = ""

        if self.log_enable:
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)

            stamp = time.strftime("%Y%m%d_%H%M%S")
            self.log_path = os.path.join(
                self.log_dir,
                "tracer_goal_tracer_%s_%s.csv" % (self.tag, stamp)
            )

            self.file = open(self.log_path, "w")
            self.writer = csv.writer(self.file)
            self.writer.writerow([
                "ros_time",
                "phase",
                "mode",
                "goal_x",
                "goal_y",
                "x",
                "y",
                "z",
                "roll",
                "pitch",
                "yaw",
                "distance",
                "heading_error",
                "raw_vx_axis",
                "raw_yaw_axis",
                "vx_axis",
                "yaw_axis",
                "beta_v",
                "beta_s",
                "beta_e",
                "rho",
                "sigma",
                "v_cmd",
                "v_meas",
                "rho_v_inst",
                "rho_v_mean",
                "sigma_v",
                "reached",
                "emergency_stop",
            ])

        rospy.on_shutdown(self.on_shutdown)

    def odom_cb(self, msg):
        self.latest_odom = msg

    def get_pose_state(self):
        if self.latest_odom is None:
            return None

        msg = self.latest_odom
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        v = msg.twist.twist.linear

        roll, pitch, yaw = quat_to_roll_pitch_yaw(q.x, q.y, q.z, q.w)

        state = {
            "base_height": p.z,
            "orientation": (q.x, q.y, q.z, q.w),
            "linear_velocity": (v.x, v.y, v.z),
        }

        return p.x, p.y, p.z, roll, pitch, yaw, state

    def initialize_relative_goal(self):
        if not self.relative_goal or self.goal_initialized:
            return

        pose = self.get_pose_state()
        if pose is None:
            return

        x0, y0, z0, roll0, pitch0, yaw0, state = pose

        # relative goal is in initial body-heading frame
        self.goal_x = (
            x0
            + self.relative_goal_x * math.cos(yaw0)
            - self.relative_goal_y * math.sin(yaw0)
        )
        self.goal_y = (
            y0
            + self.relative_goal_x * math.sin(yaw0)
            + self.relative_goal_y * math.cos(yaw0)
        )
        self.goal_initialized = True

    def toggle_button0(self):
        self.pub.publish(make_joy(0.0, 0.0, toggle=True))
        rospy.sleep(0.25)
        self.pub.publish(make_joy(0.0, 0.0, toggle=False))
        rospy.sleep(0.25)

    def publish_zero_for(self, sec):
        rate = rospy.Rate(self.rate_hz)
        t0 = rospy.Time.now().to_sec()
        while not rospy.is_shutdown():
            if rospy.Time.now().to_sec() - t0 > sec:
                break
            self.pub.publish(make_joy(0.0, 0.0, toggle=False))
            rate.sleep()

    def compute_goal_raw_cmd(self, x, y, yaw):
        dx = self.goal_x - x
        dy = self.goal_y - y
        dist = math.sqrt(dx * dx + dy * dy)

        desired_heading = math.atan2(dy, dx)
        heading_error = wrap_to_pi(desired_heading - yaw)

        if dist < self.goal_tolerance:
            return dist, heading_error, 0.0, 0.0, True

        raw_yaw_axis = clamp(
            self.k_yaw * heading_error,
            -self.max_yaw_axis,
            self.max_yaw_axis,
        )

        # Rotate first when heading error is large.
        yaw_factor = max(
            0.0,
            1.0 - abs(heading_error) / max(1e-6, self.yaw_slowdown_angle),
        )

        raw_vx_axis = clamp(
            self.k_v * dist,
            0.0,
            self.max_vx_axis,
        )
        raw_vx_axis = raw_vx_axis * yaw_factor

        return dist, heading_error, raw_vx_axis, raw_yaw_axis, False

    def beta_to_vx_axis(self, raw_vx_axis, beta_v, beta_s, beta_e):
        # If goal tracker asks for small speed, conservative speed should not exceed it.
        nominal = clamp(raw_vx_axis, 0.0, self.max_vx_axis)
        conservative = min(self.conservative_vx_axis, nominal)
        energy = 0.5 * (nominal + conservative)

        vx = (
            beta_v * nominal +
            beta_s * conservative +
            beta_e * energy
        )

        return clamp(vx, 0.0, self.max_vx_axis)

    def compute_tracer_modulation(self, state, z, roll, pitch, raw_vx_axis):
        rp_abs = max(abs(roll), abs(pitch))

        ram_out = self.ram.update(state, self.last_target_vx_axis)

        rho_posture = clamp(rp_abs / self.max_roll_pitch_hard, 0.0, 1.0)
        sigma_posture = clamp(
            (rp_abs - self.max_roll_pitch_soft)
            / max(1e-6, self.max_roll_pitch_hard - self.max_roll_pitch_soft),
            0.0,
            1.0,
        )

        rho_v_mean = ram_out["rho_v_mean"]
        sigma_v = ram_out["sigma_v"]

        rho = max(rho_posture, rho_v_mean)
        sigma = max(sigma_posture, sigma_v)

        if z < self.min_base_height or rp_abs > self.max_roll_pitch_hard:
            return {
                "mode": "recovery",
                "beta_v": 0.0,
                "beta_s": 1.0,
                "beta_e": 0.0,
                "rho": rho,
                "sigma": sigma,
                "ram_out": ram_out,
                "emergency_stop": True,
                "target_vx_axis": 0.0,
            }

        obj = self.objective_selector.select(
            posture_conservative=(rp_abs > self.max_roll_pitch_soft),
            rho_v_mean=rho_v_mean,
            sigma_v=sigma_v,
        )

        beta_v = obj["beta_v"]
        beta_s = obj["beta_s"]
        beta_e = obj["beta_e"]

        target_vx_axis = self.beta_to_vx_axis(
            raw_vx_axis,
            beta_v,
            beta_s,
            beta_e,
        )

        return {
            "mode": obj["mode"],
            "beta_v": beta_v,
            "beta_s": beta_s,
            "beta_e": beta_e,
            "rho": rho,
            "sigma": sigma,
            "ram_out": ram_out,
            "emergency_stop": False,
            "target_vx_axis": target_vx_axis,
        }

    def log_row(self, phase, mode, goal_x, goal_y,
                x, y, z, roll, pitch, yaw,
                dist, heading_error,
                raw_vx_axis, raw_yaw_axis,
                vx_axis, yaw_axis,
                beta_v, beta_s, beta_e,
                rho, sigma, ram_out,
                reached, emergency_stop):
        if self.writer is None:
            return

        self.writer.writerow([
            rospy.Time.now().to_sec(),
            phase,
            mode,
            goal_x,
            goal_y,
            x,
            y,
            z,
            roll,
            pitch,
            yaw,
            dist,
            heading_error,
            raw_vx_axis,
            raw_yaw_axis,
            vx_axis,
            yaw_axis,
            beta_v,
            beta_s,
            beta_e,
            rho,
            sigma,
            ram_out.get("v_cmd", 0.0),
            ram_out.get("v_meas", 0.0),
            ram_out.get("rho_v_inst", 0.0),
            ram_out.get("rho_v_mean", 0.0),
            ram_out.get("sigma_v", 0.0),
            str(reached),
            str(emergency_stop),
        ])

    def on_shutdown(self):
        rospy.loginfo("GoalTracer shutdown: sending safe stop commands.")
        try:
            if self.shutdown_zero:
                self.publish_zero_for(0.5)

            if self.shutdown_toggle_stand:
                self.toggle_button0()
                self.publish_zero_for(0.5)
        except Exception as e:
            rospy.logwarn("GoalTracer shutdown failed: %s", str(e))

    def run(self):
        rospy.loginfo("GoalTracer: waiting for odometry...")
        while not rospy.is_shutdown() and self.latest_odom is None:
            rospy.sleep(0.05)

        self.initialize_relative_goal()

        rospy.loginfo(
            "GoalTracer start: goal=(%.3f, %.3f), relative=%s",
            self.goal_x,
            self.goal_y,
            str(self.relative_goal),
        )

        self.publish_zero_for(1.0)

        if self.toggle_walk:
            rospy.loginfo("GoalTracer: toggling walking mode on.")
            self.toggle_button0()

        self.publish_zero_for(2.0)

        rate = rospy.Rate(self.rate_hz)
        t0 = rospy.Time.now().to_sec()

        while not rospy.is_shutdown():
            if rospy.Time.now().to_sec() - t0 > self.duration_limit:
                rospy.loginfo("GoalTracer: duration limit reached.")
                break

            pose = self.get_pose_state()
            if pose is None:
                rate.sleep()
                continue

            x, y, z, roll, pitch, yaw, state = pose

            dist, heading_error, raw_vx_axis, raw_yaw_axis, reached = \
                self.compute_goal_raw_cmd(x, y, yaw)

            self.reached = reached

            if reached:
                self.pub.publish(make_joy(0.0, 0.0, toggle=False))
                self.log_row(
                    "reached",
                    "reached",
                    self.goal_x,
                    self.goal_y,
                    x, y, z, roll, pitch, yaw,
                    dist, heading_error,
                    raw_vx_axis, raw_yaw_axis,
                    0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0,
                    {
                        "v_cmd": 0.0,
                        "v_meas": 0.0,
                        "rho_v_inst": 0.0,
                        "rho_v_mean": 0.0,
                        "sigma_v": 0.0,
                    },
                    True,
                    False,
                )
                rospy.loginfo("GoalTracer: goal reached. dist=%.3f", dist)
                break

            tracer = self.compute_tracer_modulation(
                state,
                z,
                roll,
                pitch,
                raw_vx_axis,
            )

            vx_axis = tracer["target_vx_axis"]

            # M11: keep yaw mostly from GoalTracker.
            # Later we can also beta-modulate yaw.
            yaw_axis = raw_yaw_axis

            if tracer["emergency_stop"]:
                vx_axis = 0.0
                yaw_axis = 0.0

            self.pub.publish(make_joy(vx_axis, yaw_axis, toggle=False))

            self.last_target_vx_axis = vx_axis

            self.log_row(
                "track",
                tracer["mode"],
                self.goal_x,
                self.goal_y,
                x, y, z, roll, pitch, yaw,
                dist, heading_error,
                raw_vx_axis, raw_yaw_axis,
                vx_axis, yaw_axis,
                tracer["beta_v"],
                tracer["beta_s"],
                tracer["beta_e"],
                tracer["rho"],
                tracer["sigma"],
                tracer["ram_out"],
                False,
                tracer["emergency_stop"],
            )

            rospy.loginfo_throttle(
                1.0,
                "GoalTracer | mode=%s dist=%.3f h_err=%.3f "
                "raw_vx=%.3f vx=%.3f yaw=%.3f beta=[%.2f %.2f %.2f] "
                "rho=%.2f sigma=%.2f",
                tracer["mode"],
                dist,
                heading_error,
                raw_vx_axis,
                vx_axis,
                yaw_axis,
                tracer["beta_v"],
                tracer["beta_s"],
                tracer["beta_e"],
                tracer["rho"],
                tracer["sigma"],
            )

            rate.sleep()

        self.publish_zero_for(1.0)

        if self.toggle_stand_on_exit:
            rospy.loginfo("GoalTracer: toggling walking mode off.")
            self.toggle_button0()

        self.publish_zero_for(1.0)

        if self.file is not None:
            self.file.close()
            rospy.loginfo("GoalTracer log: %s", self.log_path)
            print(self.log_path)


if __name__ == "__main__":
    rospy.init_node("tracer_goal_tracer_bridge")
    GoalTracerBridge().run()
