#!/usr/bin/env python
from __future__ import print_function

import os
import sys
import csv
import time

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
    def __init__(self, vx_axis_sign=1.0):
        self.vx_axis_sign = float(vx_axis_sign)

    def to_joy(self, ref, force_toggle=False):
        msg = Joy()
        msg.header.stamp = rospy.Time.now()
        msg.axes = [
            float(ref.yaw_rate),     # axes[0]: yaw
            float(ref.height_rate),  # axes[1]: height rate
            float(ref.vy),           # axes[2]: lateral
            0.0,
            0.0,
            float(self.vx_axis_sign * ref.vx),  # axes[5]: forward
            0.0,
            0.0,
        ]
        msg.buttons = [1 if force_toggle else 0, 0, 0, 0, 0]
        return msg


class CsvLogger(object):
    def __init__(self, log_dir):
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(log_dir, "tracer_closed_loop_%s.csv" % stamp)
        self.f = open(self.path, "w")
        self.writer = csv.writer(self.f)

        self.writer.writerow([
            "ros_time", "wall_time",
            "phase", "mode",
            "base_x", "base_y", "base_z",
            "roll", "pitch",
            "vx_axis", "vy_axis", "yaw_axis",
            "walk", "emergency_stop",
            "beta_v", "beta_s", "beta_e",
            "rho", "sigma",
            "v_cmd", "v_meas", "rho_v",
            "rho_v_inst", "rho_v_mean", "sigma_v",
        ])
        self.f.flush()

    def write(self, phase, state, theta, ref):
        pos = state.get("position", (0.0, 0.0, 0.0))
        self.writer.writerow([
            rospy.Time.now().to_sec(),
            time.time(),
            phase,
            theta.get("mode", "unknown"),
            pos[0], pos[1], pos[2],
            theta.get("roll", 0.0),
            theta.get("pitch", 0.0),
            ref.vx,
            ref.vy,
            ref.yaw_rate,
            ref.walk,
            ref.emergency_stop,
            theta.get("beta_v", 0.0),
            theta.get("beta_s", 0.0),
            theta.get("beta_e", 0.0),
            theta.get("rho", 0.0),
            theta.get("sigma", 0.0),
            theta.get("v_cmd", 0.0),
            theta.get("v_meas", 0.0),
            theta.get("rho_v", 0.0),
            theta.get("rho_v_inst", 0.0),
            theta.get("rho_v_mean", 0.0),
            theta.get("sigma_v", 0.0),
        ])
        self.f.flush()


class TracerCmdBridge(object):
    def __init__(self):
        self.state = {}
        self.received_odom = False

        self.rate_hz = rospy.get_param("~rate", 20.0)

        self.nominal_vx_axis = rospy.get_param("~nominal_vx_axis", 0.01)
        self.conservative_vx_axis = rospy.get_param("~conservative_vx_axis", 0.005)
        self.vx_max = rospy.get_param("~vx_max", 0.03)
        self.mismatch_soft = rospy.get_param("~mismatch_soft", 0.65)
        self.mismatch_hard = rospy.get_param("~mismatch_hard", 0.90)
        self.sigma_soft = rospy.get_param("~sigma_soft", 0.18)
        self.sigma_hard = rospy.get_param("~sigma_hard", 0.30)
        self.mismatch_history_len = rospy.get_param("~mismatch_history_len", 40)
        self.vx_axis_sign = rospy.get_param("~vx_axis_sign", 1.0)

        self.auto_walk = rospy.get_param("~auto_walk", True)
        self.warmup_sec = rospy.get_param("~warmup_sec", 2.0)
        self.toggle_after_warmup = rospy.get_param("~toggle_after_warmup", True)

        self.stable_min_height = rospy.get_param("~stable_min_height", 0.25)
        self.stable_max_abs_roll = rospy.get_param("~stable_max_abs_roll", 0.25)
        self.stable_max_abs_pitch = rospy.get_param("~stable_max_abs_pitch", 0.25)

        self.log_enable = rospy.get_param("~log_enable", True)
        self.log_dir = rospy.get_param("~log_dir", "/root/TRACER/logs")

        # Safety behavior when the bridge is stopped with Ctrl-C.
        # zero command is always safe.
        # toggle_stand is useful only when the low-level controller is known to be in walking mode,
        # because button[0] is a toggle, not an explicit stand command.
        self.shutdown_zero = rospy.get_param("~shutdown_zero", True)
        self.shutdown_toggle_stand = rospy.get_param("~shutdown_toggle_stand", False)

        self.phase = "waiting"
        self.t0 = None
        self.walk_toggled = False

        self.planner = RuleBasedHighLevelPlanner(
            nominal_vx_axis=self.nominal_vx_axis,
            conservative_vx_axis=self.conservative_vx_axis,
            mismatch_soft=self.mismatch_soft,
            mismatch_hard=self.mismatch_hard,
            sigma_soft=self.sigma_soft,
            sigma_hard=self.sigma_hard,
            history_len=self.mismatch_history_len,
        )
        self.mapper = ThetaToRefMapper(vx_max=self.vx_max)
        self.adapter = GazeboJoyAdapter(vx_axis_sign=self.vx_axis_sign)

        self.logger = CsvLogger(self.log_dir) if self.log_enable else None
        if self.logger:
            rospy.loginfo("TRACER bridge CSV log: %s", self.logger.path)

        self.pub = rospy.Publisher("/joy", Joy, queue_size=10)
        self.sub = rospy.Subscriber(
            "/body_pose_ground_truth",
            Odometry,
            self.odom_callback,
            queue_size=10,
        )

        rospy.on_shutdown(self.on_shutdown)

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

    def make_zero_theta(self, mode="warmup"):
        return {
            "mode": mode,
            "walk": False,
            "target_vx_axis": 0.0,
            "target_yaw_axis": 0.0,
            "emergency_stop": False,
            "beta_v": 0.0,
            "beta_s": 1.0,
            "beta_e": 0.0,
            "rho": 0.0,
            "sigma": 0.0,
            "roll": 0.0,
            "pitch": 0.0,
        }

    def is_stable(self, theta):
        z = self.state.get("base_height", 0.0)
        roll = abs(theta.get("roll", 0.0))
        pitch = abs(theta.get("pitch", 0.0))

        return (
            z > self.stable_min_height and
            roll < self.stable_max_abs_roll and
            pitch < self.stable_max_abs_pitch and
            not theta.get("emergency_stop", False)
        )

    def publish_ref(self, theta, force_toggle=False):
        ref = self.mapper.map(theta)
        self.pub.publish(self.adapter.to_joy(ref, force_toggle=force_toggle))

        if self.logger:
            self.logger.write(self.phase, self.state, theta, ref)

        rospy.loginfo_throttle(
            1.0,
            "TRACER cmd | phase=%s mode=%s z=%.3f roll=%.3f pitch=%.3f "
            "vx=%.3f beta=[%.2f %.2f %.2f] rho=%.2f sigma=%.2f "
            "v_cmd=%.3f v_meas=%.3f rho_v=%.2f estop=%s",
            str(self.phase),
            str(theta.get("mode", "unknown")),
            float(self.state.get("base_height", -1.0)),
            float(theta.get("roll", 0.0)),
            float(theta.get("pitch", 0.0)),
            float(ref.vx),
            float(theta.get("beta_v", 0.0)),
            float(theta.get("beta_s", 0.0)),
            float(theta.get("beta_e", 0.0)),
            float(theta.get("rho", 0.0)),
            float(theta.get("sigma", 0.0)),
            float(theta.get("v_cmd", 0.0)),
            float(theta.get("v_meas", 0.0)),
            float(theta.get("rho_v", 0.0)),
            str(ref.emergency_stop),
        )

    def toggle_walk_once(self):
        rospy.loginfo("TRACER bridge: toggling walking mode after warmup.")
        zero_theta = self.make_zero_theta(mode="toggle")
        self.phase = "toggle"
        self.publish_ref(zero_theta, force_toggle=True)
        rospy.sleep(0.25)
        self.publish_ref(zero_theta, force_toggle=False)
        rospy.sleep(0.25)
        self.walk_toggled = True

    def publish_raw_zero_joy(self, repeat=20, rate_hz=20.0):
        rate = rospy.Rate(rate_hz)
        msg = Joy()
        msg.axes = [0.0] * 8
        msg.buttons = [0, 0, 0, 0, 0]

        for _ in range(repeat):
            if rospy.is_shutdown():
                break
            msg.header.stamp = rospy.Time.now()
            self.pub.publish(msg)
            rate.sleep()

    def publish_toggle_button0(self):
        msg = Joy()
        msg.header.stamp = rospy.Time.now()
        msg.axes = [0.0] * 8
        msg.buttons = [1, 0, 0, 0, 0]
        self.pub.publish(msg)
        rospy.sleep(0.25)

        msg.header.stamp = rospy.Time.now()
        msg.buttons = [0, 0, 0, 0, 0]
        self.pub.publish(msg)
        rospy.sleep(0.25)

    def on_shutdown(self):
        rospy.loginfo("TRACER bridge shutdown: sending safe stop commands.")

        try:
            if self.shutdown_zero:
                self.publish_raw_zero_joy(repeat=10, rate_hz=20.0)

            if self.shutdown_toggle_stand:
                rospy.loginfo("TRACER bridge shutdown: toggling walking mode once.")
                self.publish_toggle_button0()
                self.publish_raw_zero_joy(repeat=10, rate_hz=20.0)

        except Exception as e:
            rospy.logwarn("TRACER bridge shutdown hook failed: %s", str(e))

    def run(self):
        rate = rospy.Rate(self.rate_hz)

        rospy.loginfo("TRACER bridge started. Waiting for odometry...")
        while not rospy.is_shutdown() and not self.received_odom:
            zero_theta = self.make_zero_theta(mode="waiting")
            self.pub.publish(self.adapter.to_joy(self.mapper.map(zero_theta), force_toggle=False))
            rate.sleep()

        rospy.loginfo("TRACER bridge received odometry.")
        self.t0 = rospy.Time.now().to_sec()
        self.phase = "warmup"

        while not rospy.is_shutdown():
            now = rospy.Time.now().to_sec()
            theta_live = self.planner.compute_theta(self.state)

            if self.phase == "warmup":
                zero_theta = self.make_zero_theta(mode="warmup")
                zero_theta["roll"] = theta_live.get("roll", 0.0)
                zero_theta["pitch"] = theta_live.get("pitch", 0.0)
                zero_theta["rho"] = theta_live.get("rho", 0.0)
                zero_theta["sigma"] = theta_live.get("sigma", 0.0)

                self.publish_ref(zero_theta, force_toggle=False)

                warmup_done = (now - self.t0) >= self.warmup_sec
                if warmup_done:
                    if self.is_stable(theta_live):
                        rospy.loginfo("TRACER bridge: warmup stable.")
                        if self.auto_walk and self.toggle_after_warmup and not self.walk_toggled:
                            self.toggle_walk_once()
                        self.phase = "walk" if self.auto_walk else "stand"
                    else:
                        rospy.logwarn("TRACER bridge: warmup failed stability check. Staying in stand.")
                        self.phase = "stand"

            elif self.phase == "stand":
                zero_theta = self.make_zero_theta(mode="stand")
                zero_theta["roll"] = theta_live.get("roll", 0.0)
                zero_theta["pitch"] = theta_live.get("pitch", 0.0)
                self.publish_ref(zero_theta, force_toggle=False)

            elif self.phase == "walk":
                theta = theta_live
                if not self.is_stable(theta):
                    rospy.logwarn("TRACER bridge: instability detected. Switching to recovery.")
                    self.phase = "recovery"
                    theta = self.make_zero_theta(mode="recovery")
                    theta["emergency_stop"] = True
                self.publish_ref(theta, force_toggle=False)

            elif self.phase == "recovery":
                theta = self.make_zero_theta(mode="recovery")
                theta["emergency_stop"] = True
                self.publish_ref(theta, force_toggle=False)

            rate.sleep()


if __name__ == "__main__":
    rospy.init_node("tracer_cmd_bridge")
    TracerCmdBridge().run()
