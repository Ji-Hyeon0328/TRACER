#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, Int32, String


def finite(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def clamp(value: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(value)))


def load_json(path: str) -> dict[str, Any]:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = (Path.cwd() / resolved).resolve()
    with resolved.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def array_message(values: list[float]) -> Float64MultiArray:
    message = Float64MultiArray()
    message.data = [float(value) for value in values]
    return message


class PhaseBLL1BankJ7MapperV2(Node):
    """
    Mapper between a categorical high-level action and the frozen J7 gate.

    The selected action controls only:
      - far-goal forward speed
      - swing clearance

    Near-goal speed, body height, stop/slow distances, and yaw bound remain
    fixed by the runtime contract.

    The frozen J7 gate treats theta[0] as a legacy gate action token and
    rejects value 8. This mapper therefore publishes active token 1 in
    theta[0] and stores the true 0..8 bank action ID in theta[1].

    J7 compares the projected reference against the empirical reference from
    the same sequence number. The empirical far-speed/clearance anchor is the
    center action of the bank:
      vx=0.065 m/s, clearance=0.045 m.

    Every runtime bank v1 action remains within the frozen J7 defaults:
      |delta vx| <= 0.025 m/s < 0.030 m/s
      |delta clearance| <= 0.005 m < 0.006 m
      |delta yaw| <= 0.020 rad/s
    """

    def __init__(self) -> None:
        super().__init__("tracer_phase_b_ll1_bank_j7_mapper_node_v2")

        self.declare_parameter(
            "action_bank_path",
            (
                "configs/phase_b_theta_lite_rl_v1/"
                "ll1_j7_compatible_action_bank_v1.json"
            ),
        )
        self.declare_parameter(
            "runtime_contract_path",
            (
                "configs/phase_b_theta_lite_rl_v1/"
                "ll1_bank_j7_runtime_contract_v2.json"
            ),
        )
        self.declare_parameter("initial_action_id", 4)
        self.declare_parameter("enable_action_index_sub", True)
        self.declare_parameter("publish_hz", 20.0)
        self.declare_parameter("goal_timeout_sec", 1.0)
        self.declare_parameter("odom_timeout_sec", 1.0)
        self.declare_parameter("hold_on_stale_input", True)

        bank_path = str(self.get_parameter("action_bank_path").value)
        contract_path = str(self.get_parameter("runtime_contract_path").value)

        self.bank = load_json(bank_path)
        self.contract = load_json(contract_path)

        actions = self.bank.get("actions")
        if not isinstance(actions, list) or not actions:
            raise RuntimeError("action bank contains no actions")

        self.actions: list[dict[str, Any]] = []
        self.actions_by_id: dict[int, dict[str, Any]] = {}

        for row in actions:
            if not isinstance(row, dict):
                raise RuntimeError("action bank row is not an object")
            action_id = int(row["action_id"])
            if action_id in self.actions_by_id:
                raise RuntimeError(f"duplicate action_id={action_id}")
            copied = dict(row)
            self.actions.append(copied)
            self.actions_by_id[action_id] = copied

        self.topics = dict(self.contract["topics"])
        self.anchor = dict(self.contract["empirical_anchor"])
        self.goal_logic = dict(self.contract["goal_logic"])
        theta_layout = dict(self.contract["theta_shadow_layout"])
        self.j7_active_action_token = finite(
            theta_layout["j7_active_action_token"],
            1.0,
        )
        self.legacy_reserved_token = finite(
            theta_layout["legacy_reserved_rejected_token"],
            8.0,
        )
        if abs(
            self.j7_active_action_token
            - self.legacy_reserved_token
        ) < 1.0e-12:
            raise RuntimeError(
                "J7 active action token collides with reserved token"
            )

        self.selected_action_id = int(
            self.get_parameter("initial_action_id").value
        )
        self.selected_action = self._resolve_action(self.selected_action_id)

        self.sequence = 0.0
        self.last_goal: Optional[list[float]] = None
        self.last_goal_time: Optional[float] = None
        self.last_odom: Optional[list[float]] = None
        self.last_odom_time: Optional[float] = None
        self.reached_latch = False

        self.empirical_pub = self.create_publisher(
            Float64MultiArray,
            self.topics["empirical_reference"],
            10,
        )
        self.projected_pub = self.create_publisher(
            Float64MultiArray,
            self.topics["projected_reference"],
            10,
        )
        self.theta_pub = self.create_publisher(
            Float64MultiArray,
            self.topics["theta_shadow"],
            10,
        )
        self.debug_pub = self.create_publisher(
            String,
            self.topics["debug"],
            10,
        )

        self.create_subscription(
            Float64MultiArray,
            self.topics["relative_goal"],
            self._on_goal,
            10,
        )
        self.create_subscription(
            Float64MultiArray,
            self.topics["robot_odom"],
            self._on_odom,
            10,
        )

        if bool(self.get_parameter("enable_action_index_sub").value):
            self.create_subscription(
                Int32,
                self.topics["action_index"],
                self._on_action_index,
                10,
            )

        publish_hz = max(
            1.0,
            finite(self.get_parameter("publish_hz").value, 20.0),
        )
        self.timer = self.create_timer(
            1.0 / publish_hz,
            self._on_timer,
        )

        self.get_logger().info(
            "Phase-B LL1 bank J7 mapper started: "
            f"action_id={self.selected_action_id} "
            f"name={self.selected_action['name']} "
            f"gate_token={self.j7_active_action_token:.0f} "
            f"bank={bank_path} contract={contract_path}"
        )

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1.0e-9

    def _resolve_action(self, action_id: int) -> dict[str, Any]:
        if action_id not in self.actions_by_id:
            valid = sorted(self.actions_by_id)
            raise ValueError(
                f"unknown action_id={action_id}; valid={valid}"
            )
        return self.actions_by_id[action_id]

    def _on_action_index(self, message: Int32) -> None:
        requested = int(message.data)
        try:
            action = self._resolve_action(requested)
        except ValueError as error:
            self.get_logger().error(str(error))
            return

        if requested == self.selected_action_id:
            return

        self.selected_action_id = requested
        self.selected_action = action
        self.reached_latch = False

        self.get_logger().info(
            "action update: "
            f"action_id={requested} name={action['name']}"
        )

    def _on_goal(self, message: Float64MultiArray) -> None:
        data = [finite(value) for value in message.data]
        if len(data) < 2:
            self.get_logger().warning(
                f"relative_goal expects >=2 values, got {len(data)}"
            )
            return

        while len(data) < 4:
            data.append(1.0)

        self.last_goal = data[:4]
        self.last_goal_time = self._now()

    def _on_odom(self, message: Float64MultiArray) -> None:
        data = [finite(value) for value in message.data]
        if len(data) < 6:
            self.get_logger().warning(
                f"robot_odom_flat expects >=6 values, got {len(data)}"
            )
            return

        self.last_odom = data[:6]
        self.last_odom_time = self._now()

    def _fresh(
        self,
        timestamp: Optional[float],
        timeout_parameter: str,
    ) -> bool:
        if timestamp is None:
            return False
        timeout = max(
            0.01,
            finite(self.get_parameter(timeout_parameter).value, 1.0),
        )
        return self._now() - timestamp <= timeout

    def _hold_references(
        self,
        reason: str,
        hold_override: bool,
    ) -> tuple[list[float], list[float], list[float], dict[str, Any]]:
        clearance = finite(
            self.anchor["swing_clearance"],
            0.045,
        )
        body_height = finite(
            self.anchor["body_height"],
            0.320,
        )
        enable = finite(self.anchor.get("enable", 1.0), 1.0)

        empirical = [
            self.sequence,
            0.0,
            0.0,
            body_height,
            clearance,
            enable,
        ]
        projected = list(empirical)
        theta = [0.0] * 9
        theta[0] = float(self.j7_active_action_token)
        theta[1] = float(self.selected_action_id)
        theta[8] = 1.0 if hold_override else 0.0

        debug = {
            "mode": "hold",
            "reason": reason,
            "sequence": self.sequence,
            "action_id": self.selected_action_id,
            "j7_gate_action_token": self.j7_active_action_token,
            "action_name": self.selected_action["name"],
            "hold_override": bool(hold_override),
            "empirical": empirical,
            "projected": projected,
        }
        return empirical, projected, theta, debug

    def _compute_references(
        self,
    ) -> tuple[list[float], list[float], list[float], dict[str, Any]]:
        stale_goal = not self._fresh(
            self.last_goal_time,
            "goal_timeout_sec",
        )
        stale_odom = not self._fresh(
            self.last_odom_time,
            "odom_timeout_sec",
        )

        if self.last_goal is None or stale_goal:
            return self._hold_references(
                "missing_or_stale_goal",
                hold_override=bool(
                    self.get_parameter("hold_on_stale_input").value
                ),
            )

        if self.last_odom is None or stale_odom:
            return self._hold_references(
                "missing_or_stale_odom",
                hold_override=bool(
                    self.get_parameter("hold_on_stale_input").value
                ),
            )

        x_rel = finite(self.last_goal[0])
        y_rel = finite(self.last_goal[1])
        goal_enable = finite(self.last_goal[3], 1.0)

        distance = math.hypot(x_rel, y_rel)
        heading_error = math.atan2(
            y_rel,
            max(1.0e-6, x_rel),
        )

        stop_distance = finite(
            self.goal_logic["goal_stop_distance"],
            0.15,
        )
        slow_distance = finite(
            self.goal_logic["goal_slow_distance"],
            0.25,
        )
        yaw_gain = finite(
            self.goal_logic["yaw_gain"],
            1.2,
        )
        yaw_max = abs(
            finite(self.goal_logic["yaw_rate_max"], 0.020)
        )

        reached_latch_enabled = bool(
            self.goal_logic.get("reached_latch", True)
        )

        if self.reached_latch and reached_latch_enabled:
            return self._hold_references(
                "reached_latch",
                hold_override=False,
            )

        if goal_enable <= 0.5 or distance <= stop_distance:
            if distance <= stop_distance and reached_latch_enabled:
                self.reached_latch = True
            return self._hold_references(
                (
                    "goal_disabled"
                    if goal_enable <= 0.5
                    else "goal_reached"
                ),
                hold_override=False,
            )

        if distance <= slow_distance:
            mode = "near"
            empirical_vx = finite(self.anchor["vx_near"], 0.040)
            projected_vx = finite(
                self.selected_action["vx_near"],
                0.040,
            )
        else:
            mode = "far"
            empirical_vx = finite(self.anchor["vx_far"], 0.065)
            projected_vx = finite(
                self.selected_action["vx_far"],
                0.065,
            )

        yaw_rate = clamp(
            yaw_gain * heading_error,
            -yaw_max,
            yaw_max,
        )

        body_height = finite(
            self.anchor["body_height"],
            0.320,
        )
        anchor_clearance = finite(
            self.anchor["swing_clearance"],
            0.045,
        )
        projected_clearance = finite(
            self.selected_action["swing_clearance"],
            anchor_clearance,
        )
        enable = finite(self.anchor.get("enable", 1.0), 1.0)

        empirical = [
            self.sequence,
            empirical_vx,
            0.0,
            body_height,
            anchor_clearance,
            enable,
        ]
        projected = [
            self.sequence,
            projected_vx,
            yaw_rate,
            body_height,
            projected_clearance,
            enable,
        ]

        theta = [0.0] * 9
        theta[0] = float(self.j7_active_action_token)
        theta[1] = float(self.selected_action_id)
        theta[8] = 0.0

        debug = {
            "mode": mode,
            "sequence": self.sequence,
            "action_id": self.selected_action_id,
            "j7_gate_action_token": self.j7_active_action_token,
            "action_name": self.selected_action["name"],
            "distance": distance,
            "heading_error": heading_error,
            "empirical": empirical,
            "projected": projected,
            "delta": [
                projected[index] - empirical[index]
                for index in range(1, 5)
            ],
            "hold_override": False,
        }

        return empirical, projected, theta, debug

    def _on_timer(self) -> None:
        empirical, projected, theta, debug = self._compute_references()

        self.empirical_pub.publish(array_message(empirical))
        self.projected_pub.publish(array_message(projected))
        self.theta_pub.publish(array_message(theta))

        debug_message = String()
        debug_message.data = json.dumps(
            debug,
            sort_keys=True,
        )
        self.debug_pub.publish(debug_message)

        self.sequence += 1.0


def main() -> None:
    rclpy.init(args=None)
    node = PhaseBLL1BankJ7MapperV2()

    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
