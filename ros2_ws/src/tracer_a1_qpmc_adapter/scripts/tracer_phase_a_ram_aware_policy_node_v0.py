#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String


def to_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None or str(v).strip() == "":
            return default
        return float(v)
    except Exception:
        return default


def to_bool(v: Any, default: bool = False) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return default
    s = str(v).strip().lower()
    if s in {"true", "1", "yes", "y"}:
        return True
    if s in {"false", "0", "no", "n"}:
        return False
    return default


def resolve_path(root: Path, path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return root / p


class PhaseARamAwarePolicyNodeV0(Node):
    """
    Phase-A runtime node.

    v0 intentionally does not import torch.
    It consumes pre-exported RAM-aware stack JSON files from:
      scripts/training/tracer_export_phase_a_highlevel_stack_ram_v2.py

    Output:
      /tracer/mpc_reference: Float64MultiArray
        [counter, vx, yaw_rate, body_height, swing_clearance, enable]

      /tracer/phase_a_policy_status: String(JSON)
    """

    def __init__(self) -> None:
        super().__init__("tracer_phase_a_ram_aware_policy_node_v0")

        self.declare_parameter("tracer_root", os.getcwd())
        self.declare_parameter("terrain", "flat_normal")

        self.declare_parameter("stack_json", "")
        self.declare_parameter(
            "flat_stack_json",
            "configs/highlevel_stack/phase_a_stack_flat_ram_v2.json",
        )
        self.declare_parameter(
            "sponge_stack_json",
            "configs/highlevel_stack/phase_a_stack_sponge_ram_v2.json",
        )
        self.declare_parameter(
            "slippery_stack_json",
            "configs/highlevel_stack/phase_a_stack_slippery_ram_v2.json",
        )

        self.declare_parameter("mpc_topic", "/tracer/mpc_reference")
        self.declare_parameter("status_topic", "/tracer/phase_a_policy_status")
        self.declare_parameter("publish_hz", 20.0)

        self.declare_parameter("require_final_guard", True)
        self.declare_parameter("reload_json_each_tick", True)

        # Fallback hold reference.
        # We keep enable=1.0 by default so the low-level controller remains active
        # with zero forward velocity. Change to 0.0 only if your controller treats
        # enable=0 as a safe hold mode.
        self.declare_parameter("hold_vx", 0.0)
        self.declare_parameter("hold_yaw_rate", 0.0)
        self.declare_parameter("hold_body_height", 0.30)
        self.declare_parameter("hold_swing_clearance", 0.03)
        self.declare_parameter("hold_enable", 1.0)

        self.root = Path(str(self.get_parameter("tracer_root").value)).expanduser().resolve()

        self.mpc_topic = str(self.get_parameter("mpc_topic").value)
        self.status_topic = str(self.get_parameter("status_topic").value)
        self.publish_hz = float(self.get_parameter("publish_hz").value)

        self.require_final_guard = bool(self.get_parameter("require_final_guard").value)
        self.reload_json_each_tick = bool(self.get_parameter("reload_json_each_tick").value)

        self.pub = self.create_publisher(Float64MultiArray, self.mpc_topic, 10)
        self.status_pub = self.create_publisher(String, self.status_topic, 10)

        self.counter = 0.0
        self.last_load_error = ""
        self.last_stack_path = ""
        self.last_stack_mtime = 0.0
        self.cached_stack: dict[str, Any] | None = None

        timer_period = 1.0 / max(1e-6, self.publish_hz)
        self.timer = self.create_timer(timer_period, self.on_timer)

        self.get_logger().info(
            f"Phase-A RAM-aware policy node started. root={self.root} "
            f"mpc_topic={self.mpc_topic} status_topic={self.status_topic} hz={self.publish_hz}"
        )

    def selected_stack_path(self) -> Path:
        explicit = str(self.get_parameter("stack_json").value).strip()
        terrain = str(self.get_parameter("terrain").value).strip()

        if explicit:
            return resolve_path(self.root, explicit)

        if terrain == "flat_normal":
            return resolve_path(self.root, str(self.get_parameter("flat_stack_json").value))
        if terrain == "sponge_firm_flat":
            return resolve_path(self.root, str(self.get_parameter("sponge_stack_json").value))
        if terrain == "slippery_mild_flat":
            return resolve_path(self.root, str(self.get_parameter("slippery_stack_json").value))

        # Fallback convention.
        return resolve_path(
            self.root,
            f"configs/highlevel_stack/phase_a_stack_{terrain}_ram_v2.json",
        )

    def load_stack(self) -> dict[str, Any] | None:
        path = self.selected_stack_path()
        self.last_stack_path = str(path)

        try:
            if not path.exists():
                self.last_load_error = f"stack_json_not_found: {path}"
                return None

            mtime = path.stat().st_mtime
            if (
                self.cached_stack is not None
                and not self.reload_json_each_tick
                and mtime == self.last_stack_mtime
            ):
                return self.cached_stack

            data = json.loads(path.read_text(encoding="utf-8"))
            self.cached_stack = data
            self.last_stack_mtime = mtime
            self.last_load_error = ""
            return data

        except Exception as e:
            self.last_load_error = f"stack_json_load_error: {type(e).__name__}: {e}"
            return None

    def hold_reference(self) -> tuple[list[float], dict[str, Any]]:
        ref = [
            self.counter,
            float(self.get_parameter("hold_vx").value),
            float(self.get_parameter("hold_yaw_rate").value),
            float(self.get_parameter("hold_body_height").value),
            float(self.get_parameter("hold_swing_clearance").value),
            float(self.get_parameter("hold_enable").value),
        ]
        reason = {
            "mode": "hold_or_recovery_needed",
            "reason": self.last_load_error or "no_safe_candidate_or_guard_failed",
        }
        return ref, reason

    def action_reference_from_stack(self, stack: dict[str, Any]) -> tuple[list[float], dict[str, Any]]:
        selected = stack.get("selected_action", {})
        selection_status = str(stack.get("selection_status", ""))

        final_guard = to_bool(
            selected.get(
                "final_guard_pass",
                selected.get("ram_guard_pass", False),
            )
        )

        is_safe_status = selection_status in {
            "ram_and_aggregate_guard_pass",
            "ram_guard_pass",
            "safe_filter_pass",
        }

        if self.require_final_guard and not final_guard:
            ref, reason = self.hold_reference()
            reason.update({
                "selection_status": selection_status,
                "final_guard_pass": final_guard,
                "selected_action": selected,
            })
            return ref, reason

        if self.require_final_guard and not is_safe_status:
            ref, reason = self.hold_reference()
            reason.update({
                "selection_status": selection_status,
                "final_guard_pass": final_guard,
                "selected_action": selected,
            })
            return ref, reason

        vx = to_float(selected.get("vx"))
        yaw_rate = to_float(selected.get("yaw_rate"))
        body_height = to_float(selected.get("body_height"), 0.30)
        swing_clearance = to_float(selected.get("swing_clearance"), 0.03)
        enable = to_float(selected.get("enable"), 1.0)

        ref = [
            self.counter,
            vx,
            yaw_rate,
            body_height,
            swing_clearance,
            enable,
        ]

        reason = {
            "mode": "ram_aware_stack_action",
            "selection_status": selection_status,
            "final_guard_pass": final_guard,
            "selected_action": selected,
        }
        return ref, reason

    def publish_status(self, ref: list[float], info: dict[str, Any]) -> None:
        terrain = str(self.get_parameter("terrain").value)
        msg = String()
        msg.data = json.dumps(
            {
                "stamp_wall": time.time(),
                "node": "tracer_phase_a_ram_aware_policy_node_v0",
                "terrain": terrain,
                "stack_path": self.last_stack_path,
                "mpc_reference": {
                    "counter": ref[0],
                    "vx": ref[1],
                    "yaw_rate": ref[2],
                    "body_height": ref[3],
                    "swing_clearance": ref[4],
                    "enable": ref[5],
                },
                **info,
            },
            sort_keys=True,
        )
        self.status_pub.publish(msg)

    def on_timer(self) -> None:
        self.counter += 1.0

        stack = self.load_stack()
        if stack is None:
            ref, info = self.hold_reference()
        else:
            ref, info = self.action_reference_from_stack(stack)

        msg = Float64MultiArray()
        msg.data = [float(x) for x in ref]
        self.pub.publish(msg)
        self.publish_status(ref, info)


def main() -> None:
    rclpy.init()
    node = PhaseARamAwarePolicyNodeV0()

    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
