#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import socket
import sys
import time
from pathlib import Path
from typing import Any

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String


def find_repo_root() -> Path:
    here = Path(__file__).resolve()
    # .../TRACER/ros2_ws/src/tracer_a1_qpmc_adapter/scripts/file.py
    return here.parents[4]


ROOT = find_repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tracer_core.highlevel.gait_mode_selector import (  # noqa: E402
    input_from_policy_entry,
    select_gait_mode,
)
from tracer_core.highlevel.decoder_mapper import decode_meta_gait_to_low_level_ref  # noqa: E402
from tracer_core.highlevel.meta_gait_policy import make_meta_gait_policy  # noqa: E402
from tracer_core.highlevel.policy_input_builder import (  # noqa: E402
    build_high_level_policy_input,
    high_level_policy_input_summary,
)
from tracer_core.highlevel.objective_selector import (  # noqa: E402
    ObjectiveSelectorInput,
    RuntimeBaselineObjectiveSelector,
    infer_terrain_family,
)


def find_default_policy() -> Path:
    return ROOT / "configs/highlevel_policy/tracer_fusion_policy_v0.json"


GATE_LEVEL_FROM_CODE = {
    0: "stable",
    1: "caution",
    2: "unstable",
}

ACTION_FROM_CODE = {
    0: "keep",
    1: "would_cautious",
    2: "would_conservative_probe",
    3: "prior_avoid_keep",
    4: "prior_recovery_keep",
    5: "no_valid_keep",
    6: "failed_candidate_keep",
    -1: "unknown",
}


def _bool_env(name: str, default: str = "0") -> bool:
    return bool(int(os.environ.get(name, default)))


def _as_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _jsonable(obj):
    """Best-effort conversion for dataclasses / simple objects / dicts."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if hasattr(obj, "__dict__"):
        return {str(k): _jsonable(v) for k, v in obj.__dict__.items()}
    return str(obj)


LEARNED_STACK_V3_GMS_CODE = {
    "fast": 0,
    "cautious_probe": 1,
    "high_clearance_slow_probe": 2,
    "conservative": 3,
    "disabled": 4,
    "unknown": 5,
}

LEARNED_STACK_V3_LEVEL_CODE = {
    "unknown": 0,
    "stable": 1,
    "caution": 2,
    "unstable": 3,
}

LEARNED_STACK_V3_GATE_ACTION_CODE = {
    "unknown": 0,
    "keep": 1,
    "would_cautious": 1,
    "would_conservative_probe": 2,
    "force_conservative_probe": 3,
    "disable": 4,
}


def _ls3_label(x: Any) -> str:
    s = str(x or "").strip()
    aliases = {
        "validated_locomotion": "fast",
        "validated_fast": "fast",
        "locomotion": "fast",
        "high_clearance": "high_clearance_slow_probe",
        "slow_probe": "high_clearance_slow_probe",
        "cautious": "cautious_probe",
    }
    return aliases.get(s, s)


def _ls3_code_norm(label: Any, table: dict, denom: float, default: str = "unknown") -> float:
    key = _ls3_label(label) if table is LEARNED_STACK_V3_GMS_CODE else str(label or default).strip()
    return float(table.get(key, table.get(default, 0))) / max(1.0, float(denom))


def _ls3_terrain_onehot(terrain: str) -> list[float]:
    t = str(terrain or "unknown")
    return [
        1.0 if t == "flat_normal" else 0.0,
        1.0 if t == "rough_mid" else 0.0,
        1.0 if t == "slope_5deg" else 0.0,
    ]


def _ls3_beta_from_policy_input(policy_input: Any, entry: dict[str, Any]) -> list[float]:
    try:
        beta = policy_input.beta
        return [float(beta.motion), float(beta.stability), float(beta.energy)]
    except Exception:
        b = entry.get("beta", {})
        return [
            float(b.get("motion", 1.0 / 3.0)),
            float(b.get("stability", 1.0 / 3.0)),
            float(b.get("energy", 1.0 / 3.0)),
        ]




def _ls3_objective_v2_beta_prior_for_ram_window(terrain: str) -> list[float]:
    # Match learned_stack_shadow_node_v3 RAM-window feature distribution.
    # This is used only for RAM-shadow-v2 input construction, not for active command.
    t = str(terrain or "unknown")
    if t == "flat_normal":
        return [0.436, 0.327, 0.237]
    if t == "rough_mid":
        return [0.108, 0.745, 0.147]
    if t == "slope_5deg":
        return [0.143, 0.659, 0.198]
    return [0.35, 0.55, 0.10]

def _ls3_ram_sigma(policy_input: Any, gate: dict[str, Any]) -> float:
    try:
        return float(policy_input.ram.sigma)
    except Exception:
        return _as_float(gate.get("sigma_mean", 0.0), 0.0)


def _ls3_ram_rho_norm(policy_input: Any, gate: dict[str, Any]) -> float:
    try:
        rho = policy_input.ram.rho
        if rho:
            return min(1.0, float(len(rho)) / 16.0)
    except Exception:
        pass
    return _as_float(gate.get("rho_norm", 0.0), 0.0)


class TracerFusionPolicyMpcRefNode(Node):
    def __init__(self):
        super().__init__("tracer_fusion_policy_mpc_ref_node")

        default_policy = str(find_default_policy())

        self.declare_parameter("policy_json", os.environ.get("TRACER_FUSION_POLICY_JSON", default_policy))
        self.declare_parameter("terrain", os.environ.get("TRACER_TERRAIN_KEY", "flat_normal"))
        self.declare_parameter("hz", float(os.environ.get("TRACER_MPC_REF_HZ", "10.0")))
        self.declare_parameter("duration_sec", float(os.environ.get("TRACER_PUBLISH_DURATION", "0.0")))

        # Optional Gait Mode Selector.
        # Default is disabled so existing experiments remain unchanged.
        self.declare_parameter("enable_gms", int(os.environ.get("TRACER_ENABLE_GMS", "0")))
        self.declare_parameter("gms_use_ram_gate", int(os.environ.get("TRACER_GMS_USE_RAM_GATE", "1")))
        self.declare_parameter("gms_gate_freshness_sec", float(os.environ.get("TRACER_GMS_GATE_FRESHNESS_SEC", "2.0")))

        # Meta-gait policy interface.
        # rule_based is the current runtime-safe implementation.
        # learned/torch is reserved for a future exported model.
        self.declare_parameter("meta_gait_policy_kind", os.environ.get("TRACER_META_GAIT_POLICY_KIND", "rule_based"))
        self.declare_parameter("meta_gait_policy_model", os.environ.get("TRACER_META_GAIT_POLICY_MODEL", ""))

        self.declare_parameter("objective_selector_kind", os.environ.get("TRACER_OBJECTIVE_SELECTOR_KIND", "disabled"))
        self.declare_parameter(
            "objective_selector_model",
            os.environ.get(
                "TRACER_OBJECTIVE_SELECTOR_MODEL",
                "data/preference_datasets/tracer_objective_selector_runtime_v0_baseline_model.json",
            ),
        )
        self.declare_parameter("objective_selector_apply_semantic", int(os.environ.get("TRACER_OBJECTIVE_SELECTOR_APPLY_SEMANTIC", "1")))
        self.declare_parameter("objective_selector_apply_beta", int(os.environ.get("TRACER_OBJECTIVE_SELECTOR_APPLY_BETA", "1")))
        self.declare_parameter("objective_selector_block_no_deploy", int(os.environ.get("TRACER_OBJECTIVE_SELECTOR_BLOCK_NO_DEPLOY", "1")))
        self.declare_parameter("objective_selector_verbose", int(os.environ.get("TRACER_OBJECTIVE_SELECTOR_VERBOSE", "0")))
        # Learned high-level stack v3.
        # Safe default: query disabled and never deploy.
        # enable=1 only adds learned_stack_v3 to /tracer/highlevel_debug.
        # deploy=1 is reserved for a later active override patch.
        self.declare_parameter("enable_learned_stack_v3", int(os.environ.get("TRACER_ENABLE_LEARNED_STACK_V3", "0")))
        self.declare_parameter("deploy_learned_stack_v3", int(os.environ.get("TRACER_DEPLOY_LEARNED_STACK_V3", "0")))
        self.declare_parameter("learned_stack_v3_host", os.environ.get("TRACER_LEARNED_STACK_V3_HOST", "127.0.0.1"))
        self.declare_parameter("learned_stack_v3_port", int(os.environ.get("TRACER_LEARNED_STACK_V3_PORT", "50430")))
        self.declare_parameter("learned_stack_v3_timeout_sec", float(os.environ.get("TRACER_LEARNED_STACK_V3_TIMEOUT_SEC", "0.05")))
        self.declare_parameter("learned_stack_v3_min_gms_prob", float(os.environ.get("TRACER_LEARNED_STACK_V3_MIN_GMS_PROB", "0.55")))
        self.declare_parameter("learned_stack_v3_min_episode_success", float(os.environ.get("TRACER_LEARNED_STACK_V3_MIN_EPISODE_SUCCESS", "0.50")))


        # Optional terrain-aware command transition ramp.
        # Used for slippery active fallback experiments.
        self.declare_parameter("ramp_body_height_enable", int(os.environ.get("TRACER_RAMP_BODY_HEIGHT_ENABLE", "0")))
        self.declare_parameter("ramp_body_height_start", float(os.environ.get("TRACER_RAMP_BODY_HEIGHT_START", "0.325")))
        self.declare_parameter("ramp_body_height_duration", float(os.environ.get("TRACER_RAMP_BODY_HEIGHT_DURATION", "2.0")))

        # Optional velocity ramp.
        # This lets the robot complete the body-height transition first,
        # then gradually apply a small micro-brake / backstep velocity.
        self.declare_parameter("ramp_vx_enable", int(os.environ.get("TRACER_RAMP_VX_ENABLE", "0")))
        self.declare_parameter("ramp_vx_start", float(os.environ.get("TRACER_RAMP_VX_START", "0.0")))
        self.declare_parameter("ramp_vx_delay", float(os.environ.get("TRACER_RAMP_VX_DELAY", "0.0")))
        self.declare_parameter("ramp_vx_duration", float(os.environ.get("TRACER_RAMP_VX_DURATION", "2.0")))

        self.policy_json = Path(self.get_parameter("policy_json").value)
        self.terrain = str(self.get_parameter("terrain").value)
        self.hz = float(self.get_parameter("hz").value)
        self.duration_sec = float(self.get_parameter("duration_sec").value)

        self.enable_gms = bool(int(self.get_parameter("enable_gms").value))
        self.gms_use_ram_gate = bool(int(self.get_parameter("gms_use_ram_gate").value))
        self.gms_gate_freshness_sec = max(0.0, float(self.get_parameter("gms_gate_freshness_sec").value))
        self.meta_gait_policy_kind = str(self.get_parameter("meta_gait_policy_kind").value)
        self.meta_gait_policy_model = str(self.get_parameter("meta_gait_policy_model").value)
        self.meta_gait_policy = make_meta_gait_policy(
            self.meta_gait_policy_kind,
            self.meta_gait_policy_model or None,
        )

        self.objective_selector_kind = str(self.get_parameter("objective_selector_kind").value)
        self.objective_selector_model = str(self.get_parameter("objective_selector_model").value)
        self.objective_selector_apply_semantic = bool(int(self.get_parameter("objective_selector_apply_semantic").value))
        self.objective_selector_apply_beta = bool(int(self.get_parameter("objective_selector_apply_beta").value))
        self.objective_selector_block_no_deploy = bool(int(self.get_parameter("objective_selector_block_no_deploy").value))
        self.objective_selector_verbose = bool(int(self.get_parameter("objective_selector_verbose").value))
        self.enable_learned_stack_v3 = bool(int(self.get_parameter("enable_learned_stack_v3").value))
        self.deploy_learned_stack_v3 = bool(int(self.get_parameter("deploy_learned_stack_v3").value))
        self.learned_stack_v3_host = str(self.get_parameter("learned_stack_v3_host").value)
        self.learned_stack_v3_port = int(self.get_parameter("learned_stack_v3_port").value)
        self.learned_stack_v3_timeout_sec = float(self.get_parameter("learned_stack_v3_timeout_sec").value)
        self.learned_stack_v3_min_gms_prob = float(self.get_parameter("learned_stack_v3_min_gms_prob").value)
        self.learned_stack_v3_min_episode_success = float(self.get_parameter("learned_stack_v3_min_episode_success").value)

        self.learned_stack_v3_sock = None
        self.learned_stack_v3_window: list[dict[str, float]] = []
        self.latest_learned_stack_v3 = None
        if self.enable_learned_stack_v3:
            self.learned_stack_v3_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.learned_stack_v3_sock.settimeout(self.learned_stack_v3_timeout_sec)


        self.objective_selector = None
        if self.objective_selector_kind in {"runtime_baseline_json", "baseline_json"}:
            objective_model_path = Path(self.objective_selector_model)
            if not objective_model_path.is_absolute():
                objective_model_path = ROOT / objective_model_path
            self.objective_selector = RuntimeBaselineObjectiveSelector(objective_model_path)
        elif self.objective_selector_kind not in {"", "disabled", "none"}:
            raise RuntimeError(f"Unknown objective_selector_kind={self.objective_selector_kind}")

        self.ramp_body_height_enable = bool(int(self.get_parameter("ramp_body_height_enable").value))
        self.ramp_body_height_start = float(self.get_parameter("ramp_body_height_start").value)
        self.ramp_body_height_duration = max(1e-6, float(self.get_parameter("ramp_body_height_duration").value))

        self.ramp_vx_enable = bool(int(self.get_parameter("ramp_vx_enable").value))
        self.ramp_vx_start = float(self.get_parameter("ramp_vx_start").value)
        self.ramp_vx_delay = max(0.0, float(self.get_parameter("ramp_vx_delay").value))
        self.ramp_vx_duration = max(1e-6, float(self.get_parameter("ramp_vx_duration").value))

        if self.hz <= 0.0:
            raise RuntimeError("hz must be positive")

        if not self.policy_json.exists():
            raise FileNotFoundError(self.policy_json)

        self.policy = json.loads(self.policy_json.read_text())
        terrains = self.policy.get("terrains", {})

        if self.terrain not in terrains:
            keys = "\n".join(sorted(terrains.keys()))
            raise RuntimeError(
                f"Unknown terrain key: {self.terrain}\n"
                f"Available terrain keys:\n{keys}"
            )

        self.entry = terrains[self.terrain]
        self.command = self.entry["command"]
        self.counter = 0.0
        self.t0 = time.time()
        self.last_print = 0.0
        self.stop_requested = False

        self.latest_gate: dict[str, float | str] | None = None
        self.latest_gate_wall_time = 0.0
        self.latest_objective_selector_output = None

        self.pub = self.create_publisher(Float64MultiArray, "/tracer/mpc_reference", 10)
        self.debug_pub = self.create_publisher(String, "/tracer/highlevel_debug", 10)
        self.beta_pub = self.create_publisher(Float64MultiArray, "/tracer/objective_weights", 10)

        if self.enable_gms and self.gms_use_ram_gate:
            self.gate_sub = self.create_subscription(
                Float64MultiArray,
                "/tracer/ram_gate_advice",
                self.on_ram_gate_advice,
                10,
            )
        else:
            self.gate_sub = None

        self.timer = self.create_timer(1.0 / self.hz, self.on_timer)

        self.get_logger().info(f"policy_json={self.policy_json}")
        self.get_logger().info(
            f"terrain={self.terrain} "
            f"mode={self.entry['fused_mode']} "
            f"semantic={self.entry.get('semantic_mode', 'unknown')} "
            f"style={self.entry['suggested_style']} "
            f"risk={self.entry['fused_risk']:.3f}"
        )
        self.get_logger().info(
            f"command vx={self.command['vx']:.3f} "
            f"yaw={self.command['yaw_rate']:.3f} "
            f"h={self.command['body_height']:.3f} "
            f"clr={self.command['swing_clearance']:.3f} "
            f"enable={self.command['enable']:.1f}"
        )
        self.get_logger().info(
            f"GMS enable={int(self.enable_gms)} "
            f"use_ram_gate={int(self.gms_use_ram_gate)} "
            f"gate_freshness={self.gms_gate_freshness_sec:.2f}s"
        )
        self.get_logger().info(
            f"meta_gait_policy kind={self.meta_gait_policy_kind} "
            f"model={self.meta_gait_policy_model or '<none>'}"
        )
        self.get_logger().info(
            f"objective_selector kind={self.objective_selector_kind} "
            f"model={self.objective_selector_model or '<none>'} "
            f"apply_semantic={int(self.objective_selector_apply_semantic)} "
            f"apply_beta={int(self.objective_selector_apply_beta)} "
            f"block_no_deploy={int(self.objective_selector_block_no_deploy)} "
            f"verbose={int(self.objective_selector_verbose)}"
        )

        self.get_logger().info(
            f"learned_stack_v3 enable={int(self.enable_learned_stack_v3)} "
            f"deploy={int(self.deploy_learned_stack_v3)} "
            f"udp={self.learned_stack_v3_host}:{self.learned_stack_v3_port} "
            f"timeout={self.learned_stack_v3_timeout_sec:.3f}s "
            f"min_gms_prob={self.learned_stack_v3_min_gms_prob:.2f} "
            f"min_success={self.learned_stack_v3_min_episode_success:.2f}"
        )

        if self.ramp_body_height_enable:
            self.get_logger().info(
                f"body-height ramp enabled: "
                f"start_h={self.ramp_body_height_start:.3f} "
                f"target_h={self.command['body_height']:.3f} "
                f"duration={self.ramp_body_height_duration:.3f}s"
            )

        if self.ramp_vx_enable:
            self.get_logger().info(
                f"vx ramp enabled: "
                f"start_vx={self.ramp_vx_start:.4f} "
                f"target_vx={self.command['vx']:.4f} "
                f"delay={self.ramp_vx_delay:.3f}s "
                f"duration={self.ramp_vx_duration:.3f}s"
            )

    def _predict_objective_selector(self, gate: dict[str, Any], entry: dict[str, Any]):
        if self.objective_selector is None:
            return None

        semantic_prior = str(entry.get("semantic_mode", entry.get("semantic", "unknown")))
        style_prior = str(entry.get("suggested_style", entry.get("style", "unknown")))
        world = str(entry.get("world_name", entry.get("world", "")))
        terrain_family = infer_terrain_family(self.terrain, world, style_prior)

        ram_level = str(gate.get("ram_level", "unknown"))
        gate_action = str(gate.get("ram_gate_action", "unknown"))

        x = ObjectiveSelectorInput(
            terrain_key=self.terrain,
            world=world or "unknown",
            terrain_family=terrain_family,
            semantic_prior=semantic_prior,
            style_prior=style_prior,
            ramgate_enabled=self.gms_use_ram_gate,
            ramgate_calibrated=self.objective_selector_kind not in {"", "disabled", "none"},
            ramgate_observed=ram_level != "unknown",
            ram_monitor_observed=ram_level != "unknown",
            ram_ctrl_risk=_as_float(gate.get("control_risk", gate.get("ctrl_ema", 0.0)), 0.0),
            ram_fallen_prob=_as_float(gate.get("fallen_prob", gate.get("fallen", 0.0)), 0.0),
            ram_recovery_prob=_as_float(gate.get("recovery_prob", gate.get("recovery", 0.0)), 0.0),
            gate_stable_prob=1.0 if ram_level == "stable" else 0.0,
            gate_caution_prob=1.0 if ram_level == "caution" else 0.0,
            gate_unstable_prob=1.0 if ram_level == "unstable" else 0.0,
            gate_override_prob=_as_float(gate.get("would_override", 0.0), 0.0),
            gate_first_action=gate_action,
            gate_last_action=gate_action,
        )

        return self.objective_selector.predict(x)

    def on_ram_gate_advice(self, msg: Float64MultiArray) -> None:
        data = list(msg.data)
        if len(data) < 16:
            self.get_logger().warn(f"/tracer/ram_gate_advice too short: len={len(data)}")
            return

        gate_level_code = int(round(float(data[2])))
        action_code = int(round(float(data[3])))

        self.latest_gate = {
            "ram_level": GATE_LEVEL_FROM_CODE.get(gate_level_code, "unknown"),
            "ram_gate_action": ACTION_FROM_CODE.get(action_code, "unknown"),
            "would_override": float(data[4]),
            "control_risk": float(data[5]),
            "ctrl_ema": float(data[6]),
            "future_risk": float(data[7]),
            "fallen_prob": float(data[8]),
            "recovery_prob": float(data[9]),
            "sigma_mean": float(data[10]),
            "rho_norm": float(data[11]),
            "style_score": float(data[12]),
            "gate_vx_scale": float(data[13]),
            "gate_body_height_delta": float(data[14]),
            "gate_clearance_delta": float(data[15]),
        }
        self.latest_gate_wall_time = time.time()

    def _ramped_base_command(self, elapsed: float) -> dict[str, float]:
        body_height = float(self.command["body_height"])
        if self.ramp_body_height_enable:
            alpha = min(1.0, max(0.0, elapsed / self.ramp_body_height_duration))
            body_height = (
                (1.0 - alpha) * self.ramp_body_height_start
                + alpha * float(self.command["body_height"])
            )

        vx = float(self.command["vx"])
        if self.ramp_vx_enable:
            if elapsed < self.ramp_vx_delay:
                vx = self.ramp_vx_start
            else:
                vx_elapsed = elapsed - self.ramp_vx_delay
                beta = min(1.0, max(0.0, vx_elapsed / self.ramp_vx_duration))
                vx = (
                    (1.0 - beta) * self.ramp_vx_start
                    + beta * float(self.command["vx"])
                )

        return {
            "vx": float(vx),
            "yaw_rate": float(self.command["yaw_rate"]),
            "body_height": float(body_height),
            "swing_clearance": float(self.command["swing_clearance"]),
            "enable": float(self.command["enable"]),
        }

    def _gate_context(self, now_wall: float) -> dict[str, float | str]:
        if not self.enable_gms or not self.gms_use_ram_gate or self.latest_gate is None:
            return {
                "ram_level": "unknown",
                "ram_gate_action": "unknown",
                "control_risk": 0.0,
                "fallen_prob": 0.0,
                "recovery_prob": 0.0,
                "sigma_mean": None,
            "rho_norm": None,
            }

        age = now_wall - self.latest_gate_wall_time
        if age > self.gms_gate_freshness_sec:
            return {
                "ram_level": "unknown",
                "ram_gate_action": "unknown",
                "control_risk": 0.0,
                "fallen_prob": 0.0,
                "recovery_prob": 0.0,
                "sigma_mean": None,
                "rho_norm": None,
            }

        return {
            "ram_level": str(self.latest_gate.get("ram_level", "unknown")),
            "ram_gate_action": str(self.latest_gate.get("ram_gate_action", "unknown")),
            "control_risk": _as_float(self.latest_gate.get("control_risk"), 0.0),
            "fallen_prob": _as_float(self.latest_gate.get("fallen_prob"), 0.0),
            "recovery_prob": _as_float(self.latest_gate.get("recovery_prob"), 0.0),
            "sigma_mean": _as_float(self.latest_gate.get("sigma_mean"), 0.0),
            "rho_norm": _as_float(self.latest_gate.get("rho_norm"), 0.0),
        }


    def _learned_stack_v3_objective_x(
        self,
        selection_entry: dict[str, Any],
        gate: dict[str, Any],
        final_command: dict[str, float],
        elapsed: float,
    ) -> list[float]:
        duration_norm = min(1.0, max(0.0, float(elapsed) / 30.0))
        target_vx = max(1e-6, abs(float(self.command.get("vx", 0.28))))
        vx_over_target = min(1.5, max(0.0, abs(float(final_command.get("vx", 0.0))) / target_vx))
        body_height_delta = float(final_command.get("body_height", 0.295)) - 0.295
        clearance_delta = float(final_command.get("swing_clearance", 0.030)) - 0.030

        return _ls3_terrain_onehot(self.terrain) + [
            duration_norm,
            vx_over_target,
            float(final_command.get("enable", 1.0)),
            body_height_delta,
            clearance_delta,
            _ls3_code_norm(gate.get("ram_level", "unknown"), LEARNED_STACK_V3_LEVEL_CODE, 3.0),
            _ls3_code_norm(gate.get("ram_gate_action", "unknown"), LEARNED_STACK_V3_GATE_ACTION_CODE, 4.0),
            _as_float(gate.get("would_override", 0.0), 0.0),
            _as_float(gate.get("fallen_prob", 0.0), 0.0),
            _as_float(gate.get("recovery_prob", 0.0), 0.0),
            1.0,
            0.75,
        ]

    def _learned_stack_v3_gms_x(
        self,
        selection_entry: dict[str, Any],
        gate: dict[str, Any],
        gms_out: Any,
        policy_input: Any,
        final_command: dict[str, float],
    ) -> list[float]:
        beta = _ls3_beta_from_policy_input(policy_input, selection_entry)
        return _ls3_terrain_onehot(self.terrain) + [
            beta[0],
            beta[1],
            beta[2],
            _ls3_code_norm(gate.get("ram_level", "unknown"), LEARNED_STACK_V3_LEVEL_CODE, 3.0),
            _ls3_code_norm(gate.get("ram_gate_action", "unknown"), LEARNED_STACK_V3_GATE_ACTION_CODE, 4.0),
            _as_float(gate.get("would_override", 0.0), 0.0),
            _as_float(gate.get("control_risk", 0.0), 0.0),
            _as_float(gate.get("future_risk", 0.0), 0.0),
            _as_float(gate.get("fallen_prob", 0.0), 0.0),
            _as_float(gate.get("recovery_prob", 0.0), 0.0),
            _ls3_ram_sigma(policy_input, gate),
            _ls3_ram_rho_norm(policy_input, gate),
            _as_float(getattr(gms_out, "vx_scale", 1.0), 1.0),
            _as_float(getattr(gms_out, "body_height_delta", 0.0), 0.0),
            _as_float(getattr(gms_out, "clearance_delta", 0.0), 0.0),
            float(final_command.get("vx", 0.0)),
            float(final_command.get("body_height", 0.0)),
            float(final_command.get("swing_clearance", 0.0)),
            float(final_command.get("enable", 1.0)),
        ]

    def _learned_stack_v3_ram_step(
        self,
        selection_entry: dict[str, Any],
        gate: dict[str, Any],
        gms_out: Any,
        policy_input: Any,
        final_command: dict[str, float],
    ) -> dict[str, float]:
        beta = _ls3_objective_v2_beta_prior_for_ram_window(self.terrain)
        rule_label = _ls3_label(getattr(gms_out, "mode", "unknown"))
        proxy_intervention = max(
            _as_float(gate.get("would_override", 0.0), 0.0),
            _as_float(gate.get("control_risk", 0.0), 0.0),
            _as_float(gate.get("fallen_prob", 0.0), 0.0),
            _as_float(gate.get("recovery_prob", 0.0), 0.0),
        )
        return {
            "final_vx": float(final_command.get("vx", 0.0)),
            "final_yaw": float(final_command.get("yaw_rate", 0.0)),
            "final_body_height": float(final_command.get("body_height", 0.0)),
            "final_clearance": float(final_command.get("swing_clearance", 0.0)),
            "final_enable": float(final_command.get("enable", 1.0)),
            "rule_gms_code": _ls3_code_norm(rule_label, LEARNED_STACK_V3_GMS_CODE, 5.0),
            "learned_gms_code": _ls3_code_norm(rule_label, LEARNED_STACK_V3_GMS_CODE, 5.0),
            "learned_gms_prob": 1.0,
            "gms_disagree": 0.0,
            "learned_beta_motion": beta[0],
            "learned_beta_stability": beta[1],
            "learned_beta_energy": beta[2],
            "learned_ram_intervention_score": proxy_intervention,
            "learned_ram_future_override_mean": _as_float(gate.get("would_override", 0.0), 0.0),
            "rule_ram_level_code": _ls3_code_norm(gate.get("ram_level", "unknown"), LEARNED_STACK_V3_LEVEL_CODE, 3.0),
            "rule_gate_action_code": _ls3_code_norm(gate.get("ram_gate_action", "unknown"), LEARNED_STACK_V3_GATE_ACTION_CODE, 4.0),
            "rule_control_risk": _as_float(gate.get("control_risk", 0.0), 0.0),
            "rule_fallen_prob_weak": _as_float(gate.get("fallen_prob", 0.0), 0.0),
            "rule_sigma_mean": _ls3_ram_sigma(policy_input, gate),
        }

    def _learned_stack_v3_flatten_ram_window(self) -> list[float]:
        features = [
            "final_vx",
            "final_yaw",
            "final_body_height",
            "final_clearance",
            "final_enable",
            "rule_gms_code",
            "learned_gms_code",
            "learned_gms_prob",
            "gms_disagree",
            "learned_beta_motion",
            "learned_beta_stability",
            "learned_beta_energy",
            "learned_ram_intervention_score",
            "learned_ram_future_override_mean",
            "rule_ram_level_code",
            "rule_gate_action_code",
            "rule_control_risk",
            "rule_fallen_prob_weak",
            "rule_sigma_mean",
        ]
        target_len = 30
        if not self.learned_stack_v3_window:
            dummy = {k: 0.0 for k in features}
            use = [dummy] * target_len
        elif len(self.learned_stack_v3_window) < target_len:
            use = [self.learned_stack_v3_window[0]] * (target_len - len(self.learned_stack_v3_window)) + self.learned_stack_v3_window
        else:
            use = self.learned_stack_v3_window[-target_len:]

        x: list[float] = []
        for step in use:
            x.extend([float(step.get(k, 0.0)) for k in features])
        return x

    def _query_learned_stack_v3(
        self,
        selection_entry: dict[str, Any],
        gate: dict[str, Any],
        gms_out: Any,
        policy_input: Any,
        final_command: dict[str, float],
        elapsed: float,
    ) -> dict[str, Any] | None:
        if not self.enable_learned_stack_v3:
            self.latest_learned_stack_v3 = None
            return None

        if self.learned_stack_v3_sock is None:
            return {
                "ok": False,
                "error": "socket_not_initialized",
                "deploy_candidate": False,
                "deploy_active": False,
            }

        ram_step = self._learned_stack_v3_ram_step(
            selection_entry=selection_entry,
            gate=gate,
            gms_out=gms_out,
            policy_input=policy_input,
            final_command=final_command,
        )
        self.learned_stack_v3_window.append(ram_step)
        if len(self.learned_stack_v3_window) > 30:
            self.learned_stack_v3_window = self.learned_stack_v3_window[-30:]

        payload = {
            "request_id": f"fusion_v3_{self.terrain}_{int(self.counter)}",
            "terrain": self.terrain,
            "objective_x": self._learned_stack_v3_objective_x(
                selection_entry=selection_entry,
                gate=gate,
                final_command=final_command,
                elapsed=elapsed,
            ),
            "ram_x": self._learned_stack_v3_flatten_ram_window(),
            "gms_x": self._learned_stack_v3_gms_x(
                selection_entry=selection_entry,
                gate=gate,
                gms_out=gms_out,
                policy_input=policy_input,
                final_command=final_command,
            ),
        }

        try:
            data = json.dumps(payload).encode("utf-8")
            self.learned_stack_v3_sock.sendto(data, (self.learned_stack_v3_host, self.learned_stack_v3_port))
            resp_bytes, _ = self.learned_stack_v3_sock.recvfrom(65535)
            resp = json.loads(resp_bytes.decode("utf-8"))
        except Exception as e:
            resp = {
                "ok": False,
                "error": f"udp_query_failed: {e!r}",
            }

        gms = resp.get("gms") or {}
        ram = resp.get("ram") or {}
        gms_prob = _as_float(gms.get("prob", 0.0), 0.0)
        episode_success = _as_float(ram.get("episode_success", 0.0), 0.0)

        deploy_candidate = bool(
            resp.get("ok", False)
            and (resp.get("objective") or {}).get("ok", False)
            and ram.get("ok", False)
            and gms.get("ok", False)
            and gms_prob >= self.learned_stack_v3_min_gms_prob
            and episode_success >= self.learned_stack_v3_min_episode_success
        )

        out = {
            "ok": bool(resp.get("ok", False)),
            "deploy_candidate": deploy_candidate,
            "deploy_active": bool(self.deploy_learned_stack_v3 and deploy_candidate),
            "raw": resp,
            "payload_dims": {
                "objective_x": len(payload["objective_x"]),
                "ram_x": len(payload["ram_x"]),
                "gms_x": len(payload["gms_x"]),
            },
            "gms_label": gms.get("label", None),
            "gms_prob": gms_prob,
            "ram_intervention_score": ram.get("intervention_score", None),
            "ram_future_override_mean": ram.get("future_override_mean", None),
            "episode_success": episode_success,
        }

        self.latest_learned_stack_v3 = out
        return out


    def _select_command(self, base_command: dict[str, float], now_wall: float):
        semantic_mode = str(self.entry.get("semantic_mode", self.entry.get("semantic", "unknown")))
        raw_vx = float(base_command.get("vx", 0.0))

        # Primitive-evaluation commands are intentionally hand-authored low-level
        # references. In particular, micro/true backstep tests need vx < 0 to be
        # preserved exactly. Do not let GMS/meta-gait rewrite them into a
        # conservative forward command.
        primitive_bypass = (
            semantic_mode.startswith("primitive_eval_")
            or semantic_mode in {
                "backstep_braking_locomotion",
                "backstep_locomotion",
                "micro_backstep_locomotion",
                "true_backstep_locomotion",
            }
            or raw_vx < -1e-9
        )
        if primitive_bypass:
            return base_command, None, None, None, None, None, None

        if not self.enable_gms:
            return base_command, None, None, None, None, None, None

        gate = self._gate_context(now_wall)
        selection_entry = dict(self.entry)

        objective_out = self._predict_objective_selector(gate, selection_entry)
        self.latest_objective_selector_output = objective_out
        objective_hard_protect = False

        if objective_out is not None:
            # V0 runtime protection:
            # If RAM gate monitor has already hard-protected validated flat/fast
            # as stable/keep, do not let the Objective Selector re-block from
            # the same raw RAM scalars.
            objective_hard_protect = (
                self.terrain == "flat_normal"
                and semantic_mode == "validated_locomotion"
                and str(selection_entry.get("suggested_style", selection_entry.get("style", "unknown"))) == "fast"
                and gate.get("ram_level", "unknown") == "stable"
                and gate.get("ram_gate_action", "unknown") == "keep"
                and objective_out.deploy_label == "do_not_deploy_forward"
                and objective_out.reason == "safety_override"
            )
            if objective_hard_protect:
                self.get_logger().info(
                    "objective_selector_hard_protect "
                    "terrain=flat_normal semantic=validated_locomotion style=fast "
                    "gate=stable/keep raw_safety_override_suppressed"
                )
                objective_out.semantic_target = "validated_locomotion"
                objective_out.deploy_label = "deploy_forward"
                objective_out.beta_v = 0.55
                objective_out.beta_s = 0.25
                objective_out.beta_e = 0.20
                objective_out.safety_override = False
                objective_out.reason = "validated_flat_fast_hard_protect"

            # V0 calibrated-gate authority:
            # For non-flat cautious/high-clearance probes, RAM gate should choose
            # how to become conservative. Do not let raw RAM heads independently
            # convert a calibrated probe into no-deploy.
            objective_probe_protect = (
                str(selection_entry.get("fused_mode", self.entry.get("fused_mode", "unknown"))) == "locomotion"
                and semantic_mode in {"cautious_probe", "high_clearance_slow_probe"}
                and gate.get("ram_gate_action", "unknown") in {
                    "keep",
                    "would_cautious",
                    "would_conservative_probe",
                }
                and objective_out.deploy_label == "do_not_deploy_forward"
                and objective_out.reason == "safety_override"
            )
            if objective_probe_protect:
                self.get_logger().info(
                    "objective_selector_probe_protect "
                    f"terrain={self.terrain} semantic={semantic_mode} "
                    f"gate={gate.get('ram_level', 'unknown')}/"
                    f"{gate.get('ram_gate_action', 'unknown')} "
                    "raw_safety_override_suppressed"
                )
                objective_out.semantic_target = semantic_mode
                objective_out.deploy_label = "deploy_forward"
                beta = selection_entry.get("beta", {})
                objective_out.beta_v = float(beta.get("motion", objective_out.beta_v))
                objective_out.beta_s = float(beta.get("stability", objective_out.beta_s))
                objective_out.beta_e = float(beta.get("energy", objective_out.beta_e))
                objective_out.safety_override = False
                objective_out.reason = "calibrated_probe_gate_protect"

            if self.objective_selector_verbose:
                self.get_logger().info(
                    "objective_selector "
                    f"semantic={objective_out.semantic_target} "
                    f"deploy={objective_out.deploy_label} "
                    f"beta=({objective_out.beta_v:.3f},{objective_out.beta_s:.3f},{objective_out.beta_e:.3f}) "
                    f"safety={int(objective_out.safety_override)} "
                    f"reason={objective_out.reason} "
                    f"gate_level={gate.get('ram_level', 'unknown')} "
                    f"gate_action={gate.get('ram_gate_action', 'unknown')} "
                    f"ctrl={float(gate.get('control_risk', 0.0)):.3f} "
                    f"fallen={float(gate.get('fallen_prob', 0.0)):.3f} "
                    f"recovery={float(gate.get('recovery_prob', 0.0)):.3f}"
                )
            if self.objective_selector_apply_semantic:
                selection_entry["semantic_mode"] = objective_out.semantic_target
                selection_entry["semantic"] = objective_out.semantic_target
                selection_entry["decision"] = objective_out.semantic_target

            if self.objective_selector_apply_beta:
                selection_entry["beta"] = {
                    "motion": float(objective_out.beta_v),
                    "stability": float(objective_out.beta_s),
                    "energy": float(objective_out.beta_e),
                }

            if (
                self.objective_selector_block_no_deploy
                and objective_out.deploy_label == "do_not_deploy_forward"
            ):
                final_command = dict(base_command)
                final_command["vx"] = 0.0
                final_command["yaw_rate"] = 0.0
                final_command["enable"] = 0.0
                return final_command, None, None, None, None, None, None

        if objective_hard_protect:
            # Keep the raw RAM values visible in RAM/gate logs, but do not feed
            # the known false-positive flat RAM scalars into the meta-gait policy.
            gate = dict(gate)
            gate["ram_level"] = "stable"
            gate["ram_gate_action"] = "keep"
            gate["control_risk"] = 0.0
            gate["fallen_prob"] = 0.0
            gate["recovery_prob"] = 0.0
            gate["sigma_mean"] = 0.0
            gate["rho_norm"] = 0.0

        gms_in = input_from_policy_entry(
            selection_entry,
            ram_level=str(gate["ram_level"]),
            ram_gate_action=str(gate["ram_gate_action"]),
            control_risk=float(gate["control_risk"]),
            fallen_prob=float(gate["fallen_prob"]),
            recovery_prob=float(gate["recovery_prob"]),
            sigma_mean=gate["sigma_mean"],  # type: ignore[arg-type]
        )
        gms_out = select_gait_mode(gms_in)

        policy_input = build_high_level_policy_input(
            policy_entry=selection_entry,
            gms_in=gms_in,
            gms_out=gms_out,
            gate=gate,
            context=[],
            robot_state=[],
            goal=[],
        )

        # Current implementation uses RuleBasedMetaGaitPolicy. Later this call will
        # be replaced by an exported learned policy without changing the mapper.
        meta = self.meta_gait_policy.predict(policy_input)

        # Preserve the base command's yaw if the current rule-based policy did not
        # explicitly choose one.
        if abs(meta.yaw_rate) < 1e-12 and abs(base_command.get("yaw_rate", 0.0)) > 1e-12:
            meta = meta.__class__(
                **{**meta.__dict__, "yaw_rate": float(base_command.get("yaw_rate", 0.0))}
            )

        low_ref = decode_meta_gait_to_low_level_ref(meta)

        final_command = {
            "vx": float(low_ref.vx),
            "yaw_rate": float(low_ref.yaw_rate),
            "body_height": float(low_ref.body_height),
            "swing_clearance": float(low_ref.swing_clearance),
            "enable": float(low_ref.enable),
        }
        learned_stack_v3 = self._query_learned_stack_v3(
            selection_entry=selection_entry,
            gate=gate,
            gms_out=gms_out,
            policy_input=policy_input,
            final_command=final_command,
            elapsed=max(0.0, now_wall - self.t0),
        )

        return final_command, gms_in, gms_out, meta, low_ref, policy_input, learned_stack_v3

    def on_timer(self):
        now = time.time()
        elapsed = now - self.t0

        if self.duration_sec > 0.0 and elapsed > self.duration_sec:
            self.get_logger().info("finished publishing fusion policy command")
            self.stop_requested = True
            try:
                self.timer.cancel()
            except Exception:
                pass
            return

        base_command = self._ramped_base_command(elapsed)
        final_command, gms_in, gms_out, meta, low_ref, policy_input, learned_stack_v3 = self._select_command(base_command, now)

        msg = Float64MultiArray()
        msg.data = [
            float(self.counter),
            float(final_command["vx"]),
            float(final_command["yaw_rate"]),
            float(final_command["body_height"]),
            float(final_command["swing_clearance"]),
            float(final_command["enable"]),
        ]

        try:
            dbg = {
                "counter": int(self.counter),
                "terrain": str(self.terrain),
                "elapsed_sec": float(elapsed),
                "base_command": _jsonable(base_command),
                "final_command": _jsonable(final_command),
                "gms_in": _jsonable(gms_in),
                "gms_out": _jsonable(gms_out),
                "gate": _jsonable(gate) if "gate" in locals() else None,
                "meta": _jsonable(meta),
                "low_ref": _jsonable(low_ref),
                "policy_input": _jsonable(policy_input),
                "learned_stack_v3": _jsonable(learned_stack_v3),
            }
            dbg_msg = String()
            dbg_msg.data = json.dumps(dbg)
            self.debug_pub.publish(dbg_msg)
        except Exception as e:
            self.get_logger().warn(f"failed to publish highlevel debug: {e!r}")
        self.pub.publish(msg)

        beta_msg = Float64MultiArray()
        if (
            self.latest_objective_selector_output is not None
            and self.objective_selector_apply_beta
        ):
            beta_msg.data = [
                float(self.latest_objective_selector_output.beta_v),
                float(self.latest_objective_selector_output.beta_s),
                float(self.latest_objective_selector_output.beta_e),
            ]
        else:
            beta = self.entry.get("beta", {})
            beta_msg.data = [
                float(beta.get("motion", 1.0 / 3.0)),
                float(beta.get("stability", 1.0 / 3.0)),
                float(beta.get("energy", 1.0 / 3.0)),
            ]
        self.beta_pub.publish(beta_msg)

        self.counter += 1.0

        if now - self.last_print >= 1.0:
            self.last_print = now
            if gms_out is None:
                self.get_logger().info(
                    f"publishing /tracer/mpc_reference "
                    f"terrain={self.terrain} "
                    f"style={self.entry['suggested_style']} "
                    f"data={msg.data}"
                )
            else:
                self.get_logger().info(
                    f"publishing /tracer/mpc_reference "
                    f"terrain={self.terrain} "
                    f"style={self.entry['suggested_style']} "
                    f"gait_mode={gms_out.mode} "
                    f"gait_reason={gms_out.reason} "
                    f"ram_level={gms_in.ram_level if gms_in else 'unknown'} "
                    f"gate_action={gms_in.ram_gate_action if gms_in else 'unknown'} "
                    f"raw_vx={base_command['vx']:.3f} "
                    f"final_vx={final_command['vx']:.3f} "
                    f"final_h={final_command['body_height']:.3f} "
                    f"final_clr={final_command['swing_clearance']:.3f} "
                    f"enable={final_command['enable']:.1f} "
                    f"theta_period={meta.gait_period if meta else -1.0:.3f} "
                    f"theta_duty={meta.duty_factor if meta else -1.0:.3f} "
                    f"theta_imp={meta.impedance_scale if meta else -1.0:.3f} "
                    f"pi_beta_s={policy_input.beta.stability if policy_input else -1.0:.3f} "
                    f"pi_sigma={policy_input.ram.sigma if policy_input else -1.0:.3f} "
                    f"pi_rho_dim={len(policy_input.ram.rho) if policy_input else -1} "
                    f"data={msg.data}"
                )


def main():
    rclpy.init()
    node = TracerFusionPolicyMpcRefNode()
    try:
        while rclpy.ok() and not getattr(node, "stop_requested", False):
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
