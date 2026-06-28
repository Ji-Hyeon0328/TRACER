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


def _forced_style_to_semantic_style(style):
    """Map a compact sweep style name to policy semantic/style fields."""
    s = str(style or "").strip()
    if not s:
        return None

    aliases = {
        "fast": ("validated_locomotion", "fast"),
        "validated": ("validated_locomotion", "fast"),
        "validated_locomotion": ("validated_locomotion", "fast"),

        "cautious": ("cautious_probe", "cautious"),
        "cautious_probe": ("cautious_probe", "cautious"),

        "high_clearance": ("high_clearance_slow_probe", "high_clearance"),
        "high_clearance_slow_probe": ("high_clearance_slow_probe", "high_clearance"),
        "slow_probe": ("high_clearance_slow_probe", "high_clearance"),

        "conservative": ("conservative_probe", "conservative"),
        "conservative_probe": ("conservative_probe", "conservative"),

        "safe_stop": ("recovery_needed", "safe_stop"),
        "active_hold": ("active_hold", "active_hold"),
    }
    return aliases.get(s)


def _apply_forced_style_to_entry(entry, forced_style):
    mapped = _forced_style_to_semantic_style(forced_style)
    if mapped is None:
        return entry

    semantic, style = mapped
    out = dict(entry)
    out["manual_override_style"] = forced_style
    out["semantic_mode"] = semantic
    out["suggested_style"] = style

    if style == "safe_stop":
        out["fused_mode"] = "recovery_needed"
    else:
        out["fused_mode"] = "locomotion"

    out["override_reason"] = "TRACER_FORCE_STYLE=" + str(forced_style)
    return out


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










LEARNED_STACK_V3_BETA_BLEND_ALPHA_DEFAULT = 0.10
LEARNED_STACK_V3_BETA_BLEND_MAX_DELTA = {
    "flat_normal": {"motion": 0.030, "stability": 0.030, "energy": 0.030},
    "rough_mid": {"motion": 0.025, "stability": 0.025, "energy": 0.025},
    "slope_5deg": {"motion": 0.020, "stability": 0.020, "energy": 0.020},
    "default": {"motion": 0.020, "stability": 0.020, "energy": 0.020},
}


def _ls3_limited_beta_blend(
    rule_beta: dict[str, float],
    learned_beta: dict[str, float],
    terrain: str,
    alpha: float,
) -> dict[str, float]:
    rule = _ls3_normalize_beta_dict(rule_beta)
    learned = _ls3_normalize_beta_dict(learned_beta)

    limits = LEARNED_STACK_V3_BETA_BLEND_MAX_DELTA.get(
        str(terrain),
        LEARNED_STACK_V3_BETA_BLEND_MAX_DELTA["default"],
    )

    out = {}
    for k in ["motion", "stability", "energy"]:
        proposed = rule[k] + float(alpha) * (learned[k] - rule[k])
        lo = rule[k] - limits[k]
        hi = rule[k] + limits[k]
        out[k] = _ls3_clamp(proposed, lo, hi)

    return _ls3_normalize_beta_dict(out)


LEARNED_STACK_V3_BETA_CLAMP_BY_TERRAIN = {
    # These are conservative shadow-only clamps.
    # They should not be interpreted as final learned-beta deployment limits.
    "flat_normal": {
        "motion": (0.35, 0.65),
        "stability": (0.20, 0.45),
        "energy": (0.05, 0.30),
    },
    "rough_mid": {
        "motion": (0.05, 0.30),
        "stability": (0.55, 0.85),
        "energy": (0.05, 0.30),
    },
    "slope_5deg": {
        "motion": (0.05, 0.35),
        "stability": (0.50, 0.85),
        "energy": (0.05, 0.30),
    },
    "default": {
        "motion": (0.05, 0.70),
        "stability": (0.20, 0.90),
        "energy": (0.02, 0.40),
    },
}


def _ls3_clamp(x: float, lo: float, hi: float) -> float:
    return min(float(hi), max(float(lo), float(x)))


def _ls3_normalize_beta_dict(beta: dict[str, Any]) -> dict[str, float]:
    m = _as_float(beta.get("motion", beta.get("beta_motion", 0.0)), 0.0)
    st = _as_float(beta.get("stability", beta.get("beta_stability", 0.0)), 0.0)
    e = _as_float(beta.get("energy", beta.get("beta_energy", 0.0)), 0.0)

    m = max(0.0, m)
    st = max(0.0, st)
    e = max(0.0, e)
    z = m + st + e

    if z <= 1e-8:
        return {"motion": 0.34, "stability": 0.56, "energy": 0.10}

    return {"motion": m / z, "stability": st / z, "energy": e / z}


def _ls3_clamp_beta_for_terrain(beta: dict[str, float], terrain: str) -> dict[str, float]:
    limits = LEARNED_STACK_V3_BETA_CLAMP_BY_TERRAIN.get(
        str(terrain),
        LEARNED_STACK_V3_BETA_CLAMP_BY_TERRAIN["default"],
    )

    out = {
        k: _ls3_clamp(float(beta.get(k, 0.0)), *limits[k])
        for k in ["motion", "stability", "energy"]
    }
    return _ls3_normalize_beta_dict(out)


LEARNED_STACK_V3_TERRAIN_ALLOWED_LABELS = {
    "flat_normal": {"fast"},
    "rough_mid": {"cautious_probe", "conservative"},
    "slope_5deg": {"high_clearance_slow_probe", "conservative"},
}

LEARNED_STACK_V3_TERRAIN_LABEL_ORDER = {
    "flat_normal": ["fast"],
    "rough_mid": ["cautious_probe", "conservative"],
    "slope_5deg": ["high_clearance_slow_probe", "conservative"],
    "default": ["fast", "cautious_probe", "high_clearance_slow_probe", "conservative"],
}

LEARNED_STACK_V3_LABEL_TO_SEMANTIC_STYLE = {
    "fast": ("validated_locomotion", "fast"),
    "cautious_probe": ("cautious_probe", "cautious"),
    "high_clearance_slow_probe": ("high_clearance_slow_probe", "high_clearance"),
    # select_gait_mode maps cautious_locomotion to mode="conservative".
    "conservative": ("cautious_locomotion", "cautious"),
}


def _ls3_rule_label_from_selection_entry(entry: dict[str, Any]) -> str:
    semantic = str(entry.get("semantic_mode", entry.get("semantic", "unknown")))
    style = str(entry.get("suggested_style", entry.get("style", "unknown")))

    if semantic == "validated_locomotion" and style == "fast":
        return "fast"
    if semantic == "cautious_probe":
        return "cautious_probe"
    if semantic == "high_clearance_slow_probe":
        return "high_clearance_slow_probe"
    if semantic in {
        "cautious_locomotion",
        "conservative_probe_recommended",
        "candidate_conditional_micro_brake",
    }:
        return "conservative"
    return _ls3_label(semantic)


def _ls3_is_same_or_more_conservative_for_terrain(
    terrain: str,
    rule_label: str,
    learned_label: str,
) -> bool:
    allowed = LEARNED_STACK_V3_TERRAIN_ALLOWED_LABELS.get(
        str(terrain),
        set(LEARNED_STACK_V3_TERRAIN_LABEL_ORDER["default"]),
    )
    if learned_label not in allowed:
        return False

    order = LEARNED_STACK_V3_TERRAIN_LABEL_ORDER.get(
        str(terrain),
        LEARNED_STACK_V3_TERRAIN_LABEL_ORDER["default"],
    )

    if rule_label not in order or learned_label not in order:
        return False

    return order.index(learned_label) >= order.index(rule_label)


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



def _objective_conditioned_force_style_from_env(terrain_key: str, entry: dict, logger=None) -> str:
    """Return a style selected by objective-conditioned selector, or empty string.

    This intentionally plugs into the already-working TRACER_FORCE_STYLE path.
    It is controlled by env vars and does not depend on ROS parameter plumbing.

    Env:
      TRACER_OBJECTIVE_CONDITIONED_SELECTOR_ENABLE=1
      TRACER_OBJECTIVE_CONDITIONED_SELECTOR_APPLY=1
      TRACER_OBJECTIVE_CONDITIONED_SELECTOR_MIN_CONFIDENCE=0.05
      TRACER_OBJECTIVE_CONDITIONED_SELECTOR_MODEL=configs/highlevel_policy/tracer_objective_conditioned_selector_v0.json

    Optional smoke/debug override:
      TRACER_OBJECTIVE_CONDITIONED_SELECTOR_OBJECTIVE=deploy_objective
      choices: motion_objective, stability_objective, deploy_objective
    """
    try:
        enable = bool(int(os.environ.get("TRACER_OBJECTIVE_CONDITIONED_SELECTOR_ENABLE", "0")))
        apply = bool(int(os.environ.get("TRACER_OBJECTIVE_CONDITIONED_SELECTOR_APPLY", "0")))
        min_conf = float(os.environ.get("TRACER_OBJECTIVE_CONDITIONED_SELECTOR_MIN_CONFIDENCE", "0.05"))
        model = os.environ.get(
            "TRACER_OBJECTIVE_CONDITIONED_SELECTOR_MODEL",
            "configs/highlevel_policy/tracer_objective_conditioned_selector_v0.json",
        )
        objective_override = os.environ.get("TRACER_OBJECTIVE_CONDITIONED_SELECTOR_OBJECTIVE", "").strip()

        if not enable:
            return ""

        from tracer_core.highlevel.objective_conditioned_selector import ObjectiveConditionedStyleSelector

        model_path = Path(model)
        if not model_path.is_absolute():
            model_path = Path.cwd() / model_path

        selector = ObjectiveConditionedStyleSelector(model_path)

        beta = entry.get("beta", None)
        if objective_override:
            prof = selector.objective_profiles.get(objective_override)
            if prof is not None:
                beta = prof

        sel = selector.select(
            terrain_key,
            beta,
            min_confidence=min_conf,
        )

        msg = (
            "objective_conditioned_selector enable=%d apply=%d terrain=%s "
            "objective=%s selected=%s semantic=%s conf=%.3f active=%d override=%s"
            % (
                int(enable),
                int(apply),
                terrain_key,
                sel.get("objective_name", "unknown"),
                sel.get("selected_style", "unknown"),
                sel.get("semantic_mode", "unknown"),
                float(sel.get("vote_confidence", 0.0)),
                int(bool(sel.get("active", False))),
                objective_override or "<none>",
            )
        )

        if logger is not None:
            logger.info(msg)
        else:
            print(msg)

        if not apply:
            return ""

        if not bool(sel.get("active", False)):
            return ""

        return str(sel.get("selected_style", "")).strip()

    except Exception as exc:
        if logger is not None:
            logger.error(f"objective_conditioned_selector failed: {exc}")
        else:
            print(f"objective_conditioned_selector failed: {exc}")
        return ""



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
        self.latest_learned_stack_v3_override = None
        self.latest_learned_stack_v3_beta_shadow = None
        self.latest_learned_stack_v3_beta_blend = None

        # Shadow-only RAM recovery gate. This never changes the active command path.
        # It only records whether a recovery gate would have triggered.
        self.ram_recovery_shadow_enable = bool(int(os.environ.get("TRACER_RAM_RECOVERY_SHADOW_ENABLE", "0")))
        self.ram_recovery_shadow_threshold = float(os.environ.get("TRACER_RAM_RECOVERY_SHADOW_THRESHOLD", "0.60"))
        self.ram_recovery_shadow_consecutive = int(os.environ.get("TRACER_RAM_RECOVERY_SHADOW_CONSECUTIVE", "3"))
        self.ram_recovery_shadow_streak = 0
        self.ram_recovery_shadow_trigger_count = 0
        self.ram_recovery_shadow_active_rows = 0
        self.latest_ram_recovery_shadow = {
            "enabled": self.ram_recovery_shadow_enable,
            "score": 0.0,
            "threshold": self.ram_recovery_shadow_threshold,
            "consecutive": self.ram_recovery_shadow_consecutive,
            "streak": 0,
            "would_recover": False,
            "trigger_count": 0,
            "active_rows": 0,
            "reason": "not_updated",
        }

        # Active RAM recovery candidate v0.
        # Default is fully disabled. ENABLE=1 computes an active decision.
        # APPLY=1 is required to modify the published MPC reference.
        # FORCE=1 is only for plumbing smoke tests on safe terrains.
        self.ram_recovery_active_enable = bool(int(os.environ.get("TRACER_RAM_RECOVERY_ACTIVE_ENABLE", "0")))
        self.ram_recovery_active_apply = bool(int(os.environ.get("TRACER_RAM_RECOVERY_ACTIVE_APPLY", "0")))
        self.ram_recovery_active_force = bool(int(os.environ.get("TRACER_RAM_RECOVERY_ACTIVE_FORCE", "0")))
        self.ram_recovery_active_vx = float(os.environ.get("TRACER_RAM_RECOVERY_ACTIVE_VX", "0.0"))
        self.ram_recovery_active_yaw_rate = float(os.environ.get("TRACER_RAM_RECOVERY_ACTIVE_YAW_RATE", "0.0"))
        self.ram_recovery_active_body_height = float(os.environ.get("TRACER_RAM_RECOVERY_ACTIVE_BODY_HEIGHT", "0.345"))
        self.ram_recovery_active_clearance = float(os.environ.get("TRACER_RAM_RECOVERY_ACTIVE_CLEARANCE", "0.090"))
        self.ram_recovery_active_trigger_count = 0
        self.ram_recovery_active_rows = 0
        self.ram_recovery_active_last_active = False
        self.latest_ram_recovery_active = {
            "enabled": self.ram_recovery_active_enable,
            "apply": self.ram_recovery_active_apply,
            "force": self.ram_recovery_active_force,
            "active": False,
            "applied": False,
            "trigger_count": 0,
            "active_rows": 0,
            "protected": False,
            "reason": "not_updated",
            "selected_recovery_command": None,
        }
        self.deploy_learned_stack_v3_beta_blend = bool(int(os.environ.get("TRACER_DEPLOY_LEARNED_STACK_V3_BETA_BLEND", "0")))
        self.learned_stack_v3_beta_blend_alpha = float(os.environ.get("TRACER_LEARNED_STACK_V3_BETA_BLEND_ALPHA", "0.10"))
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
        self.force_style = str(os.environ.get("TRACER_FORCE_STYLE", "")).strip()
        if not self.force_style:
            self.force_style = _objective_conditioned_force_style_from_env(self.terrain, self.entry, self.get_logger())
        if self.force_style:
            forced_entry = _apply_forced_style_to_entry(self.entry, self.force_style)
            if forced_entry is self.entry:
                self.get_logger().warn(
                    "unknown TRACER_FORCE_STYLE=%r; keeping policy entry style=%s"
                    % (
                        self.force_style,
                        self.entry.get("suggested_style", self.entry.get("style", "unknown")),
                    )
                )
            else:
                old_style = self.entry.get("suggested_style", self.entry.get("style", "unknown"))
                self.entry = forced_entry
                self.get_logger().warn(
                    "TRACER_FORCE_STYLE applied: %s -> %s semantic=%s"
                    % (
                        old_style,
                        self.entry.get("suggested_style"),
                        self.entry.get("semantic_mode"),
                    )
                )
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

        self.get_logger().info(
            f"RAM recovery shadow gate enable={int(self.ram_recovery_shadow_enable)} "
            f"threshold={self.ram_recovery_shadow_threshold:.3f} "
            f"consecutive={self.ram_recovery_shadow_consecutive}"
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




    def _learned_stack_v3_beta_shadow(
        self,
        selection_entry: dict[str, Any],
    ) -> dict[str, Any]:
        """Compute learned beta shadow proposal.

        This is intentionally shadow-only:
          - it reads the previous learned-stack-v3 objective beta,
          - applies terrain clamps,
          - compares against current rule/runtime beta,
          - but does not modify policy_input.beta or command output.
        """

        info: dict[str, Any] = {
            "enabled": bool(self.enable_learned_stack_v3),
            "active": False,
            "reason": "disabled",
            "terrain": str(self.terrain),
            "rule_beta": None,
            "learned_beta_raw": None,
            "learned_beta_clamped": None,
            "delta": None,
        }

        if not self.enable_learned_stack_v3:
            info["reason"] = "learned_stack_v3_not_enabled"
            return info

        prev = self.latest_learned_stack_v3
        if not isinstance(prev, dict) or not bool(prev.get("ok", False)):
            info["reason"] = "no_valid_previous_learned_result"
            return info

        objective = prev.get("objective") or {}

        # In the current fusion debug payload, the full supervised-stack UDP response
        # is preserved under learned_stack_v3["raw"]. The objective beta may therefore
        # live at either:
        #   prev["objective"]
        # or:
        #   prev["raw"]["objective"]
        if not objective:
            raw_resp = prev.get("raw") or {}
            if isinstance(raw_resp, dict):
                objective = raw_resp.get("objective") or {}

        raw_beta = objective.get("beta_dict") or {}

        # Fallback for payloads that only expose a beta list.
        if not raw_beta and isinstance(objective.get("beta"), list) and len(objective.get("beta")) >= 3:
            b = objective.get("beta")
            raw_beta = {"motion": b[0], "stability": b[1], "energy": b[2]}

        if not raw_beta:
            info["reason"] = "missing_learned_beta"
            return info

        rule_beta = _ls3_normalize_beta_dict(dict(selection_entry.get("beta", {}) or {}))
        learned_raw = _ls3_normalize_beta_dict(dict(raw_beta))
        learned_clamped = _ls3_clamp_beta_for_terrain(learned_raw, self.terrain)

        info["active"] = False
        info["reason"] = "shadow_only"
        info["rule_beta"] = rule_beta
        info["learned_beta_raw"] = learned_raw
        info["learned_beta_clamped"] = learned_clamped
        info["delta"] = {
            k: learned_clamped[k] - rule_beta.get(k, 0.0)
            for k in ["motion", "stability", "energy"]
        }
        return info



    def _learned_stack_v3_beta_blend_shadow_active(
        self,
        selection_entry: dict[str, Any],
    ) -> dict[str, Any]:
        """Apply a low-alpha learned-beta blend to selection_entry.

        This is the first limited beta-active path:
          - only uses learned beta after the shadow path succeeds,
          - alpha is small by default,
          - per-terrain max delta limits are applied,
          - RAM-triggered recovery remains inactive.
        """

        info: dict[str, Any] = {
            "enabled": False,
            "applied": False,
            "reason": "disabled",
            "terrain": str(self.terrain),
            "alpha": 0.0,
            "rule_beta": None,
            "learned_beta_clamped": None,
            "blended_beta": None,
            "delta_from_rule": None,
        }

        enabled = bool(getattr(self, "deploy_learned_stack_v3_beta_blend", False))
        info["enabled"] = enabled

        if not enabled:
            info["reason"] = "beta_blend_disabled"
            return info

        bs = getattr(self, "latest_learned_stack_v3_beta_shadow", None) or {}
        if not isinstance(bs, dict) or bs.get("reason") != "shadow_only":
            info["reason"] = "no_valid_beta_shadow"
            return info

        rule_beta = bs.get("rule_beta") or _ls3_normalize_beta_dict(dict(selection_entry.get("beta", {}) or {}))
        learned_beta = bs.get("learned_beta_clamped") or {}

        if not learned_beta:
            info["reason"] = "missing_learned_beta_clamped"
            return info

        alpha = float(getattr(self, "learned_stack_v3_beta_blend_alpha", LEARNED_STACK_V3_BETA_BLEND_ALPHA_DEFAULT))
        alpha = _ls3_clamp(alpha, 0.0, 0.20)

        blended = _ls3_limited_beta_blend(
            rule_beta=dict(rule_beta),
            learned_beta=dict(learned_beta),
            terrain=str(self.terrain),
            alpha=alpha,
        )

        # Mutate selection_entry in-place. Downstream input_from_policy_entry()
        # will now see the blended beta, but all other learned components remain gated.
        selection_entry["beta"] = {
            "motion": float(blended["motion"]),
            "stability": float(blended["stability"]),
            "energy": float(blended["energy"]),
        }

        info["applied"] = True
        info["reason"] = "low_alpha_limited_beta_blend"
        info["alpha"] = alpha
        info["rule_beta"] = dict(rule_beta)
        info["learned_beta_clamped"] = dict(learned_beta)
        info["blended_beta"] = dict(blended)
        info["delta_from_rule"] = {
            k: blended[k] - float(rule_beta.get(k, 0.0))
            for k in ["motion", "stability", "energy"]
        }
        return info


    def _learned_stack_v3_apply_gms_only_override(
        self,
        selection_entry: dict[str, Any],
        gate: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Apply one-step-delayed learned-stack-v3 GMS override.

        Safety contract:
          - Only active when TRACER_ENABLE_LEARNED_STACK_V3=1 and
            TRACER_DEPLOY_LEARNED_STACK_V3=1.
          - Uses the previous UDP query result, not a same-tick query.
          - Only allows same-or-more-conservative labels for the current terrain.
          - Does not use learned beta or learned RAM as active command inputs.
        """

        info: dict[str, Any] = {
            "enabled": bool(self.enable_learned_stack_v3),
            "deploy_requested": bool(self.deploy_learned_stack_v3),
            "used": False,
            "reason": "disabled",
            "terrain": str(self.terrain),
            "rule_label": None,
            "learned_label": None,
            "gms_prob": None,
            "episode_success": None,
        }

        if not self.enable_learned_stack_v3:
            info["reason"] = "learned_stack_v3_not_enabled"
            return selection_entry, info

        if not self.deploy_learned_stack_v3:
            info["reason"] = "deploy_not_requested"
            return selection_entry, info

        prev = self.latest_learned_stack_v3
        if not isinstance(prev, dict):
            info["reason"] = "no_previous_learned_result"
            return selection_entry, info

        if not bool(prev.get("ok", False)) or not bool(prev.get("deploy_candidate", False)):
            info["reason"] = "previous_result_not_candidate"
            return selection_entry, info

        learned_label = _ls3_label(prev.get("gms_label", "unknown"))
        gms_prob = _as_float(prev.get("gms_prob", 0.0), 0.0)
        episode_success = _as_float(prev.get("episode_success", 0.0), 0.0)

        info["learned_label"] = learned_label
        info["gms_prob"] = gms_prob
        info["episode_success"] = episode_success

        if gms_prob < self.learned_stack_v3_min_gms_prob:
            info["reason"] = "gms_prob_below_threshold"
            return selection_entry, info

        if episode_success < self.learned_stack_v3_min_episode_success:
            info["reason"] = "episode_success_below_threshold"
            return selection_entry, info

        rule_label = _ls3_rule_label_from_selection_entry(selection_entry)
        info["rule_label"] = rule_label

        if not _ls3_is_same_or_more_conservative_for_terrain(
            self.terrain,
            rule_label,
            learned_label,
        ):
            info["reason"] = "label_not_allowed_or_more_aggressive"
            return selection_entry, info

        semantic_style = LEARNED_STACK_V3_LABEL_TO_SEMANTIC_STYLE.get(learned_label)
        if semantic_style is None:
            info["reason"] = "no_semantic_mapping"
            return selection_entry, info

        semantic, style = semantic_style
        out = dict(selection_entry)
        out["semantic_mode"] = semantic
        out["semantic"] = semantic
        out["decision"] = semantic
        out["suggested_style"] = style
        out["learned_stack_v3_gms_override"] = {
            "from_rule_label": rule_label,
            "to_learned_label": learned_label,
            "semantic_mode": semantic,
            "suggested_style": style,
        }

        info["used"] = True
        info["reason"] = "applied_gms_only_same_or_more_conservative"
        info["semantic_mode"] = semantic
        info["suggested_style"] = style
        return out, info


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
            objective = resp.get("objective") or {}
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
            "deploy_requested": bool(self.deploy_learned_stack_v3),
            # This UDP response is query-time evidence only.
            # Actual active use is reported separately in learned_stack_v3_override.
            "deploy_active": False,
            "raw": resp,
            "payload_dims": {
            "objective": _jsonable(objective),
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

        selection_entry, learned_stack_v3_override = self._learned_stack_v3_apply_gms_only_override(
            selection_entry=selection_entry,
            gate=gate,
        )
        self.latest_learned_stack_v3_override = learned_stack_v3_override
        self.latest_learned_stack_v3_beta_shadow = self._learned_stack_v3_beta_shadow(selection_entry)
        self.latest_learned_stack_v3_beta_blend = self._learned_stack_v3_beta_blend_shadow_active(selection_entry)

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

    def _update_ram_recovery_shadow(self, gate: dict[str, Any]) -> dict[str, Any]:
        """Update shadow-only RAM recovery gate state.

        This method intentionally does not modify final_command, gms_out, or any
        active policy decision. It only produces a debug payload.
        """
        enabled = bool(self.ram_recovery_shadow_enable)
        threshold = float(self.ram_recovery_shadow_threshold)
        consecutive = max(1, int(self.ram_recovery_shadow_consecutive))

        score = _as_float(gate.get("recovery_prob", gate.get("recovery", 0.0)), 0.0)
        gate_level = str(gate.get("ram_level", "unknown"))
        gate_action = str(gate.get("ram_gate_action", "unknown"))

        if not enabled:
            self.latest_ram_recovery_shadow = {
                "enabled": False,
                "score": float(score),
                "threshold": threshold,
                "consecutive": consecutive,
                "streak": int(self.ram_recovery_shadow_streak),
                "would_recover": False,
                "trigger_count": int(self.ram_recovery_shadow_trigger_count),
                "active_rows": int(self.ram_recovery_shadow_active_rows),
                "gate_level": gate_level,
                "gate_action": gate_action,
                "reason": "disabled",
            }
            return self.latest_ram_recovery_shadow

        if gate_level == "unknown" or gate_action == "unknown":
            self.ram_recovery_shadow_streak = 0
            self.latest_ram_recovery_shadow = {
                "enabled": True,
                "score": float(score),
                "threshold": threshold,
                "consecutive": consecutive,
                "streak": 0,
                "would_recover": False,
                "trigger_count": int(self.ram_recovery_shadow_trigger_count),
                "active_rows": int(self.ram_recovery_shadow_active_rows),
                "gate_level": gate_level,
                "gate_action": gate_action,
                "reason": "no_fresh_gate",
            }
            return self.latest_ram_recovery_shadow

        if score >= threshold:
            self.ram_recovery_shadow_streak += 1
        else:
            self.ram_recovery_shadow_streak = 0

        would_recover = self.ram_recovery_shadow_streak >= consecutive

        if self.ram_recovery_shadow_streak == consecutive:
            self.ram_recovery_shadow_trigger_count += 1

        if would_recover:
            self.ram_recovery_shadow_active_rows += 1

        self.latest_ram_recovery_shadow = {
            "enabled": True,
            "score": float(score),
            "threshold": threshold,
            "consecutive": consecutive,
            "streak": int(self.ram_recovery_shadow_streak),
            "would_recover": bool(would_recover),
            "trigger_count": int(self.ram_recovery_shadow_trigger_count),
            "active_rows": int(self.ram_recovery_shadow_active_rows),
            "gate_level": gate_level,
            "gate_action": gate_action,
            "reason": "would_recover" if would_recover else "below_consecutive_gate",
        }
        return self.latest_ram_recovery_shadow

    def _ram_recovery_active_protect_reason(self, gms_in: Any, gms_out: Any) -> str | None:
        """Return a reason string when active recovery v0 should not override.

        V0 active recovery is only a conservative command plumbing test.
        It must not be applied to branches already marked as no-valid/avoid-required,
        because zero-vx hold was previously observed to fail there.
        """
        tokens: list[str] = []
        for key in ("semantic_mode", "fused_mode", "suggested_style", "runtime_fallback"):
            try:
                tokens.append(str(self.entry.get(key, "")))
            except Exception:
                pass

        for obj in (gms_in, gms_out, self.latest_learned_stack_v3_override, self.latest_learned_stack_v3):
            if obj is None:
                continue
            if isinstance(obj, dict):
                for key in ("mode", "reason", "semantic_mode", "fused_mode", "suggested_style", "ram_gate_action"):
                    if key in obj:
                        tokens.append(str(obj.get(key, "")))
            else:
                for key in ("mode", "reason", "semantic_mode", "fused_mode", "suggested_style", "ram_gate_action"):
                    if hasattr(obj, key):
                        try:
                            tokens.append(str(getattr(obj, key)))
                        except Exception:
                            pass

        joined = " ".join(tokens)
        protected_terms = (
            "no_valid_forward_recovery_needed",
            "no_valid_high_level_velocity_primitive",
            "avoid_required",
            "no_deployable_high_level_fallback",
            # Do not let active candidate v0 override already-declared recovery/stop branches.
            # V0 is only a conservative override plumbing test inside deployable locomotion branches.
            "recovery_needed",
            "safe_stop",
        )
        for term in protected_terms:
            if term in joined:
                return f"protected_semantic:{term}"
        return None

    def _update_ram_recovery_active(
        self,
        shadow: dict[str, Any],
        final_command: dict[str, float],
        gms_in: Any,
        gms_out: Any,
    ) -> tuple[dict[str, float], dict[str, Any]]:
        """Compute active recovery candidate v0 and optionally override command.

        This is not a recovery primitive. It is a guarded active-gate plumbing test.
        """
        enabled = bool(self.ram_recovery_active_enable)
        apply = bool(self.ram_recovery_active_apply)
        force = bool(self.ram_recovery_active_force)

        base = dict(final_command)
        selected = dict(final_command)

        shadow_would_recover = bool(shadow.get("would_recover", False))
        score = _as_float(shadow.get("score", 0.0), 0.0)
        threshold = _as_float(shadow.get("threshold", self.ram_recovery_shadow_threshold), self.ram_recovery_shadow_threshold)
        consecutive = int(shadow.get("consecutive", self.ram_recovery_shadow_consecutive))
        streak = int(shadow.get("streak", 0))

        protect_reason = self._ram_recovery_active_protect_reason(gms_in, gms_out)
        protected = protect_reason is not None

        if not enabled:
            active = False
            reason = "disabled"
        elif protected:
            active = False
            reason = protect_reason
        elif force:
            active = True
            reason = "force_active"
        elif shadow_would_recover:
            active = True
            reason = "shadow_would_recover"
        else:
            active = False
            reason = "no_trigger"

        if active:
            selected["vx"] = float(self.ram_recovery_active_vx)
            selected["yaw_rate"] = float(self.ram_recovery_active_yaw_rate)
            selected["body_height"] = max(
                float(base.get("body_height", 0.0)),
                float(self.ram_recovery_active_body_height),
            )
            selected["swing_clearance"] = max(
                float(base.get("swing_clearance", 0.0)),
                float(self.ram_recovery_active_clearance),
            )
            selected["enable"] = 1.0

            self.ram_recovery_active_rows += 1
            if not self.ram_recovery_active_last_active:
                self.ram_recovery_active_trigger_count += 1
            self.ram_recovery_active_last_active = True
        else:
            self.ram_recovery_active_last_active = False

        applied = bool(active and apply)
        out_command = selected if applied else base

        self.latest_ram_recovery_active = {
            "enabled": enabled,
            "apply": apply,
            "force": force,
            "active": bool(active),
            "applied": bool(applied),
            "score": float(score),
            "threshold": float(threshold),
            "consecutive": int(consecutive),
            "streak": int(streak),
            "shadow_would_recover": bool(shadow_would_recover),
            "trigger_count": int(self.ram_recovery_active_trigger_count),
            "active_rows": int(self.ram_recovery_active_rows),
            "protected": bool(protected),
            "reason": str(reason),
            "selected_recovery_command": _jsonable(selected) if active else None,
        }
        return out_command, self.latest_ram_recovery_active

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

        # Shadow-only recovery gate. It observes the latest RAM gate context.
        ram_recovery_shadow = self._update_ram_recovery_shadow(self._gate_context(now))

        # Active recovery candidate v0. By default this only computes debug.
        # It modifies final_command only when TRACER_RAM_RECOVERY_ACTIVE_APPLY=1.
        final_command, ram_recovery_active = self._update_ram_recovery_active(
            ram_recovery_shadow,
            final_command,
            gms_in,
            gms_out,
        )

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
                "learned_stack_v3_override": _jsonable(self.latest_learned_stack_v3_override),
                "learned_stack_v3_beta_shadow": _jsonable(self.latest_learned_stack_v3_beta_shadow),
                "learned_stack_v3_beta_blend": _jsonable(self.latest_learned_stack_v3_beta_blend),
                "ram_recovery_shadow": _jsonable(ram_recovery_shadow),
                "ram_recovery_active": _jsonable(ram_recovery_active),
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
                    f"active_rec={int(self.latest_ram_recovery_active.get('active', False))} "
                    f"active_applied={int(self.latest_ram_recovery_active.get('applied', False))} "
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
