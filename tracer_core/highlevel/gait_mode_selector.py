from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


NO_VALID_SEMANTICS = {
    "no_valid_high_level_velocity_primitive",
    "no_valid_simple_primitive",
    "failed_high_level_velocity_primitive",
    "no_valid",
}

CAUTIOUS_SEMANTICS = {
    "cautious_locomotion",
    "conservative_probe_recommended",
    "candidate_conditional_micro_brake",
}

CAUTIOUS_PROBE_SEMANTICS = {
    "cautious_probe",
}

HIGH_CLEARANCE_SLOW_PROBE_SEMANTICS = {
    "high_clearance_slow_probe",
}

RECOVERY_SEMANTICS = {
    "recovery_needed",
    "recovery_locomotion",
}

# Raw online RAM recovery probability is currently treated as advisory.
# Hard-stop recovery should be triggered only by explicit high-level/gate decisions,
# because current online RAM scalars can contain false positives during long rollouts.
RECOVERY_GATE_ACTIONS = {
    "recovery_needed",
    "recovery",
    "force_recovery",
    "disable",
    "force_stop",
}


@dataclass(frozen=True)
class GaitModeInput:
    fused_mode: str = "unknown"
    semantic_mode: str = "unknown"
    suggested_style: str = "unknown"

    ram_level: str = "unknown"
    ram_gate_action: str = "unknown"
    control_risk: float = 0.0
    fallen_prob: float = 0.0
    recovery_prob: float = 0.0
    sigma_mean: float = 0.0


@dataclass(frozen=True)
class GaitModeOutput:
    mode: str
    reason: str
    vx_scale: float = 1.0
    yaw_scale: float = 1.0
    body_height_delta: float = 0.0
    clearance_delta: float = 0.0
    allow_motion: bool = True
    command_enable_override: float | None = None


def _as_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def input_from_policy_entry(
    entry: Mapping[str, Any],
    *,
    ram_level: str = "unknown",
    ram_gate_action: str = "unknown",
    control_risk: float = 0.0,
    fallen_prob: float = 0.0,
    recovery_prob: float = 0.0,
    sigma_mean: float | None = None,
) -> GaitModeInput:
    return GaitModeInput(
        fused_mode=str(entry.get("fused_mode", "unknown")),
        semantic_mode=str(entry.get("semantic_mode", "unknown")),
        suggested_style=str(entry.get("suggested_style", "unknown")),
        ram_level=str(ram_level),
        ram_gate_action=str(ram_gate_action),
        control_risk=_as_float(control_risk, 0.0),
        fallen_prob=_as_float(fallen_prob, 0.0),
        recovery_prob=_as_float(recovery_prob, 0.0),
        sigma_mean=_as_float(
            sigma_mean if sigma_mean is not None else entry.get("sigma_mean", 0.0),
            0.0,
        ),
    )


def select_gait_mode(gms_in: GaitModeInput) -> GaitModeOutput:
    fused_mode = gms_in.fused_mode
    semantic = gms_in.semantic_mode
    style = gms_in.suggested_style
    ram_level = gms_in.ram_level
    action = gms_in.ram_gate_action

    if semantic in NO_VALID_SEMANTICS:
        return GaitModeOutput(
            mode="no_valid",
            reason=f"semantic_mode={semantic}",
            vx_scale=0.0,
            yaw_scale=0.0,
            allow_motion=False,
            command_enable_override=0.0,
        )

    if fused_mode == "avoid_required":
        return GaitModeOutput(
            mode="avoid",
            reason=f"fused_mode={fused_mode}",
            vx_scale=0.0,
            yaw_scale=0.0,
            allow_motion=False,
            command_enable_override=0.0,
        )

    if fused_mode == "recovery_needed" or semantic in RECOVERY_SEMANTICS or action in RECOVERY_GATE_ACTIONS:
        return GaitModeOutput(
            mode="recovery",
            reason=(
                f"explicit recovery condition: fused_mode={fused_mode}, "
                f"semantic={semantic}, gate_action={action}, "
                f"recovery_prob={gms_in.recovery_prob:.3f}"
            ),
            vx_scale=0.0,
            yaw_scale=0.0,
            body_height_delta=0.02,
            clearance_delta=0.02,
            allow_motion=False,
            command_enable_override=0.0,
        )

    if action == "would_conservative_probe":
        return GaitModeOutput(
            mode="conservative",
            reason=f"ram_gate_action={action}",
            vx_scale=0.35,
            yaw_scale=0.5,
            body_height_delta=0.015,
            clearance_delta=0.03,
            allow_motion=True,
            command_enable_override=None,
        )

    if semantic in CAUTIOUS_PROBE_SEMANTICS:
        return GaitModeOutput(
            mode="cautious_probe",
            reason=f"cautious_probe semantic: semantic={semantic}, ram_level={ram_level}",
            vx_scale=0.20,
            yaw_scale=0.6,
            body_height_delta=0.010,
            clearance_delta=0.020,
            allow_motion=True,
            command_enable_override=None,
        )

    if semantic in HIGH_CLEARANCE_SLOW_PROBE_SEMANTICS:
        return GaitModeOutput(
            mode="high_clearance_slow_probe",
            reason=f"high_clearance_slow_probe semantic: semantic={semantic}, ram_level={ram_level}",
            vx_scale=0.16,
            yaw_scale=0.6,
            body_height_delta=0.020,
            clearance_delta=0.030,
            allow_motion=True,
            command_enable_override=None,
        )

    if semantic in CAUTIOUS_SEMANTICS or ram_level in {"unstable", "caution"}:
        return GaitModeOutput(
            mode="conservative",
            reason=f"cautious condition: semantic={semantic}, ram_level={ram_level}",
            vx_scale=0.35,
            yaw_scale=0.5,
            body_height_delta=0.015,
            clearance_delta=0.03,
            allow_motion=True,
            command_enable_override=None,
        )

    if semantic == "validated_locomotion" and style == "fast" and ram_level in {"stable", "unknown"}:
        return GaitModeOutput(
            mode="fast",
            reason=f"validated fast locomotion: style={style}, ram_level={ram_level}",
            vx_scale=1.0,
            yaw_scale=1.0,
            allow_motion=True,
            command_enable_override=None,
        )

    if semantic == "validated_locomotion":
        return GaitModeOutput(
            mode="normal",
            reason=f"validated locomotion: semantic={semantic}",
            vx_scale=1.0,
            yaw_scale=1.0,
            allow_motion=True,
            command_enable_override=None,
        )

    return GaitModeOutput(
        mode="conservative",
        reason=f"default conservative: fused_mode={fused_mode}, semantic={semantic}",
        vx_scale=0.35,
        yaw_scale=0.5,
        body_height_delta=0.015,
        clearance_delta=0.03,
        allow_motion=True,
        command_enable_override=None,
    )


def apply_gait_mode_to_command(
    command: Mapping[str, Any],
    gms_out: GaitModeOutput,
) -> dict[str, float]:
    vx = _as_float(command.get("vx", 0.0), 0.0)
    yaw_rate = _as_float(command.get("yaw_rate", 0.0), 0.0)
    body_height = _as_float(command.get("body_height", 0.30), 0.30)
    clearance = _as_float(
        command.get("swing_clearance", command.get("clearance", 0.035)),
        0.035,
    )
    enable = _as_float(command.get("enable", 1.0), 1.0)

    out = {
        "vx": vx * gms_out.vx_scale,
        "yaw_rate": yaw_rate * gms_out.yaw_scale,
        "body_height": body_height + gms_out.body_height_delta,
        "swing_clearance": clearance + gms_out.clearance_delta,
        "enable": enable,
    }

    if not gms_out.allow_motion:
        out["vx"] = 0.0
        out["yaw_rate"] = 0.0

    if gms_out.command_enable_override is not None:
        out["enable"] = float(gms_out.command_enable_override)

    return out
