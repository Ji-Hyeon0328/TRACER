from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from tracer_core.highlevel.gait_mode_selector import (
    GaitModeOutput,
    apply_gait_mode_to_command,
)
from tracer_core.highlevel.meta_gait import (
    LowLevelReference,
    MetaGaitCommand,
    ObjectiveWeights,
)


def _as_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _clamp(x: float, lo: float, hi: float) -> float:
    return min(hi, max(lo, x))


@dataclass(frozen=True)
class DecoderMapperConfig:
    """Safety clamps for the current A1-QP-MPC reference interface."""

    vx_min: float = -0.20
    vx_max: float = 0.35
    yaw_rate_min: float = -0.80
    yaw_rate_max: float = 0.80
    body_height_min: float = 0.24
    body_height_max: float = 0.39
    swing_clearance_min: float = 0.00
    swing_clearance_max: float = 0.16


MODE_DEFAULTS = {
    "fast": {
        "gait_period": 0.32,
        "duty_factor": 0.52,
        "step_length": 0.12,
        "stance_width": 0.23,
        "impedance_scale": 0.95,
        "residual_gain_scale": 0.90,
        "risk_scale": 0.80,
    },
    "normal": {
        "gait_period": 0.40,
        "duty_factor": 0.58,
        "step_length": 0.08,
        "stance_width": 0.24,
        "impedance_scale": 1.00,
        "residual_gain_scale": 1.00,
        "risk_scale": 1.00,
    },
    "conservative": {
        "gait_period": 0.55,
        "duty_factor": 0.68,
        "step_length": 0.045,
        "stance_width": 0.28,
        "impedance_scale": 1.25,
        "residual_gain_scale": 1.15,
        "risk_scale": 1.35,
    },
    "recovery": {
        "gait_period": 0.70,
        "duty_factor": 0.75,
        "step_length": 0.00,
        "stance_width": 0.30,
        "impedance_scale": 1.40,
        "residual_gain_scale": 1.30,
        "risk_scale": 1.75,
    },
    "avoid": {
        "gait_period": 0.70,
        "duty_factor": 0.75,
        "step_length": 0.00,
        "stance_width": 0.30,
        "impedance_scale": 1.00,
        "residual_gain_scale": 1.00,
        "risk_scale": 2.00,
    },
    "no_valid": {
        "gait_period": 0.70,
        "duty_factor": 0.75,
        "step_length": 0.00,
        "stance_width": 0.30,
        "impedance_scale": 1.00,
        "residual_gain_scale": 1.00,
        "risk_scale": 2.00,
    },
}


def _mode_defaults(mode: str) -> dict[str, float]:
    return dict(MODE_DEFAULTS.get(mode, MODE_DEFAULTS["conservative"]))


def beta_adjusted_defaults(mode: str, beta: ObjectiveWeights | None = None) -> dict[str, float]:
    """Apply a small interpretable beta effect to mode defaults.

    This is not the final learned objective-conditioned policy. It only makes the
    skeleton consistent with TRACER's beta semantics:
      - more stability -> longer stance, higher impedance, shorter step
      - more motion -> slightly shorter period, longer step
      - more energy -> slightly lower impedance / residual gain
    """

    defaults = _mode_defaults(mode)
    if beta is None:
        return defaults

    b = beta.normalized()
    stability_bias = b.stability - (1.0 / 3.0)
    motion_bias = b.motion - (1.0 / 3.0)
    energy_bias = b.energy - (1.0 / 3.0)

    defaults["duty_factor"] = _clamp(defaults["duty_factor"] + 0.12 * stability_bias, 0.45, 0.82)
    defaults["gait_period"] = _clamp(defaults["gait_period"] - 0.10 * motion_bias + 0.08 * stability_bias, 0.25, 0.85)
    defaults["step_length"] = _clamp(defaults["step_length"] + 0.06 * motion_bias - 0.05 * stability_bias, 0.00, 0.18)
    defaults["impedance_scale"] = _clamp(defaults["impedance_scale"] + 0.50 * stability_bias - 0.20 * energy_bias, 0.60, 1.80)
    defaults["residual_gain_scale"] = _clamp(defaults["residual_gain_scale"] + 0.35 * stability_bias - 0.20 * energy_bias, 0.50, 1.70)

    return defaults


def meta_gait_from_gms(
    base_command: Mapping[str, Any],
    gms_out: GaitModeOutput,
    *,
    beta: ObjectiveWeights | None = None,
) -> MetaGaitCommand:
    """Convert current command + GMS decision into meta-gait theta_t."""

    command = apply_gait_mode_to_command(base_command, gms_out)
    defaults = beta_adjusted_defaults(gms_out.mode, beta)

    return MetaGaitCommand(
        vx=_as_float(command.get("vx", 0.0), 0.0),
        yaw_rate=_as_float(command.get("yaw_rate", 0.0), 0.0),
        body_height=_as_float(command.get("body_height", 0.30), 0.30),
        swing_clearance=_as_float(command.get("swing_clearance", 0.035), 0.035),
        enable=_as_float(command.get("enable", 1.0), 1.0),
        gait_period=defaults["gait_period"],
        duty_factor=defaults["duty_factor"],
        step_length=defaults["step_length"],
        stance_width=defaults["stance_width"],
        impedance_scale=defaults["impedance_scale"],
        residual_gain_scale=defaults["residual_gain_scale"],
        risk_scale=defaults["risk_scale"],
        source_mode=gms_out.mode,
        source_reason=gms_out.reason,
        extras={
            "beta_motion": beta.motion if beta else None,
            "beta_stability": beta.stability if beta else None,
            "beta_energy": beta.energy if beta else None,
        },
    )


def decode_meta_gait_to_low_level_ref(
    meta: MetaGaitCommand,
    *,
    cfg: DecoderMapperConfig = DecoderMapperConfig(),
) -> LowLevelReference:
    """Map meta-gait theta_t to the current low-level reference interface."""

    enable = 1.0 if meta.enable >= 0.5 else 0.0

    vx = _clamp(meta.vx, cfg.vx_min, cfg.vx_max)
    yaw_rate = _clamp(meta.yaw_rate, cfg.yaw_rate_min, cfg.yaw_rate_max)
    body_height = _clamp(meta.body_height, cfg.body_height_min, cfg.body_height_max)
    swing_clearance = _clamp(
        meta.swing_clearance,
        cfg.swing_clearance_min,
        cfg.swing_clearance_max,
    )

    if enable <= 0.0:
        vx = 0.0
        yaw_rate = 0.0

    return LowLevelReference(
        vx=vx,
        yaw_rate=yaw_rate,
        body_height=body_height,
        swing_clearance=swing_clearance,
        enable=enable,
        meta=meta,
    )
