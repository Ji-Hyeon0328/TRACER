from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from tracer_core.highlevel.meta_gait import MetaGaitCommand, ObjectiveWeights


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))


def _denorm_with_default(a: float, lo: float, default: float, hi: float) -> float:
    """
    Map normalized theta in [-1, 1] to physical value while preserving:
      a = -1 -> lo
      a =  0 -> default
      a = +1 -> hi

    This matches configs/meta_gait/theta_action_space_v0.yaml, where each dim
    has an explicit default that is not always the midpoint of its range.
    """
    a = _clamp(float(a), -1.0, 1.0)
    if a >= 0.0:
        return default + a * (hi - default)
    return default + (-a) * (lo - default)


@dataclass(frozen=True)
class ThetaDirectConfig:
    nominal_vx: float = 0.06
    nominal_yaw_rate: float = 0.0
    nominal_body_height: float = 0.300
    nominal_clearance: float = 0.035
    nominal_gait_period: float = 0.40
    nominal_duty_factor: float = 0.58
    nominal_step_length: float = 0.08
    nominal_stance_width: float = 0.24

    min_vx: float = -0.05
    max_vx: float = 0.16
    min_body_height: float = 0.24
    max_body_height: float = 0.36
    min_clearance: float = 0.02
    max_clearance: float = 0.12


THETA_SPECS = {
    "gait_period_scale": {"range": (0.75, 1.35), "default": 1.0},
    "duty_factor_delta": {"range": (-0.12, 0.16), "default": 0.0},
    "step_length_scale": {"range": (0.35, 1.25), "default": 1.0},
    "stance_width_delta": {"range": (-0.04, 0.06), "default": 0.0},
    "body_height_delta": {"range": (-0.04, 0.06), "default": 0.0},
    "clearance_delta": {"range": (0.00, 0.09), "default": 0.0},
    "impedance_scale": {"range": (0.70, 1.60), "default": 1.0},
    "residual_scale": {"range": (0.00, 1.50), "default": 0.5},
}

THETA_NAMES = [
    "gait_period_scale",
    "duty_factor_delta",
    "step_length_scale",
    "stance_width_delta",
    "body_height_delta",
    "clearance_delta",
    "impedance_scale",
    "residual_scale",
]


def normalized_theta_to_physical(theta_norm: Sequence[float]) -> dict[str, float]:
    if len(theta_norm) != 8:
        raise ValueError(f"theta_norm must have length 8, got {len(theta_norm)}")

    out: dict[str, float] = {}
    for name, a in zip(THETA_NAMES, theta_norm):
        spec = THETA_SPECS[name]
        lo, hi = spec["range"]
        default = float(spec["default"])
        out[name] = _denorm_with_default(float(a), float(lo), default, float(hi))
    return out


def meta_gait_from_normalized_theta(
    theta_norm: Sequence[float],
    *,
    beta: ObjectiveWeights | Mapping[str, Any] | None = None,
    cfg: ThetaDirectConfig = ThetaDirectConfig(),
    source_reason: str = "theta_direct_mapper_v0",
) -> MetaGaitCommand:
    theta = normalized_theta_to_physical(theta_norm)

    if isinstance(beta, ObjectiveWeights):
        b = beta.normalized()
    else:
        b = ObjectiveWeights.from_mapping(beta).normalized()

    gait_period = _clamp(
        cfg.nominal_gait_period * theta["gait_period_scale"],
        0.24,
        0.75,
    )
    duty_factor = _clamp(
        cfg.nominal_duty_factor + theta["duty_factor_delta"],
        0.45,
        0.78,
    )
    step_length = _clamp(
        cfg.nominal_step_length * theta["step_length_scale"],
        0.00,
        0.18,
    )
    stance_width = _clamp(
        cfg.nominal_stance_width + theta["stance_width_delta"],
        0.18,
        0.32,
    )

    # Current Gazebo A1-QP-MPC bridge consumes only:
    # [counter, vx, yaw_rate, body_height, swing_clearance, enable].
    # Until gait period / duty / stance width / impedance are wired to LL,
    # step_length_scale is used as a safe proxy for forward command magnitude.
    vx = _clamp(
        cfg.nominal_vx * theta["step_length_scale"],
        cfg.min_vx,
        cfg.max_vx,
    )
    body_height = _clamp(
        cfg.nominal_body_height + theta["body_height_delta"],
        cfg.min_body_height,
        cfg.max_body_height,
    )
    swing_clearance = _clamp(
        cfg.nominal_clearance + theta["clearance_delta"],
        cfg.min_clearance,
        cfg.max_clearance,
    )

    return MetaGaitCommand(
        vx=vx,
        yaw_rate=cfg.nominal_yaw_rate,
        body_height=body_height,
        swing_clearance=swing_clearance,
        enable=1.0,
        gait_period=gait_period,
        duty_factor=duty_factor,
        step_length=step_length,
        stance_width=stance_width,
        impedance_scale=theta["impedance_scale"],
        residual_gain_scale=theta["residual_scale"],
        risk_scale=1.0 + max(0.0, b.stability - b.motion),
        source_mode="theta_direct",
        source_reason=source_reason,
        extras={
            "theta_norm": [float(x) for x in theta_norm],
            "theta_physical": theta,
            "beta_motion": b.motion,
            "beta_stability": b.stability,
            "beta_energy": b.energy,
            "runtime_note": (
                "Current bridge only applies vx/yaw/body_height/clearance/enable; "
                "other theta fields are logged for future LL integration."
            ),
        },
    )
