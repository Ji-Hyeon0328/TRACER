from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class ThetaSafetyProjectorConfig:
    # Conservative normalized-theta bounds for current Gazebo A1-QP-MPC bridge.
    # These are not the final robot constraints. They are a rollout-time safety
    # projection to avoid unstable interpolation regions in BC/RL outputs.
    gait_period_min: float = -0.10
    gait_period_max: float = 0.60

    duty_min: float = -0.10
    duty_max: float = 0.70

    step_scale_min: float = -0.85
    step_scale_max: float = 0.25

    stance_min: float = 0.10
    stance_max: float = 0.70

    body_min: float = 0.15
    body_max: float = 0.75

    clearance_min: float = 0.20
    clearance_max: float = 0.75

    impedance_min: float = -0.20
    impedance_max: float = 0.75

    residual_min: float = -0.20
    residual_max: float = 0.60


DEFAULT_THETA_SAFETY_PROJECTOR_CONFIG = ThetaSafetyProjectorConfig()


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))


def project_theta_safety_v0(
    theta: Sequence[float],
    cfg: ThetaSafetyProjectorConfig = DEFAULT_THETA_SAFETY_PROJECTOR_CONFIG,
) -> list[float]:
    if len(theta) != 8:
        raise ValueError(f"theta must have len=8, got {len(theta)}")

    t = [float(x) for x in theta]

    bounds = [
        (cfg.gait_period_min, cfg.gait_period_max),
        (cfg.duty_min, cfg.duty_max),
        (cfg.step_scale_min, cfg.step_scale_max),
        (cfg.stance_min, cfg.stance_max),
        (cfg.body_min, cfg.body_max),
        (cfg.clearance_min, cfg.clearance_max),
        (cfg.impedance_min, cfg.impedance_max),
        (cfg.residual_min, cfg.residual_max),
    ]

    return [_clamp(x, lo, hi) for x, (lo, hi) in zip(t, bounds)]


def projection_delta_l1(theta: Sequence[float], theta_projected: Sequence[float]) -> float:
    return sum(abs(float(a) - float(b)) for a, b in zip(theta, theta_projected))
