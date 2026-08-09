from __future__ import annotations

from dataclasses import dataclass
from math import cos
from typing import Any, Sequence

import numpy as np


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))


@dataclass(frozen=True)
class M7RewardConfig:
    """
    Fixed M7-v0 task reward.

    Objective Selector beta is intentionally NOT used here.
    Reward components are kept separate so beta weighting can
    be introduced later without changing the environment boundary.
    """

    max_forward_speed: float = 0.40

    roll_scale_rad: float = 0.35
    pitch_scale_rad: float = 0.35

    progress_weight: float = 2.0
    heading_weight: float = 0.05
    tilt_weight: float = 0.25
    action_smooth_weight: float = 0.05
    effort_proxy_weight: float = 0.02

    intervention_penalty: float = 2.0
    success_bonus: float = 5.0
    failure_penalty: float = 5.0
    time_cost: float = 0.01


DEFAULT_REWARD_CONFIG = M7RewardConfig()


def compute_m7_reward(
    *,
    previous_goal_distance: float,
    goal_distance: float,
    heading_error: float,
    roll: float,
    pitch: float,
    normalized_action: Sequence[float],
    previous_normalized_action: Sequence[float],
    decision_dt: float,
    override_active: bool = False,
    safety_state: str = "normal",
    success: bool = False,
    terminated: bool = False,
    truncated: bool = False,
    cfg: M7RewardConfig = DEFAULT_REWARD_CONFIG,
) -> tuple[float, dict[str, Any]]:
    """
    Compute one high-level reward tick.

    The simulator's native reward is not used as the M7 task reward.
    """

    dt = max(float(decision_dt), 1e-6)

    action = np.asarray(
        normalized_action,
        dtype=np.float64,
    ).reshape(-1)

    previous_action = np.asarray(
        previous_normalized_action,
        dtype=np.float64,
    ).reshape(-1)

    if action.shape != (4,):
        raise ValueError(
            f"normalized_action must have shape (4,), "
            f"got {action.shape}"
        )

    if previous_action.shape != (4,):
        raise ValueError(
            "previous_normalized_action must have shape "
            f"(4,), got {previous_action.shape}"
        )

    # Progress normalized so approximately max commanded forward
    # speed corresponds to O(1) progress reward.
    progress_m = (
        float(previous_goal_distance)
        - float(goal_distance)
    )

    progress = _clip(
        progress_m
        / max(
            cfg.max_forward_speed * dt,
            1e-6,
        ),
        -1.0,
        1.0,
    )

    # Small orientation-shaping term. Its weight is deliberately
    # much smaller than actual goal progress.
    heading = float(cos(float(heading_error)))

    tilt_cost = (
        (float(roll) / cfg.roll_scale_rad) ** 2
        + (float(pitch) / cfg.pitch_scale_rad) ** 2
    )

    stability_tilt = -_clip(
        tilt_cost,
        0.0,
        4.0,
    )

    delta_action = action - previous_action

    smooth_action = -float(
        np.mean(
            delta_action * delta_action
        )
    )

    # Not claimed as physical energy.
    # This is only a small command-magnitude proxy in M7-v0.
    effort_proxy = -float(
        np.mean(
            action * action
        )
    )

    unsafe = (
        str(safety_state).strip().lower()
        == "unsafe"
    )

    intervention = (
        1.0
        if bool(override_active) or unsafe
        else 0.0
    )

    failed = bool(terminated) and not bool(success)

    total = (
        cfg.progress_weight * progress
        + cfg.heading_weight * heading
        + cfg.tilt_weight * stability_tilt
        + cfg.action_smooth_weight * smooth_action
        + cfg.effort_proxy_weight * effort_proxy
        - cfg.intervention_penalty * intervention
        + cfg.success_bonus * float(bool(success))
        - cfg.failure_penalty * float(failed)
        - cfg.time_cost
    )

    components = {
        "motion_progress": float(progress),
        "motion_progress_m": float(progress_m),
        "motion_heading": float(heading),

        "stability_tilt": float(stability_tilt),

        "smooth_action": float(smooth_action),
        "effort_proxy": float(effort_proxy),

        "m4_intervention": float(intervention),

        "success": bool(success),
        "failure": bool(failed),
        "terminated": bool(terminated),
        "truncated": bool(truncated),

        "total": float(total),

        "note":
            "effort_proxy is command magnitude, "
            "not physical energy",
    }

    return float(total), components
