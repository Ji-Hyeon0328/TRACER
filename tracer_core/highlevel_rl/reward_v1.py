from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from tracer_core.highlevel_rl.reward import (
    DEFAULT_REWARD_CONFIG,
    M7RewardConfig,
    compute_m7_reward,
)


@dataclass(frozen=True)
class TRACERRewardConfig:
    """
    ICRA27 structured high-level reward v1.

    The three beta-controlled objectives are:

      0: motion
      1: stability
      2: command-economy proxy

    IMPORTANT:
    The third objective is NOT physical energy.
    Physical energy requires joint torque / velocity telemetry
    and will replace this proxy in a later stage.

    Safety, terminal bonuses/penalties, smoothness and time
    cost remain outside beta.
    """

    motion_progress_mix: float = 0.95
    motion_heading_mix: float = 0.05

    objective_scale: float = 2.0

    action_smooth_weight: float = 0.05

    intervention_penalty: float = 2.0
    success_bonus: float = 5.0
    failure_penalty: float = 5.0
    time_cost: float = 0.01


DEFAULT_TRACER_REWARD_CONFIG = (
    TRACERRewardConfig()
)


UNIFORM_BETA = (
    1.0 / 3.0,
    1.0 / 3.0,
    1.0 / 3.0,
)


def _as_action(
    x: Sequence[float],
    *,
    name: str,
) -> np.ndarray:
    a = np.asarray(
        x,
        dtype=np.float64,
    ).reshape(-1)

    if a.shape != (4,):
        raise ValueError(
            f"{name} must have shape (4,), "
            f"got {a.shape}"
        )

    return a


def _validate_beta(
    beta: Sequence[float],
) -> np.ndarray:
    b = np.asarray(
        beta,
        dtype=np.float64,
    ).reshape(-1)

    if b.shape != (3,):
        raise ValueError(
            "beta must have shape (3,), "
            f"got {b.shape}"
        )

    if not np.all(
        np.isfinite(b)
    ):
        raise ValueError(
            "beta must contain finite values"
        )

    if np.any(b < 0.0):
        raise ValueError(
            "beta must be non-negative"
        )

    total = float(
        np.sum(b)
    )

    if not np.isclose(
        total,
        1.0,
        atol=1e-8,
    ):
        raise ValueError(
            "beta must sum to 1.0; "
            f"got {total}"
        )

    return b


def compute_structured_objectives(
    *,
    motion_progress: float,
    motion_heading: float,
    stability_tilt: float,
    effort_proxy: float,
    cfg: TRACERRewardConfig = (
        DEFAULT_TRACER_REWARD_CONFIG
    ),
) -> dict[str, float]:
    """
    Convert existing M7-v0 components into comparable
    semantic objective scores.

    Expected source ranges:

      motion_progress : [-1, +1]
      motion_heading  : [-1, +1]
      stability_tilt  : [-4, 0]
      effort_proxy    : [-1, 0]
    """

    progress_mix = float(
        cfg.motion_progress_mix
    )

    heading_mix = float(
        cfg.motion_heading_mix
    )

    if not np.isclose(
        progress_mix + heading_mix,
        1.0,
        atol=1e-8,
    ):
        raise ValueError(
            "motion component mixes must sum to 1"
        )

    motion = (
        progress_mix
        * float(motion_progress)
        + heading_mix
        * float(motion_heading)
    )

    # Existing tilt component is clipped to [-4, 0].
    # Divide by four to obtain approximately [-1, 0].
    stability = (
        float(stability_tilt)
        / 4.0
    )

    # Existing command-magnitude proxy already lies in
    # approximately [-1, 0] for normalized actions.
    command_economy_proxy = float(
        effort_proxy
    )

    return {
        "motion":
            float(motion),

        "stability":
            float(stability),

        "command_economy_proxy":
            float(
                command_economy_proxy
            ),
    }


def compute_tracer_uniform_reward(
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
    beta: Sequence[float] = UNIFORM_BETA,
    source_cfg: M7RewardConfig = (
        DEFAULT_REWARD_CONFIG
    ),
    cfg: TRACERRewardConfig = (
        DEFAULT_TRACER_REWARD_CONFIG
    ),
) -> tuple[float, dict[str, Any]]:
    """
    Structured TRACER reward using the already validated
    M7-v0 primitive reward components.

    This function intentionally reuses compute_m7_reward()
    only as a component extractor. Its additive total is
    NOT used.
    """

    action = _as_action(
        normalized_action,
        name="normalized_action",
    )

    previous_action = _as_action(
        previous_normalized_action,
        name="previous_normalized_action",
    )

    b = _validate_beta(
        beta
    )

    (
        _baseline_total,
        base,
    ) = compute_m7_reward(
        previous_goal_distance=(
            previous_goal_distance
        ),
        goal_distance=goal_distance,
        heading_error=heading_error,
        roll=roll,
        pitch=pitch,
        normalized_action=action,
        previous_normalized_action=(
            previous_action
        ),
        decision_dt=decision_dt,
        override_active=override_active,
        safety_state=safety_state,
        success=success,
        terminated=terminated,
        truncated=truncated,
        cfg=source_cfg,
    )

    objectives = (
        compute_structured_objectives(
            motion_progress=(
                base[
                    "motion_progress"
                ]
            ),
            motion_heading=(
                base[
                    "motion_heading"
                ]
            ),
            stability_tilt=(
                base[
                    "stability_tilt"
                ]
            ),
            effort_proxy=(
                base[
                    "effort_proxy"
                ]
            ),
            cfg=cfg,
        )
    )

    objective_term = float(
        cfg.objective_scale
        * (
            b[0]
            * objectives["motion"]
            + b[1]
            * objectives["stability"]
            + b[2]
            * objectives[
                "command_economy_proxy"
            ]
        )
    )

    # Smoothness is kept outside beta.
    auxiliary_term = float(
        cfg.action_smooth_weight
        * base[
            "smooth_action"
        ]
        - cfg.time_cost
    )

    intervention = float(
        base[
            "m4_intervention"
        ]
    )

    failed = bool(
        base[
            "failure"
        ]
    )

    terminal_term = float(
        - cfg.intervention_penalty
        * intervention
        + cfg.success_bonus
        * float(bool(success))
        - cfg.failure_penalty
        * float(failed)
    )

    total = float(
        objective_term
        + auxiliary_term
        + terminal_term
    )

    components = {
        **base,

        "reward_schema":
            "icra27_tracer_structured_v1",

        "beta": [
            float(x)
            for x in b
        ],

        "objective_motion":
            objectives["motion"],

        "objective_stability":
            objectives["stability"],

        "objective_command_economy_proxy":
            objectives[
                "command_economy_proxy"
            ],

        "objective_term":
            objective_term,

        "auxiliary_term":
            auxiliary_term,

        "terminal_term":
            terminal_term,

        "total":
            total,

        "note":
            "third beta objective is command-economy "
            "proxy, not physical energy",
    }

    return total, components


def compute_fixed_additive_baseline(
    **kwargs,
) -> tuple[float, dict[str, Any]]:
    """
    Exact compatibility wrapper around the frozen M7-v0
    fixed-additive engineering reward.
    """

    total, components = (
        compute_m7_reward(
            **kwargs
        )
    )

    components = {
        **components,
        "reward_schema":
            "icra27_fixed_additive_v0",
    }

    return total, components
