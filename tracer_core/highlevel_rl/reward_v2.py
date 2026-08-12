from __future__ import annotations

from dataclasses import (
    dataclass,
    fields,
)
import math
from typing import Sequence

import numpy as np


# ----------------------------------------------------------------------
# ICRA27 simplified semantic objective.
#
# Historical reward.py / reward_v1.py remain immutable baselines.
#
# Convention:
#
#     C_motion >= 0
#     C_stability >= 0
#     C_energy >= 0
#
# and zero is best for every objective.
#
# PPO maximizes:
#
#     R_objective = - beta^T C
#
# Task completion / time horizon / M4 UNSAFE are intentionally
# outside this objective decomposition.
# ----------------------------------------------------------------------


MAX_FORWARD_PROGRESS_MPS = 0.40

ENERGY_REFERENCE_POWER_W = (
    21.847058714679683
)

ENERGY_SELECTOR_SCALE = 2.0

# Frozen M7 high-level decision interval:
# 5 Hz -> 0.20 s.
#
# Selector-facing energy cost is normalized by this nominal
# decision interval so that summing C_E over an episode is
# exactly proportional to accumulated absolute mechanical
# work, rather than to a sum of average-power samples.
NOMINAL_HIGH_LEVEL_DT_S = 0.20

# Absorbing cost assigned to each remaining high-level
# decision after an unsafe / native task failure.
#
# The value 1.0 is preference-independent and aligned with
# the semantic O(1) normalization of the objective costs:
#   C_m = 1 : maximum normalized reverse-progress burden
#   C_s = 1 : frozen M4 attitude unsafe boundary
#   C_E ~ 1 : upper feasible training energy scale
#
# This task-feasibility scale is intentionally outside beta.
TASK_FAILURE_ABSORBING_COST = 1.0


@dataclass(frozen=True)
class SimplifiedTRACERCosts:
    motion: float
    stability: float
    energy: float

    progress_rate_mps: float

    progress_normalized: float

    energy_power_ratio: float

    roll_fraction_of_unsafe: float
    pitch_fraction_of_unsafe: float

    def as_vector(self) -> tuple[float, float, float]:
        return (
            float(self.motion),
            float(self.stability),
            float(self.energy),
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "cost_motion":
                float(self.motion),

            "cost_stability":
                float(self.stability),

            "cost_energy":
                float(self.energy),

            "progress_rate_mps":
                float(self.progress_rate_mps),

            "progress_normalized":
                float(self.progress_normalized),

            "energy_power_ratio":
                float(self.energy_power_ratio),

            "roll_fraction_of_unsafe":
                float(
                    self.roll_fraction_of_unsafe
                ),

            "pitch_fraction_of_unsafe":
                float(
                    self.pitch_fraction_of_unsafe
                ),
        }


def _finite(
    name: str,
    value: float,
) -> float:
    value = float(value)

    if not math.isfinite(value):
        raise ValueError(
            f"{name} must be finite, got {value!r}"
        )

    return value


def _positive(
    name: str,
    value: float,
) -> float:
    value = _finite(
        name,
        value,
    )

    if value <= 0.0:
        raise ValueError(
            f"{name} must be > 0, got {value}"
        )

    return value


def validate_beta(
    beta: Sequence[float],
) -> tuple[float, float, float]:
    values = np.asarray(
        beta,
        dtype=np.float64,
    ).reshape(-1)

    if values.shape != (3,):
        raise ValueError(
            "beta must have shape (3,), "
            f"got {values.shape}"
        )

    if not np.all(
        np.isfinite(values)
    ):
        raise ValueError(
            "beta must be finite"
        )

    if np.any(
        values < 0.0
    ):
        raise ValueError(
            "beta must be non-negative"
        )

    total = float(
        np.sum(values)
    )

    if not math.isclose(
        total,
        1.0,
        rel_tol=0.0,
        abs_tol=1e-8,
    ):
        raise ValueError(
            "beta must sum to 1; "
            f"got {total}"
        )

    return tuple(
        float(x)
        for x in values
    )


def motion_cost(
    *,
    previous_goal_distance: float,
    goal_distance: float,
    decision_dt: float,
    max_forward_progress_mps: float = (
        MAX_FORWARD_PROGRESS_MPS
    ),
) -> tuple[float, float, float]:
    """
    Zero-best goal-directed motion cost.

    progress_rate > 0:
        moving toward goal

    progress_rate = 0:
        no progress

    progress_rate < 0:
        moving away from goal

    normalized progress is clipped to [-1, 1]:

        +1 -> maximum useful progress
         0 -> no progress
        -1 -> maximum reverse progress

    The corresponding cost is:

        C_m = 0.5 * (1 - normalized_progress)

    giving a bounded C_m in [0, 1].
    """

    previous = _finite(
        "previous_goal_distance",
        previous_goal_distance,
    )

    current = _finite(
        "goal_distance",
        goal_distance,
    )

    dt = _positive(
        "decision_dt",
        decision_dt,
    )

    vmax = _positive(
        "max_forward_progress_mps",
        max_forward_progress_mps,
    )

    progress_rate = (
        previous
        - current
    ) / dt

    progress_normalized = float(
        np.clip(
            progress_rate / vmax,
            -1.0,
            1.0,
        )
    )

    cost = (
        0.5
        * (
            1.0
            - progress_normalized
        )
    )

    return (
        float(cost),
        float(progress_rate),
        float(progress_normalized),
    )


def stability_cost(
    *,
    roll: float,
    pitch: float,
    roll_unsafe_rad: float,
    pitch_unsafe_rad: float,
) -> tuple[float, float, float]:
    """
    Zero-best attitude stability cost.

    This uses the same axis-wise geometry as M4 attitude
    safety limits:

        C_s = max(
            (|roll|  / roll_unsafe)^2,
            (|pitch| / pitch_unsafe)^2,
        )

    Therefore:

        level body             -> 0
        half of unsafe limit   -> 0.25
        either axis at limit   -> 1

    M4 UNSAFE itself remains a separate hard feasibility
    condition and is not replaced by this cost.
    """

    roll = _finite(
        "roll",
        roll,
    )

    pitch = _finite(
        "pitch",
        pitch,
    )

    roll_limit = _positive(
        "roll_unsafe_rad",
        roll_unsafe_rad,
    )

    pitch_limit = _positive(
        "pitch_unsafe_rad",
        pitch_unsafe_rad,
    )

    roll_fraction = (
        abs(roll)
        / roll_limit
    )

    pitch_fraction = (
        abs(pitch)
        / pitch_limit
    )

    cost = max(
        roll_fraction ** 2,
        pitch_fraction ** 2,
    )

    return (
        float(cost),
        float(roll_fraction),
        float(pitch_fraction),
    )


def energy_cost(
    *,
    applied_abs_energy_j: float,
    energy_dt_s: float,
    reference_power_w: float = (
        ENERGY_REFERENCE_POWER_W
    ),
    selector_scale: float = (
        ENERGY_SELECTOR_SCALE
    ),
    nominal_high_level_dt_s: float = (
        NOMINAL_HIGH_LEVEL_DT_S
    ),
) -> tuple[float, float]:
    """
    Zero-best physical mechanical-energy cost.

    Two related quantities are intentionally separated.

    Diagnostic physical power ratio:

        e_hat_power =
            Delta E_abs /
            (P_ref * Delta t_E)

    where Delta t_E is the actual ACK-aligned energy
    measurement interval.

    Selector-facing integrated mechanical-energy cost:

        C_E =
            Delta E_abs /
            (
                P_ref
                * Delta t_HL_nom
                * s_E
            )

    where Delta t_HL_nom = 0.20 s is the frozen 5 Hz
    high-level decision interval.

    Therefore:

        sum_t C_E,t
            proportional to
        total absolute mechanical work.

    The actual interval remains visible through
    energy_power_ratio for diagnostics.

    No clipping is performed here.
    """

    energy_j = _finite(
        "applied_abs_energy_j",
        applied_abs_energy_j,
    )

    if energy_j < 0.0:
        raise ValueError(
            "applied_abs_energy_j must be >= 0"
        )

    dt = _positive(
        "energy_dt_s",
        energy_dt_s,
    )

    p_ref = _positive(
        "reference_power_w",
        reference_power_w,
    )

    scale = _positive(
        "selector_scale",
        selector_scale,
    )

    nominal_dt = _positive(
        "nominal_high_level_dt_s",
        nominal_high_level_dt_s,
    )

    # Diagnostic only:
    # normalized average mechanical power.
    power_ratio = (
        energy_j
        / (
            p_ref
            * dt
        )
    )

    # Selector-facing objective:
    # normalized mechanical work per nominal HL decision.
    cost = (
        energy_j
        / (
            p_ref
            * nominal_dt
            * scale
        )
    )

    return (
        float(cost),
        float(power_ratio),
    )


def compute_simplified_tracer_costs(
    *,
    previous_goal_distance: float,
    goal_distance: float,
    decision_dt: float,

    roll: float,
    pitch: float,

    roll_unsafe_rad: float,
    pitch_unsafe_rad: float,

    applied_abs_energy_j: float,
    energy_dt_s: float,

    max_forward_progress_mps: float = (
        MAX_FORWARD_PROGRESS_MPS
    ),

    reference_power_w: float = (
        ENERGY_REFERENCE_POWER_W
    ),

    energy_selector_scale: float = (
        ENERGY_SELECTOR_SCALE
    ),

    nominal_high_level_dt_s: float = (
        NOMINAL_HIGH_LEVEL_DT_S
    ),
) -> SimplifiedTRACERCosts:
    (
        c_motion,
        progress_rate,
        progress_normalized,
    ) = motion_cost(
        previous_goal_distance=(
            previous_goal_distance
        ),

        goal_distance=(
            goal_distance
        ),

        decision_dt=(
            decision_dt
        ),

        max_forward_progress_mps=(
            max_forward_progress_mps
        ),
    )

    (
        c_stability,
        roll_fraction,
        pitch_fraction,
    ) = stability_cost(
        roll=roll,
        pitch=pitch,

        roll_unsafe_rad=(
            roll_unsafe_rad
        ),

        pitch_unsafe_rad=(
            pitch_unsafe_rad
        ),
    )

    (
        c_energy,
        energy_power_ratio,
    ) = energy_cost(
        applied_abs_energy_j=(
            applied_abs_energy_j
        ),

        energy_dt_s=(
            energy_dt_s
        ),

        reference_power_w=(
            reference_power_w
        ),

        selector_scale=(
            energy_selector_scale
        ),

        nominal_high_level_dt_s=(
            nominal_high_level_dt_s
        ),
    )

    return SimplifiedTRACERCosts(
        motion=c_motion,
        stability=c_stability,
        energy=c_energy,

        progress_rate_mps=(
            progress_rate
        ),

        progress_normalized=(
            progress_normalized
        ),

        energy_power_ratio=(
            energy_power_ratio
        ),

        roll_fraction_of_unsafe=(
            roll_fraction
        ),

        pitch_fraction_of_unsafe=(
            pitch_fraction
        ),
    )


def task_feasibility_tail_cost(
    *,
    episode_step: int,
    max_episode_steps: int,
    success: bool,
    m4_terminal: bool,
    native_terminated: bool,
    native_truncated: bool,
    absorbing_cost_per_step: float = (
        TASK_FAILURE_ABSORBING_COST
    ),
) -> tuple[float, bool, int]:
    """
    Beta-independent absorbing tail cost for premature
    task failure.

    A physical failure terminates the simulator immediately,
    so the remaining high-level decisions are unobserved.
    Assign one normalized failure cost to each missed
    decision:

        C_tail =
            I_failure
            * c_F
            * N_remaining

        N_remaining =
            max_episode_steps - episode_step

    Success and ordinary horizon exhaustion receive no
    additional task-feasibility cost.

    The caller must pass native_truncated rather than the
    combined Gymnasium `truncated` flag so that an ordinary
    time limit is not mistaken for premature failure.
    """

    step = int(episode_step)
    horizon = int(max_episode_steps)

    if step < 0:
        raise ValueError(
            "episode_step must be >= 0"
        )

    if horizon <= 0:
        raise ValueError(
            "max_episode_steps must be > 0"
        )

    if step > horizon:
        raise ValueError(
            "episode_step must not exceed "
            "max_episode_steps"
        )

    failure_cost = _positive(
        "absorbing_cost_per_step",
        absorbing_cost_per_step,
    )

    failure_terminal = bool(
        (not bool(success))
        and (
            bool(m4_terminal)
            or bool(native_terminated)
            or bool(native_truncated)
        )
    )

    remaining_steps = max(
        horizon - step,
        0,
    )

    tail_cost = (
        failure_cost
        * float(remaining_steps)
        if failure_terminal
        else 0.0
    )

    return (
        float(tail_cost),
        bool(failure_terminal),
        int(remaining_steps),
    )


def objective_cost(
    *,
    costs: SimplifiedTRACERCosts,
    beta: Sequence[float],
) -> float:
    beta_m, beta_s, beta_e = (
        validate_beta(
            beta
        )
    )

    return float(
        beta_m * costs.motion
        + beta_s * costs.stability
        + beta_e * costs.energy
    )


def objective_reward(
    *,
    costs: SimplifiedTRACERCosts,
    beta: Sequence[float],
) -> float:
    """
    PPO-facing objective reward only.

        R_obj = - beta^T C

    No task success bonus, timeout penalty, or M4 penalty
    belongs in this function.
    """

    return -objective_cost(
        costs=costs,
        beta=beta,
    )


def _find_m4_unsafe_limit(
    *,
    axis: str,
) -> tuple[str, float]:
    """
    Read the frozen M4 source-of-truth rather than duplicate
    its attitude safety constants.

    This helper deliberately fails closed if the current
    M4 config does not expose one unambiguous unsafe field
    for the requested axis.
    """

    from tracer_core.lowlevel.pympc_safety_monitor import (
        DEFAULT_PYMPC_SAFETY_MONITOR_CONFIG,
    )

    cfg = (
        DEFAULT_PYMPC_SAFETY_MONITOR_CONFIG
    )

    axis = str(
        axis
    ).lower()

    candidate_names = []

    try:
        field_names = [
            field.name
            for field in fields(cfg)
        ]

    except TypeError:
        field_names = list(
            vars(cfg).keys()
        )

    for name in field_names:
        lower = name.lower()

        if (
            axis in lower
            and "unsafe" in lower
        ):
            value = getattr(
                cfg,
                name,
            )

            try:
                number = float(
                    value
                )

            except (
                TypeError,
                ValueError,
            ):
                continue

            if (
                math.isfinite(number)
                and number > 0.0
            ):
                candidate_names.append(
                    (
                        name,
                        number,
                    )
                )

    if len(candidate_names) != 1:
        raise RuntimeError(
            "Could not uniquely identify "
            f"{axis} M4 unsafe limit. "
            f"Candidates={candidate_names}. "
            f"Available fields={field_names}"
        )

    name, value = (
        candidate_names[0]
    )

    # M4 receives roll/pitch in radians. Refuse clearly
    # implausible values rather than silently interpreting
    # degrees as radians.
    if value > math.pi:
        raise RuntimeError(
            f"M4 {axis} unsafe value "
            f"{value} from {name!r} "
            "does not look like radians."
        )

    return (
        str(name),
        float(value),
    )


def m4_unsafe_attitude_limits(
) -> tuple[float, float]:
    (
        _,
        roll_limit,
    ) = _find_m4_unsafe_limit(
        axis="roll"
    )

    (
        _,
        pitch_limit,
    ) = _find_m4_unsafe_limit(
        axis="pitch"
    )

    return (
        float(roll_limit),
        float(pitch_limit),
    )
