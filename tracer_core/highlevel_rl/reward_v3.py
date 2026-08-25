from __future__ import annotations

from dataclasses import dataclass
import math

from tracer_core.highlevel_rl.reward_v2 import (
    ENERGY_REFERENCE_POWER_W,
    ENERGY_SELECTOR_SCALE,
    MAX_FORWARD_PROGRESS_MPS,
    NOMINAL_HIGH_LEVEL_DT_S,
    energy_cost,
    motion_cost,
    stability_cost,
)


# ----------------------------------------------------------------------
# ICRA27 simplified semantic objective v3.
#
# v2 remains immutable.
#
# Main change:
#
#   OLD:
#       C_S = C_posture
#
#   NEW:
#       C_S = max(C_posture, C_traction)
#
# where:
#
#   C_posture
#       = endpoint roll/pitch attitude burden
#
#   C_traction
#       = established-stance contact-time-average
#         pointwise traction burden over the current
#         high-level decision interval.
#
# No clipping is applied.
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class SimplifiedTRACERCostsV3:
    motion: float
    stability: float
    energy: float

    posture: float
    traction: float

    progress_rate_mps: float
    progress_normalized: float

    energy_power_ratio: float

    roll_fraction_of_unsafe: float
    pitch_fraction_of_unsafe: float

    def as_vector(
        self,
    ) -> tuple[float, float, float]:
        return (
            float(self.motion),
            float(self.stability),
            float(self.energy),
        )

    def as_dict(
        self,
    ) -> dict[str, float]:
        return {
            "cost_motion":
                float(self.motion),

            "cost_stability":
                float(self.stability),

            "cost_energy":
                float(self.energy),

            "cost_posture":
                float(self.posture),

            "cost_traction":
                float(self.traction),

            "progress_rate_mps":
                float(
                    self.progress_rate_mps
                ),

            "progress_normalized":
                float(
                    self.progress_normalized
                ),

            "energy_power_ratio":
                float(
                    self.energy_power_ratio
                ),

            "roll_fraction_of_unsafe":
                float(
                    self.roll_fraction_of_unsafe
                ),

            "pitch_fraction_of_unsafe":
                float(
                    self.pitch_fraction_of_unsafe
                ),
        }


def _nonnegative_finite(
    name: str,
    value: float,
) -> float:
    value = float(value)

    if not math.isfinite(value):
        raise ValueError(
            f"{name} must be finite, got "
            f"{value!r}"
        )

    if value < 0.0:
        raise ValueError(
            f"{name} must be >= 0, got "
            f"{value}"
        )

    return value


def compute_simplified_tracer_costs_v3(
    *,
    previous_goal_distance: float,
    goal_distance: float,
    decision_dt: float,

    roll: float,
    pitch: float,

    roll_unsafe_rad: float,
    pitch_unsafe_rad: float,

    traction_cost: float,

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
) -> SimplifiedTRACERCostsV3:
    """
    Compute the v3 objective basis.

    Motion and energy semantics are exactly inherited from v2.

    Posture:
        endpoint attitude burden

            C_posture =
                max(
                    (|roll| / roll_unsafe)^2,
                    (|pitch| / pitch_unsafe)^2,
                )

    Traction:
        established-contact-time-average pointwise
        traction cost supplied by the M7 transport.

    Combined locomotion stability:

        C_S =
            max(
                C_posture,
                C_traction,
            )

    The max composition introduces no additional weighting
    hyperparameter and preserves the existing "dominant active
    instability mode" semantics of the v2 posture cost.
    """

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
        c_posture,
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

    c_traction = _nonnegative_finite(
        "traction_cost",
        traction_cost,
    )

    c_stability = max(
        float(c_posture),
        float(c_traction),
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

    return SimplifiedTRACERCostsV3(
        motion=float(
            c_motion
        ),

        stability=float(
            c_stability
        ),

        energy=float(
            c_energy
        ),

        posture=float(
            c_posture
        ),

        traction=float(
            c_traction
        ),

        progress_rate_mps=float(
            progress_rate
        ),

        progress_normalized=float(
            progress_normalized
        ),

        energy_power_ratio=float(
            energy_power_ratio
        ),

        roll_fraction_of_unsafe=float(
            roll_fraction
        ),

        pitch_fraction_of_unsafe=float(
            pitch_fraction
        ),
    )
