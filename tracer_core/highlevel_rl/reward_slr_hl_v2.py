from __future__ import annotations

"""
SLR-HL-adapted-v2 reward for ICRA27 Phase 1.5.

Purpose
-------
Adapt the dense reward structure from the official SLR Go2
implementation to TRACER's frozen high-level M7 interface.

This is NOT an exact reproduction of SLR:

- SLR acts at 12D joint-command level.
- TRACER M7 policy acts at the 3D high-level command level
  [vx, yaw_rate, body_height].
- swing clearance is frozen by the current policy wrapper.
- joint acceleration and foot-clearance reward terms are
  intentionally omitted because they are unavailable at the
  frozen M7 high-level boundary.
- velocity tracking uses a policy-independent point-goal
  reference rather than the policy's own selected command.
- mechanical power is reconstructed from the ACK-aligned
  applied absolute-work interval.

The original positive SLR aggregation is retained as a
diagnostic. For variable-length point-goal training we subtract
the maximum positive tracking reward per transition so early
success does not lose future positive survival reward.
"""

from dataclasses import dataclass
import math
from typing import Sequence


REWARD_SCHEMA = "icra27_slr_hl_adapted_v2"

# ------------------------------------------------------------
# Official SLR reward semantics retained where portable.
# ------------------------------------------------------------

TRACKING_SIGMA = 0.25

SCALE_TRACKING_LIN_VEL = 1.0
SCALE_TRACKING_ANG_VEL = 0.5

SCALE_LIN_VEL_Z = -2.0
SCALE_ANG_VEL_XY = -0.05
SCALE_BASE_HEIGHT = -10.0
SCALE_POWER = -2.0e-5
SCALE_ACTION_RATE = -0.01
SCALE_ACTION_SMOOTHNESS = -0.01
SCALE_ORIENTATION = -0.2

# Omitted at the frozen M7 high-level boundary:
#
#   dof_acc        official Go2 scale = -2.5e-7
#   foot_clearance official Go2 scale = -0.5
#
# They remain explicit provenance rather than silently becoming
# zero-weight claims of exact SLR reproduction.

OMITTED_DOF_ACC_SCALE = -2.5e-7
OMITTED_FOOT_CLEARANCE_SCALE = -0.5


# ------------------------------------------------------------
# High-level adaptation contract.
# ------------------------------------------------------------

# M7 physical-command nominal/bounds.
REFERENCE_VX_MPS = 0.20
REFERENCE_VY_MPS = 0.0

REFERENCE_YAW_GAIN = 0.5
REFERENCE_YAW_LIMIT_RAD_S = 0.40

# Base-height target retained from the official SLR-Go2 reward.
#
# Flat nominal validation with the frozen M7/PyMPC stack measured
# an actual world-frame base height of approximately 0.3265 m
# (10-step mean), so the original 0.32 m target is also physically
# compatible with the current frozen controller operating point.
OFFICIAL_BASE_HEIGHT_TARGET_M = 0.32

# At perfect tracking with all penalty terms equal to zero:
#
#   tracking_lin_vel = 1
#   tracking_ang_vel = 1
#
# weighted rate = 1.0 + 0.5 = 1.5.
MAX_POSITIVE_REWARD_RATE = (
    SCALE_TRACKING_LIN_VEL
    + SCALE_TRACKING_ANG_VEL
)


def _finite(
    name: str,
    x: float,
) -> float:
    value = float(x)

    if not math.isfinite(value):
        raise ValueError(
            f"{name} must be finite, got {x!r}"
        )

    return value


def _positive(
    name: str,
    x: float,
) -> float:
    value = _finite(name, x)

    if value <= 0.0:
        raise ValueError(
            f"{name} must be > 0, got {value}"
        )

    return value


def _vector4(
    name: str,
    values: Sequence[float],
) -> tuple[float, float, float, float]:
    x = tuple(
        _finite(
            f"{name}[{i}]",
            value,
        )
        for i, value in enumerate(values)
    )

    if len(x) != 4:
        raise ValueError(
            f"{name} must have length 4, "
            f"got {len(x)}"
        )

    return (
        x[0],
        x[1],
        x[2],
        x[3],
    )


def _clip(
    x: float,
    lo: float,
    hi: float,
) -> float:
    return min(
        max(float(x), float(lo)),
        float(hi),
    )


def goal_tracking_reference(
    *,
    heading_error: float,
    vx_reference_mps: float = (
        REFERENCE_VX_MPS
    ),
    yaw_gain: float = (
        REFERENCE_YAW_GAIN
    ),
    yaw_limit_rad_s: float = (
        REFERENCE_YAW_LIMIT_RAD_S
    ),
) -> tuple[float, float, float]:
    """
    Policy-independent tracking command used by SLR-HL.

    The forward reference is the frozen M7 nominal velocity.

    SLR's official heading-command implementation uses

        yaw_cmd = clip(
            0.5 * heading_error,
            yaw_min,
            yaw_max,
        )

    We retain the 0.5 heading gain but adapt the saturation to
    M7's frozen physical yaw-rate authority +/-0.40 rad/s.
    """

    heading = _finite(
        "heading_error",
        heading_error,
    )

    vx = _finite(
        "vx_reference_mps",
        vx_reference_mps,
    )

    gain = _finite(
        "yaw_gain",
        yaw_gain,
    )

    limit = _positive(
        "yaw_limit_rad_s",
        yaw_limit_rad_s,
    )

    yaw_ref = _clip(
        gain * heading,
        -limit,
        limit,
    )

    return (
        float(vx),
        float(REFERENCE_VY_MPS),
        float(yaw_ref),
    )


def projected_gravity_xy_cost(
    *,
    roll: float,
    pitch: float,
) -> float:
    """
    Reconstruct the SLR projected-gravity XY orientation penalty
    from roll/pitch.

    With the conventional yaw-pitch-roll rotation,

        g_body_x^2 + g_body_y^2
        =
        sin(pitch)^2
        + sin(roll)^2 cos(pitch)^2.

    Signs of the projected gravity components do not matter
    because SLR squares them.
    """

    phi = _finite(
        "roll",
        roll,
    )

    theta = _finite(
        "pitch",
        pitch,
    )

    return float(
        math.sin(theta) ** 2
        + (
            math.sin(phi)
            * math.cos(theta)
        ) ** 2
    )


@dataclass(frozen=True)
class SLRHLReward:
    centered_reward: float

    positive_reward: float
    positive_reward_rate: float

    weighted_reward_rate_before_clip: float
    maximum_positive_reward: float

    tracking_lin_vel_reward: float
    tracking_ang_vel_reward: float

    lin_vel_z_cost: float
    ang_vel_xy_cost: float
    base_height_cost: float
    power_w: float
    action_rate_cost: float
    action_smoothness_cost: float
    orientation_cost: float

    target_vx_mps: float
    target_vy_mps: float
    target_yaw_rate_rad_s: float

    decision_dt_s: float
    energy_dt_s: float

    def as_dict(self) -> dict:
        return {
            "reward_schema":
                REWARD_SCHEMA,

            "slr_hl_centered_reward":
                float(
                    self.centered_reward
                ),

            "slr_hl_positive_reward":
                float(
                    self.positive_reward
                ),

            "slr_hl_positive_reward_rate":
                float(
                    self.positive_reward_rate
                ),

            "slr_hl_weighted_rate_before_clip":
                float(
                    self.weighted_reward_rate_before_clip
                ),

            "slr_hl_maximum_positive_reward":
                float(
                    self.maximum_positive_reward
                ),

            "slr_tracking_lin_vel":
                float(
                    self.tracking_lin_vel_reward
                ),

            "slr_tracking_ang_vel":
                float(
                    self.tracking_ang_vel_reward
                ),

            "slr_cost_lin_vel_z":
                float(
                    self.lin_vel_z_cost
                ),

            "slr_cost_ang_vel_xy":
                float(
                    self.ang_vel_xy_cost
                ),

            "slr_cost_base_height":
                float(
                    self.base_height_cost
                ),

            "slr_power_w":
                float(
                    self.power_w
                ),

            "slr_cost_action_rate":
                float(
                    self.action_rate_cost
                ),

            "slr_cost_action_smoothness":
                float(
                    self.action_smoothness_cost
                ),

            "slr_cost_orientation":
                float(
                    self.orientation_cost
                ),

            "slr_target_vx_mps":
                float(
                    self.target_vx_mps
                ),

            "slr_target_vy_mps":
                float(
                    self.target_vy_mps
                ),

            "slr_target_yaw_rate_rad_s":
                float(
                    self.target_yaw_rate_rad_s
                ),

            "decision_dt_s":
                float(
                    self.decision_dt_s
                ),

            "energy_dt_s":
                float(
                    self.energy_dt_s
                ),

            "slr_tracking_sigma":
                TRACKING_SIGMA,

            "slr_official_base_height_target_m":
                OFFICIAL_BASE_HEIGHT_TARGET_M,

            "slr_omitted_dof_acc":
                True,

            "slr_omitted_dof_acc_official_scale":
                OMITTED_DOF_ACC_SCALE,

            "slr_omitted_foot_clearance":
                True,

            "slr_omitted_foot_clearance_official_scale":
                OMITTED_FOOT_CLEARANCE_SCALE,

            "slr_adaptation_tracking_frame":
                "m7_body_yaw",

            "slr_height_measurement":
                "pympc_robot_height_estimate",

            "slr_height_target_adaptation":
                "current_applied_high_level_body_height",

            "slr_height_reference_semantics":
                "terrain_relative_tracking",

            "slr_adaptation_power":
                "ack_interval_average_applied_abs_power",

            "slr_termination_neutralization":
                "subtract_max_positive_reward_per_transition",
        }


def compute_slr_hl_reward_v2(
    *,
    heading_error: float,

    base_vx_body: float,
    base_vy_body: float,
    base_vz_world: float,

    base_wx: float,
    base_wy: float,
    base_wz: float,

    pympc_robot_height_estimate: float,
    applied_body_height: float,

    roll: float,
    pitch: float,

    applied_abs_energy_j: float,
    energy_dt_s: float,

    normalized_action: Sequence[float],
    previous_normalized_action: Sequence[float],
    previous_previous_normalized_action: Sequence[float],

    decision_dt_s: float,

    tracking_sigma: float = TRACKING_SIGMA,
) -> SLRHLReward:
    """
    Compute SLR-HL-adapted-v2.

    Official-style dense reward rate:

        tracking_lin_vel          +1.0
        tracking_ang_vel          +0.5
        lin_vel_z                 -2.0
        ang_vel_xy                -0.05
        base_height               -10.0
        powers                    -2e-5
        action_rate               -0.01
        action_smoothness         -0.01
        orientation               -0.2

    Then preserve SLR's only-positive aggregation:

        R_positive
        =
        dt * max(0, weighted_rate)

    For variable-length point-goal training:

        R_centered
        =
        R_positive
        - dt * 1.5

    so a mathematically perfect SLR tracking transition has
    centered reward zero rather than positive survival reward.
    """

    dt = _positive(
        "decision_dt_s",
        decision_dt_s,
    )

    e_dt = _positive(
        "energy_dt_s",
        energy_dt_s,
    )

    sigma = _positive(
        "tracking_sigma",
        tracking_sigma,
    )

    robot_height = _finite(
        "pympc_robot_height_estimate",
        pympc_robot_height_estimate,
    )

    applied_height = _finite(
        "applied_body_height",
        applied_body_height,
    )

    vx = _finite(
        "base_vx_body",
        base_vx_body,
    )

    vy = _finite(
        "base_vy_body",
        base_vy_body,
    )

    vz = _finite(
        "base_vz_world",
        base_vz_world,
    )

    wx = _finite(
        "base_wx",
        base_wx,
    )

    wy = _finite(
        "base_wy",
        base_wy,
    )

    wz = _finite(
        "base_wz",
        base_wz,
    )

    energy_j = _finite(
        "applied_abs_energy_j",
        applied_abs_energy_j,
    )

    if energy_j < 0.0:
        raise ValueError(
            "applied_abs_energy_j must be >= 0"
        )

    a_t = _vector4(
        "normalized_action",
        normalized_action,
    )

    a_tm1 = _vector4(
        "previous_normalized_action",
        previous_normalized_action,
    )

    a_tm2 = _vector4(
        "previous_previous_normalized_action",
        previous_previous_normalized_action,
    )

    (
        target_vx,
        target_vy,
        target_wz,
    ) = goal_tracking_reference(
        heading_error=heading_error,
    )

    lin_error = (
        (target_vx - vx) ** 2
        + (target_vy - vy) ** 2
    )

    ang_error = (
        target_wz - wz
    ) ** 2

    tracking_lin = math.exp(
        -lin_error / sigma
    )

    tracking_ang = math.exp(
        -ang_error / sigma
    )

    lin_vel_z_cost = vz ** 2

    ang_vel_xy_cost = (
        wx ** 2
        + wy ** 2
    )

    # SLR-HL-adapted-v2:
    #
    # Body height is itself a high-level policy action in M7.
    # Therefore compare the frozen PyMPC terrain-relative
    # robot-height estimate against the ACTUALLY APPLIED
    # high-level body-height reference.
    #
    # Do not use world-frame base z and do not force the
    # original fixed SLR-Go2 0.32 m target.
    base_height_cost = (
        robot_height
        - applied_height
    ) ** 2

    power_w = (
        energy_j / e_dt
    )

    action_rate_cost = sum(
        (
            a_t[i]
            - a_tm1[i]
        ) ** 2
        for i in range(4)
    )

    action_smoothness_cost = sum(
        (
            a_t[i]
            - 2.0 * a_tm1[i]
            + a_tm2[i]
        ) ** 2
        for i in range(4)
    )

    orientation_cost = (
        projected_gravity_xy_cost(
            roll=roll,
            pitch=pitch,
        )
    )

    weighted_rate = (
        SCALE_TRACKING_LIN_VEL
        * tracking_lin

        + SCALE_TRACKING_ANG_VEL
        * tracking_ang

        + SCALE_LIN_VEL_Z
        * lin_vel_z_cost

        + SCALE_ANG_VEL_XY
        * ang_vel_xy_cost

        + SCALE_BASE_HEIGHT
        * base_height_cost

        + SCALE_POWER
        * power_w

        + SCALE_ACTION_RATE
        * action_rate_cost

        + SCALE_ACTION_SMOOTHNESS
        * action_smoothness_cost

        + SCALE_ORIENTATION
        * orientation_cost
    )

    positive_rate = max(
        0.0,
        weighted_rate,
    )

    positive_reward = (
        dt * positive_rate
    )

    maximum_positive_reward = (
        dt
        * MAX_POSITIVE_REWARD_RATE
    )

    centered_reward = (
        positive_reward
        - maximum_positive_reward
    )

    return SLRHLReward(
        centered_reward=float(
            centered_reward
        ),

        positive_reward=float(
            positive_reward
        ),

        positive_reward_rate=float(
            positive_rate
        ),

        weighted_reward_rate_before_clip=float(
            weighted_rate
        ),

        maximum_positive_reward=float(
            maximum_positive_reward
        ),

        tracking_lin_vel_reward=float(
            tracking_lin
        ),

        tracking_ang_vel_reward=float(
            tracking_ang
        ),

        lin_vel_z_cost=float(
            lin_vel_z_cost
        ),

        ang_vel_xy_cost=float(
            ang_vel_xy_cost
        ),

        base_height_cost=float(
            base_height_cost
        ),

        power_w=float(
            power_w
        ),

        action_rate_cost=float(
            action_rate_cost
        ),

        action_smoothness_cost=float(
            action_smoothness_cost
        ),

        orientation_cost=float(
            orientation_cost
        ),

        target_vx_mps=float(
            target_vx
        ),

        target_vy_mps=float(
            target_vy
        ),

        target_yaw_rate_rad_s=float(
            target_wz
        ),

        decision_dt_s=float(dt),
        energy_dt_s=float(e_dt),
    )
