from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Sequence

import numpy as np

from tracer_core.highlevel.meta_gait import MetaGaitCommand


ACTION_NAMES = (
    "vx",
    "yaw_rate",
    "body_height",
    "swing_clearance",
)


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))


def _check_finite_vector(
    name: str,
    values: Sequence[float],
    expected_len: int | None = None,
) -> tuple[float, ...]:
    xs = tuple(float(x) for x in values)

    if expected_len is not None and len(xs) != expected_len:
        raise ValueError(
            f"{name} must have length {expected_len}, "
            f"got {len(xs)}"
        )

    for x in xs:
        if not isfinite(x):
            raise ValueError(
                f"{name} contains non-finite value: {x!r}"
            )

    return xs


def _denorm_about_default(
    a: float,
    lo: float,
    default: float,
    hi: float,
) -> float:
    """
    Map normalized action [-1,1] to a physical interval while
    preserving action=0 as the characterized nominal command.

      -1 -> lo
       0 -> default
      +1 -> hi
    """

    a = _clip(a, -1.0, 1.0)

    if a >= 0.0:
        return default + a * (hi - default)

    return default + (-a) * (lo - default)


def _norm_about_default(
    x: float,
    lo: float,
    default: float,
    hi: float,
) -> float:
    """
    Inverse of _denorm_about_default().
    """

    x = _clip(x, lo, hi)

    if x >= default:
        span = hi - default
        if span <= 1e-12:
            return 0.0
        return (x - default) / span

    span = default - lo
    if span <= 1e-12:
        return 0.0

    return -(default - x) / span


@dataclass(frozen=True)
class M7ActionSpec:
    """
    M7-v0 continuous high-level action contract.

    Physical bounds come from the current ICRA27 PyMPC
    meta-gait adapter, not from legacy Gazebo theta bounds.

    Structural timing is intentionally fixed in M7-v0.
    """

    vx_min: float = 0.00
    vx_nominal: float = 0.20
    vx_max: float = 0.40

    yaw_min: float = -0.40
    yaw_nominal: float = 0.00
    yaw_max: float = 0.40

    height_min: float = 0.24
    height_nominal: float = 0.30
    height_max: float = 0.32

    clearance_min: float = 0.03
    clearance_nominal: float = 0.06
    clearance_max: float = 0.09

    gait_period: float = 1.0 / 1.4
    duty_factor: float = 0.65


DEFAULT_ACTION_SPEC = M7ActionSpec()


def normalized_action_to_meta_gait(
    action: Sequence[float],
    spec: M7ActionSpec = DEFAULT_ACTION_SPEC,
    *,
    source_reason: str = "icra27_m7_rl_v0",
) -> MetaGaitCommand:
    """
    Convert normalized M7 policy action

        [vx, yaw_rate, body_height, swing_clearance] in [-1,1]^4

    into the six-field MetaGaitCommand consumed by frozen M5.

    gait_period / duty_factor remain at characterized nominal
    values in M7-v0.
    """

    a = _check_finite_vector(
        "normalized_action",
        action,
        expected_len=4,
    )

    vx = _denorm_about_default(
        a[0],
        spec.vx_min,
        spec.vx_nominal,
        spec.vx_max,
    )

    yaw = _denorm_about_default(
        a[1],
        spec.yaw_min,
        spec.yaw_nominal,
        spec.yaw_max,
    )

    height = _denorm_about_default(
        a[2],
        spec.height_min,
        spec.height_nominal,
        spec.height_max,
    )

    clearance = _denorm_about_default(
        a[3],
        spec.clearance_min,
        spec.clearance_nominal,
        spec.clearance_max,
    )

    return MetaGaitCommand(
        vx=vx,
        yaw_rate=yaw,
        body_height=height,
        swing_clearance=clearance,
        gait_period=spec.gait_period,
        duty_factor=spec.duty_factor,
        enable=1.0,
        source_mode="m7_rl",
        source_reason=source_reason,
        extras={
            "m7_normalized_action": [
                float(x)
                for x in a
            ],
            "structural_authority":
                "fixed_nominal_m7_v0",
        },
    )


def physical_action_to_normalized(
    values: Sequence[float],
    spec: M7ActionSpec = DEFAULT_ACTION_SPEC,
) -> np.ndarray:
    """
    Convert the four physical continuous command values back
    to the M7 normalized action coordinates.
    """

    x = _check_finite_vector(
        "physical_action",
        values,
        expected_len=4,
    )

    return np.asarray(
        [
            _norm_about_default(
                x[0],
                spec.vx_min,
                spec.vx_nominal,
                spec.vx_max,
            ),
            _norm_about_default(
                x[1],
                spec.yaw_min,
                spec.yaw_nominal,
                spec.yaw_max,
            ),
            _norm_about_default(
                x[2],
                spec.height_min,
                spec.height_nominal,
                spec.height_max,
            ),
            _norm_about_default(
                x[3],
                spec.clearance_min,
                spec.clearance_nominal,
                spec.clearance_max,
            ),
        ],
        dtype=np.float32,
    )


@dataclass(frozen=True)
class M7Observation:
    """
    Minimal high-level state for M7-v0.

    No joint-level q/dq are included intentionally.
    """

    oracle_context: tuple[float, ...]

    goal_dx_body: float
    goal_dy_body: float
    goal_distance: float
    heading_error: float

    base_vx_body: float
    base_vy_body: float
    yaw_rate: float
    base_z: float
    roll: float
    pitch: float

    applied_vx: float
    applied_yaw: float
    applied_height: float
    applied_clearance: float

    previous_action: tuple[float, float, float, float]


def build_observation(
    obs: M7Observation,
) -> np.ndarray:
    """
    Observation layout:

      oracle_context[K]
      goal[4]
      physical base response[6]
      applied command[4]
      previous normalized action[4]

    total dimension = K + 18
    """

    context = _check_finite_vector(
        "oracle_context",
        obs.oracle_context,
    )

    previous_action = _check_finite_vector(
        "previous_action",
        obs.previous_action,
        expected_len=4,
    )

    values = (
        *context,

        float(obs.goal_dx_body),
        float(obs.goal_dy_body),
        float(obs.goal_distance),
        float(obs.heading_error),

        float(obs.base_vx_body),
        float(obs.base_vy_body),
        float(obs.yaw_rate),
        float(obs.base_z),
        float(obs.roll),
        float(obs.pitch),

        float(obs.applied_vx),
        float(obs.applied_yaw),
        float(obs.applied_height),
        float(obs.applied_clearance),

        *previous_action,
    )

    _check_finite_vector(
        "observation",
        values,
    )

    return np.asarray(
        values,
        dtype=np.float32,
    )


def observation_names(
    context_dim: int,
) -> tuple[str, ...]:
    if context_dim < 0:
        raise ValueError("context_dim must be >= 0")

    return (
        *tuple(
            f"oracle_context_{i}"
            for i in range(context_dim)
        ),

        "goal_dx_body",
        "goal_dy_body",
        "goal_distance",
        "heading_error",

        "base_vx_body",
        "base_vy_body",
        "yaw_rate",
        "base_z",
        "roll",
        "pitch",

        "applied_vx",
        "applied_yaw",
        "applied_height",
        "applied_clearance",

        "previous_action_vx",
        "previous_action_yaw",
        "previous_action_height",
        "previous_action_clearance",
    )
