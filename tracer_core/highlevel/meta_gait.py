from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


def _as_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


@dataclass(frozen=True)
class ObjectiveWeights:
    """TRACER objective weights beta = [motion, stability, energy]."""

    motion: float = 1.0 / 3.0
    stability: float = 1.0 / 3.0
    energy: float = 1.0 / 3.0

    @staticmethod
    def from_mapping(data: Mapping[str, Any] | None) -> "ObjectiveWeights":
        if not data:
            return ObjectiveWeights()
        return ObjectiveWeights(
            motion=_as_float(data.get("motion", 1.0 / 3.0), 1.0 / 3.0),
            stability=_as_float(data.get("stability", 1.0 / 3.0), 1.0 / 3.0),
            energy=_as_float(data.get("energy", 1.0 / 3.0), 1.0 / 3.0),
        )

    def normalized(self) -> "ObjectiveWeights":
        vals = [max(0.0, self.motion), max(0.0, self.stability), max(0.0, self.energy)]
        s = sum(vals)
        if s <= 1e-9:
            return ObjectiveWeights()
        return ObjectiveWeights(vals[0] / s, vals[1] / s, vals[2] / s)


@dataclass(frozen=True)
class RamSignal:
    """Online RAM summary.

    rho is optional here because the current runtime mainly exposes scalar risk/gate
    values. Later we can pass the full RAM latent vector rho through this field.
    """

    rho: tuple[float, ...] = ()
    sigma: float = 0.0
    ram_level: str = "unknown"
    control_risk: float = 0.0
    fallen_prob: float = 0.0
    recovery_prob: float = 0.0


@dataclass(frozen=True)
class HighLevelPolicyInput:
    """Future learned high-level policy input schema.

    This is intentionally broader than the current runtime. The learned policy will
    eventually consume context c_t, RAM rho/sigma, objective beta, gait mode z, and
    robot/goal state, then output MetaGaitCommand theta_t.
    """

    context: tuple[float, ...] = ()
    ram: RamSignal = RamSignal()
    beta: ObjectiveWeights = ObjectiveWeights()
    gait_mode: str = "normal"
    robot_state: tuple[float, ...] = ()
    goal: tuple[float, ...] = ()


@dataclass(frozen=True)
class MetaGaitCommand:
    """Meta-gait parameter vector theta_t.

    Current runtime only consumes vx, yaw_rate, body_height, swing_clearance, enable.
    The extra fields are included now so the later RL policy has a stable target
    interface and the future MPC/WBC mapper can consume richer gait information.
    """

    vx: float = 0.0
    yaw_rate: float = 0.0
    body_height: float = 0.30
    swing_clearance: float = 0.035
    enable: float = 1.0

    gait_period: float = 0.40
    duty_factor: float = 0.58
    step_length: float = 0.08
    stance_width: float = 0.24

    impedance_scale: float = 1.0
    residual_gain_scale: float = 1.0
    risk_scale: float = 1.0

    source_mode: str = "normal"
    source_reason: str = ""
    extras: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LowLevelReference:
    """Current low-level reference.

    V0 maps to the existing /tracer/mpc_reference schema:
      [counter, vx, yaw_rate, body_height, swing_clearance, enable]

    Future versions can extend this class with COM, footstep, contact schedule,
    WBC task weights, and impedance references.
    """

    vx: float
    yaw_rate: float
    body_height: float
    swing_clearance: float
    enable: float
    meta: MetaGaitCommand | None = None

    def to_mpc_reference_array(self, counter: float = 0.0) -> list[float]:
        return [
            float(counter),
            float(self.vx),
            float(self.yaw_rate),
            float(self.body_height),
            float(self.swing_clearance),
            float(self.enable),
        ]


def tuple_from_sequence(x: Sequence[Any] | None) -> tuple[float, ...]:
    if x is None:
        return ()
    return tuple(_as_float(v, 0.0) for v in x)
