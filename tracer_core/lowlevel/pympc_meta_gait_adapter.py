from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))


def _finite(name: str, value: Any) -> float:
    x = float(value)
    if not isfinite(x):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return x


@dataclass(frozen=True)
class PyMPCMetaGaitLimits:
    """
    Conservative M2 limits.

    These bounds are intentionally based on the command-authority region
    validated in MuJoCo during ICRA27 M1, not on the legacy A1-QP-MPC
    normalized-theta bounds.
    """

    vx_min: float = 0.00
    vx_max: float = 0.40

    yaw_rate_min: float = -0.40
    yaw_rate_max: float = 0.40

    body_height_min: float = 0.24
    body_height_max: float = 0.32

    swing_clearance_min: float = 0.03
    swing_clearance_max: float = 0.09

    # Validated:
    #   1.0 Hz <-> 1.0 s
    #   1.4 Hz <-> 0.714285... s
    #   2.0 Hz <-> 0.5 s
    gait_period_min: float = 0.50
    gait_period_max: float = 1.00

    duty_factor_min: float = 0.50
    duty_factor_max: float = 0.80


DEFAULT_PYMPC_META_GAIT_LIMITS = PyMPCMetaGaitLimits()


@dataclass(frozen=True)
class PyMPCMetaGaitReference:
    vx: float
    yaw_rate: float
    body_height: float
    swing_clearance: float
    gait_period: float
    duty_factor: float

    @property
    def gait_frequency(self) -> float:
        return 1.0 / self.gait_period

    @property
    def stance_time(self) -> float:
        return self.duty_factor / self.gait_frequency

    @property
    def swing_period(self) -> float:
        return (1.0 - self.duty_factor) / self.gait_frequency


def project_meta_gait_for_pympc(
    command: Any,
    limits: PyMPCMetaGaitLimits = DEFAULT_PYMPC_META_GAIT_LIMITS,
) -> PyMPCMetaGaitReference:
    """
    Project the PyMPC-controlled subset of MetaGaitCommand into the
    experimentally validated M1 region.

    `command` is intentionally attribute-based so the adapter remains
    decoupled from the high-level module implementation.
    """

    vx = _finite("vx", command.vx)
    yaw_rate = _finite("yaw_rate", command.yaw_rate)
    body_height = _finite("body_height", command.body_height)
    swing_clearance = _finite(
        "swing_clearance",
        command.swing_clearance,
    )
    gait_period = _finite("gait_period", command.gait_period)
    duty_factor = _finite("duty_factor", command.duty_factor)

    return PyMPCMetaGaitReference(
        vx=_clamp(
            vx,
            limits.vx_min,
            limits.vx_max,
        ),
        yaw_rate=_clamp(
            yaw_rate,
            limits.yaw_rate_min,
            limits.yaw_rate_max,
        ),
        body_height=_clamp(
            body_height,
            limits.body_height_min,
            limits.body_height_max,
        ),
        swing_clearance=_clamp(
            swing_clearance,
            limits.swing_clearance_min,
            limits.swing_clearance_max,
        ),
        gait_period=_clamp(
            gait_period,
            limits.gait_period_min,
            limits.gait_period_max,
        ),
        duty_factor=_clamp(
            duty_factor,
            limits.duty_factor_min,
            limits.duty_factor_max,
        ),
    )


class PyMPCMetaGaitAdapter:
    """
    TRACER MetaGaitCommand -> Quadruped-PyMPC runtime adapter.

    Structural timing synchronization follows the same dependency used
    by upstream Quadruped-PyMPC:

        frequency
          -> PGG step frequency
          -> foothold-generator stance time
          -> swing-controller swing period

        duty factor
          -> PGG duty factor
          -> foothold-generator stance time
          -> swing-controller swing period

    The swing trajectory generator is regenerated only when timing or
    clearance actually changes. This is important because repeatedly
    regenerating it at every 2 ms simulation step would reset the
    trajectory unnecessarily.
    """

    def __init__(
        self,
        limits: PyMPCMetaGaitLimits = DEFAULT_PYMPC_META_GAIT_LIMITS,
        change_epsilon: float = 1e-9,
    ) -> None:
        self.limits = limits
        self.change_epsilon = float(change_epsilon)
        self._last_reference: PyMPCMetaGaitReference | None = None

    def reset(self) -> None:
        self._last_reference = None

    def project(self, command: Any) -> PyMPCMetaGaitReference:
        return project_meta_gait_for_pympc(
            command,
            self.limits,
        )

    def velocity_command(
        self,
        reference: PyMPCMetaGaitReference,
    ) -> tuple[float, float]:
        """
        Return scalar forward velocity and yaw-rate references.

        Simulation/ROS wrappers can map these onto the corresponding
        PyMPC ref_base_lin_vel/ref_base_ang_vel inputs.
        """
        return reference.vx, reference.yaw_rate

    def apply_runtime(
        self,
        wb_interface: Any,
        pympc_cfg: Any,
        command: Any,
    ) -> PyMPCMetaGaitReference:
        ref = self.project(command)

        previous = self._last_reference

        timing_changed = (
            previous is None
            or abs(
                ref.gait_period
                - previous.gait_period
            ) > self.change_epsilon
            or abs(
                ref.duty_factor
                - previous.duty_factor
            ) > self.change_epsilon
        )

        clearance_changed = (
            previous is None
            or abs(
                ref.swing_clearance
                - previous.swing_clearance
            ) > self.change_epsilon
        )

        # Body-height reference is read by WBInterface on each control update.
        pympc_cfg.simulation_params["ref_z"] = ref.body_height

        # Gait/contact timing.
        wb_interface.pgg.step_freq = ref.gait_frequency
        wb_interface.pgg.duty_factor = ref.duty_factor

        wb_interface.frg.stance_time = ref.stance_time

        # Swing geometry.
        wb_interface.step_height = ref.swing_clearance

        if timing_changed or clearance_changed:
            wb_interface.stc.regenerate_swing_trajectory_generator(
                step_height=ref.swing_clearance,
                swing_period=ref.swing_period,
            )

        self._last_reference = ref
        return ref
