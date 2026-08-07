from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from tracer_core.lowlevel.pympc_meta_gait_adapter import (
    PyMPCMetaGaitReference,
)


def _move_towards(
    current: float,
    target: float,
    max_rate: float,
    dt: float,
) -> float:
    if max_rate <= 0.0:
        raise ValueError(
            f"max_rate must be positive, got {max_rate}"
        )

    if dt <= 0.0:
        raise ValueError(
            f"dt must be positive, got {dt}"
        )

    max_delta = float(max_rate) * float(dt)
    error = float(target) - float(current)

    if abs(error) <= max_delta:
        return float(target)

    return float(
        current + np.sign(error) * max_delta
    )


@dataclass(frozen=True)
class PyMPCTransitionRates:
    """
    Conservative initial M2 slew limits.

    These are transition-management parameters, not claims about
    physical actuator limits. M3 can later characterize/tune them.
    """

    vx_rate_mps2: float = 0.60
    yaw_rate_rate_rad_s2: float = 1.00
    body_height_rate_mps: float = 0.05


DEFAULT_PYMPC_TRANSITION_RATES = PyMPCTransitionRates()


class PyMPCMetaGaitTransitionManager:
    """
    Safe-ish transition layer for TRACER -> PyMPC.

    Continuous references:
      - vx
      - yaw_rate
      - body_height

    are slew-rate limited continuously.

    Gait-structural references:
      - swing_clearance
      - gait_period
      - duty_factor

    are held pending and committed atomically only when the
    currently planned contact state is full stance.

    The manager intentionally does not modify PGG/STC objects itself.
    It produces one applied PyMPCMetaGaitReference which is then
    consumed by PyMPCMetaGaitAdapter.
    """

    def __init__(
        self,
        rates: PyMPCTransitionRates
        = DEFAULT_PYMPC_TRANSITION_RATES,
        structural_epsilon: float = 1e-9,
    ) -> None:
        self.rates = rates
        self.structural_epsilon = float(
            structural_epsilon
        )

        self._current: (
            PyMPCMetaGaitReference | None
        ) = None

        self._pending_structural: (
            PyMPCMetaGaitReference | None
        ) = None

        self.last_structural_commit = False
        self.structural_commit_count = 0

    def reset(self) -> None:
        self._current = None
        self._pending_structural = None
        self.last_structural_commit = False
        self.structural_commit_count = 0

    @property
    def current_reference(
        self,
    ) -> PyMPCMetaGaitReference | None:
        return self._current

    @property
    def has_pending_structural_update(self) -> bool:
        return self._pending_structural is not None

    @staticmethod
    def is_safe_structural_boundary(
        current_contact: Any,
    ) -> bool:
        if current_contact is None:
            return False

        contact = np.asarray(
            current_contact,
            dtype=float,
        ).reshape(-1)

        if contact.size != 4:
            return False

        # Planned full stance.
        return bool(
            np.all(contact > 0.5)
        )

    def _structural_differs(
        self,
        target: PyMPCMetaGaitReference,
        current: PyMPCMetaGaitReference,
    ) -> bool:
        eps = self.structural_epsilon

        return (
            abs(
                target.swing_clearance
                - current.swing_clearance
            ) > eps
            or abs(
                target.gait_period
                - current.gait_period
            ) > eps
            or abs(
                target.duty_factor
                - current.duty_factor
            ) > eps
        )

    def update(
        self,
        target: PyMPCMetaGaitReference,
        current_contact: Any,
        dt: float,
    ) -> PyMPCMetaGaitReference:
        self.last_structural_commit = False

        # First command defines the initial gait directly.
        if self._current is None:
            self._current = target
            self._pending_structural = None
            return target

        current = self._current

        # --------------------------------------------------------
        # Continuous references: slew-rate limited every tick.
        # --------------------------------------------------------
        vx = _move_towards(
            current.vx,
            target.vx,
            self.rates.vx_rate_mps2,
            dt,
        )

        yaw_rate = _move_towards(
            current.yaw_rate,
            target.yaw_rate,
            self.rates.yaw_rate_rate_rad_s2,
            dt,
        )

        body_height = _move_towards(
            current.body_height,
            target.body_height,
            self.rates.body_height_rate_mps,
            dt,
        )

        # --------------------------------------------------------
        # Structural references:
        # latest target wins while waiting for full stance.
        # --------------------------------------------------------
        if self._structural_differs(
            target,
            current,
        ):
            self._pending_structural = target
        else:
            self._pending_structural = None

        swing_clearance = (
            current.swing_clearance
        )
        gait_period = current.gait_period
        duty_factor = current.duty_factor

        if (
            self._pending_structural is not None
            and self.is_safe_structural_boundary(
                current_contact
            )
        ):
            pending = self._pending_structural

            swing_clearance = (
                pending.swing_clearance
            )
            gait_period = pending.gait_period
            duty_factor = pending.duty_factor

            self._pending_structural = None

            self.last_structural_commit = True
            self.structural_commit_count += 1

        applied = PyMPCMetaGaitReference(
            vx=vx,
            yaw_rate=yaw_rate,
            body_height=body_height,
            swing_clearance=swing_clearance,
            gait_period=gait_period,
            duty_factor=duty_factor,
        )

        self._current = applied
        return applied
