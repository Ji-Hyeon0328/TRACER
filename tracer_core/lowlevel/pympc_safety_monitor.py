from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from math import isfinite, radians
from typing import Any

import numpy as np


class PyMPCSafetyState(str, Enum):
    NORMAL = "normal"
    WATCH = "watch"
    UNSAFE = "unsafe"


@dataclass(frozen=True)
class PyMPCSafetyMonitorConfig:
    """
    Initial empirical M4 runtime-monitor thresholds.

    These values are NOT formal stability/safety guarantees.
    They are intentionally configurable and will first be
    validated in monitor-only shadow mode against M3 traces.
    """

    support_window_s: float = 0.20

    watch_support_deficit_fraction: float = 0.15
    unsafe_support_deficit_fraction: float = 0.30

    watch_roll_rad: float = radians(8.0)
    watch_pitch_rad: float = radians(8.0)

    unsafe_roll_rad: float = radians(15.0)
    unsafe_pitch_rad: float = radians(15.0)

    min_base_height_m: float = 0.22

    # Support-mismatch monitoring is emphasized after a
    # structural gait transition.
    post_transition_monitor_s: float = 3.0


DEFAULT_PYMPC_SAFETY_MONITOR_CONFIG = (
    PyMPCSafetyMonitorConfig()
)


@dataclass(frozen=True)
class PyMPCSafetyStatus:
    state: PyMPCSafetyState

    planned_support_count: int
    physical_support_count: int
    current_support_deficit_count: int

    support_deficit_fraction: float
    support_window_samples: int

    roll_rad: float
    pitch_rad: float
    base_height_m: float

    post_transition_active: bool
    time_since_structural_commit_s: float | None

    reasons: tuple[str, ...]


def _contact_vector(
    name: str,
    value: Any,
) -> np.ndarray:
    x = np.asarray(
        value,
        dtype=float,
    ).reshape(-1)

    if x.size != 4:
        raise ValueError(
            f"{name} must contain 4 leg contacts, "
            f"got shape {x.shape}"
        )

    if not np.all(np.isfinite(x)):
        raise ValueError(
            f"{name} contains non-finite values: {x}"
        )

    return x > 0.5


def _finite_scalar(
    name: str,
    value: Any,
) -> float:
    x = float(value)

    if not isfinite(x):
        raise ValueError(
            f"{name} must be finite, got {value!r}"
        )

    return x


class PyMPCSafetyMonitor:
    """
    Empirical runtime health supervisor for the TRACER
    MetaGaitCommand -> PyMPC execution boundary.

    v0 is monitor-only:
      - it does not modify torque;
      - it does not claim CLF/CBF guarantees;
      - UNSAFE is latched until reset.

    Support mismatch is evaluated over a short sliding
    window and emphasized for a finite period after a
    structural gait commit.
    """

    def __init__(
        self,
        config: PyMPCSafetyMonitorConfig
        = DEFAULT_PYMPC_SAFETY_MONITOR_CONFIG,
    ) -> None:
        self.config = config
        self.reset()

    def reset(self) -> None:
        self._support_deficit_history: deque[bool] = (
            deque()
        )

        self._time_since_structural_commit_s: (
            float | None
        ) = None

        self._unsafe_latched = False

        self._last_status = PyMPCSafetyStatus(
            state=PyMPCSafetyState.NORMAL,
            planned_support_count=0,
            physical_support_count=0,
            current_support_deficit_count=0,
            support_deficit_fraction=0.0,
            support_window_samples=0,
            roll_rad=0.0,
            pitch_rad=0.0,
            base_height_m=0.0,
            post_transition_active=False,
            time_since_structural_commit_s=None,
            reasons=(),
        )

    @property
    def last_status(self) -> PyMPCSafetyStatus:
        return self._last_status

    @property
    def unsafe_latched(self) -> bool:
        return self._unsafe_latched

    def notify_structural_commit(self) -> None:
        # Start a fresh post-transition evidence window.
        self._time_since_structural_commit_s = 0.0
        self._support_deficit_history.clear()

    def update(
        self,
        *,
        planned_contact: Any,
        physical_contact: Any,
        roll_rad: float,
        pitch_rad: float,
        base_height_m: float,
        dt: float,
    ) -> PyMPCSafetyStatus:
        dt = _finite_scalar(
            "dt",
            dt,
        )

        if dt <= 0.0:
            raise ValueError(
                f"dt must be positive, got {dt}"
            )

        planned = _contact_vector(
            "planned_contact",
            planned_contact,
        )

        physical = _contact_vector(
            "physical_contact",
            physical_contact,
        )

        roll = _finite_scalar(
            "roll_rad",
            roll_rad,
        )

        pitch = _finite_scalar(
            "pitch_rad",
            pitch_rad,
        )

        base_height = _finite_scalar(
            "base_height_m",
            base_height_m,
        )

        planned_count = int(
            np.count_nonzero(planned)
        )

        physical_count = int(
            np.count_nonzero(physical)
        )

        deficit_count = max(
            0,
            planned_count - physical_count,
        )

        self._support_deficit_history.append(
            deficit_count > 0
        )

        window_steps = max(
            1,
            int(
                round(
                    self.config.support_window_s
                    / dt
                )
            ),
        )

        while (
            len(self._support_deficit_history)
            > window_steps
        ):
            self._support_deficit_history.popleft()

        # Normalize by the configured full window, not by the
        # partially filled deque.  Immediately after a structural
        # commit the history is intentionally cleared; using the
        # current deque length would turn one mismatch sample into
        # a deficit fraction of 1.0.
        #
        # With dt=2 ms and a 200 ms window:
        #   1 mismatch sample  -> 0.01
        #  15 mismatch samples -> 0.15
        #  35 mismatch samples -> 0.35
        deficit_fraction = float(
            np.count_nonzero(
                self._support_deficit_history
            )
            / window_steps
        )

        time_since_commit = (
            self._time_since_structural_commit_s
        )

        post_transition_active = (
            time_since_commit is not None
            and time_since_commit
            <= self.config.post_transition_monitor_s
        )

        reasons: list[str] = []

        abs_roll = abs(roll)
        abs_pitch = abs(pitch)

        # -----------------------------------------------------
        # Hard empirical UNSAFE conditions.
        # -----------------------------------------------------
        if (
            base_height
            < self.config.min_base_height_m
        ):
            reasons.append(
                "base_height_below_limit"
            )

        if (
            abs_roll
            >= self.config.unsafe_roll_rad
        ):
            reasons.append(
                "roll_above_unsafe_limit"
            )

        if (
            abs_pitch
            >= self.config.unsafe_pitch_rad
        ):
            reasons.append(
                "pitch_above_unsafe_limit"
            )

        if (
            post_transition_active
            and deficit_fraction
            >= self.config
            .unsafe_support_deficit_fraction
        ):
            reasons.append(
                "sustained_support_deficit_unsafe"
            )

        hard_unsafe = bool(reasons)

        if hard_unsafe:
            self._unsafe_latched = True

        # -----------------------------------------------------
        # WATCH conditions.
        # -----------------------------------------------------
        watch_reasons: list[str] = []

        if (
            abs_roll
            >= self.config.watch_roll_rad
        ):
            watch_reasons.append(
                "roll_above_watch_limit"
            )

        if (
            abs_pitch
            >= self.config.watch_pitch_rad
        ):
            watch_reasons.append(
                "pitch_above_watch_limit"
            )

        if (
            post_transition_active
            and deficit_fraction
            >= self.config
            .watch_support_deficit_fraction
        ):
            watch_reasons.append(
                "sustained_support_deficit_watch"
            )

        if self._unsafe_latched:
            state = PyMPCSafetyState.UNSAFE

        elif watch_reasons:
            state = PyMPCSafetyState.WATCH
            reasons.extend(watch_reasons)

        else:
            state = PyMPCSafetyState.NORMAL

        status = PyMPCSafetyStatus(
            state=state,
            planned_support_count=planned_count,
            physical_support_count=physical_count,
            current_support_deficit_count=(
                deficit_count
            ),
            support_deficit_fraction=(
                deficit_fraction
            ),
            support_window_samples=len(
                self._support_deficit_history
            ),
            roll_rad=roll,
            pitch_rad=pitch,
            base_height_m=base_height,
            post_transition_active=(
                post_transition_active
            ),
            time_since_structural_commit_s=(
                time_since_commit
            ),
            reasons=tuple(reasons),
        )

        self._last_status = status

        if time_since_commit is not None:
            self._time_since_structural_commit_s = (
                time_since_commit + dt
            )

        return status
