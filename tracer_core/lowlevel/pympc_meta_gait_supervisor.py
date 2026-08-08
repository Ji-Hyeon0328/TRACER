from __future__ import annotations

from dataclasses import dataclass

from tracer_core.highlevel.meta_gait import (
    MetaGaitCommand,
)
from tracer_core.lowlevel.pympc_safety_monitor import (
    PyMPCSafetyState,
    PyMPCSafetyStatus,
)


@dataclass(frozen=True)
class PyMPCSupervisorSelection:
    requested_label: str
    selected_label: str
    command: MetaGaitCommand
    override_active: bool


class PyMPCMetaGaitSupervisor:
    """
    Runtime command-level safety supervisor.

    The supervisor does not modify torque and does not
    bypass the existing PyMPC transition manager.

    NORMAL / WATCH:
        requested MetaGaitCommand passes through.

    UNSAFE:
        latch a known nominal MetaGaitCommand.  The
        existing transition manager remains responsible
        for slew limiting and structural full-stance
        commit.
    """

    def __init__(
        self,
        *,
        fallback_command: MetaGaitCommand,
        fallback_label: str
        = "m4_backoff_nominal",
    ) -> None:
        self.fallback_command = fallback_command
        self.fallback_label = str(
            fallback_label
        )
        self.reset()

    def reset(self) -> None:
        self.override_active = False
        self.override_time_s: float | None = None
        self.override_reasons: tuple[str, ...] = ()

    def observe(
        self,
        status: PyMPCSafetyStatus,
        *,
        time_s: float | None = None,
    ) -> bool:
        """
        Return True only on the tick that the override
        becomes newly latched.
        """
        if (
            self.override_active
            or status.state
            != PyMPCSafetyState.UNSAFE
        ):
            return False

        self.override_active = True

        self.override_time_s = (
            None
            if time_s is None
            else float(time_s)
        )

        self.override_reasons = tuple(
            status.reasons
        )

        return True

    def select(
        self,
        requested_label: str,
        requested_command: MetaGaitCommand,
    ) -> PyMPCSupervisorSelection:
        if self.override_active:
            return PyMPCSupervisorSelection(
                requested_label=str(
                    requested_label
                ),
                selected_label=(
                    self.fallback_label
                ),
                command=self.fallback_command,
                override_active=True,
            )

        return PyMPCSupervisorSelection(
            requested_label=str(
                requested_label
            ),
            selected_label=str(
                requested_label
            ),
            command=requested_command,
            override_active=False,
        )
