from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tracer_core.highlevel.meta_gait import (
    MetaGaitCommand,
)
from tracer_core.lowlevel.pympc_meta_gait_adapter import (
    PyMPCMetaGaitAdapter,
    PyMPCMetaGaitReference,
)
from tracer_core.lowlevel.pympc_meta_gait_supervisor import (
    PyMPCMetaGaitSupervisor,
)
from tracer_core.lowlevel.pympc_runtime_lifecycle import (
    PyMPCLowLevelLifecycle,
)
from tracer_core.lowlevel.pympc_safety_monitor import (
    PyMPCSafetyMonitor,
    PyMPCSafetyStatus,
)
from tracer_core.lowlevel.pympc_transition_manager import (
    PyMPCMetaGaitTransitionManager,
)


@dataclass(frozen=True)
class PyMPCCommandResolution:
    """
    One command-governance tick.

    Runtime path:
        requested
          -> supervisor selection
          -> selected projection
          -> transition manager
          -> applied

    requested_projected is diagnostic only.
    """

    requested_label: str
    selected_label: str

    requested: MetaGaitCommand
    requested_projected: PyMPCMetaGaitReference

    selected: MetaGaitCommand
    selected_projected: PyMPCMetaGaitReference

    applied: PyMPCMetaGaitReference

    override_active: bool
    structural_commit: bool


@dataclass(frozen=True)
class PyMPCPhysicalObservation:
    status: PyMPCSafetyStatus
    override_activated: bool


class PyMPCLowLevelRuntime:
    """
    Reusable TRACER -> Quadruped-PyMPC low-level runtime.

    Owns persistent execution state only.

    It intentionally does NOT own:
      - simulator reset,
      - PyMPC wrapper reset,
      - trace arrays,
      - termination history,
      - experiment summaries.

    Episode reset therefore clears only controller-side
    TRACER execution state.
    """

    def __init__(
        self,
        *,
        fallback_command: MetaGaitCommand,
        fallback_label: str = "m4_backoff_nominal",
        adapter: PyMPCMetaGaitAdapter | None = None,
        transition_manager:
            PyMPCMetaGaitTransitionManager | None = None,
        safety_monitor: PyMPCSafetyMonitor | None = None,
        supervisor: PyMPCMetaGaitSupervisor | None = None,
    ) -> None:
        self.adapter = (
            PyMPCMetaGaitAdapter()
            if adapter is None
            else adapter
        )

        self.transition_manager = (
            PyMPCMetaGaitTransitionManager()
            if transition_manager is None
            else transition_manager
        )

        self.safety_monitor = (
            PyMPCSafetyMonitor()
            if safety_monitor is None
            else safety_monitor
        )

        self.supervisor = (
            PyMPCMetaGaitSupervisor(
                fallback_command=fallback_command,
                fallback_label=fallback_label,
            )
            if supervisor is None
            else supervisor
        )

        self.lifecycle = PyMPCLowLevelLifecycle(
            adapter=self.adapter,
            transition_manager=self.transition_manager,
            safety_monitor=self.safety_monitor,
            supervisor=self.supervisor,
        )

        self.command: (
            PyMPCMetaGaitReference | None
        ) = None

        self.target_label: str | None = None

    def reset_episode(self) -> None:
        self.lifecycle.reset_episode()

        self.command = None
        self.target_label = None

    def resolve_command(
        self,
        *,
        requested_label: str,
        requested_command: MetaGaitCommand,
        current_contact: Any,
        dt: float,
    ) -> PyMPCCommandResolution:
        """
        Resolve one high-level MetaGaitCommand into the
        controller-facing guarded reference.

        requested_projected is retained only for
        observability. It does not alter selection.
        """

        selection = self.supervisor.select(
            requested_label=requested_label,
            requested_command=requested_command,
        )

        requested_projected = self.adapter.project(
            requested_command
        )

        selected_projected = self.adapter.project(
            selection.command
        )

        applied = self.transition_manager.update(
            target=selected_projected,
            current_contact=current_contact,
            dt=dt,
        )

        structural_commit = bool(
            self.transition_manager
            .last_structural_commit
        )

        if structural_commit:
            self.safety_monitor.notify_structural_commit()

        self.command = applied
        self.target_label = selection.selected_label

        return PyMPCCommandResolution(
            requested_label=str(
                requested_label
            ),
            selected_label=str(
                selection.selected_label
            ),
            requested=requested_command,
            requested_projected=requested_projected,
            selected=selection.command,
            selected_projected=selected_projected,
            applied=applied,
            override_active=bool(
                selection.override_active
            ),
            structural_commit=structural_commit,
        )

    def apply_to_pympc(
        self,
        *,
        wb_interface: Any,
        pympc_cfg: Any,
    ) -> PyMPCMetaGaitReference:
        """
        Apply the current guarded reference to PyMPC.

        This deliberately reuses the existing adapter
        runtime hook instead of duplicating PyMPC/PWG/STC
        mutation logic here.
        """

        if self.command is None:
            raise RuntimeError(
                "No applied command is available. "
                "Call resolve_command() first."
            )

        return self.adapter.apply_runtime(
            wb_interface=wb_interface,
            pympc_cfg=pympc_cfg,
            command=self.command,
        )

    def observe_physical_state(
        self,
        *,
        planned_contact: Any,
        physical_contact: Any,
        roll_rad: float,
        pitch_rad: float,
        base_height_m: float,
        dt: float,
        time_s: float | None = None,
    ) -> PyMPCPhysicalObservation:
        """
        Close the post-execution M4 health loop.

        Observation occurs after physical execution.
        A newly detected UNSAFE state affects command
        selection on the following command tick.
        """

        status = self.safety_monitor.update(
            planned_contact=planned_contact,
            physical_contact=physical_contact,
            roll_rad=roll_rad,
            pitch_rad=pitch_rad,
            base_height_m=base_height_m,
            dt=dt,
        )

        activated = self.supervisor.observe(
            status,
            time_s=time_s,
        )

        return PyMPCPhysicalObservation(
            status=status,
            override_activated=bool(activated),
        )
