from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class Resettable(Protocol):
    def reset(self) -> None:
        ...


@dataclass
class PyMPCLowLevelLifecycle:
    """
    Own the TRACER-side episode reset contract.

    This class intentionally does NOT reset:
      - the MuJoCo environment,
      - the Quadruped-PyMPC wrapper/controller,
      - experiment trace/dataset buffers.

    Those lifetimes are owned elsewhere.

    It resets only persistent TRACER low-level execution
    state that must not leak across environment resets.
    """

    adapter: Resettable
    transition_manager: Resettable
    safety_monitor: Resettable
    supervisor: Resettable

    reset_count: int = 0

    def reset_episode(self) -> None:
        self.adapter.reset()
        self.transition_manager.reset()
        self.safety_monitor.reset()
        self.supervisor.reset()

        self.reset_count += 1
