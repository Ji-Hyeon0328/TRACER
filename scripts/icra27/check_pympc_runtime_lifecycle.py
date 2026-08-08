#!/usr/bin/env python3

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


from tracer_core.lowlevel.pympc_runtime_lifecycle import (
    PyMPCLowLevelLifecycle,
)


class SpyResettable:
    def __init__(self, name, calls):
        self.name = name
        self.calls = calls
        self.count = 0

    def reset(self):
        self.calls.append(self.name)
        self.count += 1


def main():
    calls = []

    adapter = SpyResettable(
        "adapter",
        calls,
    )

    manager = SpyResettable(
        "transition_manager",
        calls,
    )

    monitor = SpyResettable(
        "safety_monitor",
        calls,
    )

    supervisor = SpyResettable(
        "supervisor",
        calls,
    )

    lifecycle = PyMPCLowLevelLifecycle(
        adapter=adapter,
        transition_manager=manager,
        safety_monitor=monitor,
        supervisor=supervisor,
    )

    assert lifecycle.reset_count == 0

    lifecycle.reset_episode()

    expected = [
        "adapter",
        "transition_manager",
        "safety_monitor",
        "supervisor",
    ]

    assert calls == expected, (
        calls,
        expected,
    )

    assert lifecycle.reset_count == 1

    assert adapter.count == 1
    assert manager.count == 1
    assert monitor.count == 1
    assert supervisor.count == 1

    lifecycle.reset_episode()

    assert calls == expected + expected
    assert lifecycle.reset_count == 2

    assert adapter.count == 2
    assert manager.count == 2
    assert monitor.count == 2
    assert supervisor.count == 2

    print(
        "PyMPC low-level lifecycle checker: PASS"
    )

    print(
        "reset order:",
        " -> ".join(expected),
    )

    print(
        "reset count:",
        lifecycle.reset_count,
    )


if __name__ == "__main__":
    main()
