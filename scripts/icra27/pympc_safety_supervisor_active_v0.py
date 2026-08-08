#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ICRA27_SCRIPTS = ROOT / "scripts" / "icra27"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ICRA27_SCRIPTS))


import pympc_safety_monitor_live_shadow_v0 as shadow

from tracer_core.highlevel.meta_gait import (
    MetaGaitCommand,
)
from tracer_core.lowlevel.pympc_meta_gait_supervisor import (
    PyMPCMetaGaitSupervisor,
)


trace = shadow.trace
guarded = trace.guarded


NOMINAL = MetaGaitCommand(
    vx=0.20,
    yaw_rate=0.0,
    body_height=0.30,
    swing_clearance=0.06,
    gait_period=1.0 / 1.4,
    duty_factor=0.65,
    source_mode="icra27_m4_backoff",
    source_reason=(
        "M4 known-nominal safety back-off"
    ),
)


SUPERVISOR = PyMPCMetaGaitSupervisor(
    fallback_command=NOMINAL,
)


ORIGINAL_TARGET_FOR_TIME = (
    guarded.target_for_time
)

ORIGINAL_SHADOW_ENV_STEP = (
    shadow.shadow_env_step
)


def cli_value(flag: str) -> str | None:
    if flag not in sys.argv:
        return None

    i = sys.argv.index(flag)

    if i + 1 >= len(sys.argv):
        return None

    return sys.argv[i + 1]


def supervised_target_for_time(t):
    label, command = (
        ORIGINAL_TARGET_FOR_TIME(t)
    )

    selection = SUPERVISOR.select(
        label,
        command,
    )

    return (
        selection.selected_label,
        selection.command,
    )


def active_env_step(
    self,
    action,
):
    result = ORIGINAL_SHADOW_ENV_STEP(
        self,
        action,
    )

    status = shadow.MONITOR.last_status

    event_time = None

    if shadow.FIRST_UNSAFE is not None:
        event_time = float(
            shadow.FIRST_UNSAFE["time_s"]
        )

    activated = SUPERVISOR.observe(
        status,
        time_s=event_time,
    )

    if activated:
        print()
        print("=" * 72)
        print(
            "[M4 ACTIVE] UNSAFE -> "
            "nominal back-off requested"
        )
        print(
            "  detection t :",
            SUPERVISOR.override_time_s,
        )
        print(
            "  reasons     :",
            list(
                SUPERVISOR.override_reasons
            ),
        )
        print(
            "  fallback    : "
            "vx=0.200 h=0.300 clr=0.060 "
            "f=1.400 D=0.650"
        )
        print(
            "  NOTE: structural change still "
            "passes through the existing "
            "full-stance transition manager."
        )
        print("=" * 72)

    return result


def reset_shadow_state():
    shadow.MONITOR.reset()

    shadow.PENDING_SAMPLE = None
    shadow.FIRST_ROLLOUT_DONE = False
    shadow.TARGET_COMMIT = None

    shadow.STATE_EVENTS.clear()

    shadow.FIRST_WATCH = None
    shadow.FIRST_UNSAFE = None
    shadow.LAST_STATE = None

    SUPERVISOR.reset()


def main():
    output_arg = cli_value(
        "--output"
    )

    if output_arg is None:
        raise SystemExit(
            "--output is required"
        )

    base_output = Path(
        output_arg
    )

    reset_shadow_state()

    # The existing guarded controller still owns:
    #   projection
    #   continuous slew
    #   structural pending/commit
    #
    # We only replace target selection and add the
    # already validated monitor observation.
    guarded.target_for_time = (
        supervised_target_for_time
    )

    trace.traced_compute_actions = (
        shadow.shadow_compute_actions
    )

    trace.instrumented_env_step = (
        active_env_step
    )

    try:
        trace.main()

    finally:
        guarded.target_for_time = (
            ORIGINAL_TARGET_FOR_TIME
        )

        trace.traced_compute_actions = (
            shadow.ORIGINAL_TRACED_COMPUTE
        )

        trace.instrumented_env_step = (
            ORIGINAL_SHADOW_ENV_STEP
        )

    if trace.TERMINATION_EVENTS:
        termination = (
            trace.TERMINATION_EVENTS[0]
        )
    else:
        termination = None

    backoff_commits = [
        event
        for event
        in guarded.STRUCTURAL_EVENTS
        if event.get("target_label")
        == SUPERVISOR.fallback_label
    ]

    first_backoff_commit = (
        backoff_commits[0]
        if backoff_commits
        else None
    )

    if (
        SUPERVISOR.override_active
        and termination is None
    ):
        classification = (
            "unsafe_detected_"
            "backoff_survived_window"
        )

    elif (
        SUPERVISOR.override_active
        and termination is not None
    ):
        classification = (
            "unsafe_detected_"
            "backoff_failed"
        )

    elif termination is not None:
        classification = (
            "failure_without_override"
        )

    else:
        classification = (
            "survived_no_override"
        )

    print()
    print("=" * 78)
    print(
        "ICRA27 M4 ACTIVE SAFETY SUPERVISOR"
    )
    print("=" * 78)

    print(
        "override active :",
        SUPERVISOR.override_active,
    )

    print(
        "UNSAFE time     :",
        SUPERVISOR.override_time_s,
    )

    print(
        "override reason :",
        list(
            SUPERVISOR.override_reasons
        ),
    )

    if first_backoff_commit is None:
        print(
            "backoff commit  : NONE"
        )
    else:
        print(
            "backoff commit  : "
            f"{float(first_backoff_commit['time_s']):.3f}s "
            f"f="
            f"{float(first_backoff_commit['applied_frequency_hz']):.3f} "
            f"D="
            f"{float(first_backoff_commit['applied_duty_factor']):.3f}"
        )

    if termination is None:
        print(
            "termination     : NONE"
        )
    else:
        print(
            "termination     : "
            f"{float(termination['simulation_time_s']):.3f}s "
            f"{termination['reason']}"
        )

    print(
        "classification  :",
        classification,
    )

    result = {
        "mode":
            "m4_active_nominal_backoff_v0",

        "override_active":
            bool(
                SUPERVISOR.override_active
            ),

        "override_time_s":
            SUPERVISOR.override_time_s,

        "override_reasons":
            list(
                SUPERVISOR.override_reasons
            ),

        "fallback": {
            "vx_mps": 0.20,
            "body_height_m": 0.30,
            "swing_clearance_m": 0.06,
            "frequency_hz": 1.40,
            "duty_factor": 0.65,
        },

        "first_backoff_commit":
            first_backoff_commit,

        "first_termination":
            termination,

        "classification":
            classification,
    }

    output = base_output.with_name(
        base_output.stem
        + "_m4_active.json"
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.write_text(
        json.dumps(
            result,
            indent=2,
        )
        + "\n"
    )

    print(
        "saved active    :",
        output,
    )
    print("=" * 78)


if __name__ == "__main__":
    main()
