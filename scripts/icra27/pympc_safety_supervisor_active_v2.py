#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
from dataclasses import replace
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
    # M4-B2:
    # immediately decelerate through the existing continuous
    # transition channel while structural gait recovery waits
    # for a planned full-stance boundary.
    vx=0.00,
    yaw_rate=0.0,
    body_height=0.30,
    swing_clearance=0.06,
    gait_period=1.0 / 1.4,
    duty_factor=0.65,
    source_mode="icra27_m4_backoff_v1",
    source_reason=(
        "M4 decelerating nominal-structure back-off"
    ),
)


SUPERVISOR = PyMPCMetaGaitSupervisor(
    fallback_command=NOMINAL,
)

SOFT_SLOWDOWN_ACTIVE = False
SOFT_SLOWDOWN_TIME_S = None
SOFT_SLOWDOWN_REASONS = ()
SOFT_SLOWDOWN_LABEL = "m4_soft_slowdown"


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

    # Full UNSAFE override has highest priority.
    if selection.override_active:
        return (
            selection.selected_label,
            selection.command,
        )

    # Attitude-WATCH soft intervention:
    # preserve the requested structural gait while reducing
    # only the continuously executable forward velocity.
    if SOFT_SLOWDOWN_ACTIVE:
        slowed = replace(
            command,
            vx=0.0,
            source_mode=(
                "icra27_m4_soft_slowdown"
            ),
            source_reason=(
                "M4 attitude-WATCH "
                "continuous slowdown"
            ),
        )

        return (
            SOFT_SLOWDOWN_LABEL,
            slowed,
        )

    return label, command


def active_env_step(
    self,
    action,
):
    global SOFT_SLOWDOWN_ACTIVE
    global SOFT_SLOWDOWN_TIME_S
    global SOFT_SLOWDOWN_REASONS

    result = ORIGINAL_SHADOW_ENV_STEP(
        self,
        action,
    )

    status = shadow.MONITOR.last_status

    if trace.TRACE["time_s"]:
        current_time = float(
            trace.TRACE["time_s"][-1]
        )
    else:
        current_time = None

    attitude_watch_reasons = {
        "roll_above_watch_limit",
        "pitch_above_watch_limit",
    }

    current_reasons = set(
        status.reasons
    )

    if (
        not SOFT_SLOWDOWN_ACTIVE
        and not SUPERVISOR.override_active
        and status.state
        == shadow.PyMPCSafetyState.WATCH
        and bool(
            current_reasons
            & attitude_watch_reasons
        )
    ):
        SOFT_SLOWDOWN_ACTIVE = True
        SOFT_SLOWDOWN_TIME_S = (
            current_time
        )
        SOFT_SLOWDOWN_REASONS = tuple(
            status.reasons
        )

        print()
        print("=" * 72)
        print(
            "[M4 SOFT] attitude WATCH -> "
            "forward slowdown requested"
        )
        print(
            "  detection t :",
            SOFT_SLOWDOWN_TIME_S,
        )
        print(
            "  reasons     :",
            list(
                SOFT_SLOWDOWN_REASONS
            ),
        )
        print(
            "  action      : "
            "vx -> 0.000 via existing slew; "
            "structural gait unchanged"
        )
        print("=" * 72)

    activated = SUPERVISOR.observe(
        status,
        time_s=current_time,
    )

    if activated:
        print()
        print("=" * 72)
        print(
            "[M4 ACTIVE] UNSAFE -> "
            "nominal structural back-off requested"
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
            "vx=0.000 h=0.300 clr=0.060 "
            "f=1.400 D=0.650"
        )
        print("=" * 72)

    return result


def reset_shadow_state():
    global SOFT_SLOWDOWN_ACTIVE
    global SOFT_SLOWDOWN_TIME_S
    global SOFT_SLOWDOWN_REASONS

    SOFT_SLOWDOWN_ACTIVE = False
    SOFT_SLOWDOWN_TIME_S = None
    SOFT_SLOWDOWN_REASONS = ()

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
        "soft slowdown   :",
        SOFT_SLOWDOWN_ACTIVE,
    )

    print(
        "soft time       :",
        SOFT_SLOWDOWN_TIME_S,
    )

    print(
        "soft reasons    :",
        list(
            SOFT_SLOWDOWN_REASONS
        ),
    )

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
            "m4_attitude_watch_soft_backoff_v2",

        "soft_slowdown_active":
            bool(SOFT_SLOWDOWN_ACTIVE),

        "soft_slowdown_time_s":
            SOFT_SLOWDOWN_TIME_S,

        "soft_slowdown_reasons":
            list(
                SOFT_SLOWDOWN_REASONS
            ),

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
            "vx_mps": 0.00,
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
