#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ICRA27_SCRIPTS = ROOT / "scripts" / "icra27"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ICRA27_SCRIPTS))


import pympc_contact_attitude_trace_v0 as trace

from tracer_core.lowlevel.pympc_safety_monitor import (
    PyMPCSafetyMonitor,
    PyMPCSafetyState,
)


ORIGINAL_TRACED_COMPUTE = (
    trace.traced_compute_actions
)

ORIGINAL_TRACE_ENV_STEP = (
    trace.instrumented_env_step
)


MONITOR = PyMPCSafetyMonitor()

PENDING_SAMPLE = None
FIRST_ROLLOUT_DONE = False

TARGET_COMMIT = None

STATE_EVENTS = []
FIRST_WATCH = None
FIRST_UNSAFE = None
LAST_STATE = None


def cli_value(flag: str) -> str | None:
    if flag not in sys.argv:
        return None

    i = sys.argv.index(flag)

    if i + 1 >= len(sys.argv):
        return None

    return sys.argv[i + 1]


def status_event(
    *,
    sample_index,
    time_s,
    status,
):
    return {
        "sample_index":
            int(sample_index),

        "time_s":
            float(time_s),

        "state":
            status.state.value,

        "support_deficit_fraction":
            float(
                status.support_deficit_fraction
            ),

        "planned_support_count":
            int(
                status.planned_support_count
            ),

        "physical_support_count":
            int(
                status.physical_support_count
            ),

        "roll_deg":
            float(
                np.degrees(
                    status.roll_rad
                )
            ),

        "pitch_deg":
            float(
                np.degrees(
                    status.pitch_rad
                )
            ),

        "base_height_m":
            float(
                status.base_height_m
            ),

        "post_transition_active":
            bool(
                status.post_transition_active
            ),

        "time_since_structural_commit_s":
            (
                None
                if (
                    status
                    .time_since_structural_commit_s
                    is None
                )
                else float(
                    status
                    .time_since_structural_commit_s
                )
            ),

        "reasons":
            list(status.reasons),
    }


def shadow_compute_actions(
    self,
    *args,
    **kwargs,
):
    global PENDING_SAMPLE
    global TARGET_COMMIT

    before_events = len(
        trace.guarded.STRUCTURAL_EVENTS
    )

    tau = ORIGINAL_TRACED_COMPUTE(
        self,
        *args,
        **kwargs,
    )

    after_events = len(
        trace.guarded.STRUCTURAL_EVENTS
    )

    if after_events > before_events:
        new_events = (
            trace.guarded
            .STRUCTURAL_EVENTS[
                before_events:after_events
            ]
        )

        for event in new_events:
            if (
                event.get("target_label")
                != "target"
            ):
                continue

            if TARGET_COMMIT is None:
                TARGET_COMMIT = dict(
                    event
                )

                MONITOR.notify_structural_commit()

                print()
                print(
                    "[M4 shadow] target structural "
                    "commit detected:"
                )

                print(
                    f"  t="
                    f"{float(event['time_s']):.3f}s "
                    f"f="
                    f"{float(event['applied_frequency_hz']):.3f} "
                    f"D="
                    f"{float(event['applied_duty_factor']):.3f}"
                )

    # ORIGINAL_TRACED_COMPUTE has just appended
    # the same-tick planned/contact/kinematic sample
    # into trace.TRACE.
    if not trace.TRACE["time_s"]:
        return tau

    i = len(
        trace.TRACE["time_s"]
    ) - 1

    PENDING_SAMPLE = {
        "sample_index":
            i,

        "time_s":
            float(
                trace.TRACE["time_s"][i]
            ),

        "dt":
            float(
                trace.get_arg(
                    args,
                    kwargs,
                    "simulation_dt",
                    10,
                )
            ),

        "planned_contact":
            np.asarray(
                trace.TRACE[
                    "planned_contact"
                ][i],
                dtype=bool,
            ).copy(),

        "roll_rad":
            float(
                trace.TRACE[
                    "base_ori_rpy"
                ][i][0]
            ),

        "pitch_rad":
            float(
                trace.TRACE[
                    "base_ori_rpy"
                ][i][1]
            ),

        "base_height_m":
            float(
                trace.TRACE[
                    "base_pos"
                ][i][2]
            ),
    }

    return tau


def shadow_env_step(
    self,
    action,
):
    global PENDING_SAMPLE
    global FIRST_ROLLOUT_DONE
    global FIRST_WATCH
    global FIRST_UNSAFE
    global LAST_STATE

    if (
        not FIRST_ROLLOUT_DONE
        and PENDING_SAMPLE is not None
    ):
        (
            physical_contact,
            _,
            _,
        ) = self.feet_contact_state(
            ground_reaction_forces=True
        )

        physical = trace.contact_array(
            physical_contact
        )

        status = MONITOR.update(
            planned_contact=(
                PENDING_SAMPLE[
                    "planned_contact"
                ]
            ),

            physical_contact=physical,

            roll_rad=(
                PENDING_SAMPLE[
                    "roll_rad"
                ]
            ),

            pitch_rad=(
                PENDING_SAMPLE[
                    "pitch_rad"
                ]
            ),

            base_height_m=(
                PENDING_SAMPLE[
                    "base_height_m"
                ]
            ),

            dt=PENDING_SAMPLE["dt"],
        )

        event = status_event(
            sample_index=(
                PENDING_SAMPLE[
                    "sample_index"
                ]
            ),

            time_s=(
                PENDING_SAMPLE[
                    "time_s"
                ]
            ),

            status=status,
        )

        if status.state != LAST_STATE:
            STATE_EVENTS.append(
                event
            )

            print(
                "[M4 shadow] "
                f"t={event['time_s']:.3f}s "
                f"-> {event['state'].upper():6s} "
                f"support="
                f"{event['support_deficit_fraction']:.3f} "
                f"roll="
                f"{event['roll_deg']:+.2f}deg "
                f"pitch="
                f"{event['pitch_deg']:+.2f}deg "
                f"z="
                f"{event['base_height_m']:.3f}"
            )

            LAST_STATE = status.state

        if (
            FIRST_WATCH is None
            and status.state
            == PyMPCSafetyState.WATCH
        ):
            FIRST_WATCH = event

        if (
            FIRST_UNSAFE is None
            and status.state
            == PyMPCSafetyState.UNSAFE
        ):
            FIRST_UNSAFE = event

    # Delegate to the already validated M3 trace hook.
    # It records exactly the same physical contact and
    # then performs the real MuJoCo step.
    result = ORIGINAL_TRACE_ENV_STEP(
        self,
        action,
    )

    terminated = bool(
        result[2]
    )

    truncated = bool(
        result[3]
    )

    if (
        not FIRST_ROLLOUT_DONE
        and (
            terminated
            or truncated
        )
    ):
        FIRST_ROLLOUT_DONE = True

    PENDING_SAMPLE = None

    return result


def print_event(
    name,
    event,
    termination_time,
):
    if event is None:
        print(
            f"{name:14s}: NONE"
        )
        return

    if termination_time is None:
        lead = "-"
    else:
        lead_s = (
            termination_time
            - event["time_s"]
        )

        lead = (
            f"{lead_s:+.3f}s "
            "before termination"
        )

    print(
        f"{name:14s}: "
        f"t={event['time_s']:.3f}s "
        f"support="
        f"{event['support_deficit_fraction']:.3f} "
        f"roll="
        f"{event['roll_deg']:+.2f}deg "
        f"pitch="
        f"{event['pitch_deg']:+.2f}deg "
        f"z="
        f"{event['base_height_m']:.3f} "
        f"{lead}"
    )

    print(
        " " * 16
        + f"reasons={event['reasons']}"
    )


def main():
    output_arg = cli_value(
        "--output"
    )

    if output_arg is None:
        raise SystemExit(
            "--output is required by the underlying "
            "contact/attitude trace runner"
        )

    base_output = Path(
        output_arg
    )

    MONITOR.reset()

    # Replace only the names that the existing M3 runner
    # installs into the PyMPC wrapper and environment.
    trace.traced_compute_actions = (
        shadow_compute_actions
    )

    trace.instrumented_env_step = (
        shadow_env_step
    )

    try:
        trace.main()

    finally:
        trace.traced_compute_actions = (
            ORIGINAL_TRACED_COMPUTE
        )

        trace.instrumented_env_step = (
            ORIGINAL_TRACE_ENV_STEP
        )

    if trace.TERMINATION_EVENTS:
        termination = (
            trace.TERMINATION_EVENTS[0]
        )

        termination_time = float(
            termination[
                "simulation_time_s"
            ]
        )

    else:
        termination = None
        termination_time = None

    if termination_time is None:
        if FIRST_UNSAFE is None:
            classification = (
                "survivor_no_false_unsafe"
            )
        else:
            classification = (
                "false_unsafe_on_survivor"
            )

    else:
        if (
            FIRST_UNSAFE is not None
            and FIRST_UNSAFE[
                "time_s"
            ] < termination_time
        ):
            classification = (
                "failure_precursor_detected"
            )
        else:
            classification = (
                "failure_not_preemptively_detected"
            )

    print()
    print("=" * 78)
    print("ICRA27 M4 LIVE SHADOW MONITOR")
    print("=" * 78)

    if TARGET_COMMIT is None:
        print(
            "target commit : NONE"
        )
    else:
        print(
            "target commit : "
            f"{float(TARGET_COMMIT['time_s']):.3f}s"
        )

    print(
        "termination   : "
        + (
            "NONE"
            if termination_time is None
            else f"{termination_time:.3f}s"
        )
    )

    print_event(
        "first WATCH",
        FIRST_WATCH,
        termination_time,
    )

    print_event(
        "first UNSAFE",
        FIRST_UNSAFE,
        termination_time,
    )

    print(
        "final monitor :",
        MONITOR.last_status.state.value,
    )

    print(
        "classification:",
        classification,
    )

    print()
    print("state transitions:")

    for event in STATE_EVENTS:
        print(
            f"  t={event['time_s']:.3f}s "
            f"-> {event['state']:6s} "
            f"support="
            f"{event['support_deficit_fraction']:.3f} "
            f"roll="
            f"{event['roll_deg']:+.2f}deg "
            f"pitch="
            f"{event['pitch_deg']:+.2f}deg "
            f"z="
            f"{event['base_height_m']:.3f}"
        )

    shadow_output = (
        base_output.with_name(
            base_output.stem
            + "_m4_shadow.json"
        )
    )

    shadow_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result = {
        "mode":
            "live_shadow_monitor",

        "monitor_config":
            asdict(MONITOR.config),

        "target_commit":
            TARGET_COMMIT,

        "first_watch":
            FIRST_WATCH,

        "first_unsafe":
            FIRST_UNSAFE,

        "first_termination":
            termination,

        "classification":
            classification,

        "state_transitions":
            STATE_EVENTS,
    }

    shadow_output.write_text(
        json.dumps(
            result,
            indent=2,
        )
        + "\n"
    )

    print()
    print(
        "saved shadow:",
        shadow_output,
    )
    print("=" * 78)


if __name__ == "__main__":
    main()
