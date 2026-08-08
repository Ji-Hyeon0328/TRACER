#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


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


COMMAND_FIELDS = (
    "vx",
    "yaw_rate",
    "body_height",
    "swing_clearance",
    "gait_period",
    "duty_factor",
)

COMMAND_TRACE = {
    "time_s": [],
    "requested_label": [],
    "selected_label": [],
    "override_active": [],
    "requested": [],
    "requested_projected": [],
    "selected": [],
    "selected_projected": [],
    "applied": [],
}

LAST_COMMAND_CONTEXT = None
COMMAND_LOG_ACTIVE = True

M4_ENABLED = True


def command_vector(command):
    return np.asarray(
        [
            float(command.vx),
            float(command.yaw_rate),
            float(command.body_height),
            float(command.swing_clearance),
            float(command.gait_period),
            float(command.duty_factor),
        ],
        dtype=float,
    )


def command_dict(vector):
    vector = np.asarray(
        vector,
        dtype=float,
    ).reshape(-1)

    period = float(vector[4])

    frequency = (
        None
        if period <= 0.0
        else 1.0 / period
    )

    return {
        "vx_mps":
            float(vector[0]),

        "yaw_rate_rad_s":
            float(vector[1]),

        "body_height_m":
            float(vector[2]),

        "swing_clearance_m":
            float(vector[3]),

        "gait_period_s":
            period,

        "gait_frequency_hz":
            frequency,

        "duty_factor":
            float(vector[5]),
    }


def build_command_events():
    events = []
    previous_key = None

    for i, time_s in enumerate(
        COMMAND_TRACE["time_s"]
    ):
        applied = np.asarray(
            COMMAND_TRACE["applied"][i],
            dtype=float,
        )

        # Structural channels currently governed atomically
        # by the transition manager:
        # clearance / gait period / duty factor.
        structural_key = tuple(
            np.round(
                applied[[3, 4, 5]],
                decimals=9,
            )
        )

        key = (
            COMMAND_TRACE[
                "requested_label"
            ][i],

            COMMAND_TRACE[
                "selected_label"
            ][i],

            bool(
                COMMAND_TRACE[
                    "override_active"
                ][i]
            ),

            structural_key,
        )

        if (
            previous_key is not None
            and key == previous_key
        ):
            continue

        events.append(
            {
                "time_s":
                    float(time_s),

                "requested_label":
                    COMMAND_TRACE[
                        "requested_label"
                    ][i],

                "selected_label":
                    COMMAND_TRACE[
                        "selected_label"
                    ][i],

                "override_active":
                    bool(
                        COMMAND_TRACE[
                            "override_active"
                        ][i]
                    ),

                "requested":
                    command_dict(
                        COMMAND_TRACE[
                            "requested"
                        ][i]
                    ),

                "requested_projected":
                    command_dict(
                        COMMAND_TRACE[
                            "requested_projected"
                        ][i]
                    ),

                "selected":
                    command_dict(
                        COMMAND_TRACE[
                            "selected"
                        ][i]
                    ),

                "selected_projected":
                    command_dict(
                        COMMAND_TRACE[
                            "selected_projected"
                        ][i]
                    ),

                "applied":
                    command_dict(
                        COMMAND_TRACE[
                            "applied"
                        ][i]
                    ),
            }
        )

        previous_key = key

    return events


def cli_value(flag: str) -> str | None:
    if flag not in sys.argv:
        return None

    i = sys.argv.index(flag)

    if i + 1 >= len(sys.argv):
        return None

    return sys.argv[i + 1]


def supervised_target_for_time(t):
    global LAST_COMMAND_CONTEXT

    requested_label, requested_command = (
        ORIGINAL_TARGET_FOR_TIME(t)
    )

    requested_projected = (
        trace.base.ADAPTER.project(
            requested_command
        )
    )

    if M4_ENABLED:
        selection = SUPERVISOR.select(
            requested_label,
            requested_command,
        )

        selected_label = (
            selection.selected_label
        )

        selected_command = (
            selection.command
        )

        override_active = bool(
            selection.override_active
        )

    else:
        selected_label = str(
            requested_label
        )

        selected_command = (
            requested_command
        )

        override_active = False

    selected_projected = (
        trace.base.ADAPTER.project(
            selected_command
        )
    )

    LAST_COMMAND_CONTEXT = {
        "time_s":
            float(t),

        "requested_label":
            str(requested_label),

        "selected_label":
            str(selected_label),

        "override_active":
            bool(override_active),

        "requested":
            command_vector(
                requested_command
            ),

        "requested_projected":
            command_vector(
                requested_projected
            ),

        "selected":
            command_vector(
                selected_command
            ),

        "selected_projected":
            command_vector(
                selected_projected
            ),
    }

    return (
        selected_label,
        selected_command,
    )


def observed_compute_actions(
    self,
    *args,
    **kwargs,
):
    tau = shadow.shadow_compute_actions(
        self,
        *args,
        **kwargs,
    )

    if (
        COMMAND_LOG_ACTIVE
        and LAST_COMMAND_CONTEXT
        is not None
        and trace.base.COMMAND
        is not None
    ):
        context = LAST_COMMAND_CONTEXT

        COMMAND_TRACE["time_s"].append(
            float(
                context["time_s"]
            )
        )

        COMMAND_TRACE[
            "requested_label"
        ].append(
            context["requested_label"]
        )

        COMMAND_TRACE[
            "selected_label"
        ].append(
            context["selected_label"]
        )

        COMMAND_TRACE[
            "override_active"
        ].append(
            context["override_active"]
        )

        for key in (
            "requested",
            "requested_projected",
            "selected",
            "selected_projected",
        ):
            COMMAND_TRACE[key].append(
                np.asarray(
                    context[key],
                    dtype=float,
                ).copy()
            )

        # guarded_compute_actions assigns
        # base.COMMAND = applied immediately before
        # entering the PyMPC integration hook.
        COMMAND_TRACE[
            "applied"
        ].append(
            command_vector(
                trace.base.COMMAND
            )
        )

    return tau


def active_env_step(
    self,
    action,
):
    global COMMAND_LOG_ACTIVE

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

    if M4_ENABLED:
        activated = SUPERVISOR.observe(
            status,
            time_s=event_time,
        )
    else:
        activated = False

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

    # run_simulation may reset and continue after a
    # termination.  Do not mix the next rollout into
    # command-level observability.
    if trace.TERMINATION_EVENTS:
        COMMAND_LOG_ACTIVE = False

    return result


def reset_shadow_state():
    global LAST_COMMAND_CONTEXT
    global COMMAND_LOG_ACTIVE

    LAST_COMMAND_CONTEXT = None
    COMMAND_LOG_ACTIVE = True

    for value in COMMAND_TRACE.values():
        value.clear()

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
    global M4_ENABLED

    output_arg = cli_value(
        "--output"
    )

    m4_arg = cli_value("--m4")

    if m4_arg is None:
        m4_arg = "on"

    m4_arg = m4_arg.lower()

    if m4_arg not in ("on", "off"):
        raise SystemExit(
            "--m4 must be 'on' or 'off'"
        )

    M4_ENABLED = (
        m4_arg == "on"
    )

    # --m4 belongs only to this wrapper.
    # The delegated M3 trace argparse parser does not know it.
    original_sys_argv = list(sys.argv)

    if "--m4" in sys.argv:
        i = sys.argv.index("--m4")

        if i + 1 >= len(sys.argv):
            raise SystemExit(
                "--m4 requires on or off"
            )

        del sys.argv[i:i + 2]

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
        observed_compute_actions
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

        sys.argv[:] = original_sys_argv

    lengths = {
        key: len(value)
        for key, value
        in COMMAND_TRACE.items()
    }

    unique_lengths = set(
        lengths.values()
    )

    if len(unique_lengths) != 1:
        raise RuntimeError(
            "Command observability length "
            f"mismatch: {lengths}"
        )

    command_samples = (
        len(COMMAND_TRACE["time_s"])
    )

    def stack_command(name):
        if command_samples == 0:
            return np.empty(
                (0, len(COMMAND_FIELDS)),
                dtype=float,
            )

        return np.vstack(
            COMMAND_TRACE[name]
        ).astype(float)

    command_npz = (
        base_output.with_name(
            base_output.stem
            + "_m4_commands.npz"
        )
    )

    command_npz.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        command_npz,

        time_s=np.asarray(
            COMMAND_TRACE["time_s"],
            dtype=float,
        ),

        requested_label=np.asarray(
            COMMAND_TRACE[
                "requested_label"
            ],
            dtype="U64",
        ),

        selected_label=np.asarray(
            COMMAND_TRACE[
                "selected_label"
            ],
            dtype="U64",
        ),

        override_active=np.asarray(
            COMMAND_TRACE[
                "override_active"
            ],
            dtype=bool,
        ),

        requested=stack_command(
            "requested"
        ),

        requested_projected=(
            stack_command(
                "requested_projected"
            )
        ),

        selected=stack_command(
            "selected"
        ),

        selected_projected=(
            stack_command(
                "selected_projected"
            )
        ),

        applied=stack_command(
            "applied"
        ),
    )

    command_events = (
        build_command_events()
    )

    if trace.TERMINATION_EVENTS:
        termination = (
            trace.TERMINATION_EVENTS[0]
        )
    else:
        termination = None

    # Only accept fallback commits belonging to the first
    # rollout.  Upstream run_simulation may reset the MuJoCo
    # environment after termination and continue collecting
    # samples; those post-reset events restart at t=0 and must
    # not be mistaken for a successful pre-failure back-off.
    if termination is None:
        termination_time = None
    else:
        termination_time = float(
            termination["simulation_time_s"]
        )

    override_time = (
        SUPERVISOR.override_time_s
    )

    backoff_commits = []

    if override_time is not None:
        for event in (
            guarded.STRUCTURAL_EVENTS
        ):
            if (
                event.get("target_label")
                != SUPERVISOR.fallback_label
            ):
                continue

            event_time = float(
                event["time_s"]
            )

            if event_time < override_time:
                continue

            if (
                termination_time is not None
                and event_time
                >= termination_time
            ):
                continue

            backoff_commits.append(
                event
            )

    first_backoff_commit = (
        backoff_commits[0]
        if backoff_commits
        else None
    )

    if (
        not M4_ENABLED
        and termination is not None
    ):
        classification = (
            "m4_off_terminated"
        )

    elif (
        not M4_ENABLED
        and termination is None
    ):
        classification = (
            "m4_off_survived"
        )

    elif (
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
        and first_backoff_commit
        is not None
    ):
        classification = (
            "unsafe_detected_"
            "backoff_committed_but_failed"
        )

    elif (
        SUPERVISOR.override_active
        and termination is not None
    ):
        classification = (
            "unsafe_detected_"
            "terminated_before_backoff_commit"
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

    monitor_unsafe_time = (
        None
        if shadow.FIRST_UNSAFE is None
        else float(
            shadow.FIRST_UNSAFE["time_s"]
        )
    )

    print(
        "M4 mode         :",
        "ON" if M4_ENABLED else "OFF",
    )

    print(
        "monitor UNSAFE  :",
        monitor_unsafe_time,
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

    print()
    print("command path events:")

    for event in command_events:
        req = event[
            "requested_projected"
        ]
        sel = event[
            "selected_projected"
        ]
        app = event[
            "applied"
        ]

        print(
            f"  t={event['time_s']:.3f}s "
            f"req={event['requested_label']} "
            f"sel={event['selected_label']} "
            f"override={event['override_active']} "
            f"| req(f,D)="
            f"({req['gait_frequency_hz']:.3f},"
            f"{req['duty_factor']:.3f}) "
            f"sel(f,D)="
            f"({sel['gait_frequency_hz']:.3f},"
            f"{sel['duty_factor']:.3f}) "
            f"app(f,D)="
            f"({app['gait_frequency_hz']:.3f},"
            f"{app['duty_factor']:.3f})"
        )

    result = {
        "mode":
            "m4_runtime_supervisor_ablation_v0",

        "m4_enabled":
            bool(M4_ENABLED),

        "monitor_unsafe_time_s":
            monitor_unsafe_time,

        "command_observability": {
            "ordering": [
                "requested",
                "selected",
                "selected_projected",
                "applied",
            ],

            "fields":
                list(COMMAND_FIELDS),

            "sample_count":
                int(command_samples),

            "npz":
                str(command_npz),

            "events":
                command_events,
        },

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
