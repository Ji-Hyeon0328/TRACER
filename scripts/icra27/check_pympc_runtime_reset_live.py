#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPT_DIR))


import pympc_safety_supervisor_ablation_v0 as ablation

from tracer_core.lowlevel.pympc_runtime_lifecycle import (
    PyMPCLowLevelLifecycle,
)


base = ablation.trace.base
guarded = ablation.guarded
shadow = ablation.shadow
sim = base.sim


LIFECYCLE = PyMPCLowLevelLifecycle(
    adapter=base.ADAPTER,
    transition_manager=guarded.MANAGER,
    safety_monitor=shadow.MONITOR,
    supervisor=ablation.SUPERVISOR,
)


RESET_EVENTS: list[dict] = []


def cli_value(flag: str) -> str | None:
    if flag not in sys.argv:
        return None

    i = sys.argv.index(flag)

    if i + 1 >= len(sys.argv):
        raise SystemExit(
            f"{flag} requires a value"
        )

    return sys.argv[i + 1]


def snapshot(stage: str) -> dict:
    status = shadow.MONITOR.last_status

    return {
        "stage": stage,

        "monitor_state":
            status.state.value,

        "monitor_unsafe_latched":
            bool(
                shadow.MONITOR.unsafe_latched
            ),

        "supervisor_override_active":
            bool(
                ablation.SUPERVISOR
                .override_active
            ),

        "manager_current_is_none":
            (
                guarded.MANAGER
                .current_reference
                is None
            ),

        "manager_pending_structural":
            bool(
                guarded.MANAGER
                .has_pending_structural_update
            ),

        "base_command_is_none":
            base.COMMAND is None,

        "current_target_label":
            guarded.CURRENT_TARGET_LABEL,

        "last_command_context_is_none":
            (
                ablation.LAST_COMMAND_CONTEXT
                is None
            ),

        "lifecycle_reset_count":
            int(
                LIFECYCLE.reset_count
            ),
    }


ORIGINAL_WRAPPER_RESET = (
    sim.QuadrupedPyMPC_Wrapper.reset
)


def reset_with_tracer(
    self,
    initial_feet_pos,
):
    # Upstream has already called env.reset()
    # immediately before entering this method.
    pre = snapshot(
        "after_env_reset_before_tracer_reset"
    )

    # Preserve the upstream PyMPC reset exactly.
    result = ORIGINAL_WRAPPER_RESET(
        self,
        initial_feet_pos,
    )

    # TRACER-owned persistent LL state.
    LIFECYCLE.reset_episode()

    # Runtime command context belongs to the
    # current episode, not to accumulated evidence.
    base.COMMAND = None
    guarded.CURRENT_TARGET_LABEL = None
    ablation.LAST_COMMAND_CONTEXT = None
    shadow.PENDING_SAMPLE = None

    post = snapshot(
        "after_pympc_and_tracer_reset"
    )

    RESET_EVENTS.append({
        "index": len(RESET_EVENTS),
        "pre": pre,
        "post": post,
    })

    print()
    print("=" * 72)
    print(
        "[ICRA27 RESET] "
        "PyMPC reset -> TRACER reset_episode"
    )
    print(
        "  pre  : "
        f"monitor={pre['monitor_state']} "
        f"unsafe="
        f"{pre['monitor_unsafe_latched']} "
        f"override="
        f"{pre['supervisor_override_active']} "
        f"manager_none="
        f"{pre['manager_current_is_none']} "
        f"command_none="
        f"{pre['base_command_is_none']}"
    )
    print(
        "  post : "
        f"monitor={post['monitor_state']} "
        f"unsafe="
        f"{post['monitor_unsafe_latched']} "
        f"override="
        f"{post['supervisor_override_active']} "
        f"manager_none="
        f"{post['manager_current_is_none']} "
        f"command_none="
        f"{post['base_command_is_none']}"
    )
    print("=" * 72)

    return result


def main() -> None:
    output_arg = cli_value("--output")

    if output_arg is None:
        raise SystemExit(
            "--output is required"
        )

    # This checker always tests M4 ON.
    # Do not pass --m4 to trace.main(), whose
    # argparse parser does not know that flag.
    if "--m4" in sys.argv:
        raise SystemExit(
            "Do not pass --m4; "
            "this checker forces M4 ON."
        )

    base_output = Path(output_arg)

    check_output = (
        base_output.with_name(
            base_output.stem
            + "_reset_check.json"
        )
    )

    check_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Clean M4-side state before the run.
    ablation.M4_ENABLED = True
    ablation.reset_shadow_state()

    original_target_for_time = (
        guarded.target_for_time
    )

    original_traced_compute = (
        ablation.trace.traced_compute_actions
    )

    original_instrumented_env_step = (
        ablation.trace.instrumented_env_step
    )

    original_wrapper_reset = (
        sim.QuadrupedPyMPC_Wrapper.reset
    )

    # Install exactly the already validated
    # M4 live-control hooks.
    guarded.target_for_time = (
        ablation.supervised_target_for_time
    )

    ablation.trace.traced_compute_actions = (
        ablation.observed_compute_actions
    )

    ablation.trace.instrumented_env_step = (
        ablation.active_env_step
    )

    sim.QuadrupedPyMPC_Wrapper.reset = (
        reset_with_tracer
    )

    try:
        # Run only the underlying trace experiment.
        #
        # This avoids the canonical ablation
        # post-processing, which intentionally
        # assumes one persistent supervisor latch.
        ablation.trace.main()

    finally:
        guarded.target_for_time = (
            original_target_for_time
        )

        ablation.trace.traced_compute_actions = (
            original_traced_compute
        )

        ablation.trace.instrumented_env_step = (
            original_instrumented_env_step
        )

        sim.QuadrupedPyMPC_Wrapper.reset = (
            original_wrapper_reset
        )

    if not RESET_EVENTS:
        raise RuntimeError(
            "No upstream controller reset was observed"
        )

    first = RESET_EVENTS[0]

    pre = first["pre"]
    post = first["post"]

    # --------------------------------------------------
    # The selected f100_d055 trajectory is known to
    # enter UNSAFE and latch M4 before termination.
    # --------------------------------------------------
    assert (
        pre["monitor_state"]
        == "unsafe"
    ), pre

    assert (
        pre["monitor_unsafe_latched"]
        is True
    ), pre

    assert (
        pre["supervisor_override_active"]
        is True
    ), pre

    assert (
        pre["manager_current_is_none"]
        is False
    ), pre

    assert (
        pre["base_command_is_none"]
        is False
    ), pre

    # --------------------------------------------------
    # After upstream PyMPC reset + TRACER reset_episode.
    # --------------------------------------------------
    assert (
        post["monitor_state"]
        == "normal"
    ), post

    assert (
        post["monitor_unsafe_latched"]
        is False
    ), post

    assert (
        post["supervisor_override_active"]
        is False
    ), post

    assert (
        post["manager_current_is_none"]
        is True
    ), post

    assert (
        post["manager_pending_structural"]
        is False
    ), post

    assert (
        post["base_command_is_none"]
        is True
    ), post

    assert (
        post["current_target_label"]
        is None
    ), post

    assert (
        post["last_command_context_is_none"]
        is True
    ), post

    assert (
        post["lifecycle_reset_count"]
        == 1
    ), post

    # --------------------------------------------------
    # Evaluation history must survive controller reset.
    # --------------------------------------------------
    termination_preserved = bool(
        ablation.trace.TERMINATION_EVENTS
    )

    first_unsafe_preserved = (
        shadow.FIRST_UNSAFE is not None
    )

    assert termination_preserved
    assert first_unsafe_preserved

    result = {
        "mode":
            "pympc_runtime_reset_live_check_v0",

        "reset_contract": [
            "upstream_env_reset",
            "upstream_pympc_reset",
            "tracer_reset_episode",
            "runtime_command_context_clear",
        ],

        "first_reset":
            first,

        "reset_count":
            len(RESET_EVENTS),

        "evidence_preserved": {
            "termination_events":
                termination_preserved,

            "first_unsafe":
                first_unsafe_preserved,
        },

        "pass": True,
    }

    check_output.write_text(
        json.dumps(
            result,
            indent=2,
        )
        + "\n"
    )

    print()
    print("=" * 72)
    print(
        "PyMPC runtime live reset checker: PASS"
    )
    print(
        "reset order:"
        " env -> PyMPC -> TRACER"
    )
    print(
        "termination evidence preserved:",
        termination_preserved,
    )
    print(
        "UNSAFE evidence preserved:",
        first_unsafe_preserved,
    )
    print(
        "saved:",
        check_output,
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
