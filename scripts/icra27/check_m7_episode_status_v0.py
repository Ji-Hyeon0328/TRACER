#!/usr/bin/env python3

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(
    0,
    str(ROOT),
)


from tracer_core.highlevel_rl.pympc_env import (
    _resolve_episode_status,
)

from tracer_core.highlevel_rl.reward import (
    compute_m7_reward,
)


def status(**kwargs):
    base = dict(
        success=False,
        native_terminated=False,
        native_truncated=False,
        episode_step=1,
        max_episode_steps=25,
        safety_state="normal",
        override_active=False,
        terminate_on_m4_unsafe=True,
    )

    base.update(kwargs)

    return _resolve_episode_status(
        **base
    )


def main():
    # Normal transition.
    x = status()

    assert not x["terminated"]
    assert not x["truncated"]
    assert not x["m4_terminal"]

    # WATCH must remain diagnostic.
    x = status(
        safety_state="watch"
    )

    assert not x["m4_intervention"]
    assert not x["m4_terminal"]
    assert not x["terminated"]

    # UNSAFE becomes terminal when enabled.
    x = status(
        safety_state="unsafe"
    )

    assert x["m4_intervention"]
    assert x["m4_terminal"]
    assert x["terminated"]
    assert not x["truncated"]

    # Override alone is sufficient.
    x = status(
        override_active=True
    )

    assert x["m4_intervention"]
    assert x["m4_terminal"]
    assert x["terminated"]

    # Backward-compatible diagnostic mode.
    x = status(
        safety_state="unsafe",
        terminate_on_m4_unsafe=False,
    )

    assert x["m4_intervention"]
    assert not x["m4_terminal"]
    assert not x["terminated"]

    # Success remains success.
    x = status(
        success=True
    )

    assert x["terminated"]
    assert not x["m4_terminal"]

    # Time limit is truncation, not failure termination.
    x = status(
        episode_step=25
    )

    assert not x["terminated"]
    assert x["truncated"]

    # M4 terminal takes precedence over time-limit truncation.
    x = status(
        safety_state="unsafe",
        episode_step=25,
    )

    assert x["terminated"]
    assert not x["truncated"]

    # Reward semantics:
    # unsafe terminal => intervention + failure.
    reward, components = compute_m7_reward(
        previous_goal_distance=1.0,
        goal_distance=1.0,
        heading_error=0.0,
        roll=0.0,
        pitch=0.0,
        normalized_action=[0, 0, 0, 0],
        previous_normalized_action=[
            0, 0, 0, 0
        ],
        decision_dt=0.2,
        override_active=True,
        safety_state="unsafe",
        success=False,
        terminated=True,
        truncated=False,
    )

    assert components[
        "m4_intervention"
    ] == 1.0

    assert components[
        "failure"
    ] is True

    print(
        "normal transition           : PASS"
    )
    print(
        "WATCH remains non-terminal  : PASS"
    )
    print(
        "UNSAFE -> terminal          : PASS"
    )
    print(
        "override -> terminal        : PASS"
    )
    print(
        "diagnostic compatibility    : PASS"
    )
    print(
        "success semantics           : PASS"
    )
    print(
        "time-limit truncation       : PASS"
    )
    print(
        "M4 terminal precedence      : PASS"
    )
    print(
        "failure + intervention reward: PASS"
    )

    print()
    print(
        "[ICRA27] M7 episode-status "
        "semantics: PASS"
    )


if __name__ == "__main__":
    main()
