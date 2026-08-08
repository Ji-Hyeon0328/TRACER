#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

RESULT_DIR = (
    ROOT
    / "results"
    / "icra27"
    / "m4_safety_monitor"
    / "standalone_v0"
)

NOMINAL_PATH = (
    RESULT_DIR
    / "nominal_f140_d065.json"
)

M4_PATH = (
    RESULT_DIR
    / "f110_d055.json"
)


def load(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)

    return json.loads(
        path.read_text()
    )


def close(
    value,
    expected,
    tol=0.010,
):
    return (
        value is not None
        and abs(
            float(value)
            - float(expected)
        ) <= tol
    )


def main():
    nominal = load(NOMINAL_PATH)
    active = load(M4_PATH)

    # --------------------------------------------------
    # Nominal standalone acceptance.
    # --------------------------------------------------
    assert (
        nominal["mode"]
        == "pympc_lowlevel_standalone_v0"
    )

    assert nominal["first_unsafe"] is None
    assert nominal["first_override"] is None

    assert (
        nominal["first_backoff_commit"]
        is None
    )

    assert (
        nominal["termination_count"]
        == 0
    )

    assert nominal["hooks_restored"] is True

    # Upstream performs a controller reset at the
    # normal episode/horizon boundary as well.
    assert len(
        nominal["reset_events"]
    ) >= 1

    # --------------------------------------------------
    # Known M4 intervention trajectory.
    # --------------------------------------------------
    assert (
        active["mode"]
        == "pympc_lowlevel_standalone_v0"
    )

    assert (
        active["first_unsafe"]
        is not None
    )

    assert (
        active["first_override"]
        is not None
    )

    assert (
        active["first_backoff_commit"]
        is not None
    )

    assert (
        active["termination_count"]
        == 0
    )

    assert active["hooks_restored"] is True

    unsafe_t = (
        active["first_unsafe"]["time_s"]
    )

    override_t = (
        active["first_override"]["time_s"]
    )

    backoff_t = (
        active[
            "first_backoff_commit"
        ]["time_s"]
    )

    # Cross-check against the frozen canonical
    # f110_d055 M4-v0 trajectory.
    assert close(
        unsafe_t,
        5.138,
    ), unsafe_t

    assert close(
        override_t,
        5.138,
    ), override_t

    assert close(
        backoff_t,
        5.488,
    ), backoff_t

    assert backoff_t > override_t

    # The episode-boundary reset must clean
    # runtime safety state.
    assert active["reset_events"]

    final_reset = (
        active["reset_events"][-1]
    )

    assert (
        final_reset["pre"][
            "unsafe_latched"
        ]
        is True
    )

    assert (
        final_reset["pre"][
            "override_active"
        ]
        is True
    )

    assert (
        final_reset["post"][
            "unsafe_latched"
        ]
        is False
    )

    assert (
        final_reset["post"][
            "override_active"
        ]
        is False
    )

    print(
        "PyMPC closed low-level "
        "E2E acceptance: PASS"
    )

    print(
        "nominal:"
        " survived, no intervention"
    )

    print(
        "M4:"
        f" unsafe={unsafe_t:.3f}s"
        f" override={override_t:.3f}s"
        f" backoff={backoff_t:.3f}s"
        " termination=none"
    )

    print(
        "episode reset:"
        " unsafe/override -> clean"
    )

    print(
        "standalone hooks:"
        " restored"
    )


if __name__ == "__main__":
    main()
