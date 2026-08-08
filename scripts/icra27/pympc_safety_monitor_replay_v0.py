#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


from tracer_core.lowlevel.pympc_safety_monitor import (
    PyMPCSafetyMonitor,
    PyMPCSafetyState,
)


TRACE_ROOT = (
    ROOT
    / "results"
    / "icra27"
    / "m3_contact_attitude_trace_v0"
)


def load_case(tag: str):
    npz_path = TRACE_ROOT / f"{tag}.npz"
    json_path = TRACE_ROOT / f"{tag}.json"

    if not npz_path.exists():
        raise FileNotFoundError(npz_path)

    if not json_path.exists():
        raise FileNotFoundError(json_path)

    x = np.load(npz_path)
    meta = json.loads(
        json_path.read_text()
    )

    required = [
        "time_s",
        "base_pos",
        "base_ori_rpy",
        "planned_contact",
        "physical_contact",
    ]

    missing = [
        key
        for key in required
        if key not in x.files
    ]

    if missing:
        raise RuntimeError(
            f"{tag}: missing NPZ keys: {missing}"
        )

    arrays = {
        key: np.asarray(x[key])
        for key in required
    }

    lengths = {
        key: len(value)
        for key, value in arrays.items()
    }

    if len(set(lengths.values())) != 1:
        raise RuntimeError(
            f"{tag}: sample-count mismatch: {lengths}"
        )

    commit = meta.get("target_commit")

    if commit is None:
        raise RuntimeError(
            f"{tag}: target_commit missing"
        )

    termination = meta.get(
        "first_termination"
    )

    n_total = lengths["time_s"]

    # First-rollout-only replay.
    if termination is None:
        stop = n_total
    else:
        idx = int(
            termination[
                "global_sample_index"
            ]
        )

        stop = min(
            idx + 1,
            n_total,
        )

    return (
        arrays,
        meta,
        stop,
    )


def event_dict(
    *,
    sample_index: int,
    time_s: float,
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


def replay_case(tag: str):
    arrays, meta, stop = load_case(tag)

    t = arrays["time_s"][:stop]
    base_pos = arrays["base_pos"][:stop]
    rpy = arrays["base_ori_rpy"][:stop]

    planned = (
        arrays["planned_contact"][:stop]
    )

    physical = (
        arrays["physical_contact"][:stop]
    )

    if len(t) < 2:
        raise RuntimeError(
            f"{tag}: insufficient samples"
        )

    dt_samples = np.diff(t)

    positive_dt = dt_samples[
        dt_samples > 0.0
    ]

    if len(positive_dt) == 0:
        raise RuntimeError(
            f"{tag}: no positive dt"
        )

    nominal_dt = float(
        np.median(positive_dt)
    )

    commit_time = float(
        meta["target_commit"]["time_s"]
    )

    monitor = PyMPCSafetyMonitor()

    commit_notified = False

    first_watch = None
    first_unsafe = None

    transitions = []

    previous_state = None

    max_deficit_fraction = 0.0
    max_abs_roll_deg = 0.0
    max_abs_pitch_deg = 0.0
    min_base_height = float("inf")

    for i in range(len(t)):
        time_s = float(t[i])

        if (
            not commit_notified
            and time_s >= commit_time
        ):
            monitor.notify_structural_commit()
            commit_notified = True

        if i == 0:
            dt = nominal_dt
        else:
            raw_dt = float(
                t[i] - t[i - 1]
            )

            dt = (
                raw_dt
                if raw_dt > 0.0
                else nominal_dt
            )

        status = monitor.update(
            planned_contact=planned[i],
            physical_contact=physical[i],
            roll_rad=float(rpy[i, 0]),
            pitch_rad=float(rpy[i, 1]),
            base_height_m=float(
                base_pos[i, 2]
            ),
            dt=dt,
        )

        max_deficit_fraction = max(
            max_deficit_fraction,
            status.support_deficit_fraction,
        )

        max_abs_roll_deg = max(
            max_abs_roll_deg,
            abs(
                float(
                    np.degrees(
                        status.roll_rad
                    )
                )
            ),
        )

        max_abs_pitch_deg = max(
            max_abs_pitch_deg,
            abs(
                float(
                    np.degrees(
                        status.pitch_rad
                    )
                )
            ),
        )

        min_base_height = min(
            min_base_height,
            status.base_height_m,
        )

        if (
            status.state
            != previous_state
        ):
            transitions.append(
                event_dict(
                    sample_index=i,
                    time_s=time_s,
                    status=status,
                )
            )

            previous_state = status.state

        if (
            first_watch is None
            and status.state
            == PyMPCSafetyState.WATCH
        ):
            first_watch = event_dict(
                sample_index=i,
                time_s=time_s,
                status=status,
            )

        if (
            first_unsafe is None
            and status.state
            == PyMPCSafetyState.UNSAFE
        ):
            first_unsafe = event_dict(
                sample_index=i,
                time_s=time_s,
                status=status,
            )

    termination = meta.get(
        "first_termination"
    )

    termination_time = (
        None
        if termination is None
        else float(
            termination[
                "simulation_time_s"
            ]
        )
    )

    def lead_time(event):
        if (
            event is None
            or termination_time is None
        ):
            return None

        return float(
            termination_time
            - event["time_s"]
        )

    result = {
        "tag":
            tag,

        "samples_replayed":
            int(len(t)),

        "nominal_dt_s":
            nominal_dt,

        "target_commit_time_s":
            commit_time,

        "termination_time_s":
            termination_time,

        "first_watch":
            first_watch,

        "first_unsafe":
            first_unsafe,

        "watch_lead_to_termination_s":
            lead_time(first_watch),

        "unsafe_lead_to_termination_s":
            lead_time(first_unsafe),

        "state_transitions":
            transitions,

        "max_support_deficit_fraction":
            float(max_deficit_fraction),

        "max_abs_roll_deg":
            float(max_abs_roll_deg),

        "max_abs_pitch_deg":
            float(max_abs_pitch_deg),

        "min_base_height_m":
            float(min_base_height),

        "final_state":
            monitor.last_status.state.value,
    }

    return result


def print_event(
    name,
    event,
    termination_time,
):
    if event is None:
        print(
            f"{name:12s}: NONE"
        )
        return

    if termination_time is None:
        lead_text = "-"
    else:
        lead = (
            termination_time
            - event["time_s"]
        )

        lead_text = (
            f"{lead:+.3f}s before termination"
        )

    print(
        f"{name:12s}: "
        f"t={event['time_s']:.3f}s "
        f"support={event['support_deficit_fraction']:.3f} "
        f"roll={event['roll_deg']:+.2f}deg "
        f"pitch={event['pitch_deg']:+.2f}deg "
        f"z={event['base_height_m']:.3f} "
        f"{lead_text}"
    )

    print(
        " " * 14
        + "reasons="
        + str(event["reasons"])
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "tags",
        nargs="*",
        default=[
            "f110_d055",
            "f110_d060",
        ],
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "results"
            / "icra27"
            / "m4_safety_monitor"
            / "offline_replay_v0.json"
        ),
    )

    args = parser.parse_args()

    results = []

    for tag in args.tags:
        result = replay_case(tag)
        results.append(result)

        print()
        print("=" * 78)
        print(tag)
        print("=" * 78)

        print(
            "samples       :",
            result["samples_replayed"],
        )

        print(
            "commit        :",
            f"{result['target_commit_time_s']:.3f}s",
        )

        print(
            "termination   :",
            (
                "NONE"
                if (
                    result[
                        "termination_time_s"
                    ]
                    is None
                )
                else (
                    f"{result['termination_time_s']:.3f}s"
                )
            ),
        )

        print_event(
            "first WATCH",
            result["first_watch"],
            result["termination_time_s"],
        )

        print_event(
            "first UNSAFE",
            result["first_unsafe"],
            result["termination_time_s"],
        )

        print(
            "max deficit   :",
            f"{result['max_support_deficit_fraction']:.3f}",
        )

        print(
            "max |roll|    :",
            f"{result['max_abs_roll_deg']:.2f} deg",
        )

        print(
            "max |pitch|   :",
            f"{result['max_abs_pitch_deg']:.2f} deg",
        )

        print(
            "min base z    :",
            f"{result['min_base_height_m']:.3f} m",
        )

        print(
            "final state   :",
            result["final_state"],
        )

        print()
        print("state transitions:")

        for event in result[
            "state_transitions"
        ]:
            print(
                f"  t={event['time_s']:.3f}s "
                f"-> {event['state']:6s} "
                f"support="
                f"{event['support_deficit_fraction']:.3f} "
                f"roll={event['roll_deg']:+.2f}deg "
                f"pitch={event['pitch_deg']:+.2f}deg "
                f"z={event['base_height_m']:.3f} "
                f"reasons={event['reasons']}"
            )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
        json.dumps(
            {
                "mode":
                    "offline_shadow_replay",

                "cases":
                    results,
            },
            indent=2,
        )
        + "\n"
    )

    print()
    print("=" * 78)
    print("saved:", args.output)
    print("=" * 78)


if __name__ == "__main__":
    main()
