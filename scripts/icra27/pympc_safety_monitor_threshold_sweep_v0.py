#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


from tracer_core.lowlevel.pympc_safety_monitor import (
    DEFAULT_PYMPC_SAFETY_MONITOR_CONFIG,
    PyMPCSafetyMonitor,
    PyMPCSafetyState,
)


TRACE_ROOT = (
    ROOT
    / "results"
    / "icra27"
    / "m3_contact_attitude_trace_v0"
)


THRESHOLDS = (
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
)


def load_case(json_path: Path):
    meta = json.loads(
        json_path.read_text()
    )

    npz_path = json_path.with_suffix(
        ".npz"
    )

    if not npz_path.exists():
        raise FileNotFoundError(npz_path)

    x = np.load(npz_path)

    required = (
        "time_s",
        "base_pos",
        "base_ori_rpy",
        "planned_contact",
        "physical_contact",
    )

    for key in required:
        if key not in x.files:
            raise RuntimeError(
                f"{json_path.stem}: missing {key}"
            )

    t = np.asarray(
        x["time_s"],
        dtype=float,
    )

    base_pos = np.asarray(
        x["base_pos"],
        dtype=float,
    )

    rpy = np.asarray(
        x["base_ori_rpy"],
        dtype=float,
    )

    planned = np.asarray(
        x["planned_contact"],
        dtype=bool,
    )

    physical = np.asarray(
        x["physical_contact"],
        dtype=bool,
    )

    n = min(
        len(t),
        len(base_pos),
        len(rpy),
        len(planned),
        len(physical),
    )

    t = t[:n]
    base_pos = base_pos[:n]
    rpy = rpy[:n]
    planned = planned[:n]
    physical = physical[:n]

    termination = meta.get(
        "first_termination"
    )

    if termination is None:
        stop = n
        termination_time = None
    else:
        stop = min(
            int(
                termination[
                    "global_sample_index"
                ]
            )
            + 1,
            n,
        )

        termination_time = float(
            termination[
                "simulation_time_s"
            ]
        )

    commit = meta.get(
        "target_commit"
    )

    if commit is None:
        raise RuntimeError(
            f"{json_path.stem}: no target_commit"
        )

    commit_time = float(
        commit["time_s"]
    )

    positive_dt = np.diff(
        t[:stop]
    )

    positive_dt = positive_dt[
        positive_dt > 0.0
    ]

    if len(positive_dt) == 0:
        raise RuntimeError(
            f"{json_path.stem}: no positive dt"
        )

    nominal_dt = float(
        np.median(positive_dt)
    )

    return {
        "tag":
            json_path.stem,

        "meta":
            meta,

        "time":
            t,

        "base_pos":
            base_pos,

        "rpy":
            rpy,

        "planned":
            planned,

        "physical":
            physical,

        "stop":
            stop,

        "commit_time":
            commit_time,

        "termination_time":
            termination_time,

        "ground_truth_failure":
            termination_time is not None,

        "nominal_dt":
            nominal_dt,
    }


def replay(
    case,
    unsafe_support_threshold,
):
    config = replace(
        DEFAULT_PYMPC_SAFETY_MONITOR_CONFIG,
        unsafe_support_deficit_fraction=(
            float(
                unsafe_support_threshold
            )
        ),
    )

    monitor = PyMPCSafetyMonitor(
        config
    )

    t = case["time"]
    stop = case["stop"]

    notified = False

    first_unsafe = None

    max_post_support = 0.0
    max_post_roll_deg = 0.0
    max_post_pitch_deg = 0.0

    for i in range(stop):
        time_s = float(t[i])

        if (
            not notified
            and time_s
            >= case["commit_time"]
        ):
            monitor.notify_structural_commit()
            notified = True

        if i == 0:
            dt = case["nominal_dt"]
        else:
            raw_dt = float(
                t[i] - t[i - 1]
            )

            dt = (
                raw_dt
                if raw_dt > 0.0
                else case["nominal_dt"]
            )

        status = monitor.update(
            planned_contact=(
                case["planned"][i]
            ),

            physical_contact=(
                case["physical"][i]
            ),

            roll_rad=float(
                case["rpy"][i, 0]
            ),

            pitch_rad=float(
                case["rpy"][i, 1]
            ),

            base_height_m=float(
                case["base_pos"][i, 2]
            ),

            dt=dt,
        )

        if (
            notified
            and status.post_transition_active
        ):
            max_post_support = max(
                max_post_support,
                float(
                    status
                    .support_deficit_fraction
                ),
            )

            max_post_roll_deg = max(
                max_post_roll_deg,
                abs(
                    float(
                        np.degrees(
                            status.roll_rad
                        )
                    )
                ),
            )

            max_post_pitch_deg = max(
                max_post_pitch_deg,
                abs(
                    float(
                        np.degrees(
                            status.pitch_rad
                        )
                    )
                ),
            )

        if (
            first_unsafe is None
            and status.state
            == PyMPCSafetyState.UNSAFE
        ):
            first_unsafe = {
                "time_s":
                    time_s,

                "support":
                    float(
                        status
                        .support_deficit_fraction
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

                "reasons":
                    list(status.reasons),
            }

    detected = (
        first_unsafe is not None
    )

    if (
        first_unsafe is not None
        and case["termination_time"]
        is not None
    ):
        lead = float(
            case["termination_time"]
            - first_unsafe["time_s"]
        )
    else:
        lead = None

    return {
        "tag":
            case["tag"],

        "ground_truth_failure":
            case["ground_truth_failure"],

        "unsafe_detected":
            detected,

        "first_unsafe":
            first_unsafe,

        "lead_time_s":
            lead,

        "max_post_support":
            max_post_support,

        "max_post_roll_deg":
            max_post_roll_deg,

        "max_post_pitch_deg":
            max_post_pitch_deg,
    }


def main():
    json_files = sorted(
        TRACE_ROOT.glob("*.json")
    )

    if not json_files:
        raise SystemExit(
            f"No traces under {TRACE_ROOT}"
        )

    cases = [
        load_case(p)
        for p in json_files
    ]

    all_results = {}

    print("=" * 112)
    print(
        "ICRA27 M4 UNSAFE SUPPORT-THRESHOLD STRESS CHECK"
    )
    print("=" * 112)

    print(
        "Fixed:"
        " support_window=0.20 s,"
        " existing attitude/base-height thresholds unchanged"
    )

    print(
        "This is a v0 engineering stress check,"
        " not statistical safety validation."
    )

    for threshold in THRESHOLDS:
        print()
        print("=" * 112)
        print(
            f"unsafe_support_threshold = "
            f"{threshold:.2f}"
        )
        print("=" * 112)

        results = [
            replay(
                case,
                threshold,
            )
            for case in cases
        ]

        all_results[
            f"{threshold:.2f}"
        ] = results

        tp = fp = tn = fn = 0
        leads = []

        print(
            f"{'case':14s} "
            f"{'GT':8s} "
            f"{'UNSAFE':8s} "
            f"{'t_unsafe':>9s} "
            f"{'lead':>9s} "
            f"{'supMax':>8s} "
            f"{'rollMax':>9s} "
            f"{'reason'}"
        )

        print("-" * 112)

        for r in results:
            gt = r[
                "ground_truth_failure"
            ]

            pred = r[
                "unsafe_detected"
            ]

            if gt and pred:
                tp += 1
            elif (
                not gt
                and pred
            ):
                fp += 1
            elif (
                not gt
                and not pred
            ):
                tn += 1
            else:
                fn += 1

            event = r[
                "first_unsafe"
            ]

            if event is None:
                unsafe_t = "-"
                reason = "-"
            else:
                unsafe_t = (
                    f"{event['time_s']:.3f}"
                )

                reason = ",".join(
                    event["reasons"]
                )

            if r["lead_time_s"] is None:
                lead_text = "-"
            else:
                lead_text = (
                    f"{r['lead_time_s']:.3f}"
                )

                if gt and pred:
                    leads.append(
                        r["lead_time_s"]
                    )

            print(
                f"{r['tag']:14s} "
                f"{'FAIL' if gt else 'PASS':8s} "
                f"{'YES' if pred else 'NO':8s} "
                f"{unsafe_t:>9s} "
                f"{lead_text:>9s} "
                f"{r['max_post_support']:8.3f} "
                f"{r['max_post_roll_deg']:9.2f} "
                f"{reason}"
            )

        print("-" * 112)

        print(
            f"TP={tp} FP={fp} TN={tn} FN={fn}"
        )

        if leads:
            print(
                "failure precursor lead:"
                f" min={min(leads):.3f}s"
                f" median={np.median(leads):.3f}s"
                f" max={max(leads):.3f}s"
            )

    output = (
        ROOT
        / "results"
        / "icra27"
        / "m4_safety_monitor"
        / "threshold_sweep_v0.json"
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.write_text(
        json.dumps(
            {
                "thresholds":
                    list(THRESHOLDS),

                "results":
                    all_results,
            },
            indent=2,
        )
        + "\n"
    )

    print()
    print("=" * 112)
    print("saved:", output)
    print("=" * 112)


if __name__ == "__main__":
    main()
