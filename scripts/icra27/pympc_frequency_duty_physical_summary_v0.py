#!/usr/bin/env python3

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = (
    ROOT
    / "results"
    / "icra27"
    / "m3_frequency_duty_physical_v1"
)

OUTPUT = (
    ROOT
    / "results"
    / "icra27"
    / "m3_authority"
    / "frequency_duty_physical_v0.json"
)


def load_points():
    points = []

    for path in sorted(INPUT_DIR.glob("*.json")):
        data = json.loads(
            path.read_text(encoding="utf-8")
        )

        c = data["command"]
        s = data["summary"]
        stability = data["stability"]

        points.append({
            "file":
                str(path.relative_to(ROOT)),

            "frequency_hz":
                float(c["gait_frequency_hz"]),

            "duty_command":
                float(c["duty_factor"]),

            "planned_duty":
                float(
                    s["mean_planned_duty_factor"]
                ),

            "physical_duty":
                float(
                    s["mean_physical_duty_factor"]
                ),

            "physical_minus_planned":
                float(
                    s["mean_physical_minus_planned"]
                ),

            "contact_agreement":
                float(
                    s["mean_contact_agreement_ratio"]
                ),

            "max_leg_duty_error":
                float(
                    s["max_abs_leg_duty_error"]
                ),

            "usable_tracking_point":
                bool(
                    stability[
                        "usable_tracking_point"
                    ]
                ),

            "num_terminations":
                int(
                    stability[
                        "num_terminations"
                    ]
                ),

            "first_termination":
                stability[
                    "first_termination"
                ],
        })

    if len(points) != 9:
        raise RuntimeError(
            f"Expected 9 grid points, got {len(points)}"
        )

    return points


def linear_fit(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    slope, intercept = np.polyfit(
        x,
        y,
        1,
    )

    pred = slope * x + intercept

    ss_res = float(
        np.sum((y - pred) ** 2)
    )

    ss_tot = float(
        np.sum(
            (y - np.mean(y)) ** 2
        )
    )

    r2 = (
        1.0 - ss_res / ss_tot
        if ss_tot > 1e-15
        else None
    )

    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": (
            float(r2)
            if r2 is not None
            else None
        ),
    }


def main():
    points = load_points()

    by_frequency = defaultdict(list)
    by_duty = defaultdict(list)

    for p in points:
        by_frequency[
            p["frequency_hz"]
        ].append(p)

        by_duty[
            p["duty_command"]
        ].append(p)

    frequency_summary = []

    for frequency in sorted(by_frequency):
        group_all = sorted(
            by_frequency[frequency],
            key=lambda p: p["duty_command"],
        )

        group = [
            p
            for p in group_all
            if p["usable_tracking_point"]
        ]

        if len(group) < 2:
            raise RuntimeError(
                f"frequency={frequency}: "
                "fewer than 2 valid tracking points"
            )

        physical_fit = linear_fit(
            [
                p["duty_command"]
                for p in group
            ],
            [
                p["physical_duty"]
                for p in group
            ],
        )

        planned_fit = linear_fit(
            [
                p["duty_command"]
                for p in group
            ],
            [
                p["planned_duty"]
                for p in group
            ],
        )

        frequency_summary.append({
            "frequency_hz":
                float(frequency),

            "planned_duty_gain":
                planned_fit["slope"],

            "physical_duty_gain":
                physical_fit["slope"],

            "physical_duty_intercept":
                physical_fit["intercept"],

            "physical_duty_r_squared":
                physical_fit["r_squared"],

            "num_valid_points":
                int(len(group)),

            "num_invalid_points":
                int(
                    len(group_all) - len(group)
                ),

            "mean_physical_minus_planned":
                float(
                    np.mean([
                        p["physical_minus_planned"]
                        for p in group
                    ])
                ),

            "mean_contact_agreement":
                float(
                    np.mean([
                        p["contact_agreement"]
                        for p in group
                    ])
                ),

            "worst_contact_agreement":
                float(
                    min(
                        p["contact_agreement"]
                        for p in group
                    )
                ),

            "max_leg_duty_error":
                float(
                    max(
                        p["max_leg_duty_error"]
                        for p in group
                    )
                ),
        })

    duty_summary = []

    for duty in sorted(by_duty):
        group_all = sorted(
            by_duty[duty],
            key=lambda p: p["frequency_hz"],
        )

        group_valid = [
            p
            for p in group_all
            if p["usable_tracking_point"]
        ]

        if not group_valid:
            raise RuntimeError(
                f"duty={duty}: no valid tracking points"
            )

        physical_values = [
            p["physical_duty"]
            for p in group_valid
        ]

        agreement_values = [
            p["contact_agreement"]
            for p in group_valid
        ]

        duty_summary.append({
            "duty_command":
                float(duty),

            "num_valid_points":
                int(len(group_valid)),

            "num_invalid_points":
                int(
                    len(group_all)
                    - len(group_valid)
                ),

            "invalid_frequencies_hz": [
                float(p["frequency_hz"])
                for p in group_all
                if not p["usable_tracking_point"]
            ],

            "physical_duty_by_frequency": {
                f"{p['frequency_hz']:.1f}": (
                    float(p["physical_duty"])
                    if p["usable_tracking_point"]
                    else None
                )
                for p in group_all
            },

            "physical_minus_planned_by_frequency": {
                f"{p['frequency_hz']:.1f}": (
                    float(
                        p["physical_minus_planned"]
                    )
                    if p["usable_tracking_point"]
                    else None
                )
                for p in group_all
            },

            "physical_duty_range_valid_only":
                float(
                    max(physical_values)
                    - min(physical_values)
                ),

            "agreement_range_valid_only":
                float(
                    max(agreement_values)
                    - min(agreement_values)
                ),
        })

    result = {
        "grid": {
            "frequency_hz":
                sorted(by_frequency.keys()),

            "duty_factor":
                sorted(by_duty.keys()),
        },

        "points": points,

        "per_frequency_summary":
            frequency_summary,

        "per_duty_summary":
            duty_summary,
    }

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT.write_text(
        json.dumps(
            result,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    print("=" * 110)
    print(
        "ICRA27 M3 frequency × duty "
        "PHYSICAL CONTACT SUMMARY"
    )
    print("=" * 110)

    print(
        f"{'freq':>8s}"
        f"{'plan gain':>14s}"
        f"{'phys gain':>14s}"
        f"{'phys bias':>14s}"
        f"{'phys R2':>12s}"
        f"{'phys-plan':>14s}"
        f"{'agreement':>14s}"
        f"{'worst agr':>14s}"
    )

    print("-" * 110)

    for r in frequency_summary:
        print(
            f"{r['frequency_hz']:8.3f}"
            f"{r['planned_duty_gain']:14.6f}"
            f"{r['physical_duty_gain']:14.6f}"
            f"{r['physical_duty_intercept']:14.6f}"
            f"{r['physical_duty_r_squared']:12.6f}"
            f"{r['mean_physical_minus_planned']:14.6f}"
            f"{r['mean_contact_agreement']:14.6f}"
            f"{r['worst_contact_agreement']:14.6f}"
        )

    print("=" * 110)

    print()
    print("Fixed-duty frequency sensitivity:")
    print()

    for r in duty_summary:
        vals = r[
            "physical_duty_by_frequency"
        ]

        formatted = []

        for frequency in ("1.0", "1.4", "2.0"):
            value = vals.get(frequency)

            if value is None:
                formatted.append(
                    f"f{frequency}=INVALID"
                )
            else:
                formatted.append(
                    f"f{frequency}={value:.6f}"
                )

        print(
            f"D={r['duty_command']:.2f} | "
            + ", ".join(formatted)
            + " | valid range="
            + f"{r['physical_duty_range_valid_only']:.6f}"
            + " | invalid="
            + str(r["invalid_frequencies_hz"])
        )

    print()
    print("Saved:", OUTPUT)


if __name__ == "__main__":
    main()
