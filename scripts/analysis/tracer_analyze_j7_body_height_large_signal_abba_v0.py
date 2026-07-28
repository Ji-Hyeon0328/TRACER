import csv
import math
import statistics
import sys


input_path, output_path, summary_path = (
    sys.argv[1:]
)


WINDOW_ORDER = [
    "A1",
    "B1",
    "B2",
    "A2",
]


SETTLED_RANGES = {
    "A1": (5.0, 9.8),
    "B1": (19.0, 23.8),
    "B2": (29.0, 33.8),
    "A2": (43.0, 47.8),
}


EXPECTED_BODY_H = {
    "A1": 0.320,
    "B1": 0.305,
    "B2": 0.305,
    "A2": 0.320,
}


def finite(value):
    try:
        value = float(value)
    except Exception:
        return None

    if not math.isfinite(value):
        return None

    return value


all_rows = []


with open(
    input_path,
    "r",
    encoding="utf-8",
    newline="",
) as stream:
    for raw in csv.DictReader(stream):
        window = raw["window"]

        if window not in WINDOW_ORDER:
            continue

        row = {
            "window": window,
            "offset_s": finite(raw["offset_s"]),
            "ref_body_h": finite(raw["ref_body_h"]),
            "base_z": finite(raw["base_z"]),
            "base_vz": finite(raw["base_vz"]),
            "roll_deg": finite(raw["roll_deg"]),
            "pitch_deg": finite(raw["pitch_deg"]),
            "yaw_deg": finite(raw["yaw_deg"]),
        }

        if any(
            row[key] is None
            for key in (
                "offset_s",
                "ref_body_h",
                "base_z",
                "base_vz",
                "roll_deg",
                "pitch_deg",
                "yaw_deg",
            )
        ):
            continue

        if not math.isclose(
            row["ref_body_h"],
            EXPECTED_BODY_H[window],
            abs_tol=1e-9,
        ):
            continue

        start, end = SETTLED_RANGES[window]

        if not (
            start
            <= row["offset_s"]
            < end
        ):
            continue

        all_rows.append(row)


by_window = {
    name: []
    for name in WINDOW_ORDER
}


for row in all_rows:
    by_window[row["window"]].append(row)


summaries = []


for name in WINDOW_ORDER:
    rows = by_window[name]

    if len(rows) < 20:
        raise RuntimeError(
            "{} has insufficient settled rows: {}".format(
                name,
                len(rows),
            )
        )

    base_z = [
        row["base_z"]
        for row in rows
    ]

    base_vz = [
        row["base_vz"]
        for row in rows
    ]

    roll = [
        abs(row["roll_deg"])
        for row in rows
    ]

    pitch = [
        abs(row["pitch_deg"])
        for row in rows
    ]

    summary = {
        "window": name,
        "condition": (
            "A"
            if name.startswith("A")
            else "B"
        ),
        "expected_body_h": EXPECTED_BODY_H[name],
        "sample_count": len(rows),
        "median_base_z": statistics.median(base_z),
        "mean_base_z": statistics.mean(base_z),
        "std_base_z": (
            statistics.stdev(base_z)
            if len(base_z) > 1
            else 0.0
        ),
        "median_abs_base_vz": statistics.median(
            abs(value)
            for value in base_vz
        ),
        "median_abs_roll_deg": statistics.median(
            roll
        ),
        "max_abs_roll_deg": max(roll),
        "median_abs_pitch_deg": statistics.median(
            pitch
        ),
        "max_abs_pitch_deg": max(pitch),
        "min_base_z": min(base_z),
        "max_base_z": max(base_z),
    }

    summaries.append(summary)


summary_by_name = {
    row["window"]: row
    for row in summaries
}


a1 = summary_by_name["A1"]["median_base_z"]
b1 = summary_by_name["B1"]["median_base_z"]
b2 = summary_by_name["B2"]["median_base_z"]
a2 = summary_by_name["A2"]["median_base_z"]


a_mean = statistics.mean([a1, a2])
b_mean = statistics.mean([b1, b2])

simple_effect = b_mean - a_mean


time_centers = {
    "A1": 5.9,
    "B1": 15.9,
    "B2": 23.9,
    "A2": 33.9,
}


def interpolated_a(time_value):
    alpha = (
        (
            time_value
            - time_centers["A1"]
        )
        /
        (
            time_centers["A2"]
            - time_centers["A1"]
        )
    )

    return a1 + alpha * (a2 - a1)


b1_effect = (
    b1
    - interpolated_a(
        time_centers["B1"]
    )
)

b2_effect = (
    b2
    - interpolated_a(
        time_centers["B2"]
    )
)

corrected_effect = statistics.mean(
    [b1_effect, b2_effect]
)

command_effect = -0.015

realized_target_ratio = (
    corrected_effect
    / command_effect
)


all_settled_z = [
    row["base_z"]
    for row in all_rows
]

all_settled_roll = [
    abs(row["roll_deg"])
    for row in all_rows
]

all_settled_pitch = [
    abs(row["pitch_deg"])
    for row in all_rows
]


fieldnames = [
    "window",
    "condition",
    "expected_body_h",
    "sample_count",
    "median_base_z",
    "mean_base_z",
    "std_base_z",
    "median_abs_base_vz",
    "median_abs_roll_deg",
    "max_abs_roll_deg",
    "median_abs_pitch_deg",
    "max_abs_pitch_deg",
    "min_base_z",
    "max_base_z",
]


with open(
    output_path,
    "w",
    encoding="utf-8",
    newline="",
) as stream:
    writer = csv.DictWriter(
        stream,
        fieldnames=fieldnames,
    )

    writer.writeheader()
    writer.writerows(summaries)


lines = [
    "TRACER J7 body-height interleaved ABBA authority",
    "=================================================",
    "",
    "A command body_h = 0.320 m",
    "B command body_h = 0.305 m",
    "command effect m = {:+.9f}".format(
        command_effect
    ),
    "",
]


for row in summaries:
    lines.append(
        (
            "{window}: n={sample_count} "
            "median_base_z={median_base_z:.9f} "
            "mean_base_z={mean_base_z:.9f} "
            "std_base_z={std_base_z:.9f} "
            "median_abs_vz={median_abs_base_vz:.9f} "
            "max_abs_roll_deg={max_abs_roll_deg:.6f} "
            "max_abs_pitch_deg={max_abs_pitch_deg:.6f}"
        ).format(**row)
    )


lines.extend([
    "",
    "A endpoint drift mm = {:+.6f}".format(
        1000.0 * (a2 - a1)
    ),
    "B half drift mm = {:+.6f}".format(
        1000.0 * (b2 - b1)
    ),
    "simple ABBA base-z effect mm = {:+.6f}".format(
        1000.0 * simple_effect
    ),
    "B1 drift-corrected effect mm = {:+.6f}".format(
        1000.0 * b1_effect
    ),
    "B2 drift-corrected effect mm = {:+.6f}".format(
        1000.0 * b2_effect
    ),
    "combined drift-corrected effect mm = {:+.6f}".format(
        1000.0 * corrected_effect
    ),
    "realized/target ratio = {:.6f}".format(
        realized_target_ratio
    ),
    "",
    "minimum settled base_z m = {:.6f}".format(
        min(all_settled_z)
    ),
    "maximum settled |roll| deg = {:.6f}".format(
        max(all_settled_roll)
    ),
    "maximum settled |pitch| deg = {:.6f}".format(
        max(all_settled_pitch)
    ),
])


failures = []


if b1_effect >= -0.0030:
    failures.append(
        "B1 did not produce a clear lower base-z response"
    )


if b2_effect >= -0.0030:
    failures.append(
        "B2 did not produce a clear lower base-z response"
    )


if corrected_effect >= -0.0050:
    failures.append(
        "combined body-height response below 5 mm"
    )


if min(all_settled_z) <= 0.20:
    failures.append(
        "gross base-height collapse observed"
    )


if max(all_settled_roll) >= 20.0:
    failures.append(
        "gross roll excursion observed"
    )


if max(all_settled_pitch) >= 20.0:
    failures.append(
        "gross pitch excursion observed"
    )


if failures:
    lines.extend([
        "",
        "[FAIL]",
    ])

    lines.extend(
        "- " + failure
        for failure in failures
    )
else:
    lines.extend([
        "",
        (
            "[EVIDENCE] J7-selected body-height command "
            "reached LL1 and produced a repeatable "
            "negative realized base-z response."
        ),
        (
            "[LIMIT] This establishes command authority "
            "only, not terrain, stability, energy, "
            "selector, or policy benefit."
        ),
    ])


text = "\n".join(lines) + "\n"


with open(
    summary_path,
    "w",
    encoding="utf-8",
) as stream:
    stream.write(text)


print(text)


if failures:
    raise RuntimeError(
        "; ".join(failures)
    )
