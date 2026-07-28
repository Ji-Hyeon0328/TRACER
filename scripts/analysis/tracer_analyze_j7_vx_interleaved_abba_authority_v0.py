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
    "A1": (1.0, 2.4),
    "B1": (4.0, 5.4),
    "B2": (6.5, 7.9),
    "A2": (9.5, 10.9),
}


EXPECTED_VX = {
    "A1": 0.090,
    "B1": 0.115,
    "B2": 0.115,
    "A2": 0.090,
}


def finite(value):
    try:
        value = float(value)
    except Exception:
        return None

    if not math.isfinite(value):
        return None

    return value


rows = []


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
            "ref_vx": finite(raw["ref_vx"]),
            "base_x": finite(raw["base_x"]),
            "base_y": finite(raw["base_y"]),
            "base_z": finite(raw["base_z"]),
            "vx_world": finite(raw["vx_world"]),
            "vy_world": finite(raw["vy_world"]),
            "vx_body": finite(raw["vx_body"]),
            "vy_body": finite(raw["vy_body"]),
            "roll_deg": finite(raw["roll_deg"]),
            "pitch_deg": finite(raw["pitch_deg"]),
            "yaw_deg": finite(raw["yaw_deg"]),
        }

        if any(
            row[key] is None
            for key in row
            if key != "window"
        ):
            continue

        if not math.isclose(
            row["ref_vx"],
            EXPECTED_VX[window],
            abs_tol=1e-9,
        ):
            continue

        start, end = SETTLED_RANGES[window]

        if not (
            start <= row["offset_s"] < end
        ):
            continue

        rows.append(row)


by_window = {
    name: []
    for name in WINDOW_ORDER
}


for row in rows:
    by_window[row["window"]].append(row)


summaries = []


for name in WINDOW_ORDER:
    window_rows = by_window[name]

    if len(window_rows) < 10:
        raise RuntimeError(
            "{} insufficient settled rows: {}".format(
                name,
                len(window_rows),
            )
        )

    vx_body = [
        row["vx_body"]
        for row in window_rows
    ]

    vx_world = [
        row["vx_world"]
        for row in window_rows
    ]

    vy_body = [
        abs(row["vy_body"])
        for row in window_rows
    ]

    roll = [
        abs(row["roll_deg"])
        for row in window_rows
    ]

    pitch = [
        abs(row["pitch_deg"])
        for row in window_rows
    ]

    ordered = sorted(
        window_rows,
        key=lambda row: row["offset_s"],
    )

    duration = (
        ordered[-1]["offset_s"]
        - ordered[0]["offset_s"]
    )

    dx = (
        ordered[-1]["base_x"]
        - ordered[0]["base_x"]
    )

    summary = {
        "window": name,
        "condition": (
            "A"
            if name.startswith("A")
            else "B"
        ),
        "expected_vx": EXPECTED_VX[name],
        "sample_count": len(window_rows),
        "median_vx_body": statistics.median(
            vx_body
        ),
        "mean_vx_body": statistics.mean(
            vx_body
        ),
        "std_vx_body": (
            statistics.stdev(vx_body)
            if len(vx_body) > 1
            else 0.0
        ),
        "median_vx_world": statistics.median(
            vx_world
        ),
        "median_abs_vy_body": statistics.median(
            vy_body
        ),
        "window_dx": dx,
        "window_duration": duration,
        "window_dx_rate": (
            dx / duration
            if duration > 1e-9
            else float("nan")
        ),
        "median_abs_roll_deg": statistics.median(
            roll
        ),
        "max_abs_roll_deg": max(roll),
        "median_abs_pitch_deg": statistics.median(
            pitch
        ),
        "max_abs_pitch_deg": max(pitch),
        "min_base_z": min(
            row["base_z"]
            for row in window_rows
        ),
        "max_base_x": max(
            row["base_x"]
            for row in window_rows
        ),
    }

    summaries.append(summary)


by_name = {
    row["window"]: row
    for row in summaries
}


a1 = by_name["A1"]["median_vx_body"]
b1 = by_name["B1"]["median_vx_body"]
b2 = by_name["B2"]["median_vx_body"]
a2 = by_name["A2"]["median_vx_body"]


centers = {
    "A1": 1.7,
    "B1": 4.7,
    "B2": 7.2,
    "A2": 10.2,
}


def interpolated_a(time_value):
    alpha = (
        (
            time_value
            - centers["A1"]
        )
        /
        (
            centers["A2"]
            - centers["A1"]
        )
    )

    return a1 + alpha * (a2 - a1)


b1_effect = (
    b1
    - interpolated_a(
        centers["B1"]
    )
)

b2_effect = (
    b2
    - interpolated_a(
        centers["B2"]
    )
)

combined_effect = statistics.mean(
    [b1_effect, b2_effect]
)

simple_effect = (
    statistics.mean([b1, b2])
    - statistics.mean([a1, a2])
)

command_effect = 0.025

realized_target_ratio = (
    combined_effect / command_effect
)


fieldnames = [
    "window",
    "condition",
    "expected_vx",
    "sample_count",
    "median_vx_body",
    "mean_vx_body",
    "std_vx_body",
    "median_vx_world",
    "median_abs_vy_body",
    "window_dx",
    "window_duration",
    "window_dx_rate",
    "median_abs_roll_deg",
    "max_abs_roll_deg",
    "median_abs_pitch_deg",
    "max_abs_pitch_deg",
    "min_base_z",
    "max_base_x",
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


all_roll = [
    abs(row["roll_deg"])
    for row in rows
]

all_pitch = [
    abs(row["pitch_deg"])
    for row in rows
]

all_z = [
    row["base_z"]
    for row in rows
]

all_x = [
    row["base_x"]
    for row in rows
]


lines = [
    "TRACER J7 VX interleaved ABBA authority",
    "=========================================",
    "",
    "A command vx = 0.090 m/s",
    "B command vx = 0.115 m/s",
    "command effect m/s = {:+.9f}".format(
        command_effect
    ),
    "",
]


for row in summaries:
    lines.append(
        (
            "{window}: n={sample_count} "
            "median_vx_body={median_vx_body:.9f} "
            "mean_vx_body={mean_vx_body:.9f} "
            "std_vx_body={std_vx_body:.9f} "
            "dx_rate={window_dx_rate:.9f} "
            "median_abs_vy_body={median_abs_vy_body:.9f} "
            "max_abs_roll_deg={max_abs_roll_deg:.6f} "
            "max_abs_pitch_deg={max_abs_pitch_deg:.6f}"
        ).format(**row)
    )


lines.extend([
    "",
    "A endpoint velocity drift m/s = {:+.9f}".format(
        a2 - a1
    ),
    "B-half velocity drift m/s = {:+.9f}".format(
        b2 - b1
    ),
    "simple ABBA body-vx effect m/s = {:+.9f}".format(
        simple_effect
    ),
    "B1 drift-corrected effect m/s = {:+.9f}".format(
        b1_effect
    ),
    "B2 drift-corrected effect m/s = {:+.9f}".format(
        b2_effect
    ),
    "combined drift-corrected effect m/s = {:+.9f}".format(
        combined_effect
    ),
    "realized/target ratio = {:.6f}".format(
        realized_target_ratio
    ),
    "",
    "maximum settled x m = {:.6f}".format(
        max(all_x)
    ),
    "minimum settled base-z m = {:.6f}".format(
        min(all_z)
    ),
    "maximum settled |roll| deg = {:.6f}".format(
        max(all_roll)
    ),
    "maximum settled |pitch| deg = {:.6f}".format(
        max(all_pitch)
    ),
])


failures = []


if b1_effect <= 0.005:
    failures.append(
        "B1 did not produce more than 0.005 m/s "
        "positive realized body-vx response"
    )


if b2_effect <= 0.005:
    failures.append(
        "B2 did not produce more than 0.005 m/s "
        "positive realized body-vx response"
    )


if combined_effect <= 0.008:
    failures.append(
        "combined realized VX response did not exceed "
        "0.008 m/s"
    )


if max(all_x) >= 1.8:
    failures.append(
        "flat-region x guard was exceeded"
    )


if min(all_z) <= 0.20:
    failures.append(
        "gross base-height collapse observed"
    )


if max(all_roll) >= 20.0:
    failures.append(
        "gross roll excursion observed"
    )


if max(all_pitch) >= 20.0:
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
            "[EVIDENCE] J7-selected VX command reached "
            "LL1 and produced a repeatable positive "
            "realized body-frame velocity response."
        ),
        (
            "[LIMIT] This establishes VX command "
            "authority only, not speed optimality, "
            "terrain benefit, stability benefit, "
            "energy benefit, or policy performance."
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
