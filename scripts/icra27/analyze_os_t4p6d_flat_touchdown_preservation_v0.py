#!/usr/bin/env python3

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

TRACE_DIR = (
    ROOT
    / "results"
    / "icra27"
    / "os_t4p6d_u30_flat_touchdown_shadow_v0"
)

BETAS = (
    "balanced",
    "stability",
    "energy",
)

LEG = "RR"

GATE_S = 0.040
DT_TOL = 1e-9


def load(beta):
    matches = list(
        TRACE_DIR.glob(
            f"u30_{beta}_*slip500.jsonl"
        )
    )

    if len(matches) != 1:
        raise RuntimeError(
            f"{beta}: expected one trace, got {matches}"
        )

    rows = [
        json.loads(line)
        for line in matches[0].read_text().splitlines()
        if line.strip()
    ]

    if len(rows) != 100:
        raise RuntimeError(
            f"{beta}: expected 100 rows, "
            f"got {len(rows)}"
        )

    return rows


def actual(row):
    return bool(
        row["feet"][LEG]["contact"]
    )


def v3_eligible(row):
    return bool(
        row["feet"][LEG]["established"]
    )


def latched_eligible(row):
    return bool(
        row["latched_stance_shadow"][
            LEG
        ]["eligible"]
    )


def v3_age(row):
    return float(
        row["feet"][LEG]["contact_age_s"]
    )


def latched_age(row):
    return float(
        row["latched_stance_shadow"][
            LEG
        ]["age_s"]
    )


def slip(row):
    if not actual(row):
        return None

    return float(
        row["feet"][LEG]["slip_speed_mps"]
    )


def analyze(beta, rows):
    # --------------------------------------------------------
    # Find a physical touchdown inside the trace.
    # Prefer an OFF->ON transition. If the trace begins during
    # contact, find the next such transition.
    # --------------------------------------------------------
    touchdown_idx = None

    previous = actual(rows[0])

    for i in range(1, len(rows)):
        now = actual(rows[i])

        if (
            not previous
            and now
        ):
            touchdown_idx = i
            break

        previous = now

    if touchdown_idx is None:
        raise RuntimeError(
            f"{beta}: no RR OFF->ON touchdown "
            "inside trace window"
        )

    # Continuous physical-contact segment after touchdown.
    segment = []

    for row in rows[
        touchdown_idx:
    ]:
        if not actual(row):
            break

        segment.append(row)

    if not segment:
        raise RuntimeError(
            f"{beta}: empty touchdown segment"
        )

    first = segment[0]

    # --------------------------------------------------------
    # During clean continuous contact, the latched candidate
    # must be semantically identical to v3.
    # --------------------------------------------------------
    max_age_error = max(
        abs(
            v3_age(row)
            - latched_age(row)
        )
        for row in segment
    )

    eligibility_mismatches = [
        row
        for row in segment
        if (
            v3_eligible(row)
            != latched_eligible(row)
        )
    ]

    pre_gate = [
        row
        for row in segment
        if (
            latched_age(row)
            < GATE_S - DT_TOL
        )
    ]

    pre_gate_bad = [
        row
        for row in pre_gate
        if (
            v3_eligible(row)
            or latched_eligible(row)
        )
    ]

    v3_first = next(
        (
            row
            for row in segment
            if v3_eligible(row)
        ),
        None,
    )

    latched_first = next(
        (
            row
            for row in segment
            if latched_eligible(row)
        ),
        None,
    )

    if v3_first is None:
        raise RuntimeError(
            f"{beta}: v3 never becomes established "
            "before first contact break"
        )

    if latched_first is None:
        raise RuntimeError(
            f"{beta}: latched never becomes eligible "
            "before first contact break"
        )

    max_pre_gate_slip = max(
        (
            slip(row)
            for row in pre_gate
            if slip(row) is not None
        ),
        default=0.0,
    )

    return {
        "touchdown_time":
            float(first["time_post_s"]),

        "first_v3_time":
            float(v3_first["time_post_s"]),

        "first_latched_time":
            float(latched_first["time_post_s"]),

        "first_v3_age":
            v3_age(v3_first),

        "first_latched_age":
            latched_age(latched_first),

        "segment_ms":
            1000.0
            * sum(
                float(row["dt_s"])
                for row in segment
            ),

        "max_age_error":
            max_age_error,

        "eligibility_mismatches":
            len(
                eligibility_mismatches
            ),

        "pre_gate_bad":
            len(pre_gate_bad),

        "max_pre_gate_slip":
            max_pre_gate_slip,

        "segment":
            segment,
    }


def main():
    print("=" * 120)
    print(
        "ICRA27 OS-T4.6d "
        "NOMINAL-FLAT TOUCHDOWN PRESERVATION"
    )
    print("=" * 120)

    all_pass = True

    for beta in BETAS:
        rows = load(beta)

        result = analyze(
            beta,
            rows,
        )

        print()
        print(beta.upper())

        print(
            f"touchdown       : "
            f"{result['touchdown_time']:.3f} s"
        )

        print(
            f"continuous seg  : "
            f"{result['segment_ms']:.1f} ms"
        )

        print(
            f"first eligible  : "
            f"v3={result['first_v3_time']:.3f}s "
            f"(age={result['first_v3_age']:.3f}) "
            f"latched={result['first_latched_time']:.3f}s "
            f"(age={result['first_latched_age']:.3f})"
        )

        print(
            f"max age error   : "
            f"{result['max_age_error']:.3e} s"
        )

        print(
            f"elig mismatch   : "
            f"{result['eligibility_mismatches']}"
        )

        print(
            f"pre-gate leak   : "
            f"{result['pre_gate_bad']}"
        )

        print(
            f"max slip <40ms  : "
            f"{result['max_pre_gate_slip']:.6f} m/s"
        )

        beta_pass = (
            result[
                "max_age_error"
            ] <= 1e-12
            and result[
                "eligibility_mismatches"
            ] == 0
            and result[
                "pre_gate_bad"
            ] == 0
            and abs(
                result[
                    "first_v3_time"
                ]
                - result[
                    "first_latched_time"
                ]
            ) <= 1e-12
        )

        print(
            "verdict         :",
            "PASS"
            if beta_pass
            else "FAIL",
        )

        all_pass = (
            all_pass
            and beta_pass
        )

        print(
            "  first-contact timeline:"
        )

        for row in result[
            "segment"
        ][:25]:
            age = v3_age(row)

            if age > 0.055:
                break

            print(
                f"    t="
                f"{float(row['time_post_s']):.3f} "
                f"v3Age={v3_age(row):.3f} "
                f"lAge={latched_age(row):.3f} "
                f"v3={int(v3_eligible(row))} "
                f"L={int(latched_eligible(row))} "
                f"slip={slip(row):.5f}"
            )

    print()
    print("=" * 120)

    if all_pass:
        print(
            "RESULT: LATCHED CANDIDATE PRESERVES "
            "THE ORIGINAL NOMINAL-FLAT "
            "TOUCHDOWN GATE"
        )

        print(
            "[ICRA27] OS-T4.6d "
            "flat touchdown preservation: PASS"
        )

    else:
        print(
            "RESULT: touchdown semantics differ; "
            "do NOT promote candidate."
        )

        print(
            "[ICRA27] OS-T4.6d "
            "flat touchdown preservation: "
            "REQUIRES FURTHER AUDIT"
        )

    print("=" * 120)


if __name__ == "__main__":
    main()
