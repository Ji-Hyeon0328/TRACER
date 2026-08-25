#!/usr/bin/env python3

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

TRACE_DIR = (
    ROOT
    / "results"
    / "icra27"
    / "os_t4p6b_u30_k11_planned_stance_shadow_v0"
)

BETAS = (
    "balanced",
    "stability",
    "energy",
)

LEGS = (
    "FL",
    "FR",
    "RL",
    "RR",
)


def load(beta):
    matches = list(
        TRACE_DIR.glob(
            f"u30_{beta}_*slip500.jsonl"
        )
    )

    if len(matches) != 1:
        raise RuntimeError(
            f"{beta}: expected one trace, "
            f"got {matches}"
        )

    rows = [
        json.loads(line)
        for line
        in matches[0].read_text().splitlines()
        if line.strip()
    ]

    if len(rows) != 100:
        raise RuntimeError(
            f"{beta}: expected 100 rows, "
            f"got {len(rows)}"
        )

    return rows


def aggregate(rows):
    sums = {
        "raw_num": 0.0,
        "raw_den": 0.0,

        "v3_num": 0.0,
        "v3_den": 0.0,

        "planned_num": 0.0,
        "planned_den": 0.0,

        "latched_num": 0.0,
        "latched_den": 0.0,
    }

    for row in rows:
        dt = float(
            row["dt_s"]
        )

        for leg in LEGS:
            foot = row[
                "feet"
            ][leg]

            if not bool(
                foot["contact"]
            ):
                continue

            cost = float(
                foot[
                    "traction_cost"
                ]
            )

            sums[
                "raw_num"
            ] += cost * dt

            sums[
                "raw_den"
            ] += dt

            if bool(
                foot["established"]
            ):
                sums[
                    "v3_num"
                ] += cost * dt

                sums[
                    "v3_den"
                ] += dt

            if bool(
                row[
                    "planned_stance_shadow"
                ][leg][
                    "eligible"
                ]
            ):
                sums[
                    "planned_num"
                ] += cost * dt

                sums[
                    "planned_den"
                ] += dt

            if bool(
                row[
                    "latched_stance_shadow"
                ][leg][
                    "eligible"
                ]
            ):
                sums[
                    "latched_num"
                ] += cost * dt

                sums[
                    "latched_den"
                ] += dt

    def mean(
        prefix,
    ):
        den = sums[
            prefix + "_den"
        ]

        return (
            sums[
                prefix + "_num"
            ] / den
            if den > 0.0
            else 0.0
        )

    return {
        "raw": mean("raw"),
        "v3": mean("v3"),
        "planned": mean(
            "planned"
        ),
        "latched": mean(
            "latched"
        ),

        **sums,
    }


def main():
    traces = {
        beta:
            load(beta)
        for beta in BETAS
    }

    result = {
        beta:
            aggregate(
                traces[beta]
            )
        for beta in BETAS
    }

    print("=" * 118)
    print(
        "ICRA27 OS-T4.6c u30 "
        "LATCHED-STANCE TRACTION SHADOW"
    )
    print("=" * 118)

    for beta in BETAS:
        r = result[beta]

        print(
            f"{beta:<9} "
            f"raw={r['raw']:.6f} "
            f"v3={r['v3']:.6f} "
            f"planned={r['planned']:.6f} "
            f"latched={r['latched']:.6f}"
        )

    def delta(key):
        return (
            result[
                "stability"
            ][key]
            - result[
                "energy"
            ][key]
        )

    d_raw = delta("raw")
    d_v3 = delta("v3")
    d_planned = delta(
        "planned"
    )
    d_latched = delta(
        "latched"
    )

    print()
    print(
        f"S-E dRaw    = {d_raw:+.6f}"
    )
    print(
        f"S-E dV3     = {d_v3:+.6f}"
    )
    print(
        f"S-E dPlanned= {d_planned:+.6f}"
    )
    print(
        f"S-E dLatched= {d_latched:+.6f}"
    )

    print()
    print("=" * 118)
    print(
        "RR MICRO-CONTACT REGION"
    )
    print("=" * 118)

    for beta in BETAS:
        print()
        print(beta.upper())

        for row in traces[beta]:
            t = float(
                row["time_post_s"]
            )

            if not (
                3.282
                <= t
                <= 3.304
            ):
                continue

            foot = (
                row[
                    "feet"
                ]["RR"]
            )

            p = (
                row[
                    "planned_stance_shadow"
                ]["RR"]
            )

            l = (
                row[
                    "latched_stance_shadow"
                ]["RR"]
            )

            speed = (
                "OFF"
                if not foot[
                    "contact"
                ]
                else (
                    f"{float(foot['slip_speed_mps']):.4f}"
                )
            )

            print(
                f"t={t:.3f} "
                f"actual={int(bool(foot['contact']))} "
                f"v3Age="
                f"{float(foot['contact_age_s']):.3f} "
                f"pAge={float(p['planned_stance_age_s']):.3f} "
                f"lAge={float(l['age_s']):.3f} "
                f"v3={int(bool(foot['established']))} "
                f"P={int(bool(p['eligible']))} "
                f"L={int(bool(l['eligible']))} "
                f"v={speed}"
            )

    print()
    print("=" * 118)

    if (
        d_raw < 0.0
        and d_v3 > 0.0
        and d_latched < 0.0
    ):
        print(
            "RESULT: LATCHED-STANCE CANDIDATE "
            "REMOVES THE v3 SIGN INVERSION"
        )
        print(
            "[ICRA27] OS-T4.6c "
            "latched-stance shadow: PASS"
        )

    else:
        print(
            "RESULT: LATCHED candidate "
            "requires further audit"
        )
        print(
            "[ICRA27] OS-T4.6c "
            "latched-stance shadow: "
            "REQUIRES FURTHER AUDIT"
        )

    print("=" * 118)


if __name__ == "__main__":
    main()
