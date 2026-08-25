#!/usr/bin/env python3

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

OUT = (
    ROOT
    / "results"
    / "icra27"
    / "os_t4p5k6f_u30_k11_perfoot_trace_v0"
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

TRACE_FILES = {
    beta:
        OUT
        / (
            f"u30_{beta}_"
            "low27200_k11_slip500.jsonl"
        )
    for beta in BETAS
}


def read_jsonl(path):
    rows = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            line = line.strip()

            if line:
                rows.append(
                    json.loads(line)
                )

    return rows


def new_stats():
    return {
        "contact_dt":
            0.0,

        "established_dt":
            0.0,

        "excluded_dt":
            0.0,

        "raw_cost_time":
            0.0,

        "established_cost_time":
            0.0,

        "excluded_cost_time":
            0.0,

        "raw_tail15_dt":
            0.0,

        "raw_tail30_dt":
            0.0,

        "est_tail15_dt":
            0.0,

        "est_tail30_dt":
            0.0,

        "excluded_tail15_dt":
            0.0,

        "excluded_tail30_dt":
            0.0,

        "contact_samples":
            0,

        "established_samples":
            0,

        "excluded_samples":
            0,
    }


def accumulate(stats, row, leg):
    foot = row["feet"][leg]

    if not bool(
        foot["contact"]
    ):
        return

    dt = float(
        row["dt_s"]
    )

    speed = float(
        foot["slip_speed_mps"]
    )

    cost = float(
        foot["traction_cost"]
    )

    established = bool(
        foot["established"]
    )

    stats["contact_dt"] += dt
    stats["raw_cost_time"] += (
        cost * dt
    )
    stats["contact_samples"] += 1

    if speed >= 0.15:
        stats["raw_tail15_dt"] += dt

    if speed >= 0.30:
        stats["raw_tail30_dt"] += dt

    if established:
        stats[
            "established_dt"
        ] += dt

        stats[
            "established_cost_time"
        ] += (
            cost * dt
        )

        stats[
            "established_samples"
        ] += 1

        if speed >= 0.15:
            stats[
                "est_tail15_dt"
            ] += dt

        if speed >= 0.30:
            stats[
                "est_tail30_dt"
            ] += dt

    else:
        stats[
            "excluded_dt"
        ] += dt

        stats[
            "excluded_cost_time"
        ] += (
            cost * dt
        )

        stats[
            "excluded_samples"
        ] += 1

        if speed >= 0.15:
            stats[
                "excluded_tail15_dt"
            ] += dt

        if speed >= 0.30:
            stats[
                "excluded_tail30_dt"
            ] += dt


def safe_mean(
    numerator,
    denominator,
):
    if denominator <= 0.0:
        return 0.0

    return (
        numerator
        / denominator
    )


def fmt_ms(value):
    return (
        1000.0 * value
    )


def main():
    traces = {
        beta:
            read_jsonl(
                TRACE_FILES[beta]
            )
        for beta in BETAS
    }

    for beta, rows in traces.items():
        if len(rows) != 100:
            raise RuntimeError(
                f"{beta}: expected 100 rows, "
                f"got {len(rows)}"
            )

    all_stats = {}

    print("=" * 118)
    print(
        "ICRA27 OS-T4.5k6g "
        "CONTACT-AGE GATE AUDIT"
    )
    print("=" * 118)
    print(
        "source : k6f u30/k11 "
        "500 Hz per-foot trace"
    )
    print(
        "scope  : OFFLINE ONLY; "
        "no simulation / no retraining"
    )
    print()

    for beta in BETAS:
        all_stats[beta] = {}

        print("=" * 118)
        print(beta.upper())
        print("=" * 118)

        for leg in LEGS:
            stats = new_stats()

            for row in traces[beta]:
                accumulate(
                    stats,
                    row,
                    leg,
                )

            all_stats[
                beta
            ][leg] = stats

            raw_mean = safe_mean(
                stats[
                    "raw_cost_time"
                ],
                stats[
                    "contact_dt"
                ],
            )

            est_mean = safe_mean(
                stats[
                    "established_cost_time"
                ],
                stats[
                    "established_dt"
                ],
            )

            excl_mean = safe_mean(
                stats[
                    "excluded_cost_time"
                ],
                stats[
                    "excluded_dt"
                ],
            )

            print(
                f"{leg} "
                f"contact={fmt_ms(stats['contact_dt']):6.1f}ms "
                f"est={fmt_ms(stats['established_dt']):6.1f}ms "
                f"excluded={fmt_ms(stats['excluded_dt']):6.1f}ms "
                f"rawCtr={raw_mean:.6f} "
                f"estCtr={est_mean:.6f} "
                f"excludedCtr={excl_mean:.6f}"
            )

            print(
                "   "
                f"raw >=.15/"
                f">=.30 = "
                f"{fmt_ms(stats['raw_tail15_dt']):5.1f}/"
                f"{fmt_ms(stats['raw_tail30_dt']):5.1f}ms   "
                f"est = "
                f"{fmt_ms(stats['est_tail15_dt']):5.1f}/"
                f"{fmt_ms(stats['est_tail30_dt']):5.1f}ms   "
                f"excluded = "
                f"{fmt_ms(stats['excluded_tail15_dt']):5.1f}/"
                f"{fmt_ms(stats['excluded_tail30_dt']):5.1f}ms"
            )

        print()

    print("=" * 118)
    print("AGGREGATE RAW vs ESTABLISHED TRACTION")
    print("=" * 118)

    aggregate = {}

    for beta in BETAS:
        raw_num = sum(
            all_stats[beta][leg][
                "raw_cost_time"
            ]
            for leg in LEGS
        )

        raw_den = sum(
            all_stats[beta][leg][
                "contact_dt"
            ]
            for leg in LEGS
        )

        est_num = sum(
            all_stats[beta][leg][
                "established_cost_time"
            ]
            for leg in LEGS
        )

        est_den = sum(
            all_stats[beta][leg][
                "established_dt"
            ]
            for leg in LEGS
        )

        excl_num = (
            raw_num
            - est_num
        )

        aggregate[beta] = {
            "raw":
                safe_mean(
                    raw_num,
                    raw_den,
                ),

            "est":
                safe_mean(
                    est_num,
                    est_den,
                ),

            "raw_num":
                raw_num,

            "est_num":
                est_num,

            "excluded_num":
                excl_num,

            "raw_den":
                raw_den,

            "est_den":
                est_den,
        }

        exclusion_share = (
            0.0
            if raw_num <= 0.0
            else excl_num / raw_num
        )

        print(
            f"{beta:<9} "
            f"rawCtr="
            f"{aggregate[beta]['raw']:.6f} "
            f"estCtr="
            f"{aggregate[beta]['est']:.6f} "
            f"contact="
            f"{1000*raw_den:.1f}ms "
            f"est_contact="
            f"{1000*est_den:.1f}ms "
            f"cost-time-excluded="
            f"{100*exclusion_share:.2f}%"
        )

    print()

    for lhs, rhs in (
        ("stability", "energy"),
        ("stability", "balanced"),
        ("balanced", "energy"),
    ):
        d_raw = (
            aggregate[lhs]["raw"]
            - aggregate[rhs]["raw"]
        )

        d_est = (
            aggregate[lhs]["est"]
            - aggregate[rhs]["est"]
        )

        print(
            f"{lhs.upper()} - "
            f"{rhs.upper()}: "
            f"dRaw={d_raw:+.6f} "
            f"dEstablished={d_est:+.6f} "
            f"gate_shift="
            f"{(d_est-d_raw):+.6f}"
        )

    print()
    print("=" * 118)
    print(
        "RR CONTACT / GATE TIMELINE "
        "3.270--3.305 s"
    )
    print("=" * 118)

    by_step = {
        beta: {
            int(row[
                "physics_step_num_post"
            ]): row
            for row in traces[beta]
        }
        for beta in BETAS
    }

    common = sorted(
        set(
            by_step["balanced"]
        )
        & set(
            by_step["stability"]
        )
        & set(
            by_step["energy"]
        )
    )

    def foot_text(row):
        foot = row["feet"]["RR"]

        if not foot["contact"]:
            return (
                "OFF"
            )

        return (
            "ON "
            f"age={float(foot['contact_age_s']):.3f} "
            f"est={int(bool(foot['established']))} "
            f"v={float(foot['slip_speed_mps']):.4f} "
            f"C={float(foot['traction_cost']):.4f}"
        )

    for step in common:
        row_b = by_step[
            "balanced"
        ][step]

        t = float(
            row_b["time_post_s"]
        )

        if not (
            3.270
            <= t
            <= 3.305
        ):
            continue

        print(
            f"t={t:.3f} "
            f"tick={step}"
        )

        print(
            "   B:",
            foot_text(
                by_step[
                    "balanced"
                ][step]
            )
        )

        print(
            "   S:",
            foot_text(
                by_step[
                    "stability"
                ][step]
            )
        )

        print(
            "   E:",
            foot_text(
                by_step[
                    "energy"
                ][step]
            )
        )

    print()
    print("=" * 118)
    print(
        "[ICRA27] OS-T4.5k6g "
        "contact-age gate audit: COMPUTE PASS"
    )
    print("=" * 118)


if __name__ == "__main__":
    main()
