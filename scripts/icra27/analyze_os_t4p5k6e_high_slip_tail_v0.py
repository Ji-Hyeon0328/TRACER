#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

K6D_ROOT = (
    ROOT
    / "results"
    / "icra27"
    / "os_t4p5k6d_interval_slip_distribution_v0"
)

SUMMARY_CSV = (
    K6D_ROOT
    / "interval_summary.csv"
)

HIST_CSV = (
    K6D_ROOT
    / "established_histogram_interval.csv"
)

OUT_DIR = (
    ROOT
    / "results"
    / "icra27"
    / "os_t4p5k6e_high_slip_tail_v0"
)

UPDATES = (
    10,
    20,
    30,
    40,
)

BETAS = (
    "balanced",
    "stability",
    "energy",
)

THRESHOLDS = (
    0.15,
    0.30,
    0.50,
    1.00,
)

EARLY_LAST_STEP = 2


def read_csv(path):
    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:
        return list(
            csv.DictReader(f)
        )


def as_int(row, key):
    return int(
        float(row[key])
    )


def as_float(row, key):
    return float(
        row[key]
    )


def mean(values):
    values = list(values)

    if not values:
        return float("nan")

    return (
        sum(values)
        / len(values)
    )


def corr(x, y):
    pairs = [
        (float(a), float(b))
        for a, b in zip(x, y)
        if (
            math.isfinite(float(a))
            and math.isfinite(float(b))
        )
    ]

    if len(pairs) < 3:
        return float("nan")

    xs = [
        a for a, _ in pairs
    ]

    ys = [
        b for _, b in pairs
    ]

    mx = mean(xs)
    my = mean(ys)

    dx = [
        x - mx for x in xs
    ]

    dy = [
        y - my for y in ys
    ]

    sx = math.sqrt(
        sum(v * v for v in dx)
    )

    sy = math.sqrt(
        sum(v * v for v in dy)
    )

    if (
        sx <= 1e-15
        or sy <= 1e-15
    ):
        return float("nan")

    return (
        sum(
            a * b
            for a, b
            in zip(dx, dy)
        )
        / (
            sx * sy
        )
    )


def threshold_key(
    threshold,
):
    return (
        "tail_ge_"
        + str(threshold).replace(
            ".",
            "p",
        )
        + "_s"
    )


def fraction_key(
    threshold,
):
    return (
        "tail_frac_ge_"
        + str(threshold).replace(
            ".",
            "p",
        )
    )


def main():
    for path in (
        SUMMARY_CSV,
        HIST_CSV,
    ):
        if not path.exists():
            raise FileNotFoundError(
                path
            )

    if OUT_DIR.exists():
        raise RuntimeError(
            "Output directory already exists; "
            "refusing overwrite: "
            f"{OUT_DIR}"
        )

    OUT_DIR.mkdir(
        parents=True
    )

    summary_rows = read_csv(
        SUMMARY_CSV
    )

    hist_rows = read_csv(
        HIST_CSV
    )

    summary_by_key = {}

    for row in summary_rows:
        key = (
            as_int(
                row,
                "update",
            ),
            row["beta"],
            as_int(
                row,
                "policy_step",
            ),
        )

        summary_by_key[key] = row

    hist_by_key = {}

    for row in hist_rows:
        key = (
            as_int(
                row,
                "update",
            ),
            row["beta"],
            as_int(
                row,
                "policy_step",
            ),
        )

        hist_by_key.setdefault(
            key,
            [],
        ).append(
            row
        )

    interval_rows = []

    print("=" * 116)
    print(
        "ICRA27 OS-T4.5k6e "
        "ESTABLISHED HIGH-SLIP TAIL AUDIT"
    )
    print("=" * 116)

    print(
        "NOTE: histogram metadata time_fraction "
        "is cumulative and is NOT used here."
    )

    print(
        "All tail fractions below are reconstructed "
        "from interval contact_time_s / "
        "interval established_contact_time_s."
    )

    print()

    for key, summary in sorted(
        summary_by_key.items()
    ):
        update, beta, step = key

        hists = hist_by_key.get(
            key,
            [],
        )

        est_dt = as_float(
            summary,
            "established_contact_time_s",
        )

        hist_dt = sum(
            as_float(
                row,
                "contact_time_s",
            )
            for row in hists
        )

        if abs(
            hist_dt - est_dt
        ) > 1e-9:
            raise RuntimeError(
                f"Histogram duration mismatch "
                f"{key}: "
                f"hist={hist_dt} est={est_dt}"
            )

        out = {
            "update":
                update,

            "beta":
                beta,

            "policy_step":
                step,

            "cost_traction":
                as_float(
                    summary,
                    "reward_cost_traction",
                ),

            "established_contact_time_s":
                est_dt,
        }

        for threshold in THRESHOLDS:
            tail_dt = 0.0

            for row in hists:
                metadata = json.loads(
                    row[
                        "metadata_json"
                    ]
                )

                low = metadata.get(
                    "low_mps"
                )

                if low is None:
                    continue

                # Chosen thresholds coincide exactly
                # with histogram bin boundaries.
                if float(low) >= (
                    threshold - 1e-12
                ):
                    tail_dt += as_float(
                        row,
                        "contact_time_s",
                    )

            out[
                threshold_key(
                    threshold
                )
            ] = tail_dt

            out[
                fraction_key(
                    threshold
                )
            ] = (
                0.0
                if est_dt <= 0.0
                else tail_dt
                / est_dt
            )

        interval_rows.append(
            out
        )

    # --------------------------------------------------------
    # Pair each specialized beta with Balanced at same
    # update/policy step.
    # --------------------------------------------------------

    indexed = {
        (
            row["update"],
            row["beta"],
            row["policy_step"],
        ):
            row
        for row in interval_rows
    }

    pair_rows = []

    for update in UPDATES:
        for specialized in (
            "stability",
            "energy",
        ):
            steps = sorted(
                step
                for (
                    u,
                    beta,
                    step,
                )
                in indexed
                if (
                    u == update
                    and beta == specialized
                )
            )

            for step in steps:
                b_key = (
                    update,
                    "balanced",
                    step,
                )

                x_key = (
                    update,
                    specialized,
                    step,
                )

                if b_key not in indexed:
                    continue

                b = indexed[b_key]
                x = indexed[x_key]

                row = {
                    "update":
                        update,

                    "specialized_beta":
                        specialized,

                    "policy_step":
                        step,

                    "delta_cost_traction":
                        (
                            x[
                                "cost_traction"
                            ]
                            - b[
                                "cost_traction"
                            ]
                        ),
                }

                for threshold in THRESHOLDS:
                    tk = threshold_key(
                        threshold
                    )

                    fk = fraction_key(
                        threshold
                    )

                    tag = str(
                        threshold
                    ).replace(
                        ".",
                        "p",
                    )

                    row[
                        f"delta_tail_ge_{tag}_s"
                    ] = (
                        x[tk] - b[tk]
                    )

                    row[
                        f"delta_tail_frac_ge_{tag}"
                    ] = (
                        x[fk] - b[fk]
                    )

                pair_rows.append(
                    row
                )

    def write_csv(
        path,
        rows,
    ):
        if not rows:
            raise RuntimeError(
                f"No rows for {path}"
            )

        fields = list(
            rows[0].keys()
        )

        with path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=fields,
            )

            writer.writeheader()
            writer.writerows(
                rows
            )

    write_csv(
        OUT_DIR
        / "interval_tail_burden.csv",
        interval_rows,
    )

    write_csv(
        OUT_DIR
        / "pairwise_tail_deltas.csv",
        pair_rows,
    )

    # --------------------------------------------------------
    # Aggregate diagnostic.
    # --------------------------------------------------------

    print("=" * 116)
    print(
        "SPECIALIZED vs BALANCED"
    )
    print("=" * 116)

    for update in UPDATES:
        for beta in (
            "stability",
            "energy",
        ):
            rows = [
                row
                for row in pair_rows
                if (
                    row[
                        "update"
                    ]
                    == update
                    and row[
                        "specialized_beta"
                    ]
                    == beta
                )
            ]

            early = [
                row
                for row in rows
                if row[
                    "policy_step"
                ] <= EARLY_LAST_STEP
            ]

            post = [
                row
                for row in rows
                if row[
                    "policy_step"
                ] > EARLY_LAST_STEP
            ]

            print()
            print(
                f"u{update:02d} "
                f"{beta.upper()}"
            )

            print(
                "  dCtr "
                f"all={mean(r['delta_cost_traction'] for r in rows):+.6f} "
                f"early={mean(r['delta_cost_traction'] for r in early):+.6f} "
                f"post={mean(r['delta_cost_traction'] for r in post):+.6f}"
            )

            for threshold in THRESHOLDS:
                tag = str(
                    threshold
                ).replace(
                    ".",
                    "p",
                )

                delta_key = (
                    f"delta_tail_ge_{tag}_s"
                )

                frac_key = (
                    f"delta_tail_frac_ge_{tag}"
                )

                print(
                    f"  >= {threshold:>4.2f} m/s "
                    f"dTail post="
                    f"{1000.0 * mean(r[delta_key] for r in post):+.3f} ms "
                    f"dFrac post="
                    f"{mean(r[frac_key] for r in post):+.5f} "
                    f"corr(dCtr,dFrac)="
                    f"{corr(
                        [r['delta_cost_traction'] for r in post],
                        [r[frac_key] for r in post],
                    ):+.3f}"
                )

    # --------------------------------------------------------
    # Rank largest post-launch Stability tail events.
    # --------------------------------------------------------

    stability_post = [
        row
        for row in pair_rows
        if (
            row[
                "specialized_beta"
            ]
            == "stability"
            and row[
                "policy_step"
            ] > EARLY_LAST_STEP
        )
    ]

    stability_post.sort(
        key=lambda row: (
            row[
                "delta_cost_traction"
            ]
        ),
        reverse=True,
    )

    print()
    print("=" * 116)
    print(
        "TOP POST-LAUNCH STABILITY "
        "TRACTION EVENTS"
    )
    print("=" * 116)

    for row in stability_post[:15]:
        print(
            f"u{row['update']:02d} "
            f"k={row['policy_step']:>2} "
            f"dCtr="
            f"{row['delta_cost_traction']:+.6f} "
            f"dTail>=.15="
            f"{1000.0 * row['delta_tail_ge_0p15_s']:+.1f}ms "
            f">=.30="
            f"{1000.0 * row['delta_tail_ge_0p3_s']:+.1f}ms "
            f">=.50="
            f"{1000.0 * row['delta_tail_ge_0p5_s']:+.1f}ms "
            f">=1.0="
            f"{1000.0 * row['delta_tail_ge_1p0_s']:+.1f}ms"
        )

    print()
    print(
        "[ICRA27] OS-T4.5k6e "
        "established high-slip tail: COMPUTE PASS"
    )


if __name__ == "__main__":
    main()
