#!/usr/bin/env python3

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


METRICS = (
    ("decision_time_s", "time"),
    ("cost_motion", "Cm"),
    ("cost_stability", "Cs"),
    ("cost_energy", "CE"),
    ("episode_energy_abs_j", "Eabs"),
)


def read_csv(path):
    with path.open(
        newline=""
    ) as f:
        return list(
            csv.DictReader(f)
        )


def describe(values):
    x = np.asarray(
        values,
        dtype=np.float64,
    )

    return {
        "min":
            float(np.min(x)),
        "q10":
            float(np.quantile(x, 0.10)),
        "q25":
            float(np.quantile(x, 0.25)),
        "median":
            float(np.median(x)),
        "q75":
            float(np.quantile(x, 0.75)),
        "q90":
            float(np.quantile(x, 0.90)),
        "max":
            float(np.max(x)),
    }


def print_stats(name, stats):
    print(
        f"{name:<6} "
        f"min={stats['min']:.5f} "
        f"q10={stats['q10']:.5f} "
        f"q25={stats['q25']:.5f} "
        f"med={stats['median']:.5f} "
        f"q75={stats['q75']:.5f} "
        f"q90={stats['q90']:.5f} "
        f"max={stats['max']:.5f}"
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--atlas",
        required=True,
    )

    args = ap.parse_args()

    p = Path(
        args.atlas
    )

    if not p.is_file():
        raise FileNotFoundError(
            p
        )

    rows = read_csv(
        p
    )

    if len(rows) != 82:
        raise RuntimeError(
            f"Expected 82 expressible rows, got {len(rows)}"
        )

    groups = defaultdict(
        list
    )

    terrains = defaultdict(
        list
    )

    for row in rows:
        groups[
            row["group"]
        ].append(
            row
        )

        terrains[
            row["terrain"]
        ].append(
            row
        )

    print(
        "=" * 92
    )
    print(
        "ICRA27 PHASE-2A MISSION-SPACE AUDIT V0"
    )
    print(
        "=" * 92
    )

    print()
    print(
        "ALL EXPRESSIBLE GOOD CANDIDATES"
    )

    for key, short in METRICS:
        print_stats(
            short,
            describe(
                [
                    float(row[key])
                    for row in rows
                ]
            ),
        )

    print()
    print(
        "=" * 92
    )
    print(
        "BY TERRAIN"
    )
    print(
        "=" * 92
    )

    for terrain in sorted(
        terrains
    ):
        subset = terrains[
            terrain
        ]

        print()
        print(
            f"{terrain} "
            f"(n={len(subset)})"
        )

        for key, short in METRICS:
            print_stats(
                short,
                describe(
                    [
                        float(row[key])
                        for row in subset
                    ]
                ),
            )

    print()
    print(
        "=" * 92
    )
    print(
        "BY EVALUATION GROUP"
    )
    print(
        "=" * 92
    )

    for group in sorted(
        groups
    ):
        subset = groups[
            group
        ]

        print()
        print(
            f"{group} "
            f"(n={len(subset)})"
        )

        for key, short in METRICS:
            print_stats(
                short,
                describe(
                    [
                        float(row[key])
                        for row in subset
                    ]
                ),
            )

    # --------------------------------------------------------
    # Counterfactual spread within each exact context.
    #
    # This is especially important:
    # if the four alternatives have almost identical values
    # for a metric, a mission threshold on that metric cannot
    # meaningfully change selection.
    # --------------------------------------------------------

    contexts = defaultdict(
        list
    )

    for row in rows:
        contexts[
            row["context_id"]
        ].append(
            row
        )

    print()
    print(
        "=" * 92
    )
    print(
        "WITHIN-CONTEXT EXPRESSIBLE SPREAD"
    )
    print(
        "=" * 92
    )

    for key, short in METRICS:
        spreads = []

        usable_contexts = 0

        for candidates in contexts.values():
            if len(candidates) < 2:
                continue

            x = np.asarray(
                [
                    float(row[key])
                    for row in candidates
                ],
                dtype=np.float64,
            )

            spreads.append(
                float(
                    np.max(x)
                    - np.min(x)
                )
            )

            usable_contexts += 1

        stats = describe(
            spreads
        )

        print(
            f"{short:<6} "
            f"contexts={usable_contexts:>2} "
            f"spread_med={stats['median']:.6f} "
            f"spread_q75={stats['q75']:.6f} "
            f"spread_max={stats['max']:.6f}"
        )

    print()
    print(
        "[ICRA27] Phase-2A mission-space audit: PASS"
    )


if __name__ == "__main__":
    main()
