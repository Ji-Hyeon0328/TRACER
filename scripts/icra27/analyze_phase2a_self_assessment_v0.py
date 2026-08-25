#!/usr/bin/env python3

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


BETA_NAMES = (
    "balanced",
    "motion",
    "stability",
    "energy",
)


def as_bool(x):
    return str(x).strip().lower() in {
        "1",
        "true",
        "yes",
    }


def as_int_or_none(x):
    x = str(x).strip()

    if not x:
        return None

    return int(x)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--candidates",
        required=True,
    )

    args = ap.parse_args()

    p = Path(
        args.candidates
    )

    if not p.is_file():
        raise FileNotFoundError(
            p
        )

    with p.open(
        newline=""
    ) as f:
        rows = list(
            csv.DictReader(f)
        )

    if len(rows) != 116:
        raise RuntimeError(
            f"Expected 116 rows, got {len(rows)}"
        )

    contexts = defaultdict(
        list
    )

    for row in rows:
        contexts[
            row["context_id"]
        ].append(
            row
        )

    if len(contexts) != 29:
        raise RuntimeError(
            f"Expected 29 contexts, got {len(contexts)}"
        )

    # --------------------------------------------------------
    # Global Pareto membership by beta.
    # --------------------------------------------------------

    pareto_by_beta = Counter()
    dominated_by_beta = Counter()

    for row in rows:
        beta = row[
            "beta_name"
        ]

        if as_bool(
            row["pareto_nondominated"]
        ):
            pareto_by_beta[
                beta
            ] += 1
        else:
            dominated_by_beta[
                beta
            ] += 1

    # --------------------------------------------------------
    # Pareto membership grouped by evaluation group.
    # --------------------------------------------------------

    group_beta_total = Counter()
    group_beta_pareto = Counter()

    for row in rows:
        key = (
            row["group"],
            row["beta_name"],
        )

        group_beta_total[
            key
        ] += 1

        if as_bool(
            row["pareto_nondominated"]
        ):
            group_beta_pareto[
                key
            ] += 1

    # --------------------------------------------------------
    # Rank-1 counts.
    #
    # These answer:
    #   Which beta actually gives the lowest Cm?
    #   Which beta actually gives the lowest Cs?
    #   Which beta actually gives the lowest CE?
    # --------------------------------------------------------

    rank1 = {
        "motion":
            Counter(),

        "stability":
            Counter(),

        "energy":
            Counter(),
    }

    for row in rows:
        beta = row[
            "beta_name"
        ]

        if (
            as_int_or_none(
                row["motion_rank"]
            )
            == 1
        ):
            rank1[
                "motion"
            ][beta] += 1

        if (
            as_int_or_none(
                row["stability_rank"]
            )
            == 1
        ):
            rank1[
                "stability"
            ][beta] += 1

        if (
            as_int_or_none(
                row["energy_rank"]
            )
            == 1
        ):
            rank1[
                "energy"
            ][beta] += 1

    # --------------------------------------------------------
    # List contexts where at least one beta is dominated.
    # --------------------------------------------------------

    partially_dominated = []

    for context_id, candidates in contexts.items():
        pareto = [
            row["beta_name"]
            for row in candidates
            if as_bool(
                row[
                    "pareto_nondominated"
                ]
            )
        ]

        dominated = [
            row["beta_name"]
            for row in candidates
            if not as_bool(
                row[
                    "pareto_nondominated"
                ]
            )
        ]

        if dominated:
            first = candidates[0]

            partially_dominated.append(
                {
                    "context_id":
                        context_id,

                    "group":
                        first["group"],

                    "terrain":
                        first["terrain"],

                    "seed":
                        first["seed"],

                    "pareto":
                        pareto,

                    "dominated":
                        dominated,
                }
            )

    # --------------------------------------------------------
    # Output.
    # --------------------------------------------------------

    print(
        "=" * 80
    )

    print(
        "ICRA27 PHASE-2A SELF-ASSESSMENT ANALYSIS V0"
    )

    print(
        "=" * 80
    )

    print()
    print(
        "GLOBAL PARETO MEMBERSHIP"
    )

    for beta in BETA_NAMES:
        pcount = pareto_by_beta[
            beta
        ]

        dcount = dominated_by_beta[
            beta
        ]

        print(
            f"{beta:<10} "
            f"Pareto={pcount:>2}/29 "
            f"dominated={dcount:>2}/29"
        )

    print()
    print(
        "PARETO MEMBERSHIP BY GROUP"
    )

    groups = sorted(
        {
            row["group"]
            for row in rows
        }
    )

    for group in groups:
        print()
        print(
            group
        )

        for beta in BETA_NAMES:
            total = group_beta_total[
                (
                    group,
                    beta,
                )
            ]

            pareto = group_beta_pareto[
                (
                    group,
                    beta,
                )
            ]

            print(
                f"  {beta:<10} "
                f"{pareto}/{total}"
            )

    print()
    print(
        "OBJECTIVE RANK-1 COUNTS"
    )

    for objective in (
        "motion",
        "stability",
        "energy",
    ):
        print()
        print(
            f"{objective}:"
        )

        for beta in BETA_NAMES:
            print(
                f"  {beta:<10} "
                f"{rank1[objective][beta]}/29"
            )

    print()
    print(
        "CONTEXTS WITH DOMINATED ANCHORS"
    )

    print(
        f"count: {len(partially_dominated)}/29"
    )

    for item in partially_dominated:
        print(
            f"{item['group']:<18} "
            f"seed={item['seed']:<6} "
            f"Pareto={','.join(item['pareto'])} "
            f"dominated={','.join(item['dominated'])}"
        )

    print()
    print(
        "[ICRA27] Phase-2A self-assessment "
        "analysis: PASS"
    )


if __name__ == "__main__":
    main()
