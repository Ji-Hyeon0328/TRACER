#!/usr/bin/env python3

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np


KEY = (
    "context_id",
    "beta_name",
)


def as_bool(x):
    return str(x).strip().lower() in {
        "1",
        "true",
        "yes",
    }


def beta_entropy(beta):
    beta = np.asarray(
        beta,
        dtype=np.float64,
    )

    positive = beta[
        beta > 0.0
    ]

    if len(positive) == 0:
        return 0.0

    h = -float(
        np.sum(
            positive
            * np.log(
                positive
            )
        )
    )

    return (
        h
        / math.log(3.0)
    )


def dominant_component(beta):
    names = (
        "motion",
        "stability",
        "energy",
    )

    beta = np.asarray(
        beta,
        dtype=np.float64,
    )

    maximum = float(
        np.max(beta)
    )

    winners = [
        names[i]
        for i, value
        in enumerate(beta)
        if abs(
            float(value)
            - maximum
        ) <= 1e-9
    ]

    if len(winners) != 1:
        return "tie"

    return winners[0]


def read_csv(path):
    with path.open(
        newline=""
    ) as f:
        return list(
            csv.DictReader(f)
        )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--self-assessment",
        required=True,
    )

    ap.add_argument(
        "--inverse-exact",
        required=True,
    )

    ap.add_argument(
        "--out-dir",
        required=True,
    )

    args = ap.parse_args()

    sa_path = Path(
        args.self_assessment
    )

    inv_path = Path(
        args.inverse_exact
    )

    out_dir = Path(
        args.out_dir
    )

    for path in (
        sa_path,
        inv_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(
                path
            )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    sa_rows = read_csv(
        sa_path
    )

    inv_rows = read_csv(
        inv_path
    )

    if len(sa_rows) != 116:
        raise RuntimeError(
            "Expected 116 self-assessment rows; "
            f"got {len(sa_rows)}"
        )

    if len(inv_rows) != 116:
        raise RuntimeError(
            "Expected 116 inverse rows; "
            f"got {len(inv_rows)}"
        )

    sa_by_key = {}

    for row in sa_rows:
        key = tuple(
            row[k]
            for k in KEY
        )

        if key in sa_by_key:
            raise RuntimeError(
                f"Duplicate self-assessment key: {key}"
            )

        sa_by_key[
            key
        ] = row

    inv_by_key = {}

    for row in inv_rows:
        key = tuple(
            row[k]
            for k in KEY
        )

        if key in inv_by_key:
            raise RuntimeError(
                f"Duplicate inverse key: {key}"
            )

        inv_by_key[
            key
        ] = row

    if set(sa_by_key) != set(
        inv_by_key
    ):
        raise RuntimeError(
            "Self-assessment/inverse key mismatch."
        )

    expressible = []
    unsupported_good = []

    for key in sorted(
        sa_by_key
    ):
        sa = sa_by_key[
            key
        ]

        inv = inv_by_key[
            key
        ]

        feasible = as_bool(
            sa["feasible"]
        )

        pareto = as_bool(
            sa[
                "pareto_nondominated"
            ]
        )

        good = (
            feasible
            and pareto
        )

        strict_supported = (
            inv[
                "support_class"
            ]
            == "strict"
        )

        beta_hat = np.asarray(
            [
                float(
                    inv[
                        "beta_hat_motion"
                    ]
                ),
                float(
                    inv[
                        "beta_hat_stability"
                    ]
                ),
                float(
                    inv[
                        "beta_hat_energy"
                    ]
                ),
            ],
            dtype=np.float64,
        )

        beta_gen = np.asarray(
            [
                float(
                    inv[
                        "beta_gen_motion"
                    ]
                ),
                float(
                    inv[
                        "beta_gen_stability"
                    ]
                ),
                float(
                    inv[
                        "beta_gen_energy"
                    ]
                ),
            ],
            dtype=np.float64,
        )

        if not np.isclose(
            beta_hat.sum(),
            1.0,
            atol=1e-8,
        ):
            raise RuntimeError(
                f"{key}: invalid beta_hat "
                f"{beta_hat}"
            )

        if not np.isclose(
            beta_gen.sum(),
            1.0,
            atol=1e-8,
        ):
            raise RuntimeError(
                f"{key}: invalid beta_gen "
                f"{beta_gen}"
            )

        base = {
            "context_id":
                sa["context_id"],

            "context_index":
                int(
                    sa[
                        "context_index"
                    ]
                ),

            "group":
                sa["group"],

            "terrain":
                sa["terrain"],

            "seed":
                int(
                    sa["seed"]
                ),

            "source_beta_name":
                sa["beta_name"],

            "cost_motion":
                float(
                    sa["cost_motion"]
                ),

            "cost_stability":
                float(
                    sa["cost_stability"]
                ),

            "cost_energy":
                float(
                    sa["cost_energy"]
                ),

            "episode_energy_abs_j":
                float(
                    sa[
                        "episode_energy_abs_j"
                    ]
                ),

            "decision_time_s":
                float(
                    sa[
                        "decision_time_s"
                    ]
                ),

            "roll_rms_rad":
                float(
                    sa[
                        "roll_rms_rad"
                    ]
                ),

            "pitch_rms_rad":
                float(
                    sa[
                        "pitch_rms_rad"
                    ]
                ),

            "mean_applied_vx_mps":
                float(
                    sa[
                        "mean_applied_vx_mps"
                    ]
                ),

            "pareto_nondominated":
                pareto,

            "support_class":
                inv[
                    "support_class"
                ],

            "max_margin":
                float(
                    inv[
                        "max_margin"
                    ]
                ),

            "beta_hat_motion":
                float(
                    beta_hat[0]
                ),

            "beta_hat_stability":
                float(
                    beta_hat[1]
                ),

            "beta_hat_energy":
                float(
                    beta_hat[2]
                ),

            "beta_hat_dominant":
                dominant_component(
                    beta_hat
                ),

            "beta_hat_entropy":
                beta_entropy(
                    beta_hat
                ),

            "beta_hat_interior_005":
                bool(
                    np.min(
                        beta_hat
                    )
                    >= 0.05
                ),

            "beta_gen_motion":
                float(
                    beta_gen[0]
                ),

            "beta_gen_stability":
                float(
                    beta_gen[1]
                ),

            "beta_gen_energy":
                float(
                    beta_gen[2]
                ),

            "beta_hat_gen_l1":
                float(
                    np.sum(
                        np.abs(
                            beta_hat
                            - beta_gen
                        )
                    )
                ),

            "original_beta_regret":
                float(
                    inv[
                        "original_beta_regret"
                    ]
                ),

            "original_beta_consistent":
                as_bool(
                    inv[
                        "original_beta_consistent"
                    ]
                ),

            # There is intentionally no preferred_beta label.
            #
            # This record becomes selector supervision only
            # after external expert/mission selection evidence
            # identifies this trajectory as preferred.
            "selection_evidence_required":
                True,
        }

        if good and strict_supported:
            expressible.append(
                base
            )

        elif good:
            unsupported_good.append(
                base
            )

    if len(expressible) != 82:
        raise RuntimeError(
            "Expected 82 strict-supported good "
            f"candidates; got {len(expressible)}"
        )

    if len(unsupported_good) != 20:
        raise RuntimeError(
            "Expected 20 unsupported good candidates; "
            f"got {len(unsupported_good)}"
        )

    atlas_csv = (
        out_dir
        / "expressible_preference_atlas.csv"
    )

    with atlas_csv.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                expressible[0]
            ),
        )

        writer.writeheader()
        writer.writerows(
            expressible
        )

    unsupported_csv = (
        out_dir
        / "unsupported_good_candidates.csv"
    )

    with unsupported_csv.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                unsupported_good[0]
            ),
        )

        writer.writeheader()
        writer.writerows(
            unsupported_good
        )

    dominant_counts = Counter(
        row[
            "beta_hat_dominant"
        ]
        for row in expressible
    )

    source_counts = Counter(
        row[
            "source_beta_name"
        ]
        for row in expressible
    )

    group_counts = Counter(
        row[
            "group"
        ]
        for row in expressible
    )

    margins = np.asarray(
        [
            row[
                "max_margin"
            ]
            for row in expressible
        ],
        dtype=np.float64,
    )

    l1 = np.asarray(
        [
            row[
                "beta_hat_gen_l1"
            ]
            for row in expressible
        ],
        dtype=np.float64,
    )

    interior_count = sum(
        int(
            row[
                "beta_hat_interior_005"
            ]
        )
        for row in expressible
    )

    summary = {
        "schema":
            "icra27_phase2a_expressible_preference_atlas_v0",

        "self_assessment_source":
            str(sa_path),

        "inverse_exact_source":
            str(inv_path),

        "good_candidate_count":
            102,

        "expressible_good_candidate_count":
            len(
                expressible
            ),

        "unsupported_good_candidate_count":
            len(
                unsupported_good
            ),

        "expressible_fraction_of_good":
            float(
                len(expressible)
                / 102.0
            ),

        "source_beta_name_counts":
            dict(
                source_counts
            ),

        "beta_hat_dominant_counts":
            dict(
                dominant_counts
            ),

        "beta_hat_interior_005_count":
            interior_count,

        "max_margin_quantiles":
            {
                "min":
                    float(
                        np.min(
                            margins
                        )
                    ),

                "q25":
                    float(
                        np.quantile(
                            margins,
                            0.25,
                        )
                    ),

                "median":
                    float(
                        np.median(
                            margins
                        )
                    ),

                "q75":
                    float(
                        np.quantile(
                            margins,
                            0.75,
                        )
                    ),

                "max":
                    float(
                        np.max(
                            margins
                        )
                    ),
            },

        "beta_hat_vs_generation_l1":
            {
                "mean":
                    float(
                        np.mean(
                            l1
                        )
                    ),

                "median":
                    float(
                        np.median(
                            l1
                        )
                    ),

                "max":
                    float(
                        np.max(
                            l1
                        )
                    ),
            },

        "group_counts":
            dict(
                group_counts
            ),

        "training_status":
            (
                "NOT selector supervision yet: "
                "external expert/mission selection "
                "evidence is still required."
            ),
    }

    summary_path = (
        out_dir
        / "summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        )
        + "\n"
    )

    print(
        "=" * 80
    )

    print(
        "ICRA27 PHASE-2A EXPRESSIBLE PREFERENCE ATLAS V0"
    )

    print(
        "=" * 80
    )

    print(
        "good candidates        : 102"
    )

    print(
        "expressible good       :",
        len(
            expressible
        ),
    )

    print(
        "unsupported good       :",
        len(
            unsupported_good
        ),
    )

    print(
        "expressible fraction   :",
        f"{len(expressible) / 102.0:.3f}",
    )

    print()
    print(
        "SOURCE BETA COUNTS"
    )

    for key in (
        "balanced",
        "motion",
        "stability",
        "energy",
    ):
        print(
            f"  {key:<10}: "
            f"{source_counts[key]}"
        )

    print()
    print(
        "INFERRED BETA-HAT DOMINANT COMPONENT"
    )

    for key in (
        "motion",
        "stability",
        "energy",
        "tie",
    ):
        print(
            f"  {key:<10}: "
            f"{dominant_counts[key]}"
        )

    print()
    print(
        "beta_hat interior >=.05:",
        f"{interior_count}/{len(expressible)}",
    )

    print()
    print(
        "max-margin quantiles    :",
        summary[
            "max_margin_quantiles"
        ],
    )

    print(
        "beta-hat/gen L1         :",
        summary[
            "beta_hat_vs_generation_l1"
        ],
    )

    print()
    print(
        "atlas csv               :",
        atlas_csv,
    )

    print(
        "unsupported csv         :",
        unsupported_csv,
    )

    print(
        "summary                 :",
        summary_path,
    )

    print()
    print(
        "[ICRA27] Phase-2A expressible "
        "preference atlas: PASS"
    )


if __name__ == "__main__":
    main()
