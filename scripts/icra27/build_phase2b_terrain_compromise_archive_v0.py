#!/usr/bin/env python3

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


COSTS = (
    "cost_motion",
    "cost_stability",
    "cost_energy",
)

BETA_ORDER = (
    "balanced",
    "motion",
    "stability",
    "energy",
)

TERRAINS = (
    "flat",
    "low_friction",
    "rough_perlin",
)

TOL = 1e-12


def as_bool(x):
    return str(x).strip().lower() in {
        "1",
        "true",
        "yes",
    }


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--self-assessment",
        required=True,
    )

    ap.add_argument(
        "--out-dir",
        required=True,
    )

    args = ap.parse_args()

    src = Path(
        args.self_assessment
    )

    out_dir = Path(
        args.out_dir
    )

    if not src.is_file():
        raise FileNotFoundError(src)

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = read_csv(src)

    if not rows:
        raise RuntimeError(
            "Self-assessment archive is empty."
        )

    # ========================================================
    # Keep only physically feasible Pareto candidates.
    #
    # This is task-independent.
    # ========================================================

    good = [
        row
        for row in rows
        if (
            as_bool(row["feasible"])
            and as_bool(
                row[
                    "pareto_nondominated"
                ]
            )
        )
    ]

    if not good:
        raise RuntimeError(
            "No feasible Pareto candidates."
        )

    # ========================================================
    # TRAIN-global normalization.
    #
    # We deliberately do NOT use context-local ranges:
    # tiny differences on easy terrain (especially Cs)
    # should not be amplified into a full [0,1] range.
    # ========================================================

    global_min = {}
    global_max = {}
    global_span = {}

    for cost in COSTS:
        values = np.asarray(
            [
                float(row[cost])
                for row in good
            ],
            dtype=float,
        )

        global_min[cost] = float(
            np.min(values)
        )

        global_max[cost] = float(
            np.max(values)
        )

        span = (
            global_max[cost]
            - global_min[cost]
        )

        if span <= TOL:
            raise RuntimeError(
                f"Degenerate TRAIN-global span "
                f"for {cost}: {span}"
            )

        global_span[cost] = float(
            span
        )

    # ========================================================
    # Context archive.
    # ========================================================

    by_context = defaultdict(list)

    for row in good:
        by_context[
            row["context_id"]
        ].append(row)

    if len(by_context) != 54:
        raise RuntimeError(
            "Expected 54 TRAIN contexts, got "
            f"{len(by_context)}"
        )

    candidate_records = []
    context_records = []

    for context_id in sorted(
        by_context
    ):
        candidates = by_context[
            context_id
        ]

        if len(candidates) not in {
            3,
            4,
        }:
            raise RuntimeError(
                f"{context_id}: expected 3 or 4 "
                f"Pareto candidates, got "
                f"{len(candidates)}"
            )

        first = candidates[0]

        terrain = first["terrain"]

        if terrain not in TERRAINS:
            raise RuntimeError(
                f"Unknown terrain: {terrain!r}"
            )

        for row in candidates[1:]:
            if row["terrain"] != terrain:
                raise RuntimeError(
                    f"{context_id}: terrain mismatch"
                )

        # ----------------------------------------------------
        # Context-specific ideal point.
        # ----------------------------------------------------

        ideal = {
            cost:
                min(
                    float(row[cost])
                    for row in candidates
                )
            for cost in COSTS
        }

        scored = []

        for row in candidates:
            regret = {}

            for cost in COSTS:
                regret[cost] = (
                    float(row[cost])
                    - ideal[cost]
                ) / global_span[cost]

                if regret[cost] < -TOL:
                    raise RuntimeError(
                        "Negative regret contract "
                        f"failure: {context_id}/{cost}"
                    )

                regret[cost] = max(
                    0.0,
                    regret[cost],
                )

            regret_vec = np.asarray(
                [
                    regret[
                        "cost_motion"
                    ],
                    regret[
                        "cost_stability"
                    ],
                    regret[
                        "cost_energy"
                    ],
                ],
                dtype=float,
            )

            # Equal-objective Euclidean compromise.
            utopia_l2 = float(
                np.linalg.norm(
                    regret_vec
                )
                / math.sqrt(3.0)
            )

            # Minimize the worst normalized objective regret.
            chebyshev = float(
                np.max(
                    regret_vec
                )
            )

            scored.append(
                {
                    "row":
                        row,

                    "regret_motion":
                        regret[
                            "cost_motion"
                        ],

                    "regret_stability":
                        regret[
                            "cost_stability"
                        ],

                    "regret_energy":
                        regret[
                            "cost_energy"
                        ],

                    "utopia_l2_score":
                        utopia_l2,

                    "chebyshev_score":
                        chebyshev,
                }
            )

        # ----------------------------------------------------
        # Do not silently break real ties.
        # ----------------------------------------------------

        def winner_info(
            key,
        ):
            ordered = sorted(
                scored,
                key=lambda item: (
                    item[key],
                    BETA_ORDER.index(
                        item["row"][
                            "beta_name"
                        ]
                    ),
                ),
            )

            best = ordered[0][key]

            tied = [
                item
                for item in ordered
                if abs(
                    item[key]
                    - best
                ) <= TOL
            ]

            if len(tied) != 1:
                return {
                    "winner":
                        None,

                    "ambiguous":
                        True,

                    "best_score":
                        best,

                    "score_gap":
                        0.0,
                }

            second = (
                ordered[1][key]
                if len(ordered) > 1
                else float("nan")
            )

            return {
                "winner":
                    tied[0],

                "ambiguous":
                    False,

                "best_score":
                    best,

                "score_gap":
                    (
                        second - best
                        if len(ordered) > 1
                        else float("nan")
                    ),
            }

        l2 = winner_info(
            "utopia_l2_score"
        )

        cheb = winner_info(
            "chebyshev_score"
        )

        l2_name = (
            ""
            if l2["winner"] is None
            else l2[
                "winner"
            ]["row"]["beta_name"]
        )

        cheb_name = (
            ""
            if cheb["winner"] is None
            else cheb[
                "winner"
            ]["row"]["beta_name"]
        )

        agreement = (
            l2_name != ""
            and l2_name == cheb_name
        )

        for item in scored:
            row = item["row"]

            candidate_records.append(
                {
                    "context_id":
                        context_id,

                    "group":
                        row["group"],

                    "terrain":
                        terrain,

                    "seed":
                        int(row["seed"]),

                    "source_beta_name":
                        row["beta_name"],

                    "cost_motion":
                        float(
                            row[
                                "cost_motion"
                            ]
                        ),

                    "cost_stability":
                        float(
                            row[
                                "cost_stability"
                            ]
                        ),

                    "cost_energy":
                        float(
                            row[
                                "cost_energy"
                            ]
                        ),

                    "ideal_cost_motion":
                        ideal[
                            "cost_motion"
                        ],

                    "ideal_cost_stability":
                        ideal[
                            "cost_stability"
                        ],

                    "ideal_cost_energy":
                        ideal[
                            "cost_energy"
                        ],

                    "regret_motion":
                        item[
                            "regret_motion"
                        ],

                    "regret_stability":
                        item[
                            "regret_stability"
                        ],

                    "regret_energy":
                        item[
                            "regret_energy"
                        ],

                    "utopia_l2_score":
                        item[
                            "utopia_l2_score"
                        ],

                    "chebyshev_score":
                        item[
                            "chebyshev_score"
                        ],

                    "selected_utopia_l2":
                        int(
                            row["beta_name"]
                            == l2_name
                            and l2_name != ""
                        ),

                    "selected_chebyshev":
                        int(
                            row["beta_name"]
                            == cheb_name
                            and cheb_name != ""
                        ),
                }
            )

        context_records.append(
            {
                "context_id":
                    context_id,

                "group":
                    first["group"],

                "terrain":
                    terrain,

                "seed":
                    int(first["seed"]),

                "pareto_candidate_count":
                    len(candidates),

                "utopia_l2_selected_source_beta":
                    l2_name,

                "utopia_l2_ambiguous":
                    int(
                        l2["ambiguous"]
                    ),

                "utopia_l2_best_score":
                    l2[
                        "best_score"
                    ],

                "utopia_l2_score_gap":
                    l2[
                        "score_gap"
                    ],

                "chebyshev_selected_source_beta":
                    cheb_name,

                "chebyshev_ambiguous":
                    int(
                        cheb["ambiguous"]
                    ),

                "chebyshev_best_score":
                    cheb[
                        "best_score"
                    ],

                "chebyshev_score_gap":
                    cheb[
                        "score_gap"
                    ],

                "rules_agree":
                    int(
                        agreement
                    ),
            }
        )

    # ========================================================
    # Terrain-level diagnostics.
    #
    # Source-beta identities are provenance only.
    # They are NOT Objective Selector labels.
    # ========================================================

    terrain_records = []

    for terrain in TERRAINS:
        subset = [
            row
            for row in context_records
            if row[
                "terrain"
            ] == terrain
        ]

        if len(subset) != 18:
            raise RuntimeError(
                f"{terrain}: expected 18 contexts, "
                f"got {len(subset)}"
            )

        l2_counts = Counter(
            row[
                "utopia_l2_selected_source_beta"
            ]
            for row in subset
            if row[
                "utopia_l2_selected_source_beta"
            ]
        )

        cheb_counts = Counter(
            row[
                "chebyshev_selected_source_beta"
            ]
            for row in subset
            if row[
                "chebyshev_selected_source_beta"
            ]
        )

        terrain_records.append(
            {
                "terrain":
                    terrain,

                "context_count":
                    len(subset),

                "rules_agree_count":
                    sum(
                        row["rules_agree"]
                        for row in subset
                    ),

                "utopia_l2_ambiguous_count":
                    sum(
                        row[
                            "utopia_l2_ambiguous"
                        ]
                        for row in subset
                    ),

                "chebyshev_ambiguous_count":
                    sum(
                        row[
                            "chebyshev_ambiguous"
                        ]
                        for row in subset
                    ),

                "utopia_l2_balanced":
                    l2_counts[
                        "balanced"
                    ],

                "utopia_l2_motion":
                    l2_counts[
                        "motion"
                    ],

                "utopia_l2_stability":
                    l2_counts[
                        "stability"
                    ],

                "utopia_l2_energy":
                    l2_counts[
                        "energy"
                    ],

                "chebyshev_balanced":
                    cheb_counts[
                        "balanced"
                    ],

                "chebyshev_motion":
                    cheb_counts[
                        "motion"
                    ],

                "chebyshev_stability":
                    cheb_counts[
                        "stability"
                    ],

                "chebyshev_energy":
                    cheb_counts[
                        "energy"
                    ],

                "utopia_l2_score_gap_median":
                    float(
                        np.median(
                            [
                                row[
                                    "utopia_l2_score_gap"
                                ]
                                for row in subset
                                if not row[
                                    "utopia_l2_ambiguous"
                                ]
                            ]
                        )
                    ),

                "chebyshev_score_gap_median":
                    float(
                        np.median(
                            [
                                row[
                                    "chebyshev_score_gap"
                                ]
                                for row in subset
                                if not row[
                                    "chebyshev_ambiguous"
                                ]
                            ]
                        )
                    ),
            }
        )

    candidates_path = (
        out_dir
        / "terrain_pareto_candidates.csv"
    )

    contexts_path = (
        out_dir
        / "terrain_compromise_contexts.csv"
    )

    terrain_path = (
        out_dir
        / "terrain_compromise_summary.csv"
    )

    with candidates_path.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                candidate_records[0]
            ),
        )

        writer.writeheader()
        writer.writerows(
            candidate_records
        )

    with contexts_path.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                context_records[0]
            ),
        )

        writer.writeheader()
        writer.writerows(
            context_records
        )

    with terrain_path.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                terrain_records[0]
            ),
        )

        writer.writeheader()
        writer.writerows(
            terrain_records
        )

    overall_agree = sum(
        row["rules_agree"]
        for row in context_records
    )

    l2_ambiguous = sum(
        row["utopia_l2_ambiguous"]
        for row in context_records
    )

    cheb_ambiguous = sum(
        row["chebyshev_ambiguous"]
        for row in context_records
    )

    summary = {
        "schema":
            (
                "icra27_phase2b_terrain_"
                "compromise_archive_v0"
            ),

        "task_conditioning":
            "none",

        "candidate_definition":
            (
                "feasible + Pareto-nondominated "
                "TRAIN trajectories"
            ),

        "context_count":
            len(
                context_records
            ),

        "candidate_count":
            len(
                candidate_records
            ),

        "train_global_normalization":
            {
                cost: {
                    "min":
                        global_min[cost],

                    "max":
                        global_max[cost],

                    "span":
                        global_span[cost],
                }
                for cost in COSTS
            },

        "primary_rule":
            "Chebyshev minimax normalized regret",

        "sensitivity_rule":
            (
                "Euclidean distance to "
                "context ideal point"
            ),

        "rules_agree_count":
            overall_agree,

        "rules_agree_fraction":
            (
                overall_agree
                / len(
                    context_records
                )
            ),

        "utopia_l2_ambiguous_count":
            l2_ambiguous,

        "chebyshev_ambiguous_count":
            cheb_ambiguous,

        "important_note":
            (
                "Source beta names are trajectory "
                "provenance only and are never used "
                "in compromise scoring."
            ),

        "normalization_note":
            (
                "Per-objective TRAIN-global spans "
                "are used so tiny context-local "
                "variations are not artificially "
                "amplified."
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

    print("=" * 104)
    print(
        "ICRA27 PHASE-2B TERRAIN-CONDITIONED "
        "PARETO COMPROMISE ARCHIVE V0"
    )
    print("=" * 104)

    print(
        "contexts              :",
        len(
            context_records
        ),
    )

    print(
        "Pareto candidates     :",
        len(
            candidate_records
        ),
    )

    print(
        "rules agree           :",
        f"{overall_agree}/"
        f"{len(context_records)}",
        (
            f"({overall_agree / len(context_records):.3f})"
        ),
    )

    print(
        "L2 ambiguous          :",
        l2_ambiguous,
    )

    print(
        "Chebyshev ambiguous   :",
        cheb_ambiguous,
    )

    print()
    print(
        "TRAIN-GLOBAL OBJECTIVE SPANS"
    )

    for cost in COSTS:
        print(
            f"  {cost:<16}: "
            f"min={global_min[cost]:.8f} "
            f"max={global_max[cost]:.8f} "
            f"span={global_span[cost]:.8f}"
        )

    print()
    print(
        "TERRAIN SELECTION PROVENANCE"
    )

    for row in terrain_records:
        print()
        print(
            row["terrain"].upper()
        )

        print(
            "  agreement       :",
            f"{row['rules_agree_count']}/18",
        )

        print(
            "  Utopia-L2       :",
            "balanced="
            f"{row['utopia_l2_balanced']} "
            "motion="
            f"{row['utopia_l2_motion']} "
            "stability="
            f"{row['utopia_l2_stability']} "
            "energy="
            f"{row['utopia_l2_energy']}",
        )

        print(
            "  Chebyshev       :",
            "balanced="
            f"{row['chebyshev_balanced']} "
            "motion="
            f"{row['chebyshev_motion']} "
            "stability="
            f"{row['chebyshev_stability']} "
            "energy="
            f"{row['chebyshev_energy']}",
        )

        print(
            "  median score gap:",
            "L2="
            f"{row['utopia_l2_score_gap_median']:.6f} "
            "Cheb="
            f"{row['chebyshev_score_gap_median']:.6f}",
        )

    print()
    print(
        "candidates:",
        candidates_path,
    )

    print(
        "contexts  :",
        contexts_path,
    )

    print(
        "terrain   :",
        terrain_path,
    )

    print(
        "summary   :",
        summary_path,
    )

    print()
    print(
        "[ICRA27] Phase-2B terrain-conditioned "
        "Pareto compromise archive: PASS"
    )


if __name__ == "__main__":
    main()
