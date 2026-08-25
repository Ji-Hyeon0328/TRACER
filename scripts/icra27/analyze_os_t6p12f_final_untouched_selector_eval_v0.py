from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


PRECOMMIT_DIR = (
    ROOT
    / "results/icra27"
    / "os_t6p12d_heldout_selector_precommit_v0"
)

PRECOMMIT_MANIFEST = (
    PRECOMMIT_DIR
    / "heldout_selector_precommit_manifest.json"
)

SELECTION_CSV = (
    PRECOMMIT_DIR
    / "precommitted_eta_selections.csv"
)

BASELINE_CSV = (
    PRECOMMIT_DIR
    / "train_only_eta_baselines.csv"
)


PHYSICAL_DIR = (
    ROOT
    / "results/icra27"
    / "os_t6p12e_heldout_physical_atlas_v0"
)

PHYSICAL_MANIFEST = (
    PHYSICAL_DIR
    / "heldout_physical_atlas_manifest.json"
)

PHYSICAL_CSV = (
    PHYSICAL_DIR
    / "heldout_physical_beta_response_atlas.csv"
)

FEASIBILITY_CSV = (
    PHYSICAL_DIR
    / "context_feasibility_summary.csv"
)


PROTOCOL_MANIFEST = (
    ROOT
    / "results/icra27"
    / "os_t6p12b_untouched_eval_protocol_v0"
    / "untouched_eval_protocol_manifest.json"
)


OUT_DIR = (
    ROOT
    / "results/icra27"
    / "os_t6p12f_final_untouched_selector_eval_v0"
)

OUT_SURFACE = (
    OUT_DIR
    / "true_eta_quality_surface.csv"
)

OUT_CONTEXT = (
    OUT_DIR
    / "context_eta_evaluation.csv"
)

OUT_SUMMARY = (
    OUT_DIR
    / "population_eta_summary.csv"
)

OUT_HARD = (
    OUT_DIR
    / "hard_feasibility_stress_summary.csv"
)

OUT_MANIFEST = (
    OUT_DIR
    / "final_untouched_selector_eval_manifest.json"
)


VAL = (
    1,
    21,
    16,
    14,
)

TEST = (
    27,
    2,
    3,
    19,
    26,
)

HARD = (
    9,
    23,
    24,
)

STANDARD = (
    VAL
    + TEST
)


EXPECTED_STANDARD = 9
EXPECTED_HARD = 3
EXPECTED_BETA = 21

TCHEBY_RHO = 0.01
TOL = 1.0e-12


ETA_ORDER = (
    "balanced",
    "motion_biased",
    "stability_biased",
    "energy_biased",
)


def read_csv(path):
    with path.open(
        "r",
        newline="",
    ) as f:
        return list(
            csv.DictReader(f)
        )


def write_csv(
    path,
    rows,
):
    if not rows:
        raise RuntimeError(
            f"No rows for {path}"
        )

    fields = []

    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(
                    key
                )

    with path.open(
        "w",
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


def sha256(path):
    h = hashlib.sha256()

    with path.open(
        "rb"
    ) as f:
        while True:
            chunk = f.read(
                1024 * 1024
            )

            if not chunk:
                break

            h.update(
                chunk
            )

    return h.hexdigest()


def as_int(x):
    return int(
        float(x)
    )


def finite(x):
    y = float(x)

    if not math.isfinite(
        y
    ):
        raise ValueError(
            f"Non-finite value: {x!r}"
        )

    return y


def context_id(seed):
    return (
        f"rough_seed_{int(seed)}"
    )


def split_for_seed(seed):
    if seed in VAL:
        return "val"

    if seed in TEST:
        return "test"

    if seed in HARD:
        return "hard"

    raise RuntimeError(
        seed
    )


def pareto_mask(J):
    """
    Minimization Pareto set.

    Point i is dominated iff there exists j such that

        J[j] <= J[i] in every objective

    and

        J[j] < J[i] in at least one objective.
    """

    J = np.asarray(
        J,
        dtype=np.float64,
    )

    n = J.shape[
        0
    ]

    keep = np.ones(
        n,
        dtype=bool,
    )

    for i in range(n):
        for j in range(n):

            if i == j:
                continue

            weak = np.all(
                J[j]
                <= J[i]
                + TOL
            )

            strict = np.any(
                J[j]
                < J[i]
                - TOL
            )

            if (
                weak
                and strict
            ):
                keep[i] = False
                break

    return keep


def physical_regret(
    J,
):
    J = np.asarray(
        J,
        dtype=np.float64,
    )

    mask = pareto_mask(
        J
    )

    if not np.any(
        mask
    ):
        raise RuntimeError(
            "Empty Pareto set."
        )

    pareto = J[
        mask
    ]

    ideal = np.min(
        pareto,
        axis=0,
    )

    nadir = np.max(
        pareto,
        axis=0,
    )

    span = (
        nadir
        - ideal
    )


    regret = np.zeros_like(
        J,
        dtype=np.float64,
    )

    for k in range(
        J.shape[1]
    ):

        if abs(
            span[k]
        ) <= TOL:
            regret[
                :,
                k
            ] = 0.0

        else:
            regret[
                :,
                k
            ] = (
                J[
                    :,
                    k
                ]
                - ideal[k]
            ) / span[k]


    # Deliberately NO clipping.
    return (
        regret,
        mask,
        ideal,
        nadir,
        span,
    )


def q_values(
    regret,
    eta,
):
    eta = np.asarray(
        eta,
        dtype=np.float64,
    )

    weighted = (
        regret
        * eta[
            None,
            :
        ]
    )

    return (
        np.max(
            weighted,
            axis=1,
        )
        +
        TCHEBY_RHO
        * np.sum(
            weighted,
            axis=1,
        )
    )


def rank_of(
    q,
    index,
):
    target = float(
        q[
            index
        ]
    )

    return (
        1
        + int(
            np.sum(
                q
                < target
                - TOL
            )
        )
    )


def classify_pair(
    selected,
    baseline,
):
    delta = (
        selected
        - baseline
    )

    if delta < -TOL:
        return "beat"

    if delta > TOL:
        return "worse"

    return "tie"


def percentile(
    values,
    q,
):
    if not values:
        return float(
            "nan"
        )

    return float(
        np.percentile(
            np.asarray(
                values,
                dtype=np.float64,
            ),
            q,
        )
    )


def main():
    if OUT_DIR.exists():
        raise SystemExit(
            f"REFUSING TO OVERWRITE: "
            f"{OUT_DIR}"
        )


    for path in (
        PRECOMMIT_MANIFEST,
        SELECTION_CSV,
        BASELINE_CSV,
        PHYSICAL_MANIFEST,
        PHYSICAL_CSV,
        FEASIBILITY_CSV,
        PROTOCOL_MANIFEST,
    ):
        if not path.exists():
            raise FileNotFoundError(
                path
            )


    # ========================================================
    # Frozen provenance.
    # ========================================================

    precommit = json.loads(
        PRECOMMIT_MANIFEST.read_text()
    )

    physical_manifest = json.loads(
        PHYSICAL_MANIFEST.read_text()
    )

    protocol = json.loads(
        PROTOCOL_MANIFEST.read_text()
    )


    if precommit.get(
        "status"
    ) != "FREEZE_PASS":
        raise RuntimeError(
            "T6.12d is not FREEZE_PASS."
        )

    if physical_manifest.get(
        "status"
    ) != "FREEZE_PASS":
        raise RuntimeError(
            "T6.12e is not FREEZE_PASS."
        )

    if protocol.get(
        "status"
    ) != "FREEZE_PASS":
        raise RuntimeError(
            "T6.12b is not FREEZE_PASS."
        )


    if not bool(
        physical_manifest.get(
            "selector_precommitted_before_outcomes",
            False,
        )
    ):
        raise RuntimeError(
            "Physical atlas does not certify "
            "selector precommit."
        )


    if physical_manifest[
        "selector_precommit_sha256"
    ] != sha256(
        PRECOMMIT_MANIFEST
    ):
        raise RuntimeError(
            "T6.12e/T6.12d precommit "
            "hash mismatch."
        )


    expected_splits = protocol[
        "evaluation_splits"
    ]

    if expected_splits[
        "val"
    ] != list(VAL):
        raise RuntimeError(
            "VAL split drift."
        )

    if expected_splits[
        "test"
    ] != list(TEST):
        raise RuntimeError(
            "TEST split drift."
        )

    if expected_splits[
        "hard"
    ] != list(HARD):
        raise RuntimeError(
            "HARD split drift."
        )


    # ========================================================
    # Feasibility structure.
    #
    # T6.12e outcome is now frozen:
    # standard 9 must be 21/21 feasible,
    # hard 3 must be 0/21 feasible.
    # This script does NOT use this to tune anything.
    # ========================================================

    feasibility_rows = read_csv(
        FEASIBILITY_CSV
    )

    feasibility = {
        row[
            "context_id"
        ]:
            row
        for row in feasibility_rows
    }


    if len(
        feasibility
    ) != 12:
        raise RuntimeError(
            "Expected 12 feasibility contexts."
        )


    for seed in STANDARD:

        cid = context_id(
            seed
        )

        count = as_int(
            feasibility[
                cid
            ][
                "feasible_beta_rows"
            ]
        )

        if count != EXPECTED_BETA:
            raise RuntimeError(
                f"{cid}: standard heldout "
                f"is not full 21-beta feasible."
            )


    for seed in HARD:

        cid = context_id(
            seed
        )

        count = as_int(
            feasibility[
                cid
            ][
                "feasible_beta_rows"
            ]
        )

        if count != 0:
            raise RuntimeError(
                f"{cid}: hard set expected "
                "zero feasible beta rows."
            )


    # ========================================================
    # Frozen selections and TRAIN-only baselines.
    # ========================================================

    selection_rows = read_csv(
        SELECTION_CSV
    )

    baseline_rows = read_csv(
        BASELINE_CSV
    )


    selections = {}

    for row in selection_rows:

        key = (
            row[
                "context_id"
            ],
            row[
                "eta_name"
            ],
        )

        if key in selections:
            raise RuntimeError(
                f"Duplicate precommit selection: "
                f"{key}"
            )

        selections[
            key
        ] = row


    baselines = {
        row[
            "eta_name"
        ]:
            row
        for row in baseline_rows
    }


    if set(
        baselines
    ) != set(
        ETA_ORDER
    ):
        raise RuntimeError(
            "Eta-specific baseline set drift."
        )


    eta_vectors = {}

    for eta_name in ETA_ORDER:

        sample = next(
            row
            for row in selection_rows
            if row[
                "eta_name"
            ] == eta_name
        )

        eta_vectors[
            eta_name
        ] = np.asarray(
            [
                finite(
                    sample[
                        "eta_motion"
                    ]
                ),

                finite(
                    sample[
                        "eta_stability"
                    ]
                ),

                finite(
                    sample[
                        "eta_energy"
                    ]
                ),
            ],
            dtype=np.float64,
        )


    # ========================================================
    # Physical atlas grouping.
    # ========================================================

    physical_rows = read_csv(
        PHYSICAL_CSV
    )

    if len(
        physical_rows
    ) != (
        12
        * EXPECTED_BETA
    ):
        raise RuntimeError(
            "Expected exactly 252 physical rows."
        )


    grouped = defaultdict(
        list
    )

    for row in physical_rows:

        grouped[
            row[
                "context_id"
            ]
        ].append(
            row
        )


    # ========================================================
    # True context-local surfaces.
    # ========================================================

    surface_rows = []
    context_eval_rows = []


    for seed in STANDARD:

        cid = context_id(
            seed
        )

        split = split_for_seed(
            seed
        )

        rows = grouped[
            cid
        ]


        if len(rows) != EXPECTED_BETA:
            raise RuntimeError(
                f"{cid}: expected 21 rows."
            )


        if any(
            as_int(
                row[
                    "feasible"
                ]
            )
            != 1
            for row in rows
        ):
            raise RuntimeError(
                f"{cid}: standard context "
                "contains infeasible candidate."
            )


        # Canonical cross-artifact key is beta_name.
        rows = sorted(
            rows,
            key=lambda row:
                row[
                    "beta_name"
                ],
        )


        names = [
            row[
                "beta_name"
            ]
            for row in rows
        ]


        if len(
            set(
                names
            )
        ) != EXPECTED_BETA:
            raise RuntimeError(
                f"{cid}: beta-name duplication."
            )


        name_to_index = {
            name:
                i
            for i, name in enumerate(
                names
            )
        }


        J = np.asarray(
            [
                [
                    finite(
                        row[
                            "J_motion_s_per_m"
                        ]
                    ),

                    finite(
                        row[
                            "J_stability"
                        ]
                    ),

                    finite(
                        row[
                            "J_energy_j_per_m"
                        ]
                    ),
                ]
                for row in rows
            ],
            dtype=np.float64,
        )


        (
            regret,
            pareto,
            ideal,
            nadir,
            span,
        ) = physical_regret(
            J
        )


        for eta_name in ETA_ORDER:

            eta = eta_vectors[
                eta_name
            ]

            q = q_values(
                regret,
                eta,
            )


            oracle_index = int(
                np.argmin(
                    q
                )
            )

            oracle_q = float(
                q[
                    oracle_index
                ]
            )


            selected_row = selections[
                (
                    cid,
                    eta_name,
                )
            ]

            selected_name = (
                selected_row[
                    "selected_beta_name"
                ]
            )


            if selected_name not in (
                name_to_index
            ):
                raise RuntimeError(
                    f"{cid}/{eta_name}: "
                    "selected beta absent from "
                    "physical atlas."
                )


            baseline_name = (
                baselines[
                    eta_name
                ][
                    "baseline_beta_name"
                ]
            )


            if baseline_name not in (
                name_to_index
            ):
                raise RuntimeError(
                    f"{cid}/{eta_name}: "
                    "baseline beta absent from "
                    "physical atlas."
                )


            selected_index = (
                name_to_index[
                    selected_name
                ]
            )

            baseline_index = (
                name_to_index[
                    baseline_name
                ]
            )


            selected_q = float(
                q[
                    selected_index
                ]
            )

            baseline_q = float(
                q[
                    baseline_index
                ]
            )


            selected_excess = (
                selected_q
                - oracle_q
            )

            baseline_excess = (
                baseline_q
                - oracle_q
            )


            selected_rank = rank_of(
                q,
                selected_index,
            )

            baseline_rank = rank_of(
                q,
                baseline_index,
            )


            outcome = classify_pair(
                selected_excess,
                baseline_excess,
            )


            sorted_q = np.sort(
                q
            )

            true_gap = float(
                sorted_q[
                    1
                ]
                - sorted_q[
                    0
                ]
            )


            context_eval_rows.append(
                {
                    "context_id":
                        cid,

                    "seed":
                        seed,

                    "split_group":
                        split,

                    "eta_name":
                        eta_name,

                    "eta_motion":
                        float(
                            eta[
                                0
                            ]
                        ),

                    "eta_stability":
                        float(
                            eta[
                                1
                            ]
                        ),

                    "eta_energy":
                        float(
                            eta[
                                2
                            ]
                        ),

                    "pareto_count":
                        int(
                            np.sum(
                                pareto
                            )
                        ),

                    "oracle_beta_name":
                        names[
                            oracle_index
                        ],

                    "oracle_Q":
                        oracle_q,

                    "true_best_second_Q_gap":
                        true_gap,

                    "selected_beta_name":
                        selected_name,

                    "selected_Q":
                        selected_q,

                    "selected_Q_excess":
                        selected_excess,

                    "selected_rank":
                        selected_rank,

                    "selected_exact":
                        int(
                            selected_excess
                            <= TOL
                        ),

                    "selected_top3":
                        int(
                            selected_rank
                            <= 3
                        ),

                    "selected_top5":
                        int(
                            selected_rank
                            <= 5
                        ),

                    "baseline_beta_name":
                        baseline_name,

                    "baseline_Q":
                        baseline_q,

                    "baseline_Q_excess":
                        baseline_excess,

                    "baseline_rank":
                        baseline_rank,

                    "selector_minus_baseline_Q_excess":
                        (
                            selected_excess
                            - baseline_excess
                        ),

                    "selector_vs_baseline":
                        outcome,

                    "Jideal_motion":
                        float(
                            ideal[
                                0
                            ]
                        ),

                    "Jideal_stability":
                        float(
                            ideal[
                                1
                            ]
                        ),

                    "Jideal_energy":
                        float(
                            ideal[
                                2
                            ]
                        ),

                    "Jnadir_motion":
                        float(
                            nadir[
                                0
                            ]
                        ),

                    "Jnadir_stability":
                        float(
                            nadir[
                                1
                            ]
                        ),

                    "Jnadir_energy":
                        float(
                            nadir[
                                2
                            ]
                        ),

                    "Jspan_motion":
                        float(
                            span[
                                0
                            ]
                        ),

                    "Jspan_stability":
                        float(
                            span[
                                1
                            ]
                        ),

                    "Jspan_energy":
                        float(
                            span[
                                2
                            ]
                        ),
                }
            )


            for i, row in enumerate(
                rows
            ):

                surface_rows.append(
                    {
                        "context_id":
                            cid,

                        "seed":
                            seed,

                        "split_group":
                            split,

                        "eta_name":
                            eta_name,

                        "beta_name":
                            names[
                                i
                            ],

                        "beta_motion":
                            finite(
                                row[
                                    "beta_motion"
                                ]
                            ),

                        "beta_stability":
                            finite(
                                row[
                                    "beta_stability"
                                ]
                            ),

                        "beta_energy":
                            finite(
                                row[
                                    "beta_energy"
                                ]
                            ),

                        "pareto_nondominated":
                            int(
                                pareto[
                                    i
                                ]
                            ),

                        "J_motion":
                            float(
                                J[
                                    i,
                                    0
                                ]
                            ),

                        "J_stability":
                            float(
                                J[
                                    i,
                                    1
                                ]
                            ),

                        "J_energy":
                            float(
                                J[
                                    i,
                                    2
                                ]
                            ),

                        "regret_motion":
                            float(
                                regret[
                                    i,
                                    0
                                ]
                            ),

                        "regret_stability":
                            float(
                                regret[
                                    i,
                                    1
                                ]
                            ),

                        "regret_energy":
                            float(
                                regret[
                                    i,
                                    2
                                ]
                            ),

                        "true_Q":
                            float(
                                q[
                                    i
                                ]
                            ),

                        "true_Q_excess":
                            float(
                                q[
                                    i
                                ]
                                - oracle_q
                            ),

                        "is_oracle":
                            int(
                                i
                                == oracle_index
                            ),

                        "is_precommitted_selection":
                            int(
                                names[
                                    i
                                ]
                                == selected_name
                            ),

                        "is_train_only_baseline":
                            int(
                                names[
                                    i
                                ]
                                == baseline_name
                            ),
                    }
                )


    # ========================================================
    # HARD is feasibility stress only.
    # No oracle, Q, rank, or selector-vs-baseline score.
    # ========================================================

    hard_rows = []


    for seed in HARD:

        cid = context_id(
            seed
        )

        rows = grouped[
            cid
        ]


        if len(
            rows
        ) != EXPECTED_BETA:
            raise RuntimeError(
                f"{cid}: expected 21 HARD rows."
            )


        status_counts = defaultdict(
            int
        )

        m4_rows = 0

        settling_fail_rows = 0


        for row in rows:

            status = row[
                "status"
            ]

            status_counts[
                status
            ] += 1

            if as_int(
                row[
                    "m4_interventions"
                ]
            ) > 0:
                m4_rows += 1

            if status == (
                "settling_fail"
            ):
                settling_fail_rows += 1


        hard_rows.append(
            {
                "context_id":
                    cid,

                "seed":
                    seed,

                "split_group":
                    "hard",

                "beta_candidates":
                    EXPECTED_BETA,

                "feasible_beta_rows":
                    0,

                "lattice_feasible":
                    0,

                "oracle_defined":
                    0,

                "selector_Q_defined":
                    0,

                "m4_rows":
                    m4_rows,

                "settling_fail_rows":
                    settling_fail_rows,

                "status_counts_json":
                    json.dumps(
                        dict(
                            status_counts
                        ),
                        sort_keys=True,
                    ),

                "interpretation":
                    (
                        "Frozen policy/controller "
                        "candidate lattice infeasible; "
                        "not a beta-selection score."
                    ),
            }
        )


    # ========================================================
    # Aggregate population summaries.
    #
    # Primary:
    #   TEST / balanced.
    #
    # Secondary:
    #   VAL, TEST, VAL+TEST across all frozen eta profiles.
    # ========================================================

    populations = {
        "VAL":
            {
                context_id(
                    seed
                )
                for seed in VAL
            },

        "TEST":
            {
                context_id(
                    seed
                )
                for seed in TEST
            },

        "STANDARD_VAL_TEST":
            {
                context_id(
                    seed
                )
                for seed in STANDARD
            },
    }


    summary_rows = []


    for population_name, cids in (
        populations.items()
    ):

        for eta_name in ETA_ORDER:

            rows = [
                row
                for row in context_eval_rows
                if (
                    row[
                        "context_id"
                    ]
                    in cids
                    and row[
                        "eta_name"
                    ]
                    == eta_name
                )
            ]


            expected_n = len(
                cids
            )

            if len(
                rows
            ) != expected_n:
                raise RuntimeError(
                    f"{population_name}/"
                    f"{eta_name}: expected "
                    f"{expected_n} rows, "
                    f"got {len(rows)}"
                )


            selector_excess = [
                finite(
                    row[
                        "selected_Q_excess"
                    ]
                )
                for row in rows
            ]

            baseline_excess = [
                finite(
                    row[
                        "baseline_Q_excess"
                    ]
                )
                for row in rows
            ]

            ranks = [
                as_int(
                    row[
                        "selected_rank"
                    ]
                )
                for row in rows
            ]


            beat = sum(
                row[
                    "selector_vs_baseline"
                ]
                == "beat"
                for row in rows
            )

            tie = sum(
                row[
                    "selector_vs_baseline"
                ]
                == "tie"
                for row in rows
            )

            worse = sum(
                row[
                    "selector_vs_baseline"
                ]
                == "worse"
                for row in rows
            )


            summary_rows.append(
                {
                    "population":
                        population_name,

                    "eta_name":
                        eta_name,

                    "n_contexts":
                        len(
                            rows
                        ),

                    "selector_Qex_mean":
                        float(
                            np.mean(
                                selector_excess
                            )
                        ),

                    "selector_Qex_median":
                        float(
                            np.median(
                                selector_excess
                            )
                        ),

                    "selector_Qex_p95":
                        percentile(
                            selector_excess,
                            95,
                        ),

                    "baseline_Qex_mean":
                        float(
                            np.mean(
                                baseline_excess
                            )
                        ),

                    "baseline_Qex_median":
                        float(
                            np.median(
                                baseline_excess
                            )
                        ),

                    "baseline_Qex_p95":
                        percentile(
                            baseline_excess,
                            95,
                        ),

                    "selector_minus_baseline_mean":
                        float(
                            np.mean(
                                np.asarray(
                                    selector_excess
                                )
                                - np.asarray(
                                    baseline_excess
                                )
                            )
                        ),

                    "beat":
                        beat,

                    "tie":
                        tie,

                    "worse":
                        worse,

                    "exact_fraction":
                        float(
                            np.mean(
                                [
                                    as_int(
                                        row[
                                            "selected_exact"
                                        ]
                                    )
                                    for row in rows
                                ]
                            )
                        ),

                    "top3_fraction":
                        float(
                            np.mean(
                                [
                                    as_int(
                                        row[
                                            "selected_top3"
                                        ]
                                    )
                                    for row in rows
                                ]
                            )
                        ),

                    "top5_fraction":
                        float(
                            np.mean(
                                [
                                    as_int(
                                        row[
                                            "selected_top5"
                                        ]
                                    )
                                    for row in rows
                                ]
                            )
                        ),

                    "rank_mean":
                        float(
                            np.mean(
                                ranks
                            )
                        ),

                    "rank_median":
                        float(
                            np.median(
                                ranks
                            )
                        ),

                    "primary_metric":
                        int(
                            (
                                population_name
                                == "TEST"
                            )
                            and (
                                eta_name
                                == "balanced"
                            )
                        ),
                }
            )


    # ========================================================
    # Persist.
    # ========================================================

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    write_csv(
        OUT_SURFACE,
        surface_rows,
    )

    write_csv(
        OUT_CONTEXT,
        context_eval_rows,
    )

    write_csv(
        OUT_SUMMARY,
        summary_rows,
    )

    write_csv(
        OUT_HARD,
        hard_rows,
    )


    primary = next(
        row
        for row in summary_rows
        if (
            row[
                "population"
            ]
            == "TEST"
            and row[
                "eta_name"
            ]
            == "balanced"
        )
    )


    manifest = {
        "schema":
            "icra27_os_t6p12f_final_untouched_selector_eval_v0",

        "status":
            "COMPUTE_PASS",

        "scientific_interpretation_pending":
            True,

        "primary_evaluation":
            {
                "population":
                    "TEST",

                "eta":
                    "balanced",

                "n":
                    5,

                "metric":
                    "true context-local Q excess",

                "comparison":
                    (
                        "frozen context-aware selector "
                        "versus frozen TRAIN-only "
                        "no-context baseline"
                    ),

                "selector_Qex_mean":
                    primary[
                        "selector_Qex_mean"
                    ],

                "baseline_Qex_mean":
                    primary[
                        "baseline_Qex_mean"
                    ],

                "beat_tie_worse":
                    [
                        primary[
                            "beat"
                        ],

                        primary[
                            "tie"
                        ],

                        primary[
                            "worse"
                        ],
                    ],
            },

        "standard_evaluation_contexts":
            list(
                STANDARD
            ),

        "hard_contexts":
            list(
                HARD
            ),

        "hard_evaluation_semantics":
            (
                "All hard contexts have zero "
                "feasible beta candidates under the "
                "frozen policy/controller lattice. "
                "They are reported as feasibility "
                "stress failures and are excluded "
                "from selector Q/rank aggregation."
            ),

        "normalization":
            (
                "context-local Pareto ideal/nadir "
                "computed from feasible beta candidates"
            ),

        "regret_clipping":
            False,

        "tcheby_rho":
            TCHEBY_RHO,

        "canonical_cross_artifact_beta_key":
            "beta_name",

        "source_hashes":
            {
                "precommit_manifest":
                    sha256(
                        PRECOMMIT_MANIFEST
                    ),

                "selection_csv":
                    sha256(
                        SELECTION_CSV
                    ),

                "baseline_csv":
                    sha256(
                        BASELINE_CSV
                    ),

                "physical_manifest":
                    sha256(
                        PHYSICAL_MANIFEST
                    ),

                "physical_atlas_csv":
                    sha256(
                        PHYSICAL_CSV
                    ),
            },

        "artifact_hashes":
            {
                "true_eta_quality_surface.csv":
                    sha256(
                        OUT_SURFACE
                    ),

                "context_eta_evaluation.csv":
                    sha256(
                        OUT_CONTEXT
                    ),

                "population_eta_summary.csv":
                    sha256(
                        OUT_SUMMARY
                    ),

                "hard_feasibility_stress_summary.csv":
                    sha256(
                        OUT_HARD
                    ),
            },

        "scientific_guard":
            (
                "No model, context, beta lattice, "
                "eta, baseline, objective, normalization, "
                "or selection is changed after opening "
                "heldout physical outcomes."
            ),
    }


    OUT_MANIFEST.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


    # ========================================================
    # Human-readable final table.
    # ========================================================

    print()
    print("=" * 146)
    print(
        "ICRA27 OS-T6.12f FINAL "
        "UNTOUCHED SELECTOR EVALUATION"
    )
    print("=" * 146)

    print(
        "population          eta               "
        "| selector Qex mean/med | baseline mean/med "
        "| beat/tie/worse | exact top3 top5 | rank mean/med"
    )

    print("-" * 146)


    for row in summary_rows:

        print(
            f"{row['population']:<19} "
            f"{row['eta_name']:<17} | "
            f"{float(row['selector_Qex_mean']):.6f}/"
            f"{float(row['selector_Qex_median']):.6f} | "
            f"{float(row['baseline_Qex_mean']):.6f}/"
            f"{float(row['baseline_Qex_median']):.6f} | "
            f"{int(row['beat'])}/"
            f"{int(row['tie'])}/"
            f"{int(row['worse'])} | "
            f"{float(row['exact_fraction']):.3f} "
            f"{float(row['top3_fraction']):.3f} "
            f"{float(row['top5_fraction']):.3f} | "
            f"{float(row['rank_mean']):.3f}/"
            f"{float(row['rank_median']):.3f}"
        )


    print()
    print("PRIMARY — TEST / balanced")
    print(
        "  selector Qex mean :",
        primary[
            "selector_Qex_mean"
        ],
    )

    print(
        "  baseline Qex mean :",
        primary[
            "baseline_Qex_mean"
        ],
    )

    print(
        "  delta             :",
        primary[
            "selector_minus_baseline_mean"
        ],
    )

    print(
        "  beat/tie/worse    :",
        (
            primary[
                "beat"
            ],
            primary[
                "tie"
            ],
            primary[
                "worse"
            ],
        ),
    )

    print(
        "  exact/top3/top5   :",
        (
            primary[
                "exact_fraction"
            ],
            primary[
                "top3_fraction"
            ],
            primary[
                "top5_fraction"
            ],
        ),
    )

    print()
    print("HARD feasibility stress:")

    for row in hard_rows:
        print(
            f"  {row['context_id']:<16} "
            f"feasible=0/21 "
            f"M4_rows={row['m4_rows']} "
            f"settling_fail_rows="
            f"{row['settling_fail_rows']} "
            f"status={row['status_counts_json']}"
        )


    print()
    print(
        "[ICRA27] OS-T6.12f final untouched "
        "selector evaluation: COMPUTE PASS"
    )

    print(
        "NOTE: scientific PASS/PARTIAL/FAIL "
        "must be interpreted from the frozen "
        "primary and secondary metrics above; "
        "do not tune after this point."
    )

    print("=" * 146)


if __name__ == "__main__":
    main()
