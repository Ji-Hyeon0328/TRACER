#!/usr/bin/env python3

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.special import logsumexp


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def build_simplex_grid(step):
    n = int(round(1.0 / step))

    if not np.isclose(
        n * step,
        1.0,
        atol=1e-12,
    ):
        raise RuntimeError(
            "--simplex-step must divide 1 exactly; "
            f"got step={step}"
        )

    rows = []

    for i in range(n + 1):
        for j in range(n - i + 1):
            k = n - i - j

            rows.append(
                (
                    i / n,
                    j / n,
                    k / n,
                )
            )

    grid = np.asarray(
        rows,
        dtype=float,
    )

    if not np.allclose(
        np.sum(grid, axis=1),
        1.0,
    ):
        raise RuntimeError(
            "Simplex-grid contract failure."
        )

    return grid


def choice_log_likelihood(
    grid,
    deltas,
    kappa,
):
    # deltas:
    #   C_alt - C_selected
    #
    # P(selected)
    # =
    # 1 / (
    #   1 + sum_j exp(
    #       -kappa beta^T delta_j
    #   )
    # )

    scores = (
        grid
        @ deltas.T
    )

    alt_log_terms = (
        -float(kappa)
        * scores
    )

    alt_log_sum = logsumexp(
        alt_log_terms,
        axis=1,
    )

    log_denom = np.logaddexp(
        0.0,
        alt_log_sum,
    )

    return -log_denom


def weighted_quantile(
    values,
    weights,
    q,
):
    order = np.argsort(
        values
    )

    values = values[
        order
    ]

    weights = weights[
        order
    ]

    cdf = np.cumsum(
        weights
    )

    cdf /= cdf[
        -1
    ]

    idx = np.searchsorted(
        cdf,
        q,
        side="left",
    )

    idx = min(
        int(idx),
        len(values) - 1,
    )

    return float(
        values[
            idx
        ]
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--preference-pairs",
        required=True,
    )

    ap.add_argument(
        "--exact-audit",
        required=True,
    )

    ap.add_argument(
        "--out-dir",
        required=True,
    )

    ap.add_argument(
        "--simplex-step",
        type=float,
        default=0.01,
    )

    ap.add_argument(
        "--kappa-min",
        type=float,
        default=0.1,
    )

    ap.add_argument(
        "--kappa-max",
        type=float,
        default=3162.2776601683795,
    )

    ap.add_argument(
        "--kappa-count",
        type=int,
        default=37,
    )

    args = ap.parse_args()

    pair_path = Path(
        args.preference_pairs
    )

    audit_path = Path(
        args.exact_audit
    )

    out_dir = Path(
        args.out_dir
    )

    for path in (
        pair_path,
        audit_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(
                path
            )

    if args.kappa_min <= 0:
        raise RuntimeError(
            "--kappa-min must be positive."
        )

    if args.kappa_max <= args.kappa_min:
        raise RuntimeError(
            "--kappa-max must exceed kappa-min."
        )

    if args.kappa_count < 3:
        raise RuntimeError(
            "--kappa-count must be >= 3."
        )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    pair_rows = read_csv(
        pair_path
    )

    audit_rows = read_csv(
        audit_path
    )

    if not pair_rows:
        raise RuntimeError(
            "Preference pair dataset is empty."
        )

    if not audit_rows:
        raise RuntimeError(
            "Exact-audit dataset is empty."
        )

    audit_by_decision = {
        int(
            row["decision_index"]
        ):
            row
        for row in audit_rows
    }

    # ========================================================
    # Reconstruct one local choice event per mission decision.
    # ========================================================

    by_decision = defaultdict(
        list
    )

    for row in pair_rows:
        by_decision[
            int(
                row["decision_index"]
            )
        ].append(
            row
        )

    if set(
        by_decision
    ) != set(
        audit_by_decision
    ):
        missing_audit = sorted(
            set(by_decision)
            - set(audit_by_decision)
        )

        missing_pairs = sorted(
            set(audit_by_decision)
            - set(by_decision)
        )

        raise RuntimeError(
            "Pair/audit decision mismatch:\n"
            f"missing audit={missing_audit[:20]}\n"
            f"missing pairs={missing_pairs[:20]}"
        )

    decisions = []

    for decision_index in sorted(
        by_decision
    ):
        rows = by_decision[
            decision_index
        ]

        rows = sorted(
            rows,
            key=lambda row:
                row[
                    "alternative_beta_name"
                ],
        )

        first = rows[
            0
        ]

        metadata_keys = (
            "context_id",
            "group",
            "terrain",
            "seed",
            "mission",
            "primary_metric",
            "selected_beta_name",
            "feasible_choice_count",
            "decision_weight",
        )

        for row in rows[
            1:
        ]:
            for key in metadata_keys:
                if row[
                    key
                ] != first[
                    key
                ]:
                    raise RuntimeError(
                        f"Decision {decision_index}: "
                        f"metadata mismatch for {key}"
                    )

        deltas = np.asarray(
            [
                [
                    float(
                        row[
                            "delta_cost_motion"
                        ]
                    ),
                    float(
                        row[
                            "delta_cost_stability"
                        ]
                    ),
                    float(
                        row[
                            "delta_cost_energy"
                        ]
                    ),
                ]
                for row in rows
            ],
            dtype=float,
        )

        alt_names = tuple(
            row[
                "alternative_beta_name"
            ]
            for row in rows
        )

        audit = audit_by_decision[
            decision_index
        ]

        decisions.append(
            {
                "decision_index":
                    decision_index,

                "context_id":
                    first[
                        "context_id"
                    ],

                "group":
                    first[
                        "group"
                    ],

                "terrain":
                    first[
                        "terrain"
                    ],

                "seed":
                    int(
                        first[
                            "seed"
                    ]
                ),

                "mission":
                    first[
                        "mission"
                    ],

                "primary_metric":
                    first[
                        "primary_metric"
                    ],

                "constraint_1":
                    first[
                        "constraint_1"
                    ],

                "constraint_1_budget":
                    float(
                        first[
                            "constraint_1_budget"
                        ]
                    ),

                "constraint_2":
                    first[
                        "constraint_2"
                    ],

                "constraint_2_budget":
                    float(
                        first[
                            "constraint_2_budget"
                        ]
                    ),

                "selected_beta_name":
                    first[
                        "selected_beta_name"
                    ],

                "alternative_beta_names":
                    alt_names,

                "feasible_choice_count":
                    int(
                        first[
                            "feasible_choice_count"
                        ]
                    ),

                "decision_weight":
                    float(
                        first[
                            "decision_weight"
                        ]
                    ),

                "deltas":
                    deltas,

                "exact_support_class":
                    audit[
                        "support_class"
                    ],

                "exact_max_margin_rho":
                    float(
                        audit[
                            "max_margin_rho"
                        ]
                    ),
            }
        )

    # ========================================================
    # Deduplicate repeated critical-budget queries that imply
    # exactly the same local preference choice.
    #
    # Within one exact context+mission, the identity of the
    # selected candidate plus feasible alternative set fully
    # determines the cost differences.
    # ========================================================

    signature_lookup = {}

    signatures = []

    for decision in decisions:
        key = (
            decision[
                "context_id"
            ],
            decision[
                "mission"
            ],
            decision[
                "selected_beta_name"
            ],
            decision[
                "alternative_beta_names"
            ],
        )

        if key not in signature_lookup:
            signature_id = len(
                signatures
            )

            signature_lookup[
                key
            ] = signature_id

            signatures.append(
                {
                    "signature_id":
                        signature_id,

                    "context_id":
                        decision[
                            "context_id"
                        ],

                    "terrain":
                        decision[
                            "terrain"
                        ],

                    "mission":
                        decision[
                            "mission"
                        ],

                    "selected_beta_name":
                        decision[
                            "selected_beta_name"
                        ],

                    "alternative_beta_names":
                        decision[
                            "alternative_beta_names"
                        ],

                    "deltas":
                        decision[
                            "deltas"
                        ].copy(),

                    "decision_indices":
                        [
                            decision[
                                "decision_index"
                            ]
                        ],
                }
            )

        else:
            signature_id = signature_lookup[
                key
            ]

            signature = signatures[
                signature_id
            ]

            if not np.allclose(
                signature[
                    "deltas"
                ],
                decision[
                    "deltas"
                ],
                rtol=0.0,
                atol=1e-12,
            ):
                raise RuntimeError(
                    "Same preference signature has "
                    "different cost deltas."
                )

            signature[
                "decision_indices"
            ].append(
                decision[
                    "decision_index"
                ]
            )

        decision[
            "signature_id"
        ] = signature_id

    # --------------------------------------------------------
    # Calibration weights:
    #
    # each exact context x mission gets total weight 1,
    # divided equally across UNIQUE preference signatures.
    #
    # Therefore arbitrary critical-budget grid multiplicity
    # does not determine kappa.
    # --------------------------------------------------------

    signatures_by_group = defaultdict(
        list
    )

    for signature in signatures:
        signatures_by_group[
            (
                signature[
                    "context_id"
                ],
                signature[
                    "mission"
                ],
            )
        ].append(
            signature[
                "signature_id"
            ]
        )

    for signature in signatures:
        group_key = (
            signature[
                "context_id"
            ],
            signature[
                "mission"
            ],
        )

        signature[
            "calibration_weight"
        ] = (
            1.0
            / len(
                signatures_by_group[
                    group_key
                ]
            )
        )

    # ========================================================
    # Simplex prior.
    # ========================================================

    grid = build_simplex_grid(
        args.simplex_step
    )

    grid_count = len(
        grid
    )

    log_prior_mass = (
        -np.log(
            grid_count
        )
    )

    # Precompute beta dot delta for every unique signature.
    score_cache = {}

    for signature in signatures:
        score_cache[
            signature[
                "signature_id"
            ]
        ] = (
            grid
            @ signature[
                "deltas"
            ].T
        )

    # ========================================================
    # Empirical-Bayes calibration of global rationality kappa.
    # ========================================================

    kappas = np.logspace(
        np.log10(
            args.kappa_min
        ),
        np.log10(
            args.kappa_max
        ),
        args.kappa_count,
    )

    calibration_records = []

    best_objective = None

    best_kappa_index = None

    for kappa_index, kappa in enumerate(
        kappas
    ):
        objective = 0.0

        for signature in signatures:
            scores = score_cache[
                signature[
                    "signature_id"
                ]
            ]

            alt_log_sum = logsumexp(
                -kappa
                * scores,
                axis=1,
            )

            log_likelihood = (
                -np.logaddexp(
                    0.0,
                    alt_log_sum,
                )
            )

            log_evidence = logsumexp(
                log_likelihood
                + log_prior_mass
            )

            objective += (
                signature[
                    "calibration_weight"
                ]
                * log_evidence
            )

        calibration_records.append(
            {
                "kappa_index":
                    kappa_index,

                "kappa":
                    float(
                        kappa
                    ),

                "weighted_log_evidence":
                    float(
                        objective
                    ),
            }
        )

        if (
            best_objective is None
            or objective > best_objective
        ):
            best_objective = float(
                objective
            )

            best_kappa_index = (
                kappa_index
            )

    selected_kappa = float(
        kappas[
            best_kappa_index
        ]
    )

    kappa_at_boundary = (
        best_kappa_index == 0
        or best_kappa_index
        == len(kappas) - 1
    )

    # ========================================================
    # Posterior per UNIQUE preference signature.
    # ========================================================

    posterior_matrix = np.empty(
        (
            len(signatures),
            grid_count,
        ),
        dtype=np.float32,
    )

    signature_records = []

    for signature in signatures:
        signature_id = signature[
            "signature_id"
        ]

        scores = score_cache[
            signature_id
        ]

        alt_log_sum = logsumexp(
            -selected_kappa
            * scores,
            axis=1,
        )

        log_likelihood = (
            -np.logaddexp(
                0.0,
                alt_log_sum,
            )
        )

        log_evidence = logsumexp(
            log_likelihood
            + log_prior_mass
        )

        log_posterior = (
            log_likelihood
            - logsumexp(
                log_likelihood
            )
        )

        posterior = np.exp(
            log_posterior
        )

        posterior_matrix[
            signature_id
        ] = posterior.astype(
            np.float32
        )

        mean = (
            posterior
            @ grid
        )

        second = (
            posterior
            @ (
                grid
                * grid
            )
        )

        var = np.maximum(
            second
            - mean
            * mean,
            0.0,
        )

        std = np.sqrt(
            var
        )

        map_index = int(
            np.argmax(
                posterior
            )
        )

        map_beta = grid[
            map_index
        ]

        entropy = float(
            -np.sum(
                posterior
                * log_posterior
            )
        )

        max_entropy = float(
            np.log(
                grid_count
            )
        )

        entropy_norm = (
            entropy
            / max_entropy
        )

        information_gain = (
            max_entropy
            - entropy
        )

        effective_fraction = float(
            np.exp(
                entropy
            )
            / grid_count
        )

        choice_prob = np.exp(
            log_likelihood
        )

        map_choice_probability = float(
            choice_prob[
                map_index
            ]
        )

        posterior_expected_choice_probability = float(
            np.sum(
                posterior
                * choice_prob
            )
        )

        support_classes = {
            audit_by_decision[
                decision_index
            ][
                "support_class"
            ]
            for decision_index
            in signature[
                "decision_indices"
            ]
        }

        if len(
            support_classes
        ) != 1:
            raise RuntimeError(
                "Identical preference signature has "
                "different exact support classes."
            )

        exact_support_class = next(
            iter(
                support_classes
            )
        )

        record = {
            "signature_id":
                signature_id,

            "context_id":
                signature[
                    "context_id"
                ],

            "terrain":
                signature[
                    "terrain"
                ],

            "mission":
                signature[
                    "mission"
                ],

            "selected_beta_name":
                signature[
                    "selected_beta_name"
                ],

            "alternative_beta_names":
                "|".join(
                    signature[
                        "alternative_beta_names"
                    ]
                ),

            "choice_size":
                1
                + len(
                    signature[
                        "alternative_beta_names"
                    ]
                ),

            "raw_decision_multiplicity":
                len(
                    signature[
                        "decision_indices"
                    ]
                ),

            "calibration_weight":
                float(
                    signature[
                        "calibration_weight"
                    ]
                ),

            "exact_support_class":
                exact_support_class,

            "posterior_mean_motion":
                float(
                    mean[
                        0
                    ]
                ),

            "posterior_mean_stability":
                float(
                    mean[
                        1
                    ]
                ),

            "posterior_mean_energy":
                float(
                    mean[
                        2
                    ]
                ),

            "posterior_std_motion":
                float(
                    std[
                        0
                    ]
                ),

            "posterior_std_stability":
                float(
                    std[
                        1
                    ]
                ),

            "posterior_std_energy":
                float(
                    std[
                        2
                    ]
                ),

            "posterior_map_motion":
                float(
                    map_beta[
                        0
                    ]
                ),

            "posterior_map_stability":
                float(
                    map_beta[
                        1
                    ]
                ),

            "posterior_map_energy":
                float(
                    map_beta[
                        2
                    ]
                ),

            "motion_q05":
                weighted_quantile(
                    grid[
                        :,
                        0
                    ],
                    posterior,
                    0.05,
                ),

            "motion_q95":
                weighted_quantile(
                    grid[
                        :,
                        0
                    ],
                    posterior,
                    0.95,
                ),

            "stability_q05":
                weighted_quantile(
                    grid[
                        :,
                        1
                    ],
                    posterior,
                    0.05,
                ),

            "stability_q95":
                weighted_quantile(
                    grid[
                        :,
                        1
                    ],
                    posterior,
                    0.95,
                ),

            "energy_q05":
                weighted_quantile(
                    grid[
                        :,
                        2
                    ],
                    posterior,
                    0.05,
                ),

            "energy_q95":
                weighted_quantile(
                    grid[
                        :,
                        2
                    ],
                    posterior,
                    0.95,
                ),

            "posterior_entropy_nats":
                entropy,

            "posterior_entropy_normalized":
                entropy_norm,

            "posterior_information_gain_nats":
                information_gain,

            "posterior_effective_fraction":
                effective_fraction,

            "log_marginal_evidence":
                float(
                    log_evidence
                ),

            "map_selected_choice_probability":
                map_choice_probability,

            "posterior_expected_choice_probability":
                posterior_expected_choice_probability,
        }

        signature_records.append(
            record
        )

    # ========================================================
    # Expand posterior summaries back onto mission decisions.
    # ========================================================

    signature_record_by_id = {
        row[
            "signature_id"
        ]:
            row
        for row in signature_records
    }

    decision_records = []

    for decision in decisions:
        signature_record = (
            signature_record_by_id[
                decision[
                    "signature_id"
                ]
            ]
        )

        record = {
            "decision_index":
                decision[
                    "decision_index"
                ],

            "signature_id":
                decision[
                    "signature_id"
                ],

            "context_id":
                decision[
                    "context_id"
                ],

            "group":
                decision[
                    "group"
                ],

            "terrain":
                decision[
                    "terrain"
                ],

            "seed":
                decision[
                    "seed"
                ],

            "mission":
                decision[
                    "mission"
                ],

            "primary_metric":
                decision[
                    "primary_metric"
                ],

            "constraint_1":
                decision[
                    "constraint_1"
                ],

            "constraint_1_budget":
                decision[
                    "constraint_1_budget"
                ],

            "constraint_2":
                decision[
                    "constraint_2"
                ],

            "constraint_2_budget":
                decision[
                    "constraint_2_budget"
                ],

            "selected_beta_name":
                decision[
                    "selected_beta_name"
                ],

            "feasible_choice_count":
                decision[
                    "feasible_choice_count"
                ],

            "decision_weight":
                decision[
                    "decision_weight"
                ],

            "exact_support_class":
                decision[
                    "exact_support_class"
                ],

            "exact_max_margin_rho":
                decision[
                    "exact_max_margin_rho"
                ],
        }

        for key in (
            "posterior_mean_motion",
            "posterior_mean_stability",
            "posterior_mean_energy",
            "posterior_std_motion",
            "posterior_std_stability",
            "posterior_std_energy",
            "posterior_map_motion",
            "posterior_map_stability",
            "posterior_map_energy",
            "posterior_entropy_normalized",
            "posterior_information_gain_nats",
            "posterior_effective_fraction",
            "log_marginal_evidence",
            "map_selected_choice_probability",
            "posterior_expected_choice_probability",
        ):
            record[
                key
            ] = signature_record[
                key
            ]

        decision_records.append(
            record
        )

    # ========================================================
    # Save.
    # ========================================================

    calibration_path = (
        out_dir
        / "kappa_calibration.csv"
    )

    signature_path = (
        out_dir
        / "irl_signature_posteriors.csv"
    )

    decision_path = (
        out_dir
        / "irl_decision_posteriors.csv"
    )

    posterior_path = (
        out_dir
        / "irl_posterior_grid.npz"
    )

    with calibration_path.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                calibration_records[
                    0
                ]
            ),
        )

        writer.writeheader()
        writer.writerows(
            calibration_records
        )

    with signature_path.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                signature_records[
                    0
                ]
            ),
        )

        writer.writeheader()
        writer.writerows(
            signature_records
        )

    with decision_path.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                decision_records[
                    0
                ]
            ),
        )

        writer.writeheader()
        writer.writerows(
            decision_records
        )

    np.savez_compressed(
        posterior_path,
        beta_grid=grid.astype(
            np.float32
        ),
        posterior_probability=(
            posterior_matrix
        ),
    )

    # ========================================================
    # Diagnostics.
    # ========================================================

    support_counts = Counter(
        row[
            "exact_support_class"
        ]
        for row in decision_records
    )

    mission_counts = Counter(
        row[
            "mission"
        ]
        for row in decision_records
    )

    signature_mission_counts = Counter(
        row[
            "mission"
        ]
        for row in signature_records
    )

    entropy_by_mission = {}

    for mission in (
        "fast",
        "stable",
        "efficient",
    ):
        values = [
            row[
                "posterior_entropy_normalized"
            ]
            for row in signature_records
            if row[
                "mission"
            ] == mission
        ]

        entropy_by_mission[
            mission
        ] = {
            "count":
                len(
                    values
                ),

            "mean":
                float(
                    np.mean(
                        values
                    )
                ),

            "median":
                float(
                    np.median(
                        values
                    )
                ),
        }

    summary = {
        "schema":
            "icra27_phase2b_preference_irl_posterior_v0",

        "irl_model":
            (
                "Bayesian multinomial Boltzmann "
                "preference model"
            ),

        "reward_cost":
            "J_beta = beta^T [Cm, Cs, CE]",

        "prior":
            "uniform simplex",

        "simplex_step":
            args.simplex_step,

        "simplex_grid_count":
            grid_count,

        "raw_informative_decision_count":
            len(
                decisions
            ),

        "unique_preference_signature_count":
            len(
                signatures
            ),

        "context_mission_group_count":
            len(
                signatures_by_group
            ),

        "selected_kappa":
            selected_kappa,

        "selected_kappa_index":
            best_kappa_index,

        "kappa_search_min":
            args.kappa_min,

        "kappa_search_max":
            args.kappa_max,

        "kappa_search_count":
            args.kappa_count,

        "kappa_at_search_boundary":
            kappa_at_boundary,

        "support_counts":
            dict(
                support_counts
            ),

        "mission_decision_counts":
            dict(
                mission_counts
            ),

        "signature_mission_counts":
            dict(
                signature_mission_counts
            ),

        "posterior_entropy_by_mission":
            entropy_by_mission,

        "calibration_weighting":
            (
                "each context x mission gets total "
                "weight 1, divided equally across "
                "unique local choice signatures"
            ),

        "important_note":
            (
                "No source beta name or max-margin "
                "beta_hat is used as an IRL target."
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
        "=" * 92
    )

    print(
        "ICRA27 PHASE-2B BAYESIAN "
        "PREFERENCE IRL POSTERIOR V0"
    )

    print(
        "=" * 92
    )

    print(
        "informative decisions :",
        len(
            decisions
        ),
    )

    print(
        "unique signatures     :",
        len(
            signatures
        ),
    )

    print(
        "context x mission     :",
        len(
            signatures_by_group
        ),
    )

    print(
        "simplex grid points   :",
        grid_count,
    )

    print(
        "selected kappa        :",
        selected_kappa,
    )

    print(
        "kappa boundary        :",
        kappa_at_boundary,
    )

    print()
    print(
        "EXACT SUPPORT"
    )

    for name in (
        "strict",
        "weak",
        "unsupported",
    ):
        print(
            f"  {name:<12}: "
            f"{support_counts[name]}"
        )

    print()
    print(
        "UNIQUE SIGNATURES BY MISSION"
    )

    for mission in (
        "fast",
        "stable",
        "efficient",
    ):
        print(
            f"  {mission:<10}: "
            f"{signature_mission_counts[mission]}"
        )

    print()
    print(
        "POSTERIOR NORMALIZED ENTROPY"
    )

    for mission in (
        "fast",
        "stable",
        "efficient",
    ):
        item = entropy_by_mission[
            mission
        ]

        print(
            f"  {mission:<10}: "
            f"mean={item['mean']:.4f} "
            f"median={item['median']:.4f}"
        )

    print()
    print(
        "calibration:",
        calibration_path,
    )

    print(
        "signatures :",
        signature_path,
    )

    print(
        "decisions  :",
        decision_path,
    )

    print(
        "posterior  :",
        posterior_path,
    )

    print(
        "summary    :",
        summary_path,
    )

    if kappa_at_boundary:
        print()
        print(
            "WARNING: selected kappa lies on "
            "the search boundary. Expand the "
            "kappa range before freezing IRL v0."
        )

    print()
    print(
        "[ICRA27] Phase-2B Bayesian "
        "preference IRL posterior: PASS"
    )


if __name__ == "__main__":
    main()
