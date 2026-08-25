#!/usr/bin/env python3

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.special import logsumexp


TERRAINS = (
    "flat",
    "low_friction",
    "rough_perlin",
)

MISSIONS = (
    "fast",
    "stable",
    "efficient",
)

TOL = 1e-9


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
            f"step={step} does not divide 1"
        )

    return np.asarray(
        [
            (
                i / n,
                j / n,
                (n - i - j) / n,
            )
            for i in range(n + 1)
            for j in range(n - i + 1)
        ],
        dtype=float,
    )


def choice_log_likelihood(
    grid,
    deltas,
    kappa,
):
    scores = grid @ deltas.T

    alt_log_sum = logsumexp(
        -kappa * scores,
        axis=1,
    )

    return -np.logaddexp(
        0.0,
        alt_log_sum,
    )


def solve_shared_exact(deltas):
    deltas = np.asarray(
        deltas,
        dtype=float,
    )

    c = np.asarray(
        [
            0.0,
            0.0,
            0.0,
            -1.0,
        ]
    )

    A_ub = np.column_stack(
        (
            -deltas,
            np.ones(
                len(deltas)
            ),
        )
    )

    b_ub = np.zeros(
        len(deltas)
    )

    A_eq = np.asarray(
        [
            [
                1.0,
                1.0,
                1.0,
                0.0,
            ]
        ]
    )

    b_eq = np.asarray(
        [1.0]
    )

    bounds = [
        (0.0, 1.0),
        (0.0, 1.0),
        (0.0, 1.0),
        (None, None),
    ]

    result = linprog(
        c=c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )

    if not result.success:
        raise RuntimeError(
            result.message
        )

    rho = float(
        result.x[3]
    )

    if rho > TOL:
        cls = "strict"
    elif rho >= -TOL:
        cls = "weak"
    else:
        cls = "unsupported"

    return (
        cls,
        rho,
        result.x[:3],
    )


def log_predictive(
    log_posterior,
    log_likelihood,
):
    return float(
        logsumexp(
            log_posterior
            + log_likelihood
        )
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--preference-pairs",
        required=True,
    )

    ap.add_argument(
        "--irl-summary",
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

    args = ap.parse_args()

    pair_path = Path(
        args.preference_pairs
    )

    summary_path = Path(
        args.irl_summary
    )

    out_dir = Path(
        args.out_dir
    )

    for path in (
        pair_path,
        summary_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(
                path
            )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = read_csv(
        pair_path
    )

    irl_summary = json.loads(
        summary_path.read_text()
    )

    kappa = float(
        irl_summary[
            "selected_kappa"
        ]
    )

    grid = build_simplex_grid(
        args.simplex_step
    )

    grid_count = len(grid)

    log_uniform = np.full(
        grid_count,
        -np.log(grid_count),
        dtype=float,
    )

    # ========================================================
    # Reconstruct one event per mission decision.
    # ========================================================

    by_decision = defaultdict(list)

    for row in rows:
        by_decision[
            int(
                row["decision_index"]
            )
        ].append(row)

    events = []

    for decision_index, decision_rows in (
        by_decision.items()
    ):
        decision_rows = sorted(
            decision_rows,
            key=lambda r:
                r[
                    "alternative_beta_name"
                ],
        )

        first = decision_rows[0]

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
                for row in decision_rows
            ],
            dtype=float,
        )

        alt_names = tuple(
            row[
                "alternative_beta_name"
            ]
            for row in decision_rows
        )

        events.append(
            {
                "decision_index":
                    decision_index,

                "context_id":
                    first[
                        "context_id"
                    ],

                "terrain":
                    first[
                        "terrain"
                    ],

                "mission":
                    first[
                        "mission"
                    ],

                "selected_beta_name":
                    first[
                        "selected_beta_name"
                    ],

                "alternative_beta_names":
                    alt_names,

                "deltas":
                    deltas,
            }
        )

    # ========================================================
    # Deduplicate critical-budget repetitions inside each
    # exact context x mission.
    # ========================================================

    unique = {}

    for event in events:
        key = (
            event[
                "context_id"
            ],
            event[
                "mission"
            ],
            event[
                "selected_beta_name"
            ],
            event[
                "alternative_beta_names"
            ],
        )

        if key not in unique:
            unique[key] = event

        else:
            if not np.allclose(
                unique[key][
                    "deltas"
                ],
                event[
                    "deltas"
                ],
                atol=1e-12,
                rtol=0.0,
            ):
                raise RuntimeError(
                    "Duplicate signature with "
                    "different deltas."
                )

    # ========================================================
    # context x mission -> unique events
    # ========================================================

    by_context_mission = defaultdict(
        list
    )

    for event in unique.values():
        by_context_mission[
            (
                event[
                    "context_id"
                ],
                event[
                    "mission"
                ],
            )
        ].append(event)

    if len(by_context_mission) != 162:
        raise RuntimeError(
            "Expected 162 exact context x mission "
            f"groups, got {len(by_context_mission)}"
        )

    # Precompute one averaged log-likelihood per exact
    # context x mission.
    #
    # Averaging gives every exact context equal total weight,
    # regardless of how many unique mission queries it has.
    context_ll = {}

    context_meta = {}

    for key, group_events in (
        by_context_mission.items()
    ):
        ll = np.zeros(
            grid_count,
            dtype=float,
        )

        for event in group_events:
            ll += choice_log_likelihood(
                grid,
                event[
                    "deltas"
                ],
                kappa,
            )

        ll /= len(
            group_events
        )

        context_ll[key] = ll

        context_meta[key] = {
            "terrain":
                group_events[0][
                    "terrain"
                ],

            "signature_count":
                len(
                    group_events
                ),
        }

    # ========================================================
    # observable terrain x mission
    # ========================================================

    observable = defaultdict(list)

    for key in context_ll:
        context_id, mission = key

        terrain = context_meta[
            key
        ][
            "terrain"
        ]

        observable[
            (
                terrain,
                mission,
            )
        ].append(key)

    expected = {
        (
            terrain,
            mission,
        )
        for terrain in TERRAINS
        for mission in MISSIONS
    }

    if set(observable) != expected:
        raise RuntimeError(
            "Observable-group mismatch."
        )

    records = []
    loo_records = []

    for terrain, mission in sorted(
        observable
    ):
        context_keys = observable[
            (
                terrain,
                mission,
            )
        ]

        if len(context_keys) != 18:
            raise RuntimeError(
                f"{terrain}/{mission}: "
                f"expected 18 contexts"
            )

        # ----------------------------------------------------
        # Exact shared support across all preference pairs.
        # ----------------------------------------------------

        all_deltas = []

        for context_key in context_keys:
            for event in (
                by_context_mission[
                    context_key
                ]
            ):
                all_deltas.extend(
                    event[
                        "deltas"
                    ]
                )

        (
            shared_class,
            shared_rho,
            shared_beta,
        ) = solve_shared_exact(
            all_deltas
        )

        # ----------------------------------------------------
        # Shared posterior.
        #
        # Each exact context contributes one averaged
        # likelihood term.
        # ----------------------------------------------------

        pooled_ll = np.zeros(
            grid_count,
            dtype=float,
        )

        for context_key in context_keys:
            pooled_ll += context_ll[
                context_key
            ]

        log_post = (
            pooled_ll
            + log_uniform
        )

        log_post -= logsumexp(
            log_post
        )

        posterior = np.exp(
            log_post
        )

        mean_beta = (
            posterior
            @ grid
        )

        entropy = float(
            -np.sum(
                posterior
                * log_post
            )
        )

        entropy_norm = (
            entropy
            / np.log(
                grid_count
            )
        )

        # ----------------------------------------------------
        # Leave-one-exact-context-out prediction.
        # ----------------------------------------------------

        deltas_vs_uniform = []

        shared_scores = []
        uniform_scores = []

        for held_key in context_keys:
            train_ll = np.zeros(
                grid_count,
                dtype=float,
            )

            for train_key in context_keys:
                if train_key == held_key:
                    continue

                train_ll += context_ll[
                    train_key
                ]

            loo_log_post = (
                train_ll
                + log_uniform
            )

            loo_log_post -= logsumexp(
                loo_log_post
            )

            held_events = (
                by_context_mission[
                    held_key
                ]
            )

            held_shared = []
            held_uniform = []

            for event in held_events:
                event_ll = (
                    choice_log_likelihood(
                        grid,
                        event[
                            "deltas"
                        ],
                        kappa,
                    )
                )

                shared_score = log_predictive(
                    loo_log_post,
                    event_ll,
                )

                uniform_score = log_predictive(
                    log_uniform,
                    event_ll,
                )

                held_shared.append(
                    shared_score
                )

                held_uniform.append(
                    uniform_score
                )

            shared_mean = float(
                np.mean(
                    held_shared
                )
            )

            uniform_mean = float(
                np.mean(
                    held_uniform
                )
            )

            delta = (
                shared_mean
                - uniform_mean
            )

            deltas_vs_uniform.append(
                delta
            )

            shared_scores.append(
                shared_mean
            )

            uniform_scores.append(
                uniform_mean
            )

            loo_records.append(
                {
                    "terrain":
                        terrain,

                    "mission":
                        mission,

                    "held_context_id":
                        held_key[0],

                    "held_signature_count":
                        len(
                            held_events
                        ),

                    "shared_log_predictive":
                        shared_mean,

                    "uniform_log_predictive":
                        uniform_mean,

                    "delta_log_predictive":
                        delta,
                }
            )

        delta_arr = np.asarray(
            deltas_vs_uniform,
            dtype=float,
        )

        records.append(
            {
                "terrain":
                    terrain,

                "mission":
                    mission,

                "exact_context_count":
                    18,

                "unique_signature_count":
                    sum(
                        len(
                            by_context_mission[
                                key
                            ]
                        )
                        for key in context_keys
                    ),

                "shared_exact_class":
                    shared_class,

                "shared_exact_rho":
                    shared_rho,

                "shared_exact_beta_motion":
                    float(
                        shared_beta[0]
                    ),

                "shared_exact_beta_stability":
                    float(
                        shared_beta[1]
                    ),

                "shared_exact_beta_energy":
                    float(
                        shared_beta[2]
                    ),

                "shared_posterior_mean_motion":
                    float(
                        mean_beta[0]
                    ),

                "shared_posterior_mean_stability":
                    float(
                        mean_beta[1]
                    ),

                "shared_posterior_mean_energy":
                    float(
                        mean_beta[2]
                    ),

                "shared_posterior_entropy_norm":
                    entropy_norm,

                "loo_delta_log_predictive_mean":
                    float(
                        np.mean(
                            delta_arr
                        )
                    ),

                "loo_delta_log_predictive_median":
                    float(
                        np.median(
                            delta_arr
                        )
                    ),

                "loo_delta_log_predictive_min":
                    float(
                        np.min(
                            delta_arr
                        )
                    ),

                "loo_delta_log_predictive_max":
                    float(
                        np.max(
                            delta_arr
                        )
                    ),

                "loo_contexts_improved":
                    int(
                        np.sum(
                            delta_arr > 0.0
                        )
                    ),

                "loo_context_count":
                    len(
                        delta_arr
                    ),
            }
        )

    group_path = (
        out_dir
        / "shared_latent_groups.csv"
    )

    loo_path = (
        out_dir
        / "shared_latent_loo.csv"
    )

    with group_path.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                records[0]
            ),
        )

        writer.writeheader()
        writer.writerows(
            records
        )

    with loo_path.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                loo_records[0]
            ),
        )

        writer.writeheader()
        writer.writerows(
            loo_records
        )

    summary = {
        "schema":
            (
                "icra27_phase2b_shared_latent_"
                "preference_v0"
            ),

        "hypothesis":
            (
                "one latent beta per observable "
                "terrain class x mission family"
            ),

        "selected_kappa":
            kappa,

        "observable_group_count":
            len(
                records
            ),

        "exact_contexts_per_group":
            18,

        "primary_test":
            (
                "leave-one-exact-context-out "
                "posterior predictive gain over "
                "uniform beta prior"
            ),

        "interpretation_note":
            (
                "Posterior entropy alone is not used "
                "to accept a shared latent preference; "
                "cross-context predictive performance "
                "is the key diagnostic."
            ),
    }

    out_summary = (
        out_dir
        / "summary.json"
    )

    out_summary.write_text(
        json.dumps(
            summary,
            indent=2,
        )
        + "\n"
    )

    print("=" * 118)
    print(
        "ICRA27 PHASE-2B SHARED-LATENT "
        "PREFERENCE PREDICTIVE AUDIT V0"
    )
    print("=" * 118)

    for row in records:
        print(
            f"{row['terrain']:<14} "
            f"{row['mission']:<10} "
            f"exact={row['shared_exact_class']:<11} "
            f"rho={row['shared_exact_rho']:+.6f}  "
            f"H={row['shared_posterior_entropy_norm']:.4f}  "
            f"LOO ΔLL(mean/med/min)="
            f"{row['loo_delta_log_predictive_mean']:+.4f}/"
            f"{row['loo_delta_log_predictive_median']:+.4f}/"
            f"{row['loo_delta_log_predictive_min']:+.4f}  "
            f"improved="
            f"{row['loo_contexts_improved']}/18"
        )

    print()
    print("groups :", group_path)
    print("LOO    :", loo_path)
    print("summary:", out_summary)

    print()
    print(
        "[ICRA27] Phase-2B shared-latent "
        "preference predictive audit: COMPUTE PASS"
    )


if __name__ == "__main__":
    main()
