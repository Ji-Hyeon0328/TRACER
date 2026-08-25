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


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--preference-pairs",
        required=True,
    )

    ap.add_argument(
        "--signature-posteriors",
        required=True,
    )

    ap.add_argument(
        "--single-summary",
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

    signature_path = Path(
        args.signature_posteriors
    )

    single_summary_path = Path(
        args.single_summary
    )

    out_dir = Path(
        args.out_dir
    )

    for path in (
        pair_path,
        signature_path,
        single_summary_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    pairs = read_csv(
        pair_path
    )

    signatures_csv = read_csv(
        signature_path
    )

    single_summary = json.loads(
        single_summary_path.read_text()
    )

    kappa = float(
        single_summary[
            "selected_kappa"
        ]
    )

    # --------------------------------------------------------
    # Reconstruct simplex grid exactly as OS-9C2a.
    # --------------------------------------------------------

    step = float(
        args.simplex_step
    )

    n = int(
        round(
            1.0 / step
        )
    )

    if not np.isclose(
        n * step,
        1.0,
        atol=1e-12,
    ):
        raise RuntimeError(
            "simplex step must divide 1."
        )

    grid = np.asarray(
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

    grid_count = len(
        grid
    )

    # --------------------------------------------------------
    # Reconstruct one choice event per decision.
    # --------------------------------------------------------

    by_decision = defaultdict(list)

    for row in pairs:
        by_decision[
            int(
                row["decision_index"]
            )
        ].append(row)

    decision_events = {}

    for decision_index, rows in (
        by_decision.items()
    ):
        rows = sorted(
            rows,
            key=lambda r:
                r["alternative_beta_name"],
        )

        first = rows[0]

        deltas = np.asarray(
            [
                [
                    float(
                        row["delta_cost_motion"]
                    ),
                    float(
                        row[
                            "delta_cost_stability"
                        ]
                    ),
                    float(
                        row["delta_cost_energy"]
                    ),
                ]
                for row in rows
            ],
            dtype=float,
        )

        alt_names = tuple(
            row["alternative_beta_name"]
            for row in rows
        )

        decision_events[
            decision_index
        ] = {
            "context_id":
                first["context_id"],

            "terrain":
                first["terrain"],

            "seed":
                int(first["seed"]),

            "mission":
                first["mission"],

            "selected_beta_name":
                first["selected_beta_name"],

            "alternative_beta_names":
                alt_names,

            "deltas":
                deltas,
        }

    # --------------------------------------------------------
    # Recover UNIQUE signature IDs using same definition as
    # OS-9C2a:
    #
    # context, mission, selected source candidate,
    # feasible alternative candidate identities.
    # --------------------------------------------------------

    signature_key_to_id = {}

    for row in signatures_csv:
        key = (
            row["context_id"],
            row["mission"],
            row["selected_beta_name"],
            tuple(
                x
                for x in row[
                    "alternative_beta_names"
                ].split("|")
                if x
            ),
        )

        signature_key_to_id[
            key
        ] = int(
            row["signature_id"]
        )

    signature_events = {}

    for event in decision_events.values():
        key = (
            event["context_id"],
            event["mission"],
            event[
                "selected_beta_name"
            ],
            event[
                "alternative_beta_names"
            ],
        )

        if key not in signature_key_to_id:
            raise RuntimeError(
                f"Could not match signature: {key}"
            )

        signature_id = (
            signature_key_to_id[key]
        )

        if signature_id not in signature_events:
            signature_events[
                signature_id
            ] = event

        else:
            if not np.allclose(
                signature_events[
                    signature_id
                ]["deltas"],
                event["deltas"],
                atol=1e-12,
                rtol=0.0,
            ):
                raise RuntimeError(
                    "Identical signature has "
                    "different deltas."
                )

    if len(signature_events) != len(
        signatures_csv
    ):
        raise RuntimeError(
            "Unique-signature reconstruction "
            "count mismatch: "
            f"{len(signature_events)} vs "
            f"{len(signatures_csv)}"
        )

    # --------------------------------------------------------
    # Group unique preference queries by latent:
    #
    #      exact context x mission family.
    # --------------------------------------------------------

    by_context_mission = defaultdict(
        list
    )

    for signature_id, event in (
        signature_events.items()
    ):
        by_context_mission[
            (
                event["context_id"],
                event["mission"],
            )
        ].append(
            (
                signature_id,
                event,
            )
        )

    if len(by_context_mission) != 162:
        raise RuntimeError(
            "Expected 162 context x mission "
            f"groups, got {len(by_context_mission)}"
        )

    # --------------------------------------------------------
    # Aggregated posterior.
    #
    # Each UNIQUE choice-set query contributes once.
    #
    # IMPORTANT:
    # posterior width is interpreted as preference
    # identifiability under the synthetic query set,
    # not as calibrated confidence from 760 independent
    # human demonstrations.
    # --------------------------------------------------------

    posterior_matrix = np.empty(
        (
            len(by_context_mission),
            grid_count,
        ),
        dtype=np.float32,
    )

    records = []

    ordered_groups = sorted(
        by_context_mission
    )

    for posterior_id, group_key in enumerate(
        ordered_groups
    ):
        context_id, mission = group_key

        events = by_context_mission[
            group_key
        ]

        log_likelihood = np.zeros(
            grid_count,
            dtype=float,
        )

        source_beta_counts = Counter()

        strict_signature_count = 0

        for signature_id, event in events:
            scores = (
                grid
                @ event["deltas"].T
            )

            alt_log_sum = logsumexp(
                -kappa * scores,
                axis=1,
            )

            choice_log_likelihood = (
                -np.logaddexp(
                    0.0,
                    alt_log_sum,
                )
            )

            # One unique preference query = one factor.
            log_likelihood += (
                choice_log_likelihood
            )

            source_beta_counts[
                event[
                    "selected_beta_name"
                ]
            ] += 1

            signature_row = next(
                row
                for row in signatures_csv
                if int(
                    row["signature_id"]
                ) == signature_id
            )

            if signature_row[
                "exact_support_class"
            ] in {
                "strict",
                "weak",
            }:
                strict_signature_count += 1

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
            posterior_id
        ] = posterior.astype(
            np.float32
        )

        mean = posterior @ grid

        centered = (
            grid
            - mean
        )

        covariance = (
            centered.T
            @ (
                centered
                * posterior[:, None]
            )
        )

        std = np.sqrt(
            np.maximum(
                np.diag(covariance),
                0.0,
            )
        )

        map_index = int(
            np.argmax(posterior)
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

        info_gain = (
            max_entropy
            - entropy
        )

        effective_fraction = float(
            np.exp(entropy)
            / grid_count
        )

        first_event = events[0][1]

        record = {
            "posterior_id":
                posterior_id,

            "context_id":
                context_id,

            "terrain":
                first_event[
                    "terrain"
                ],

            "seed":
                first_event[
                    "seed"
                ],

            "mission":
                mission,

            "unique_signature_count":
                len(events),

            "expressible_signature_count":
                strict_signature_count,

            "posterior_mean_motion":
                float(mean[0]),

            "posterior_mean_stability":
                float(mean[1]),

            "posterior_mean_energy":
                float(mean[2]),

            "posterior_std_motion":
                float(std[0]),

            "posterior_std_stability":
                float(std[1]),

            "posterior_std_energy":
                float(std[2]),

            "posterior_map_motion":
                float(map_beta[0]),

            "posterior_map_stability":
                float(map_beta[1]),

            "posterior_map_energy":
                float(map_beta[2]),

            "posterior_entropy_nats":
                entropy,

            "posterior_entropy_normalized":
                entropy_norm,

            "posterior_information_gain_nats":
                info_gain,

            "posterior_effective_fraction":
                effective_fraction,

            "source_balanced_count":
                source_beta_counts[
                    "balanced"
                ],

            "source_motion_count":
                source_beta_counts[
                    "motion"
                ],

            "source_stability_count":
                source_beta_counts[
                    "stability"
                ],

            "source_energy_count":
                source_beta_counts[
                    "energy"
                ],
        }

        records.append(record)

    # --------------------------------------------------------
    # Save.
    # --------------------------------------------------------

    csv_path = (
        out_dir
        / "context_mission_irl_posteriors.csv"
    )

    npz_path = (
        out_dir
        / "context_mission_irl_posterior_grid.npz"
    )

    with csv_path.open(
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
        writer.writerows(records)

    np.savez_compressed(
        npz_path,
        beta_grid=grid.astype(
            np.float32
        ),
        posterior_probability=(
            posterior_matrix
        ),
    )

    # --------------------------------------------------------
    # Diagnostics.
    # --------------------------------------------------------

    entropy_by_mission = {}

    for mission in (
        "fast",
        "stable",
        "efficient",
    ):
        subset = [
            row
            for row in records
            if row["mission"] == mission
        ]

        values = np.asarray(
            [
                row[
                    "posterior_entropy_normalized"
                ]
                for row in subset
            ],
            dtype=float,
        )

        gains = np.asarray(
            [
                row[
                    "posterior_information_gain_nats"
                ]
                for row in subset
            ],
            dtype=float,
        )

        signature_counts = np.asarray(
            [
                row[
                    "unique_signature_count"
                ]
                for row in subset
            ],
            dtype=float,
        )

        entropy_by_mission[
            mission
        ] = {
            "count":
                len(subset),

            "entropy_mean":
                float(
                    np.mean(values)
                ),

            "entropy_median":
                float(
                    np.median(values)
                ),

            "entropy_min":
                float(
                    np.min(values)
                ),

            "entropy_max":
                float(
                    np.max(values)
                ),

            "information_gain_mean":
                float(
                    np.mean(gains)
                ),

            "unique_signature_mean":
                float(
                    np.mean(
                        signature_counts
                    )
                ),
        }

    all_entropy = np.asarray(
        [
            row[
                "posterior_entropy_normalized"
            ]
            for row in records
        ],
        dtype=float,
    )

    summary = {
        "schema":
            (
                "icra27_phase2b_context_mission_"
                "irl_posterior_v0"
            ),

        "latent_preference_unit":
            "exact context x mission family",

        "context_mission_count":
            len(records),

        "unique_signature_count":
            len(signature_events),

        "selected_kappa":
            kappa,

        "simplex_step":
            step,

        "simplex_grid_count":
            grid_count,

        "posterior_entropy_normalized":
            {
                "mean":
                    float(
                        np.mean(
                            all_entropy
                        )
                    ),

                "median":
                    float(
                        np.median(
                            all_entropy
                        )
                    ),

                "min":
                    float(
                        np.min(
                            all_entropy
                        )
                    ),

                "max":
                    float(
                        np.max(
                            all_entropy
                        )
                    ),
            },

        "by_mission":
            entropy_by_mission,

        "interpretation_note":
            (
                "Posterior width measures reward "
                "identifiability under multiple unique "
                "synthetic mission-local choice queries. "
                "It is not statistical confidence from "
                "independent human demonstrations."
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

    print("=" * 92)
    print(
        "ICRA27 PHASE-2B CONTEXT x MISSION "
        "AGGREGATED IRL POSTERIOR V0"
    )
    print("=" * 92)

    print(
        "context x mission     :",
        len(records),
    )

    print(
        "unique signatures     :",
        len(signature_events),
    )

    print(
        "selected kappa        :",
        kappa,
    )

    print()
    print(
        "AGGREGATED NORMALIZED ENTROPY"
    )

    print(
        "  overall "
        f"mean={np.mean(all_entropy):.4f} "
        f"median={np.median(all_entropy):.4f} "
        f"min={np.min(all_entropy):.4f} "
        f"max={np.max(all_entropy):.4f}"
    )

    print()

    for mission in (
        "fast",
        "stable",
        "efficient",
    ):
        item = entropy_by_mission[
            mission
        ]

        print(
            f"  {mission:<10} "
            f"entropy_mean="
            f"{item['entropy_mean']:.4f} "
            f"median="
            f"{item['entropy_median']:.4f} "
            f"info_gain="
            f"{item['information_gain_mean']:.4f} "
            f"signatures/group="
            f"{item['unique_signature_mean']:.2f}"
        )

    print()
    print("csv      :", csv_path)
    print("posterior:", npz_path)
    print("summary  :", summary_path)

    print()
    print(
        "[ICRA27] Phase-2B context x mission "
        "aggregated IRL posterior: PASS"
    )


if __name__ == "__main__":
    main()
