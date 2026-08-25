#!/usr/bin/env python3

import argparse
import csv
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np


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


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def kl_divergence(p, q):
    mask = p > 0.0

    return float(
        np.sum(
            p[mask]
            * (
                np.log(p[mask])
                - np.log(q[mask])
            )
        )
    )


def js_divergence(p, q):
    m = 0.5 * (p + q)

    return 0.5 * (
        kl_divergence(p, m)
        + kl_divergence(q, m)
    )


def overlap_coefficient(p, q):
    return float(
        np.sum(
            np.minimum(p, q)
        )
    )


def entropy(p):
    mask = p > 0.0

    return float(
        -np.sum(
            p[mask]
            * np.log(p[mask])
        )
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--posterior-csv",
        required=True,
    )

    ap.add_argument(
        "--posterior-npz",
        required=True,
    )

    ap.add_argument(
        "--out-dir",
        required=True,
    )

    args = ap.parse_args()

    csv_path = Path(
        args.posterior_csv
    )

    npz_path = Path(
        args.posterior_npz
    )

    out_dir = Path(
        args.out_dir
    )

    for path in (
        csv_path,
        npz_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = read_csv(csv_path)

    data = np.load(npz_path)

    grid = np.asarray(
        data["beta_grid"],
        dtype=float,
    )

    posterior_matrix = np.asarray(
        data["posterior_probability"],
        dtype=float,
    )

    if len(rows) != 162:
        raise RuntimeError(
            f"Expected 162 posterior rows, "
            f"got {len(rows)}"
        )

    if posterior_matrix.shape != (
        162,
        len(grid),
    ):
        raise RuntimeError(
            "Posterior matrix shape mismatch: "
            f"{posterior_matrix.shape}"
        )

    # --------------------------------------------------------
    # Reorder explicitly by posterior_id and validate.
    # --------------------------------------------------------

    ordered = [None] * len(rows)

    for row in rows:
        posterior_id = int(
            row["posterior_id"]
        )

        if not (
            0 <= posterior_id < len(rows)
        ):
            raise RuntimeError(
                f"Invalid posterior_id={posterior_id}"
            )

        if ordered[posterior_id] is not None:
            raise RuntimeError(
                f"Duplicate posterior_id={posterior_id}"
            )

        ordered[posterior_id] = row

    if any(
        row is None
        for row in ordered
    ):
        raise RuntimeError(
            "Missing posterior IDs."
        )

    row_sums = np.sum(
        posterior_matrix,
        axis=1,
    )

    if not np.allclose(
        row_sums,
        1.0,
        atol=1e-5,
    ):
        raise RuntimeError(
            "Posterior rows do not sum to one."
        )

    # Avoid numerical zero only where a distribution is used
    # as the denominator of KL.
    tiny = np.finfo(float).tiny

    groups = defaultdict(list)

    for posterior_id, row in enumerate(
        ordered
    ):
        key = (
            row["terrain"],
            row["mission"],
        )

        groups[key].append(
            posterior_id
        )

    expected_keys = {
        (
            terrain,
            mission,
        )
        for terrain in TERRAINS
        for mission in MISSIONS
    }

    if set(groups) != expected_keys:
        raise RuntimeError(
            "Observable group-key mismatch."
        )

    for key, ids in groups.items():
        if len(ids) != 18:
            raise RuntimeError(
                f"{key}: expected 18 exact contexts, "
                f"got {len(ids)}"
            )

    pair_records = []
    group_records = []

    mixture_matrix = np.empty(
        (
            len(expected_keys),
            len(grid),
        ),
        dtype=np.float32,
    )

    ordered_group_keys = sorted(groups)

    for mixture_id, key in enumerate(
        ordered_group_keys
    ):
        terrain, mission = key

        ids = groups[key]

        posteriors = posterior_matrix[
            ids
        ]

        mixture = np.mean(
            posteriors,
            axis=0,
        )

        mixture /= np.sum(
            mixture
        )

        mixture_matrix[
            mixture_id
        ] = mixture.astype(
            np.float32
        )

        js_values = []
        overlap_values = []

        for a, b in combinations(
            ids,
            2,
        ):
            p_a = posterior_matrix[a]
            p_b = posterior_matrix[b]

            js = js_divergence(
                p_a,
                p_b,
            )

            overlap = overlap_coefficient(
                p_a,
                p_b,
            )

            js_values.append(js)
            overlap_values.append(overlap)

            pair_records.append(
                {
                    "terrain":
                        terrain,

                    "mission":
                        mission,

                    "posterior_id_a":
                        a,

                    "context_id_a":
                        ordered[a][
                            "context_id"
                        ],

                    "posterior_id_b":
                        b,

                    "context_id_b":
                        ordered[b][
                            "context_id"
                        ],

                    "js_divergence_nats":
                        js,

                    "posterior_overlap":
                        overlap,
                }
            )

        js_values = np.asarray(
            js_values,
            dtype=float,
        )

        overlap_values = np.asarray(
            overlap_values,
            dtype=float,
        )

        # KL from each exact posterior to the best arithmetic
        # mixture representation available to an observable
        # terrain-class x mission selector.
        mixture_safe = np.maximum(
            mixture,
            tiny,
        )

        mixture_safe /= np.sum(
            mixture_safe
        )

        kl_to_mixture = np.asarray(
            [
                kl_divergence(
                    posterior_matrix[i],
                    mixture_safe,
                )
                for i in ids
            ],
            dtype=float,
        )

        exact_entropies = np.asarray(
            [
                entropy(
                    posterior_matrix[i]
                )
                for i in ids
            ],
            dtype=float,
        )

        max_entropy = float(
            np.log(
                len(grid)
            )
        )

        mixture_entropy = entropy(
            mixture
        )

        mixture_mean = (
            mixture
            @ grid
        )

        exact_means = np.asarray(
            [
                posterior_matrix[i]
                @ grid
                for i in ids
            ],
            dtype=float,
        )

        mean_centroid = np.mean(
            exact_means,
            axis=0,
        )

        mean_l2 = np.linalg.norm(
            exact_means
            - mean_centroid[
                None,
                :
            ],
            axis=1,
        )

        component_min = np.min(
            exact_means,
            axis=0,
        )

        component_max = np.max(
            exact_means,
            axis=0,
        )

        group_records.append(
            {
                "mixture_id":
                    mixture_id,

                "terrain":
                    terrain,

                "mission":
                    mission,

                "exact_context_count":
                    len(ids),

                "pair_count":
                    len(js_values),

                "js_mean_nats":
                    float(
                        np.mean(js_values)
                    ),

                "js_median_nats":
                    float(
                        np.median(js_values)
                    ),

                "js_q90_nats":
                    float(
                        np.quantile(
                            js_values,
                            0.90,
                        )
                    ),

                "js_max_nats":
                    float(
                        np.max(js_values)
                    ),

                "overlap_mean":
                    float(
                        np.mean(
                            overlap_values
                        )
                    ),

                "overlap_median":
                    float(
                        np.median(
                            overlap_values
                        )
                    ),

                "overlap_min":
                    float(
                        np.min(
                            overlap_values
                        )
                    ),

                "kl_to_mixture_mean_nats":
                    float(
                        np.mean(
                            kl_to_mixture
                        )
                    ),

                "kl_to_mixture_median_nats":
                    float(
                        np.median(
                            kl_to_mixture
                        )
                    ),

                "kl_to_mixture_max_nats":
                    float(
                        np.max(
                            kl_to_mixture
                        )
                    ),

                "exact_entropy_norm_mean":
                    float(
                        np.mean(
                            exact_entropies
                        )
                        / max_entropy
                    ),

                "mixture_entropy_norm":
                    float(
                        mixture_entropy
                        / max_entropy
                    ),

                "mixture_mean_motion":
                    float(
                        mixture_mean[0]
                    ),

                "mixture_mean_stability":
                    float(
                        mixture_mean[1]
                    ),

                "mixture_mean_energy":
                    float(
                        mixture_mean[2]
                    ),

                "posterior_mean_l2_dispersion_mean":
                    float(
                        np.mean(
                            mean_l2
                        )
                    ),

                "posterior_mean_l2_dispersion_max":
                    float(
                        np.max(
                            mean_l2
                        )
                    ),

                "mean_beta_motion_min":
                    float(
                        component_min[0]
                    ),

                "mean_beta_motion_max":
                    float(
                        component_max[0]
                    ),

                "mean_beta_stability_min":
                    float(
                        component_min[1]
                    ),

                "mean_beta_stability_max":
                    float(
                        component_max[1]
                    ),

                "mean_beta_energy_min":
                    float(
                        component_min[2]
                    ),

                "mean_beta_energy_max":
                    float(
                        component_max[2]
                    ),
            }
        )

    pair_path = (
        out_dir
        / "observable_context_pairwise.csv"
    )

    group_path = (
        out_dir
        / "observable_context_compatibility.csv"
    )

    mixture_path = (
        out_dir
        / "observable_context_mixture_posteriors.npz"
    )

    with pair_path.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                pair_records[0]
            ),
        )

        writer.writeheader()
        writer.writerows(
            pair_records
        )

    with group_path.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                group_records[0]
            ),
        )

        writer.writeheader()
        writer.writerows(
            group_records
        )

    np.savez_compressed(
        mixture_path,
        beta_grid=grid.astype(
            np.float32
        ),
        mixture_probability=(
            mixture_matrix
        ),
        mixture_terrain=np.asarray(
            [
                key[0]
                for key in ordered_group_keys
            ]
        ),
        mixture_mission=np.asarray(
            [
                key[1]
                for key in ordered_group_keys
            ]
        ),
    )

    summary = {
        "schema":
            (
                "icra27_phase2b_observable_context_"
                "compatibility_v0"
            ),

        "observable_selector_input":
            "terrain class x mission family",

        "observable_group_count":
            len(
                group_records
            ),

        "exact_contexts_per_group":
            18,

        "pairwise_comparisons_per_group":
            153,

        "interpretation":
            (
                "Measures whether exact-context IRL "
                "posteriors are mutually compatible "
                "when the deployed selector can only "
                "observe terrain class and mission."
            ),

        "important_note":
            (
                "No scientific PASS threshold is "
                "defined post hoc. Results are used "
                "to decide whether richer context "
                "features are required."
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

    print("=" * 112)
    print(
        "ICRA27 PHASE-2B OBSERVABLE-CONTEXT "
        "PREFERENCE COMPATIBILITY AUDIT V0"
    )
    print("=" * 112)

    print(
        "observable groups      :",
        len(group_records),
    )

    print(
        "exact contexts / group :",
        18,
    )

    print(
        "pairs / group          :",
        153,
    )

    print()

    for row in group_records:
        print(
            f"{row['terrain']:<14} "
            f"{row['mission']:<10} "
            f"JS(mean/90/max)="
            f"{row['js_mean_nats']:.4f}/"
            f"{row['js_q90_nats']:.4f}/"
            f"{row['js_max_nats']:.4f}  "
            f"overlap(mean/min)="
            f"{row['overlap_mean']:.3f}/"
            f"{row['overlap_min']:.3f}  "
            f"KL->mix(mean/max)="
            f"{row['kl_to_mixture_mean_nats']:.4f}/"
            f"{row['kl_to_mixture_max_nats']:.4f}  "
            f"Hmix="
            f"{row['mixture_entropy_norm']:.4f}"
        )

    print()
    print(
        "pairwise :",
        pair_path,
    )

    print(
        "groups   :",
        group_path,
    )

    print(
        "mixtures :",
        mixture_path,
    )

    print(
        "summary  :",
        summary_path,
    )

    print()
    print(
        "[ICRA27] Phase-2B observable-context "
        "compatibility audit: COMPUTE PASS"
    )


if __name__ == "__main__":
    main()
