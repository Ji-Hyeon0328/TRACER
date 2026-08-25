from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]

LABEL_ROOT = (
    ROOT
    / "results/icra27"
    / "os_t5p5d_expanded_preference_labels_v0"
)

LABEL_CSV = (
    LABEL_ROOT
    / "preference_to_beta_labels.csv"
)

ATLAS_CSV = (
    ROOT
    / "results/icra27"
    / "os_t5p5c_expanded_physical_atlas_v0"
    / "physical_beta_response_atlas.csv"
)

LOCO_MANIFEST = (
    ROOT
    / "results/icra27"
    / "os_t5p5e1_selector_loco_v0"
    / "selector_loco_manifest.json"
)

OUT_DIR = (
    ROOT
    / "results/icra27"
    / "os_t5p5e1b_context_identifiability_v0"
)

OUT_NN = (
    OUT_DIR
    / "nearest_context_preference_transfer.csv"
)

OUT_PAIRWISE = (
    OUT_DIR
    / "rough_context_pairwise_similarity.csv"
)

OUT_MANIFEST = (
    OUT_DIR
    / "identifiability_manifest.json"
)


CONTEXT_COLUMNS = (
    "context_friction_mu",
    "context_height_std_m",
    "context_height_relief_p95_p05_m",
    "context_slope_rms",
    "context_slope_q95",
)

OBJECTIVE_COLUMNS = (
    "J_motion_s_per_m",
    "J_stability",
    "J_energy_j_per_m",
)

RHO = 0.01
EPS = 1.0e-12
TOL = 1.0e-10


def finite(x):
    y = float(x)

    if not math.isfinite(y):
        raise ValueError(
            f"Non-finite value: {x!r}"
        )

    return y


def read_csv(path):
    with path.open(newline="") as f:
        return list(
            csv.DictReader(f)
        )


def write_csv(path, rows):
    if not rows:
        raise ValueError(
            f"No rows for {path}"
        )

    fields = []

    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)

    with path.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(rows)


def pearson(x, y):
    x = np.asarray(
        x,
        dtype=np.float64,
    )

    y = np.asarray(
        y,
        dtype=np.float64,
    )

    x = x - np.mean(x)
    y = y - np.mean(y)

    denom = (
        np.linalg.norm(x)
        * np.linalg.norm(y)
    )

    if denom <= EPS:
        return float("nan")

    return float(
        np.dot(x, y)
        / denom
    )


def main():
    if OUT_DIR.exists():
        raise SystemExit(
            f"REFUSING TO OVERWRITE: {OUT_DIR}"
        )

    for path in (
        LABEL_CSV,
        ATLAS_CSV,
        LOCO_MANIFEST,
    ):
        if not path.exists():
            raise FileNotFoundError(
                path
            )

    loco = json.loads(
        LOCO_MANIFEST.read_text()
    )

    if loco.get(
        "status"
    ) != "COMPUTE_PASS":
        raise RuntimeError(
            "T5.5e1 LOCO is not COMPUTE_PASS."
        )

    labels = read_csv(
        LABEL_CSV
    )

    atlas = read_csv(
        ATLAS_CSV
    )

    rough_contexts = []

    first_label_by_context = {}

    for row in labels:
        cid = row[
            "context_id"
        ]

        first_label_by_context.setdefault(
            cid,
            row,
        )

        if (
            cid.startswith("rough_seed_")
            and cid not in rough_contexts
        ):
            rough_contexts.append(
                cid
            )

    if len(rough_contexts) != 18:
        raise RuntimeError(
            f"Expected 18 rough contexts; "
            f"got {len(rough_contexts)}"
        )


    # --------------------------------------------------------
    # Rough-only feature normalization.
    #
    # We intentionally audit geometric identifiability
    # inside the rough family. Friction is constant here,
    # so drop any zero-variance dimensions automatically.
    # --------------------------------------------------------

    context_matrix = np.asarray(
        [
            [
                finite(
                    first_label_by_context[
                        cid
                    ][column]
                )
                for column in (
                    CONTEXT_COLUMNS
                )
            ]
            for cid in rough_contexts
        ],
        dtype=np.float64,
    )

    mean = np.mean(
        context_matrix,
        axis=0,
    )

    std = np.std(
        context_matrix,
        axis=0,
    )

    active = (
        std > EPS
    )

    if np.sum(active) < 1:
        raise RuntimeError(
            "No varying rough context feature."
        )

    z_context = (
        (
            context_matrix[
                :,
                active
            ]
            - mean[
                active
            ]
        )
        / std[
            active
        ]
    )


    # --------------------------------------------------------
    # Atlas lookup and beta order.
    # --------------------------------------------------------

    atlas_lookup = {}

    beta_order = []

    for row in atlas:
        key = (
            row["context_id"],
            row["beta_name"],
        )

        atlas_lookup[key] = row

        if (
            row["context_id"]
            == rough_contexts[0]
        ):
            beta_order.append(
                row["beta_name"]
            )

    if len(beta_order) != 21:
        raise RuntimeError(
            "Expected 21 beta points."
        )


    # --------------------------------------------------------
    # Preference-label lookup.
    # --------------------------------------------------------

    label_lookup = {}

    preference_order = []

    first_context = rough_contexts[0]

    for row in labels:
        key = (
            row["context_id"],
            row["preference_name"],
        )

        label_lookup[key] = row

        if row["context_id"] == first_context:
            preference_order.append(
                row["preference_name"]
            )

    if len(preference_order) != 25:
        raise RuntimeError(
            "Expected 25 preferences."
        )


    # --------------------------------------------------------
    # Context-local Pareto ideal/nadir reconstructed from
    # frozen oracle label source.
    #
    # For NN transfer scoring, use all 21 atlas points with
    # the same regret reference as T5.5d:
    # ideal/nadir over nondominated points.
    # --------------------------------------------------------

    regret_reference = {}

    for cid in rough_contexts:
        rows = [
            atlas_lookup[
                (
                    cid,
                    beta_name,
                )
            ]
            for beta_name in beta_order
        ]

        values = np.asarray(
            [
                [
                    finite(
                        row[column]
                    )
                    for column in (
                        OBJECTIVE_COLUMNS
                    )
                ]
                for row in rows
            ],
            dtype=np.float64,
        )

        nondominated = []

        for i in range(
            len(rows)
        ):
            dominated = False

            for j in range(
                len(rows)
            ):
                if i == j:
                    continue

                no_worse = np.all(
                    values[j]
                    <= values[i]
                    + TOL
                )

                strictly_better = np.any(
                    values[j]
                    < values[i]
                    - TOL
                )

                if (
                    no_worse
                    and strictly_better
                ):
                    dominated = True
                    break

            if not dominated:
                nondominated.append(i)

        front_values = values[
            nondominated
        ]

        ideal = np.min(
            front_values,
            axis=0,
        )

        nadir = np.max(
            front_values,
            axis=0,
        )

        regret_reference[
            cid
        ] = (
            ideal,
            nadir - ideal,
        )


    def score(
        cid,
        beta_name,
        w,
    ):
        row = atlas_lookup[
            (
                cid,
                beta_name,
            )
        ]

        values = np.asarray(
            [
                finite(
                    row[column]
                )
                for column in (
                    OBJECTIVE_COLUMNS
                )
            ],
            dtype=np.float64,
        )

        ideal, span = (
            regret_reference[
                cid
            ]
        )

        regret = np.zeros(
            3,
            dtype=np.float64,
        )

        for k in range(3):
            if span[k] > TOL:
                regret[k] = (
                    values[k]
                    - ideal[k]
                ) / span[k]

        weighted = (
            w
            * regret
        )

        return float(
            np.max(weighted)
            + RHO
            * np.sum(weighted)
        )


    # ========================================================
    # 1-NN context transfer baseline.
    # ========================================================

    nn_rows = []

    for test_index, test_cid in enumerate(
        rough_contexts
    ):
        distances = []

        for train_index, train_cid in enumerate(
            rough_contexts
        ):
            if train_index == test_index:
                continue

            d = float(
                np.linalg.norm(
                    z_context[test_index]
                    - z_context[train_index]
                )
            )

            distances.append(
                (
                    d,
                    train_index,
                    train_cid,
                )
            )

        distances.sort()

        nearest_distance, _, nearest_cid = (
            distances[0]
        )

        for preference_name in (
            preference_order
        ):
            source_label = (
                label_lookup[
                    (
                        nearest_cid,
                        preference_name,
                    )
                ]
            )

            target_label = (
                label_lookup[
                    (
                        test_cid,
                        preference_name,
                    )
                ]
            )

            beta_name = (
                source_label[
                    "selected_beta_name"
                ]
            )

            w = np.asarray(
                [
                    finite(
                        target_label[
                            "w_motion"
                        ]
                    ),
                    finite(
                        target_label[
                            "w_stability"
                        ]
                    ),
                    finite(
                        target_label[
                            "w_energy"
                        ]
                    ),
                ],
                dtype=np.float64,
            )

            transferred_score = score(
                test_cid,
                beta_name,
                w,
            )

            oracle_score = finite(
                target_label[
                    "scalarization_score"
                ]
            )

            excess = max(
                0.0,
                transferred_score
                - oracle_score,
            )

            nn_rows.append(
                {
                    "test_context":
                        test_cid,

                    "nearest_context":
                        nearest_cid,

                    "context_distance":
                        nearest_distance,

                    "preference_name":
                        preference_name,

                    "transferred_beta_name":
                        beta_name,

                    "oracle_beta_name":
                        target_label[
                            "selected_beta_name"
                        ],

                    "exact":
                        int(
                            beta_name
                            == target_label[
                                "selected_beta_name"
                            ]
                        ),

                    "oracle_score":
                        oracle_score,

                    "transferred_score":
                        transferred_score,

                    "score_excess":
                        excess,
                }
            )


    # ========================================================
    # Pairwise descriptor distance vs response-surface distance
    # ========================================================

    # Normalize each physical objective globally over rough
    # atlas only so units do not dominate the response distance.
    rough_values = []

    for cid in rough_contexts:
        for beta_name in beta_order:
            row = atlas_lookup[
                (
                    cid,
                    beta_name,
                )
            ]

            rough_values.append(
                [
                    finite(
                        row[column]
                    )
                    for column in (
                        OBJECTIVE_COLUMNS
                    )
                ]
            )

    rough_values = np.asarray(
        rough_values,
        dtype=np.float64,
    )

    response_mean = np.mean(
        rough_values,
        axis=0,
    )

    response_std = np.std(
        rough_values,
        axis=0,
    )

    if np.any(
        response_std <= EPS
    ):
        raise RuntimeError(
            "Degenerate response objective."
        )

    response_vector = {}

    for cid in rough_contexts:
        values = np.asarray(
            [
                [
                    finite(
                        atlas_lookup[
                            (
                                cid,
                                beta_name,
                            )
                        ][column]
                    )
                    for column in (
                        OBJECTIVE_COLUMNS
                    )
                ]
                for beta_name in beta_order
            ],
            dtype=np.float64,
        )

        values = (
            values
            - response_mean
        ) / response_std

        response_vector[
            cid
        ] = values.reshape(-1)


    pairwise_rows = []

    context_distances = []
    response_distances = []

    for i in range(
        len(rough_contexts)
    ):
        for j in range(
            i + 1,
            len(rough_contexts),
        ):
            ci = rough_contexts[i]
            cj = rough_contexts[j]

            dc = float(
                np.linalg.norm(
                    z_context[i]
                    - z_context[j]
                )
            )

            dr = float(
                np.linalg.norm(
                    response_vector[ci]
                    - response_vector[cj]
                )
                / math.sqrt(
                    len(
                        response_vector[ci]
                    )
                )
            )

            context_distances.append(
                dc
            )

            response_distances.append(
                dr
            )

            pairwise_rows.append(
                {
                    "context_a":
                        ci,

                    "context_b":
                        cj,

                    "context_distance":
                        dc,

                    "response_distance":
                        dr,
                }
            )


    corr = pearson(
        context_distances,
        response_distances,
    )


    nn_excess = np.asarray(
        [
            row[
                "score_excess"
            ]
            for row in nn_rows
        ],
        dtype=np.float64,
    )

    nn_exact = np.asarray(
        [
            row[
                "exact"
            ]
            for row in nn_rows
        ],
        dtype=np.float64,
    )


    manifest = {
        "schema":
            (
                "icra27_os_t5p5e1b_"
                "context_identifiability_v0"
            ),

        "status":
            "COMPUTE_PASS",

        "rough_contexts":
            len(
                rough_contexts
            ),

        "active_rough_context_dimensions":
            [
                CONTEXT_COLUMNS[index]
                for index in range(
                    len(
                        CONTEXT_COLUMNS
                    )
                )
                if active[index]
            ],

        "nearest_context_transfer":
            {
                "rows":
                    len(
                        nn_rows
                    ),

                "exact_accuracy":
                    float(
                        np.mean(
                            nn_exact
                        )
                    ),

                "score_excess_mean":
                    float(
                        np.mean(
                            nn_excess
                        )
                    ),

                "score_excess_median":
                    float(
                        np.median(
                            nn_excess
                        )
                    ),

                "score_excess_p95":
                    float(
                        np.quantile(
                            nn_excess,
                            0.95,
                        )
                    ),

                "score_excess_max":
                    float(
                        np.max(
                            nn_excess
                        )
                    ),

                "score_excess_le_0p05_fraction":
                    float(
                        np.mean(
                            nn_excess
                            <= 0.05
                        )
                    ),
            },

        "pairwise_context_response":
            {
                "pairs":
                    len(
                        pairwise_rows
                    ),

                "pearson_distance_correlation":
                    corr,
            },

        "interpretation":
            (
                "If nearest-context transfer also "
                "has large physical score excess and "
                "descriptor distance correlates weakly "
                "with beta-response distance, the "
                "current 5D c_OS is insufficiently "
                "identifying the physical response "
                "surface. If NN transfer is strong "
                "while the MLP LOCO result is weak, "
                "model/training design is the more "
                "likely bottleneck."
            ),
    }


    OUT_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    write_csv(
        OUT_NN,
        nn_rows,
    )

    write_csv(
        OUT_PAIRWISE,
        pairwise_rows,
    )

    OUT_MANIFEST.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


    print()
    print("=" * 112)
    print(
        "ICRA27 OS-T5.5e1b "
        "CONTEXT IDENTIFIABILITY AUDIT"
    )
    print("=" * 112)

    print(
        "rough contexts:",
        len(
            rough_contexts
        ),
    )

    print(
        "active c dims :",
        int(
            np.sum(
                active
            )
        ),
    )

    print()
    print("1-NN CONTEXT TRANSFER")

    for key, value in (
        manifest[
            "nearest_context_transfer"
        ].items()
    ):
        print(
            f"  {key:<34}: "
            f"{value}"
        )

    print()
    print("CONTEXT ↔ RESPONSE GEOMETRY")

    print(
        "  pair count                       :",
        len(
            pairwise_rows
        ),
    )

    print(
        "  Pearson distance correlation     :",
        corr,
    )

    print()
    print("outputs:")
    print(" ", OUT_NN)
    print(" ", OUT_PAIRWISE)
    print(" ", OUT_MANIFEST)

    print()
    print(
        "[ICRA27] OS-T5.5e1b "
        "context identifiability: COMPUTE PASS"
    )

    print("=" * 112)


if __name__ == "__main__":
    main()
