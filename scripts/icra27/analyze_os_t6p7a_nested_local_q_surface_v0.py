from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]

T66_PATH = (
    ROOT
    / "scripts/icra27"
    / "analyze_os_t6p6a_semantic_geometry_identifiability_v0.py"
)

OUT_DIR = (
    ROOT
    / "results/icra27"
    / "os_t6p7a_nested_local_q_surface_v0"
)

OUT_FOLDS = (
    OUT_DIR
    / "nested_local_q_surface_folds.csv"
)

OUT_SUMMARY = (
    OUT_DIR
    / "nested_local_q_surface_summary.csv"
)

OUT_MANIFEST = (
    OUT_DIR
    / "nested_local_q_surface_manifest.json"
)


REPRESENTATIONS = (
    "raw_height",
    "semantic_regions",
)

K_BANK = (
    1,
    3,
    5,
    7,
)

EPS = 1.0e-6
EXPECTED_BETA = 21


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )

    if (
        spec is None
        or spec.loader is None
    ):
        raise RuntimeError(
            f"Could not load {path}"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    return module


def write_csv(path, rows):
    if not rows:
        raise RuntimeError(
            f"No rows for {path}"
        )

    with path.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            rows
        )


def weighted_q_prediction(
    *,
    distances,
    train_contexts,
    q_lookup,
    k,
):
    order = np.argsort(
        distances,
        kind="stable",
    )

    chosen = order[:k]

    d = np.asarray(
        [
            distances[i]
            for i in chosen
        ],
        dtype=np.float64,
    )

    weights = 1.0 / (
        d
        + EPS
    )

    surfaces = np.stack(
        [
            q_lookup[
                train_contexts[i]
            ]
            for i in chosen
        ],
        axis=0,
    )

    predicted = np.average(
        surfaces,
        axis=0,
        weights=weights,
    )

    neighbors = [
        train_contexts[i]
        for i in chosen
    ]

    return (
        predicted,
        neighbors,
        d,
        weights,
    )


def distance_vector(
    *,
    representation_lookup,
    train_contexts,
    held_context,
):
    X = np.stack(
        [
            representation_lookup[
                cid
            ]
            for cid in train_contexts
        ],
        axis=0,
    )

    xh = representation_lookup[
        held_context
    ]

    mean = np.mean(
        X,
        axis=0,
    )

    std = np.std(
        X,
        axis=0,
    )

    std = np.where(
        std > 1.0e-8,
        std,
        1.0,
    )

    Xn = (
        X
        - mean
    ) / std

    xhn = (
        xh
        - mean
    ) / std

    return np.linalg.norm(
        Xn
        - xhn[None, :],
        axis=1,
    )


def q_excess_for_prediction(
    true_q,
    predicted_q,
):
    oracle_index = int(
        np.argmin(
            true_q
        )
    )

    selected_index = int(
        np.argmin(
            predicted_q
        )
    )

    excess = float(
        true_q[
            selected_index
        ]
        - true_q[
            oracle_index
        ]
    )

    order = np.argsort(
        true_q,
        kind="stable",
    )

    rank = (
        int(
            np.where(
                order
                == selected_index
            )[0][0]
        )
        + 1
    )

    return (
        selected_index,
        oracle_index,
        excess,
        rank,
    )


def choose_k_inner_cv(
    *,
    representation_lookup,
    q_lookup,
    outer_train_contexts,
):
    scores = {}

    for k in K_BANK:
        excesses = []

        for inner_held in outer_train_contexts:
            inner_train = [
                cid
                for cid in outer_train_contexts
                if cid != inner_held
            ]

            distances = distance_vector(
                representation_lookup=(
                    representation_lookup
                ),
                train_contexts=inner_train,
                held_context=inner_held,
            )

            (
                predicted_q,
                _neighbors,
                _d,
                _w,
            ) = weighted_q_prediction(
                distances=distances,
                train_contexts=inner_train,
                q_lookup=q_lookup,
                k=k,
            )

            (
                _selected,
                _oracle,
                excess,
                _rank,
            ) = q_excess_for_prediction(
                q_lookup[
                    inner_held
                ],
                predicted_q,
            )

            excesses.append(
                excess
            )

        scores[
            k
        ] = float(
            np.mean(
                excesses
            )
        )

    # TRAIN-only selection.
    # Tie -> smaller k.
    best_k = min(
        K_BANK,
        key=lambda k:
            (
                scores[k],
                k,
            ),
    )

    return (
        best_k,
        scores,
    )


def main():
    if OUT_DIR.exists():
        raise SystemExit(
            f"REFUSING TO OVERWRITE: "
            f"{OUT_DIR}"
        )

    if not T66_PATH.exists():
        raise FileNotFoundError(
            T66_PATH
        )

    t66 = load_module(
        T66_PATH,
        "os_t6p7a_t66",
    )


    patch_manifest = json.loads(
        t66.PATCH_MANIFEST.read_text()
    )

    if patch_manifest.get(
        "status"
    ) != "FREEZE_PASS":
        raise RuntimeError(
            "T6.3a patch is not FREEZE_PASS."
        )

    if bool(
        patch_manifest.get(
            "heldout_used",
            False,
        )
    ):
        raise RuntimeError(
            "Patch source reports heldout use."
        )


    terrain_rows = t66.read_csv(
        t66.PATCH_CSV
    )

    patch_cols = t66.patch_columns(
        terrain_rows
    )


    original_contexts = [
        f"rough_seed_{int(seed)}"
        for seed in patch_manifest[
            "original_seeds"
        ]
    ]

    extension_contexts = [
        f"rough_seed_{int(seed)}"
        for seed in patch_manifest[
            "extension_seeds"
        ]
    ]

    if len(
        original_contexts
    ) != 18:
        raise RuntimeError(
            "Expected 18 original contexts."
        )

    if len(
        extension_contexts
    ) != 15:
        raise RuntimeError(
            "Expected 15 extension contexts."
        )


    all_contexts = (
        original_contexts
        + extension_contexts
    )

    terrain_lookup = {
        row[
            "context_id"
        ]:
            row
        for row in terrain_rows
    }


    # ========================================================
    # Rebuild frozen terrain representations.
    # ========================================================

    representations = {
        name: {}
        for name in REPRESENTATIONS
    }

    for cid in all_contexts:
        built = t66.build_representations(
            terrain_lookup[
                cid
            ],
            patch_cols,
        )

        for name in REPRESENTATIONS:
            representations[
                name
            ][
                cid
            ] = built[
                name
            ]


    # ========================================================
    # Frozen balanced-Q surfaces.
    # ========================================================

    q_rows = t66.read_csv(
        t66.Q_CSV
    )

    grouped = {}

    for row in q_rows:
        cid = row[
            "context_id"
        ]

        if cid in set(
            all_contexts
        ):
            grouped.setdefault(
                cid,
                [],
            ).append(
                row
            )


    q_lookup = {}
    beta_lookup = {}

    for cid in all_contexts:
        rows = sorted(
            grouped[
                cid
            ],
            key=lambda x:
                int(
                    x[
                        "beta_index"
                    ]
                ),
        )

        if len(
            rows
        ) != EXPECTED_BETA:
            raise RuntimeError(
                f"{cid}: expected 21 Q rows."
            )

        beta_lookup[
            cid
        ] = [
            row[
                "beta_name"
            ]
            for row in rows
        ]

        q_lookup[
            cid
        ] = np.asarray(
            [
                float(
                    row[
                        "quality_Q"
                    ]
                )
                for row in rows
            ],
            dtype=np.float64,
        )


    # ========================================================
    # Strict outer LOCO.
    # k is selected using inner CV on OUTER TRAIN only.
    # ========================================================

    fold_rows = []

    for representation in REPRESENTATIONS:

        rep_lookup = representations[
            representation
        ]

        print()
        print("=" * 112)
        print(
            f"REPRESENTATION: {representation}"
        )
        print("=" * 112)


        for held in original_contexts:

            outer_train = (
                [
                    cid
                    for cid in original_contexts
                    if cid != held
                ]
                + extension_contexts
            )

            if len(
                outer_train
            ) != 32:
                raise RuntimeError(
                    "Expected 32 outer TRAIN contexts."
                )


            # --------------------------------------------
            # TRAIN-only nested model selection.
            # --------------------------------------------

            (
                best_k,
                inner_scores,
            ) = choose_k_inner_cv(
                representation_lookup=(
                    rep_lookup
                ),
                q_lookup=q_lookup,
                outer_train_contexts=(
                    outer_train
                ),
            )


            # --------------------------------------------
            # Outer held-context prediction.
            # --------------------------------------------

            distances = distance_vector(
                representation_lookup=(
                    rep_lookup
                ),
                train_contexts=outer_train,
                held_context=held,
            )

            (
                predicted_q,
                neighbors,
                neighbor_distances,
                weights,
            ) = weighted_q_prediction(
                distances=distances,
                train_contexts=outer_train,
                q_lookup=q_lookup,
                k=best_k,
            )

            true_q = q_lookup[
                held
            ]

            (
                selected_index,
                oracle_index,
                q_excess,
                rank,
            ) = q_excess_for_prediction(
                true_q,
                predicted_q,
            )


            # Frozen no-context baseline.
            mean_train_q = np.mean(
                np.stack(
                    [
                        q_lookup[
                            cid
                        ]
                        for cid in outer_train
                    ],
                    axis=0,
                ),
                axis=0,
            )

            baseline_index = int(
                np.argmin(
                    mean_train_q
                )
            )

            baseline_q_excess = float(
                true_q[
                    baseline_index
                ]
                - true_q[
                    oracle_index
                ]
            )


            # Decision margin in predicted Q landscape.
            pred_order = np.argsort(
                predicted_q,
                kind="stable",
            )

            predicted_margin = float(
                predicted_q[
                    pred_order[1]
                ]
                - predicted_q[
                    pred_order[0]
                ]
            )


            row = {
                "representation":
                    representation,

                "held_context":
                    held,

                "selected_k":
                    int(
                        best_k
                    ),

                "inner_mean_Q_excess_k1":
                    inner_scores[1],

                "inner_mean_Q_excess_k3":
                    inner_scores[3],

                "inner_mean_Q_excess_k5":
                    inner_scores[5],

                "inner_mean_Q_excess_k7":
                    inner_scores[7],

                "neighbors":
                    "|".join(
                        neighbors
                    ),

                "neighbor_distances":
                    "|".join(
                        f"{x:.9g}"
                        for x in neighbor_distances
                    ),

                "neighbor_weights":
                    "|".join(
                        f"{x:.9g}"
                        for x in weights
                    ),

                "oracle_beta":
                    beta_lookup[
                        held
                    ][
                        oracle_index
                    ],

                "selected_beta":
                    beta_lookup[
                        held
                    ][
                        selected_index
                    ],

                "baseline_beta":
                    beta_lookup[
                        held
                    ][
                        baseline_index
                    ],

                "true_rank":
                    int(
                        rank
                    ),

                "exact":
                    int(
                        rank == 1
                    ),

                "top3":
                    int(
                        rank <= 3
                    ),

                "top5":
                    int(
                        rank <= 5
                    ),

                "Q_excess":
                    q_excess,

                "baseline_Q_excess":
                    baseline_q_excess,

                "beats_baseline":
                    int(
                        q_excess
                        < baseline_q_excess
                        - 1.0e-12
                    ),

                "predicted_Q_margin":
                    predicted_margin,

                "predicted_vs_true_Q_spearman":
                    t66.spearman(
                        predicted_q,
                        true_q,
                    ),
            }

            fold_rows.append(
                row
            )

            print(
                f"{held:<16} "
                f"k={best_k} "
                f"rank={rank:<2} "
                f"Qex={q_excess:.4f} "
                f"base={baseline_q_excess:.4f} "
                f"rhoQ="
                f"{row['predicted_vs_true_Q_spearman']:+.3f} "
                f"NNs={','.join(neighbors)}"
            )


    # ========================================================
    # Representation summaries.
    # ========================================================

    summary_rows = []

    for representation in REPRESENTATIONS:

        rows = [
            row
            for row in fold_rows
            if row[
                "representation"
            ] == representation
        ]

        def arr(key):
            return np.asarray(
                [
                    row[key]
                    for row in rows
                ],
                dtype=np.float64,
            )


        q = arr(
            "Q_excess"
        )

        summary_rows.append(
            {
                "representation":
                    representation,

                "Q_excess_mean":
                    float(
                        np.mean(q)
                    ),

                "Q_excess_median":
                    float(
                        np.median(q)
                    ),

                "Q_excess_p95":
                    float(
                        np.percentile(
                            q,
                            95,
                        )
                    ),

                "Q_excess_max":
                    float(
                        np.max(q)
                    ),

                "true_rank_mean":
                    float(
                        np.mean(
                            arr(
                                "true_rank"
                            )
                        )
                    ),

                "true_rank_median":
                    float(
                        np.median(
                            arr(
                                "true_rank"
                            )
                        )
                    ),

                "exact_fraction":
                    float(
                        np.mean(
                            arr(
                                "exact"
                            )
                        )
                    ),

                "top3_fraction":
                    float(
                        np.mean(
                            arr(
                                "top3"
                            )
                        )
                    ),

                "top5_fraction":
                    float(
                        np.mean(
                            arr(
                                "top5"
                            )
                        )
                    ),

                "beats_baseline_fraction":
                    float(
                        np.mean(
                            arr(
                                "beats_baseline"
                            )
                        )
                    ),

                "predicted_Q_spearman_mean":
                    float(
                        np.mean(
                            arr(
                                "predicted_vs_true_Q_spearman"
                            )
                        )
                    ),

                "selected_k1_fraction":
                    float(
                        np.mean(
                            arr(
                                "selected_k"
                            )
                            == 1
                        )
                    ),

                "selected_k3_fraction":
                    float(
                        np.mean(
                            arr(
                                "selected_k"
                            )
                            == 3
                        )
                    ),

                "selected_k5_fraction":
                    float(
                        np.mean(
                            arr(
                                "selected_k"
                            )
                            == 5
                        )
                    ),

                "selected_k7_fraction":
                    float(
                        np.mean(
                            arr(
                                "selected_k"
                            )
                            == 7
                        )
                    ),
            }
        )


    OUT_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    write_csv(
        OUT_FOLDS,
        fold_rows,
    )

    write_csv(
        OUT_SUMMARY,
        summary_rows,
    )


    manifest = {
        "schema":
            "icra27_os_t6p7a_nested_local_q_surface_v0",

        "status":
            "COMPUTE_PASS",

        "heldout_used":
            False,

        "representations":
            list(
                REPRESENTATIONS
            ),

        "k_bank":
            list(
                K_BANK
            ),

        "neighbor_weight":
            "inverse standardized-Euclidean distance",

        "outer_evaluation":
            (
                "Strict LOCO over 18 original rough "
                "TRAIN contexts."
            ),

        "outer_train_pool":
            (
                "Remaining 17 original rough + "
                "15 TRAIN-only extension contexts."
            ),

        "hyperparameter_selection":
            (
                "k selected independently inside each "
                "outer fold using inner leave-one-context-"
                "out mean Q-excess over OUTER TRAIN only."
            ),

        "no_context_baseline":
            (
                "argmin of mean Q over the outer "
                "TRAIN context pool."
            ),

        "summary":
            summary_rows,

        "decision_rule": {
            "semantic_nested_improves_tail":
                (
                    "Semantic terrain geometry contains "
                    "useful local response structure; "
                    "T6.6 catastrophic failures were "
                    "primarily single-neighbor brittleness."
                ),

            "semantic_nested_still_unreliable":
                (
                    "Handcrafted static terrain geometry "
                    "does not provide sufficiently stable "
                    "local structure for beta selection; "
                    "close this representation branch and "
                    "move to richer pre-decision causal "
                    "context / structured encoder."
                ),
        },
    }


    OUT_MANIFEST.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


    print()
    print("=" * 118)
    print(
        "ICRA27 OS-T6.7a NESTED LOCAL "
        "Q-SURFACE AUDIT"
    )
    print("=" * 118)

    for row in summary_rows:
        print()
        print(
            row[
                "representation"
            ]
        )

        for key, value in row.items():
            if key == "representation":
                continue

            print(
                f"  {key:<36}: {value}"
            )

    print()
    print(
        "[ICRA27] OS-T6.7a nested local "
        "Q-surface audit: COMPUTE PASS"
    )
    print("=" * 118)


if __name__ == "__main__":
    main()
