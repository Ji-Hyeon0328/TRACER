from __future__ import annotations

import csv
import importlib.util
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]

T66_PATH = (
    ROOT
    / "scripts/icra27"
    / "analyze_os_t6p6a_semantic_geometry_identifiability_v0.py"
)

T67_PATH = (
    ROOT
    / "scripts/icra27"
    / "analyze_os_t6p7a_nested_local_q_surface_v0.py"
)

T67_MANIFEST = (
    ROOT
    / "results/icra27"
    / "os_t6p7a_nested_local_q_surface_v0"
    / "nested_local_q_surface_manifest.json"
)

PROBE_CSV = (
    ROOT
    / "results/icra27"
    / "os_t6p8b_prepolicy_probe_response_atlas_v0"
    / "prepolicy_probe_response_atlas.csv"
)

PROBE_MANIFEST = (
    ROOT
    / "results/icra27"
    / "os_t6p8b_prepolicy_probe_response_atlas_v0"
    / "prepolicy_probe_response_manifest.json"
)

OUT_DIR = (
    ROOT
    / "results/icra27"
    / "os_t6p8c_probe_horizon_identifiability_v0"
)

OUT_FOLDS = (
    OUT_DIR
    / "probe_horizon_fold_results.csv"
)

OUT_SUMMARY = (
    OUT_DIR
    / "probe_horizon_summary.csv"
)

OUT_MANIFEST = (
    OUT_DIR
    / "probe_horizon_identifiability_manifest.json"
)


PROBE_FEATURES = (
    "probe_progress_m",
    "delta_vx_mps",
    "delta_vy_mps",
    "delta_yaw_rate_rps",
    "delta_base_z_m",
    "delta_roll_rad",
    "delta_pitch_rad",
)

PROBE_STEPS = tuple(
    range(6)
)

EXPECTED_BETA = 21

EPS = 1.0e-12


def load_module(
    path,
    name,
):
    spec = (
        importlib.util.spec_from_file_location(
            name,
            path,
        )
    )

    if (
        spec is None
        or spec.loader is None
    ):
        raise RuntimeError(
            f"Could not load {path}"
        )

    module = (
        importlib.util.module_from_spec(
            spec
        )
    )

    spec.loader.exec_module(
        module
    )

    return module


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


def arr(
    rows,
    key,
):
    return np.asarray(
        [
            row[key]
            for row in rows
        ],
        dtype=np.float64,
    )


def main():
    if OUT_DIR.exists():
        raise SystemExit(
            f"REFUSING TO OVERWRITE: "
            f"{OUT_DIR}"
        )

    for path in (
        T66_PATH,
        T67_PATH,
        T67_MANIFEST,
        PROBE_CSV,
        PROBE_MANIFEST,
    ):
        if not path.exists():
            raise FileNotFoundError(
                path
            )


    t66 = load_module(
        T66_PATH,
        "os_t6p8c_t66",
    )

    t67 = load_module(
        T67_PATH,
        "os_t6p8c_t67",
    )


    # ========================================================
    # Frozen context split / semantic terrain representation
    # ========================================================

    patch_manifest = json.loads(
        t66.PATCH_MANIFEST.read_text()
    )

    if patch_manifest.get(
        "status"
    ) != "FREEZE_PASS":
        raise RuntimeError(
            "T6.3a patch source is not "
            "FREEZE_PASS."
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

    all_contexts = (
        original_contexts
        + extension_contexts
    )


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

    if len(
        set(
            all_contexts
        )
    ) != 33:
        raise RuntimeError(
            "Expected 33 unique contexts."
        )


    terrain_rows = t66.read_csv(
        t66.PATCH_CSV
    )

    terrain_lookup = {
        row[
            "context_id"
        ]:
            row
        for row in terrain_rows
    }

    patch_cols = t66.patch_columns(
        terrain_rows
    )


    semantic_lookup = {}

    for cid in all_contexts:
        representations = (
            t66.build_representations(
                terrain_lookup[
                    cid
                ],
                patch_cols,
            )
        )

        semantic_lookup[
            cid
        ] = np.asarray(
            representations[
                "semantic_regions"
            ],
            dtype=np.float64,
        )


    semantic_dim = int(
        semantic_lookup[
            all_contexts[0]
        ].shape[0]
    )

    if semantic_dim != 74:
        raise RuntimeError(
            f"Expected semantic dim=74, "
            f"got {semantic_dim}"
        )


    # ========================================================
    # Frozen probe-response atlas
    # ========================================================

    probe_manifest = json.loads(
        PROBE_MANIFEST.read_text()
    )

    if probe_manifest.get(
        "status"
    ) != "FREEZE_PASS":
        raise RuntimeError(
            "T6.8b probe atlas is not "
            "FREEZE_PASS."
        )

    if bool(
        probe_manifest.get(
            "heldout_used",
            False,
        )
    ):
        raise RuntimeError(
            "T6.8b reports heldout use."
        )

    if int(
        probe_manifest[
            "contexts"
        ]
    ) != 33:
        raise RuntimeError(
            "Expected 33 probe contexts."
        )


    probe_rows = read_csv(
        PROBE_CSV
    )

    if len(
        probe_rows
    ) != 198:
        raise RuntimeError(
            f"Expected 198 probe rows, "
            f"got {len(probe_rows)}"
        )


    probe_lookup = {}
    safety_lookup = {}

    for row in probe_rows:
        cid = row[
            "context_id"
        ]

        step = int(
            row[
                "probe_step"
            ]
        )

        key = (
            cid,
            step,
        )

        if key in probe_lookup:
            raise RuntimeError(
                f"Duplicate probe key: {key}"
            )

        probe_lookup[
            key
        ] = np.asarray(
            [
                float(
                    row[name]
                )
                for name in PROBE_FEATURES
            ],
            dtype=np.float64,
        )

        safety_lookup[
            key
        ] = str(
            row[
                "safety_state"
            ]
        )


    for cid in all_contexts:
        for step in PROBE_STEPS:
            if (
                cid,
                step,
            ) not in probe_lookup:
                raise RuntimeError(
                    f"Missing probe response: "
                    f"{cid}, step={step}"
                )


    # Probe response must be exactly zero at reset.
    for cid in all_contexts:
        if not np.allclose(
            probe_lookup[
                (
                    cid,
                    0,
                )
            ],
            0.0,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise RuntimeError(
                f"{cid}: reset probe feature "
                "is not zero."
            )


    # ========================================================
    # Frozen Q surfaces
    # ========================================================

    q_rows = t66.read_csv(
        t66.Q_CSV
    )

    grouped_q = {}

    for row in q_rows:
        cid = row[
            "context_id"
        ]

        if cid in set(
            all_contexts
        ):
            grouped_q.setdefault(
                cid,
                [],
            ).append(
                row
            )


    q_lookup = {}
    beta_lookup = {}

    for cid in all_contexts:
        rows = sorted(
            grouped_q[
                cid
            ],
            key=lambda row:
                int(
                    row[
                        "beta_index"
                    ]
                ),
        )

        if len(
            rows
        ) != EXPECTED_BETA:
            raise RuntimeError(
                f"{cid}: Q beta count mismatch."
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


    reference_beta_order = beta_lookup[
        all_contexts[0]
    ]

    for cid in all_contexts:
        if beta_lookup[
            cid
        ] != reference_beta_order:
            raise RuntimeError(
                f"{cid}: beta order mismatch."
            )


    # ========================================================
    # Frozen physical J surfaces
    # ========================================================

    physical_rows = t66.read_csv(
        t66.PHYSICAL_CSV
    )

    J_by_key = {}

    for row in physical_rows:
        cid = row[
            "context_id"
        ]

        if cid not in set(
            all_contexts
        ):
            continue

        J_by_key[
            (
                cid,
                row[
                    "beta_name"
                ],
            )
        ] = np.asarray(
            [
                float(
                    row[
                        "J_motion_s_per_m"
                    ]
                ),

                float(
                    row[
                        "J_stability"
                    ]
                ),

                float(
                    row[
                        "J_energy_j_per_m"
                    ]
                ),
            ],
            dtype=np.float64,
        )


    J_lookup = {}

    for cid in all_contexts:
        J_lookup[
            cid
        ] = np.stack(
            [
                J_by_key[
                    (
                        cid,
                        beta,
                    )
                ]
                for beta in reference_beta_order
            ],
            axis=0,
        )


    # ========================================================
    # Representation constructors
    # ========================================================

    def representation_lookup(
        family,
        step,
    ):
        result = {}

        for cid in all_contexts:
            probe = probe_lookup[
                (
                    cid,
                    step,
                )
            ]

            if family == (
                "semantic_plus_probe"
            ):
                result[
                    cid
                ] = np.concatenate(
                    (
                        semantic_lookup[
                            cid
                        ],
                        probe,
                    )
                )

            elif family == "probe_only":
                result[
                    cid
                ] = probe.copy()

            else:
                raise ValueError(
                    family
                )

        return result


    cases = []

    # h=0 reproduces static semantic terrain.
    for step in PROBE_STEPS:
        cases.append(
            (
                "semantic_plus_probe",
                step,
            )
        )

    # probe-only h=0 is all-zero and has no
    # meaningful neighborhood geometry.
    for step in PROBE_STEPS[1:]:
        cases.append(
            (
                "probe_only",
                step,
            )
        )


    # ========================================================
    # Strict outer LOCO + nested TRAIN-only k selection
    # ========================================================

    fold_rows = []

    for (
        family,
        step,
    ) in cases:

        reps = representation_lookup(
            family,
            step,
        )

        print()
        print(
            "=" * 118
        )

        print(
            f"{family} "
            f"h={step * 0.2:.1f}s "
            f"dim="
            f"{reps[all_contexts[0]].shape[0]}"
        )

        print(
            "=" * 118
        )


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
                    "Expected 32 outer TRAIN "
                    "contexts."
                )


            (
                best_k,
                inner_scores,
            ) = t67.choose_k_inner_cv(
                representation_lookup=(
                    reps
                ),
                q_lookup=q_lookup,
                outer_train_contexts=(
                    outer_train
                ),
            )


            distances = t67.distance_vector(
                representation_lookup=(
                    reps
                ),
                train_contexts=(
                    outer_train
                ),
                held_context=(
                    held
                ),
            )


            (
                predicted_q,
                neighbors,
                neighbor_distances,
                weights,
            ) = (
                t67.weighted_q_prediction(
                    distances=distances,
                    train_contexts=(
                        outer_train
                    ),
                    q_lookup=q_lookup,
                    k=best_k,
                )
            )


            true_q = q_lookup[
                held
            ]


            (
                selected_index,
                oracle_index,
                q_excess,
                rank,
            ) = (
                t67.q_excess_for_prediction(
                    true_q,
                    predicted_q,
                )
            )


            # --------------------------------------------
            # Same local neighborhood/weights predict
            # each frozen physical J surface.
            # --------------------------------------------

            predicted_J = np.average(
                np.stack(
                    [
                        J_lookup[
                            cid
                        ]
                        for cid in neighbors
                    ],
                    axis=0,
                ),
                axis=0,
                weights=weights,
            )

            true_J = J_lookup[
                held
            ]


            rho_M = t66.spearman(
                predicted_J[
                    :,
                    0
                ],
                true_J[
                    :,
                    0
                ],
            )

            rho_S = t66.spearman(
                predicted_J[
                    :,
                    1
                ],
                true_J[
                    :,
                    1
                ],
            )

            rho_E = t66.spearman(
                predicted_J[
                    :,
                    2
                ],
                true_J[
                    :,
                    2
                ],
            )


            rho_Q = t66.spearman(
                predicted_q,
                true_q,
            )


            # --------------------------------------------
            # Frozen no-context baseline.
            # --------------------------------------------

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


            fold_rows.append(
                {
                    "family":
                        family,

                    "probe_step":
                        int(
                            step
                        ),

                    "probe_time_s":
                        float(
                            step
                            * 0.2
                        ),

                    "context_dim":
                        int(
                            reps[
                                held
                            ].shape[0]
                        ),

                    "held_context":
                        held,

                    "selected_k":
                        int(
                            best_k
                        ),

                    "inner_Qex_k1":
                        inner_scores[
                            1
                        ],

                    "inner_Qex_k3":
                        inner_scores[
                            3
                        ],

                    "inner_Qex_k5":
                        inner_scores[
                            5
                        ],

                    "inner_Qex_k7":
                        inner_scores[
                            7
                        ],

                    "neighbors":
                        "|".join(
                            neighbors
                        ),

                    "oracle_beta":
                        reference_beta_order[
                            oracle_index
                        ],

                    "selected_beta":
                        reference_beta_order[
                            selected_index
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
                        float(
                            q_excess
                        ),

                    "baseline_Q_excess":
                        float(
                            baseline_q_excess
                        ),

                    "beats_baseline":
                        int(
                            q_excess
                            < baseline_q_excess
                            - EPS
                        ),

                    "rho_J_motion":
                        float(
                            rho_M
                        ),

                    "rho_J_stability":
                        float(
                            rho_S
                        ),

                    "rho_J_energy":
                        float(
                            rho_E
                        ),

                    "rho_Q":
                        float(
                            rho_Q
                        ),
                }
            )


    # ========================================================
    # Horizon-level summaries
    # ========================================================

    summary_rows = []


    def safety_stats(
        step,
    ):
        all_at = 0
        original_at = 0

        all_prefix = 0
        original_prefix = 0


        for cid in all_contexts:

            at_non_normal = (
                safety_lookup[
                    (
                        cid,
                        step,
                    )
                ]
                != "normal"
            )

            prefix_non_normal = any(
                safety_lookup[
                    (
                        cid,
                        s,
                    )
                ]
                != "normal"
                for s in range(
                    step
                    + 1
                )
            )

            all_at += int(
                at_non_normal
            )

            all_prefix += int(
                prefix_non_normal
            )

            if cid in set(
                original_contexts
            ):
                original_at += int(
                    at_non_normal
                )

                original_prefix += int(
                    prefix_non_normal
                )


        progress = np.asarray(
            [
                probe_lookup[
                    (
                        cid,
                        step,
                    )
                ][0]
                for cid in all_contexts
            ],
            dtype=np.float64,
        )


        return {
            "all33_nonnormal_at_horizon":
                all_at,

            "all33_nonnormal_prefix_any":
                all_prefix,

            "original18_nonnormal_at_horizon":
                original_at,

            "original18_nonnormal_prefix_any":
                original_prefix,

            "all33_probe_progress_mean_m":
                float(
                    np.mean(
                        progress
                    )
                ),

            "all33_probe_progress_max_m":
                float(
                    np.max(
                        progress
                    )
                ),
        }


    for (
        family,
        step,
    ) in cases:

        rows = [
            row
            for row in fold_rows
            if (
                row[
                    "family"
                ]
                == family
                and int(
                    row[
                        "probe_step"
                    ]
                )
                == step
            )
        ]


        q = arr(
            rows,
            "Q_excess",
        )

        safety = safety_stats(
            step
        )


        summary = {
            "family":
                family,

            "probe_step":
                int(
                    step
                ),

            "probe_time_s":
                float(
                    step
                    * 0.2
                ),

            "context_dim":
                int(
                    rows[0][
                        "context_dim"
                    ]
                ),

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

            "baseline_Q_excess_mean":
                float(
                    np.mean(
                        arr(
                            rows,
                            "baseline_Q_excess",
                        )
                    )
                ),

            "true_rank_mean":
                float(
                    np.mean(
                        arr(
                            rows,
                            "true_rank",
                        )
                    )
                ),

            "true_rank_median":
                float(
                    np.median(
                        arr(
                            rows,
                            "true_rank",
                        )
                    )
                ),

            "exact_fraction":
                float(
                    np.mean(
                        arr(
                            rows,
                            "exact",
                        )
                    )
                ),

            "top3_fraction":
                float(
                    np.mean(
                        arr(
                            rows,
                            "top3",
                        )
                    )
                ),

            "top5_fraction":
                float(
                    np.mean(
                        arr(
                            rows,
                            "top5",
                        )
                    )
                ),

            "beats_baseline_fraction":
                float(
                    np.mean(
                        arr(
                            rows,
                            "beats_baseline",
                        )
                    )
                ),

            "rho_J_motion_mean":
                float(
                    np.mean(
                        arr(
                            rows,
                            "rho_J_motion",
                        )
                    )
                ),

            "rho_J_stability_mean":
                float(
                    np.mean(
                        arr(
                            rows,
                            "rho_J_stability",
                        )
                    )
                ),

            "rho_J_energy_mean":
                float(
                    np.mean(
                        arr(
                            rows,
                            "rho_J_energy",
                        )
                    )
                ),

            "rho_Q_mean":
                float(
                    np.mean(
                        arr(
                            rows,
                            "rho_Q",
                        )
                    )
                ),

            "selected_k1_fraction":
                float(
                    np.mean(
                        arr(
                            rows,
                            "selected_k",
                        )
                        == 1
                    )
                ),

            "selected_k3_fraction":
                float(
                    np.mean(
                        arr(
                            rows,
                            "selected_k",
                        )
                        == 3
                    )
                ),

            "selected_k5_fraction":
                float(
                    np.mean(
                        arr(
                            rows,
                            "selected_k",
                        )
                        == 5
                    )
                ),

            "selected_k7_fraction":
                float(
                    np.mean(
                        arr(
                            rows,
                            "selected_k",
                        )
                        == 7
                    )
                ),

            **safety,
        }


        summary_rows.append(
            summary
        )


    # ========================================================
    # h=0 semantic regression against T6.7a
    # ========================================================

    t67_manifest = json.loads(
        T67_MANIFEST.read_text()
    )

    t67_semantic = next(
        row
        for row in t67_manifest[
            "summary"
        ]
        if row[
            "representation"
        ]
        == "semantic_regions"
    )


    h0 = next(
        row
        for row in summary_rows
        if (
            row[
                "family"
            ]
            == "semantic_plus_probe"
            and int(
                row[
                    "probe_step"
                ]
            )
            == 0
        )
    )


    regression_pairs = (
        (
            "Q_excess_mean",
            "Q_excess_mean",
        ),
        (
            "Q_excess_median",
            "Q_excess_median",
        ),
        (
            "Q_excess_p95",
            "Q_excess_p95",
        ),
        (
            "Q_excess_max",
            "Q_excess_max",
        ),
        (
            "top5_fraction",
            "top5_fraction",
        ),
        (
            "beats_baseline_fraction",
            "beats_baseline_fraction",
        ),
        (
            "rho_Q_mean",
            "predicted_Q_spearman_mean",
        ),
    )


    regression = {}

    for new_key, old_key in regression_pairs:

        found = float(
            h0[
                new_key
            ]
        )

        expected = float(
            t67_semantic[
                old_key
            ]
        )

        diff = abs(
            found
            - expected
        )

        regression[
            new_key
        ] = {
            "found":
                found,

            "expected":
                expected,

            "abs_diff":
                diff,

            "pass":
                diff
                <= 1.0e-12,
        }


    regression_pass = all(
        item[
            "pass"
        ]
        for item in regression.values()
    )


    if not regression_pass:
        raise RuntimeError(
            "T6.7a semantic h=0 "
            f"regression failed: "
            f"{regression}"
        )


    # ========================================================
    # h-relative deltas for semantic + probe
    # ========================================================

    base = h0

    for row in summary_rows:

        if row[
            "family"
        ] != "semantic_plus_probe":
            row[
                "delta_Q_excess_mean_vs_h0"
            ] = ""

            row[
                "delta_rho_JS_vs_h0"
            ] = ""

            row[
                "delta_rho_JE_vs_h0"
            ] = ""

            row[
                "delta_rho_Q_vs_h0"
            ] = ""

            continue


        row[
            "delta_Q_excess_mean_vs_h0"
        ] = (
            row[
                "Q_excess_mean"
            ]
            - base[
                "Q_excess_mean"
            ]
        )

        row[
            "delta_rho_JS_vs_h0"
        ] = (
            row[
                "rho_J_stability_mean"
            ]
            - base[
                "rho_J_stability_mean"
            ]
        )

        row[
            "delta_rho_JE_vs_h0"
        ] = (
            row[
                "rho_J_energy_mean"
            ]
            - base[
                "rho_J_energy_mean"
            ]
        )

        row[
            "delta_rho_Q_vs_h0"
        ] = (
            row[
                "rho_Q_mean"
            ]
            - base[
                "rho_Q_mean"
            ]
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
            "icra27_os_t6p8c_probe_horizon_identifiability_v0",

        "status":
            "COMPUTE_PASS",

        "heldout_used":
            False,

        "purpose":
            (
                "Test whether short beta-independent "
                "robot-terrain interaction responses "
                "provide additional information for "
                "balanced-beta quality identification "
                "beyond static semantic terrain geometry."
            ),

        "probe_features":
            list(
                PROBE_FEATURES
            ),

        "probe_feature_semantics":
            (
                "Fixed 7D boundary response at each "
                "horizon. History is intentionally not "
                "concatenated, keeping response dimension "
                "constant across horizons."
            ),

        "families":
            {
                "semantic_plus_probe":
                    (
                        "74D frozen semantic terrain "
                        "descriptor plus 7D causal probe "
                        "boundary response."
                    ),

                "probe_only":
                    (
                        "7D causal probe boundary "
                        "response without terrain geometry."
                    ),
            },

        "h0_semantics":
            (
                "At h=0 all probe-response features "
                "are zero, so semantic_plus_probe "
                "must exactly reproduce the T6.7a "
                "nested semantic-terrain baseline."
            ),

        "t6p7a_h0_regression":
            {
                "pass":
                    regression_pass,

                "checks":
                    regression,
            },

        "evaluation":
            (
                "Strict outer LOCO over the 18 original "
                "rough TRAIN contexts; each outer fold "
                "uses remaining17 original +15 extension "
                "TRAIN contexts. k in {1,3,5,7} is "
                "selected using inner TRAIN-only LOOCV."
            ),

        "safety_fields_used_as_features":
            False,

        "safety_reporting":
            (
                "Non-normal state at each horizon and "
                "any non-normal exposure up to each "
                "horizon are reported only as diagnostic "
                "costs, never as selector features."
            ),

        "summary":
            summary_rows,

        "decision_rule":
            {
                "short_probe_improves":
                    (
                        "If 0.2-0.4 s semantic+probe "
                        "reduces Q-excess below both h=0 "
                        "and the no-context baseline while "
                        "improving J_stability/J_energy "
                        "transfer, short interaction "
                        "provides information missing from "
                        "static geometry."
                    ),

                "only_long_probe_improves":
                    (
                        "If improvements emerge only at "
                        "0.8-1.0 s with substantial "
                        "non-normal safety exposure, treat "
                        "active probing as diagnostic "
                        "motivation for future adaptive "
                        "modules rather than the current "
                        "Objective Selector."
                    ),

                "no_probe_improvement":
                    (
                        "If probe response does not "
                        "reliably improve the selector, "
                        "close this active-interaction "
                        "diagnostic branch and avoid "
                        "adding it to the current ICRA "
                        "architecture."
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
    print("=" * 140)
    print(
        "ICRA27 OS-T6.8c PROBE-HORIZON "
        "IDENTIFIABILITY AUDIT"
    )
    print("=" * 140)

    print()
    print(
        "semantic terrain + fixed 7D "
        "probe boundary response"
    )

    print(
        " h[s] | Qmean   Qmed    Qp95    "
        "rank   top5   beatB   rhoM   rhoS   "
        "rhoE   rhoQ   nonnorm/prefix"
    )

    print(
        "-" * 140
    )


    for row in summary_rows:

        if row[
            "family"
        ] != "semantic_plus_probe":
            continue

        print(
            f" {row['probe_time_s']:>3.1f} | "
            f"{row['Q_excess_mean']:>6.3f} "
            f"{row['Q_excess_median']:>6.3f} "
            f"{row['Q_excess_p95']:>6.3f} "
            f"{row['true_rank_mean']:>6.2f} "
            f"{row['top5_fraction']:>6.3f} "
            f"{row['beats_baseline_fraction']:>6.3f} "
            f"{row['rho_J_motion_mean']:>6.3f} "
            f"{row['rho_J_stability_mean']:>6.3f} "
            f"{row['rho_J_energy_mean']:>6.3f} "
            f"{row['rho_Q_mean']:>6.3f} "
            f"{row['all33_nonnormal_at_horizon']:>2}/"
            f"{row['all33_nonnormal_prefix_any']:>2}"
        )


    print()
    print(
        "probe-only"
    )

    print(
        " h[s] | Qmean   Qmed    rank   "
        "top5   beatB   rhoS   rhoE   rhoQ"
    )

    print(
        "-" * 100
    )


    for row in summary_rows:

        if row[
            "family"
        ] != "probe_only":
            continue

        print(
            f" {row['probe_time_s']:>3.1f} | "
            f"{row['Q_excess_mean']:>6.3f} "
            f"{row['Q_excess_median']:>6.3f} "
            f"{row['true_rank_mean']:>6.2f} "
            f"{row['top5_fraction']:>6.3f} "
            f"{row['beats_baseline_fraction']:>6.3f} "
            f"{row['rho_J_stability_mean']:>6.3f} "
            f"{row['rho_J_energy_mean']:>6.3f} "
            f"{row['rho_Q_mean']:>6.3f}"
        )


    print()
    print(
        "T6.7a h=0 regression:",
        regression_pass,
    )

    print()
    print(
        "[ICRA27] OS-T6.8c probe-horizon "
        "identifiability: COMPUTE PASS"
    )

    print(
        "=" * 140
    )


if __name__ == "__main__":
    main()
