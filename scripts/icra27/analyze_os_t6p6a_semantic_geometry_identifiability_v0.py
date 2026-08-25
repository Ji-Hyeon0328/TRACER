from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]

PATCH_CSV = (
    ROOT
    / "results/icra27"
    / "os_t6p3a_highres_height_patches_v0"
    / "highres_height_patches.csv"
)

PATCH_MANIFEST = (
    ROOT
    / "results/icra27"
    / "os_t6p3a_highres_height_patches_v0"
    / "highres_height_patch_manifest.json"
)

Q_CSV = (
    ROOT
    / "results/icra27"
    / "os_t6p1_balanced_physical_quality_surface_v0"
    / "balanced_beta_quality_surface.csv"
)

PHYSICAL_CSV = (
    ROOT
    / "results/icra27"
    / "os_t5p5f3_expanded_physical_atlas_v0"
    / "physical_beta_response_atlas.csv"
)

T63_MANIFEST = (
    ROOT
    / "results/icra27"
    / "os_t6p3b_highres_context_q_identifiability_v0"
    / "highres_context_q_identifiability_manifest.json"
)

OUT_DIR = (
    ROOT
    / "results/icra27"
    / "os_t6p6a_semantic_geometry_identifiability_v0"
)

OUT_FOLDS = (
    OUT_DIR
    / "representation_fold_results.csv"
)

OUT_SUMMARY = (
    OUT_DIR
    / "representation_summary.csv"
)

OUT_MANIFEST = (
    OUT_DIR
    / "semantic_geometry_identifiability_manifest.json"
)


N_LONG = 16
N_LAT = 12
EXPECTED_BETA = 21

DX = 1.62 / N_LONG
DY = 1.12 / N_LAT

EPS = 1.0e-12


def read_csv(path):
    with path.open(
        "r",
        newline="",
    ) as f:
        return list(
            csv.DictReader(f)
        )


def rank_vector(x):
    x = np.asarray(
        x,
        dtype=np.float64,
    )

    order = np.argsort(
        x,
        kind="stable",
    )

    rank = np.empty(
        len(x),
        dtype=np.float64,
    )

    rank[order] = np.arange(
        1,
        len(x) + 1,
        dtype=np.float64,
    )

    return rank


def corr(a, b):
    a = np.asarray(
        a,
        dtype=np.float64,
    )
    b = np.asarray(
        b,
        dtype=np.float64,
    )

    if (
        np.std(a) <= EPS
        or np.std(b) <= EPS
    ):
        return 0.0

    return float(
        np.corrcoef(
            a,
            b,
        )[0, 1]
    )


def spearman(a, b):
    return corr(
        rank_vector(a),
        rank_vector(b),
    )


def rms(x):
    x = np.asarray(
        x,
        dtype=np.float64,
    )

    return float(
        np.sqrt(
            np.mean(
                x * x
            )
        )
    )


def abs_p95(x):
    return float(
        np.percentile(
            np.abs(
                np.asarray(
                    x,
                    dtype=np.float64,
                )
            ),
            95.0,
        )
    )


def patch_columns(rows):
    cols = [
        key
        for key in rows[0]
        if re.fullmatch(
            r"h_l\d{2}_r\d{2}_m",
            key,
        )
    ]

    def index(name):
        m = re.fullmatch(
            r"h_l(\d{2})_r(\d{2})_m",
            name,
        )

        return (
            int(m.group(1)),
            int(m.group(2)),
        )

    cols = sorted(
        cols,
        key=index,
    )

    expected = [
        f"h_l{i:02d}_r{j:02d}_m"
        for i in range(N_LONG)
        for j in range(N_LAT)
    ]

    if cols != expected:
        raise RuntimeError(
            "16x12 patch schema mismatch."
        )

    return cols


def fields(H):
    # Axis 0 = longitudinal.
    # Axis 1 = lateral.
    gx, gy = np.gradient(
        H,
        DX,
        DY,
        edge_order=2,
    )

    gxx = np.gradient(
        gx,
        DX,
        axis=0,
        edge_order=2,
    )

    gyy = np.gradient(
        gy,
        DY,
        axis=1,
        edge_order=2,
    )

    lap = (
        gxx
        + gyy
    )

    return gx, gy, lap


def semantic_region_features(
    H,
    gx,
    gy,
    lap,
):
    out = []

    grad = np.sqrt(
        gx * gx
        + gy * gy
    )

    # --------------------------------------------------------
    # Global terrain semantics.
    # --------------------------------------------------------

    global_values = (
        float(np.mean(H)),
        float(np.std(H)),
        float(np.max(H) - np.min(H)),

        float(np.mean(gx)),
        rms(gx),
        abs_p95(gx),

        float(np.mean(gy)),
        rms(gy),
        abs_p95(gy),

        rms(grad),
        abs_p95(grad),

        rms(lap),
        abs_p95(lap),
    )

    out.extend(
        global_values
    )

    # --------------------------------------------------------
    # 3 longitudinal segments x left/right halves.
    #
    # This makes longitudinal progression and lateral
    # asymmetry explicit without using rollout outcomes.
    # --------------------------------------------------------

    long_segments = np.array_split(
        np.arange(N_LONG),
        3,
    )

    lateral_segments = (
        np.arange(
            0,
            N_LAT // 2,
        ),
        np.arange(
            N_LAT // 2,
            N_LAT,
        ),
    )

    for ii in long_segments:
        for jj in lateral_segments:
            index = np.ix_(
                ii,
                jj,
            )

            h = H[index]
            x = gx[index]
            y = gy[index]
            l = lap[index]

            g = np.sqrt(
                x * x
                + y * y
            )

            out.extend(
                (
                    float(np.mean(h)),
                    float(np.std(h)),
                    float(
                        np.max(h)
                        - np.min(h)
                    ),

                    float(np.mean(x)),
                    rms(x),

                    float(np.mean(y)),
                    rms(y),

                    rms(g),
                    abs_p95(g),

                    rms(l),
                )
            )

    return np.asarray(
        out,
        dtype=np.float64,
    )


def build_representations(
    row,
    patch_cols,
):
    mu = float(
        row[
            "friction_mu"
        ]
    )

    H = np.asarray(
        [
            float(
                row[name]
            )
            for name in patch_cols
        ],
        dtype=np.float64,
    ).reshape(
        N_LONG,
        N_LAT,
    )

    gx, gy, lap = fields(
        H
    )

    raw = np.concatenate(
        (
            np.asarray(
                [mu],
                dtype=np.float64,
            ),
            H.reshape(-1),
        )
    )

    derivative = np.concatenate(
        (
            np.asarray(
                [mu],
                dtype=np.float64,
            ),
            H.reshape(-1),
            gx.reshape(-1),
            gy.reshape(-1),
            lap.reshape(-1),
        )
    )

    semantic = np.concatenate(
        (
            np.asarray(
                [mu],
                dtype=np.float64,
            ),
            semantic_region_features(
                H,
                gx,
                gy,
                lap,
            ),
        )
    )

    return {
        "raw_height":
            raw,

        "derivative_field":
            derivative,

        "semantic_regions":
            semantic,
    }


def standardize_distance(
    X,
    xh,
):
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
        - xhn[
            None,
            :
        ],
        axis=1,
    )


def main():
    if OUT_DIR.exists():
        raise SystemExit(
            f"REFUSING TO OVERWRITE: "
            f"{OUT_DIR}"
        )

    for path in (
        PATCH_CSV,
        PATCH_MANIFEST,
        Q_CSV,
        PHYSICAL_CSV,
        T63_MANIFEST,
    ):
        if not path.exists():
            raise FileNotFoundError(
                path
            )


    patch_manifest = json.loads(
        PATCH_MANIFEST.read_text()
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
            "T6.3a reports heldout use."
        )

    if list(
        patch_manifest[
            "patch_shape"
        ]
    ) != [16, 12]:
        raise RuntimeError(
            "Expected 16x12 patch."
        )


    t63 = json.loads(
        T63_MANIFEST.read_text()
    )

    if t63.get(
        "status"
    ) != "COMPUTE_PASS":
        raise RuntimeError(
            "T6.3b is not COMPUTE_PASS."
        )


    terrain_rows = read_csv(
        PATCH_CSV
    )

    patch_cols = patch_columns(
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


    terrain_lookup = {
        row[
            "context_id"
        ]:
            row
        for row in terrain_rows
    }

    context_ids = (
        original_contexts
        + extension_contexts
    )

    if set(
        terrain_lookup
    ) != set(
        context_ids
    ):
        raise RuntimeError(
            "Terrain context set mismatch."
        )


    reps = {}

    for cid in context_ids:
        reps[
            cid
        ] = build_representations(
            terrain_lookup[
                cid
            ],
            patch_cols,
        )


    representation_names = tuple(
        reps[
            context_ids[0]
        ].keys()
    )

    print(
        "representation dimensions:"
    )

    for name in representation_names:
        print(
            f"  {name:<24}: "
            f"{reps[context_ids[0]][name].shape[0]}"
        )


    # ========================================================
    # Q surfaces
    # ========================================================

    q_rows = read_csv(
        Q_CSV
    )

    q_group = {}

    for row in q_rows:
        cid = row[
            "context_id"
        ]

        if cid in set(
            context_ids
        ):
            q_group.setdefault(
                cid,
                [],
            ).append(
                row
            )

    q_lookup = {}
    beta_lookup = {}

    for cid in context_ids:
        rows = sorted(
            q_group[
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


    # ========================================================
    # Physical J surfaces
    # ========================================================

    physical_rows = read_csv(
        PHYSICAL_CSV
    )

    J = {}

    for row in physical_rows:
        cid = row[
            "context_id"
        ]

        if cid not in set(
            context_ids
        ):
            continue

        beta = row[
            "beta_name"
        ]

        J[
            (
                cid,
                beta,
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

    for cid in context_ids:
        beta_order = beta_lookup[
            cid
        ]

        J_lookup[
            cid
        ] = np.stack(
            [
                J[
                    (
                        cid,
                        beta,
                    )
                ]
                for beta in beta_order
            ],
            axis=0,
        )


    # ========================================================
    # Strict LOCO representation audit
    # ========================================================

    fold_rows = []

    for representation in representation_names:

        for held in original_contexts:

            train_contexts = (
                [
                    cid
                    for cid in original_contexts
                    if cid != held
                ]
                + extension_contexts
            )

            if len(
                train_contexts
            ) != 32:
                raise RuntimeError(
                    "Expected 32 TRAIN contexts."
                )

            X = np.stack(
                [
                    reps[
                        cid
                    ][
                        representation
                    ]
                    for cid in train_contexts
                ],
                axis=0,
            )

            xh = reps[
                held
            ][
                representation
            ]

            distance = standardize_distance(
                X,
                xh,
            )

            nn_index = int(
                np.argmin(
                    distance
                )
            )

            near = train_contexts[
                nn_index
            ]


            held_q = q_lookup[
                held
            ]

            near_q = q_lookup[
                near
            ]

            held_J = J_lookup[
                held
            ]

            near_J = J_lookup[
                near
            ]


            oracle_index = int(
                np.argmin(
                    held_q
                )
            )

            nn_selected_index = int(
                np.argmin(
                    near_q
                )
            )


            mean_train_q = np.mean(
                np.stack(
                    [
                        q_lookup[
                            cid
                        ]
                        for cid in train_contexts
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


            true_order = np.argsort(
                held_q,
                kind="stable",
            )

            true_rank = (
                int(
                    np.where(
                        true_order
                        == nn_selected_index
                    )[0][0]
                )
                + 1
            )


            q_excess = float(
                held_q[
                    nn_selected_index
                ]
                - held_q[
                    oracle_index
                ]
            )

            baseline_excess = float(
                held_q[
                    baseline_index
                ]
                - held_q[
                    oracle_index
                ]
            )


            fold_rows.append(
                {
                    "representation":
                        representation,

                    "held_context":
                        held,

                    "nearest_context":
                        near,

                    "nearest_distance":
                        float(
                            distance[
                                nn_index
                            ]
                        ),

                    "rho_J_motion":
                        spearman(
                            held_J[:, 0],
                            near_J[:, 0],
                        ),

                    "rho_J_stability":
                        spearman(
                            held_J[:, 1],
                            near_J[:, 1],
                        ),

                    "rho_J_energy":
                        spearman(
                            held_J[:, 2],
                            near_J[:, 2],
                        ),

                    "rho_Q":
                        spearman(
                            held_q,
                            near_q,
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
                            nn_selected_index
                        ],

                    "true_rank":
                        true_rank,

                    "exact":
                        int(
                            true_rank == 1
                        ),

                    "top3":
                        int(
                            true_rank <= 3
                        ),

                    "top5":
                        int(
                            true_rank <= 5
                        ),

                    "Q_excess":
                        q_excess,

                    "baseline_Q_excess":
                        baseline_excess,

                    "beats_baseline":
                        int(
                            q_excess
                            < baseline_excess
                            - EPS
                        ),
                }
            )


    # ========================================================
    # Aggregate each representation
    # ========================================================

    summary_rows = []

    for representation in representation_names:
        rows = [
            row
            for row in fold_rows
            if row[
                "representation"
            ] == representation
        ]

        def values(key):
            return np.asarray(
                [
                    row[
                        key
                    ]
                    for row in rows
                ],
                dtype=np.float64,
            )

        summary_rows.append(
            {
                "representation":
                    representation,

                "context_dim":
                    int(
                        reps[
                            context_ids[0]
                        ][
                            representation
                        ].shape[0]
                    ),

                "rho_J_motion_mean":
                    float(
                        np.mean(
                            values(
                                "rho_J_motion"
                            )
                        )
                    ),

                "rho_J_stability_mean":
                    float(
                        np.mean(
                            values(
                                "rho_J_stability"
                            )
                        )
                    ),

                "rho_J_energy_mean":
                    float(
                        np.mean(
                            values(
                                "rho_J_energy"
                            )
                        )
                    ),

                "rho_Q_mean":
                    float(
                        np.mean(
                            values(
                                "rho_Q"
                            )
                        )
                    ),

                "exact_fraction":
                    float(
                        np.mean(
                            values(
                                "exact"
                            )
                        )
                    ),

                "top3_fraction":
                    float(
                        np.mean(
                            values(
                                "top3"
                            )
                        )
                    ),

                "top5_fraction":
                    float(
                        np.mean(
                            values(
                                "top5"
                            )
                        )
                    ),

                "true_rank_mean":
                    float(
                        np.mean(
                            values(
                                "true_rank"
                            )
                        )
                    ),

                "Q_excess_mean":
                    float(
                        np.mean(
                            values(
                                "Q_excess"
                            )
                        )
                    ),

                "Q_excess_median":
                    float(
                        np.median(
                            values(
                                "Q_excess"
                            )
                        )
                    ),

                "beats_baseline_fraction":
                    float(
                        np.mean(
                            values(
                                "beats_baseline"
                            )
                        )
                    ),
            }
        )


    # ========================================================
    # Raw baseline must reproduce T6.3b.
    # ========================================================

    raw = next(
        row
        for row in summary_rows
        if row[
            "representation"
        ] == "raw_height"
    )

    t63_agg = t63[
        "aggregate"
    ]

    checks = (
        (
            raw[
                "rho_Q_mean"
            ],
            float(
                t63_agg[
                    "nn_q_spearman_mean"
                ]
            ),
        ),
        (
            raw[
                "Q_excess_mean"
            ],
            float(
                t63_agg[
                    "nn_Q_excess_mean"
                ]
            ),
        ),
        (
            raw[
                "top5_fraction"
            ],
            float(
                t63_agg[
                    "nn_top5_fraction"
                ]
            ),
        ),
        (
            raw[
                "beats_baseline_fraction"
            ],
            float(
                t63_agg[
                    "nn_beats_baseline_fraction"
                ]
            ),
        ),
    )

    for found, expected in checks:
        if not math.isclose(
            found,
            expected,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise RuntimeError(
                "Raw-height regression against "
                "T6.3b failed: "
                f"found={found}, "
                f"expected={expected}"
            )


    OUT_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )


    def write(path, rows):
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


    write(
        OUT_FOLDS,
        fold_rows,
    )

    write(
        OUT_SUMMARY,
        summary_rows,
    )


    manifest = {
        "schema":
            "icra27_os_t6p6a_semantic_geometry_identifiability_v0",

        "status":
            "COMPUTE_PASS",

        "heldout_used":
            False,

        "purpose":
            (
                "Test whether weak J_stability/J_energy "
                "transfer arises from inappropriate raw "
                "height-space geometry rather than "
                "insufficient spatial resolution."
            ),

        "representations": {
            "raw_height":
                (
                    "friction + flattened 16x12 "
                    "relative-height patch; exact T6.3b "
                    "baseline."
                ),

            "derivative_field":
                (
                    "friction + height + longitudinal "
                    "slope + lateral slope + Laplacian "
                    "fields."
                ),

            "semantic_regions":
                (
                    "friction + physically predefined "
                    "global and 3x2 regional summaries "
                    "of height, slope, gradient magnitude "
                    "and curvature."
                ),
        },

        "sampling_spacing_m": {
            "longitudinal":
                DX,
            "lateral":
                DY,
        },

        "evaluation":
            (
                "Same strict 18-fold original-rough LOCO "
                "with remaining17+extension15 TRAIN-only "
                "nearest-neighbor pool. No learned model "
                "and no outcome-derived terrain features."
            ),

        "raw_t6p3b_regression":
            "PASS",

        "summary":
            summary_rows,

        "decision_rule":
            {
                "semantic_improves_JS_JE":
                    (
                        "Raw terrain contained useful "
                        "information but the previous "
                        "flattened representation/metric "
                        "was poorly aligned with contact-"
                        "relevant geometry."
                    ),

                "all_remain_weak":
                    (
                        "Simple static terrain geometry "
                        "descriptors are insufficient to "
                        "identify stability/energy beta "
                        "response; next inspect richer "
                        "pre-decision causal context or "
                        "structured spatial models rather "
                        "than more grid resolution."
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
        "ICRA27 OS-T6.6a SEMANTIC GEOMETRY "
        "IDENTIFIABILITY AUDIT"
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
            if key in (
                "representation",
                "context_dim",
            ):
                continue

            print(
                f"  {key:<34}: {value}"
            )

    print()
    print(
        "[ICRA27] OS-T6.6a semantic geometry "
        "identifiability: COMPUTE PASS"
    )
    print("=" * 118)


if __name__ == "__main__":
    main()
