from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


HIGHRES_CSV = (
    ROOT
    / "results/icra27"
    / "os_t6p3a_highres_height_patches_v0"
    / "highres_height_patches.csv"
)

HIGHRES_MANIFEST = (
    ROOT
    / "results/icra27"
    / "os_t6p3a_highres_height_patches_v0"
    / "highres_height_patch_manifest.json"
)


QUALITY_SURFACE = (
    ROOT
    / "results/icra27"
    / "os_t6p1_balanced_physical_quality_surface_v0"
    / "balanced_beta_quality_surface.csv"
)

QUALITY_MANIFEST = (
    ROOT
    / "results/icra27"
    / "os_t6p1_balanced_physical_quality_surface_v0"
    / "balanced_quality_surface_manifest.json"
)


LOWRES_SUMMARY = (
    ROOT
    / "results/icra27"
    / "os_t6p2b_raw_context_q_identifiability_v0"
    / "context_identifiability_summary.csv"
)

LOWRES_MANIFEST = (
    ROOT
    / "results/icra27"
    / "os_t6p2b_raw_context_q_identifiability_v0"
    / "raw_context_q_identifiability_manifest.json"
)


OUT_DIR = (
    ROOT
    / "results/icra27"
    / "os_t6p3b_highres_context_q_identifiability_v0"
)

OUT_CONTEXT = (
    OUT_DIR
    / "highres_context_identifiability_summary.csv"
)

OUT_COMPARE = (
    OUT_DIR
    / "lowres_vs_highres_context_comparison.csv"
)

OUT_MANIFEST = (
    OUT_DIR
    / "highres_context_q_identifiability_manifest.json"
)


EXPECTED_OLD = 18
EXPECTED_EXT = 15
EXPECTED_TOTAL = 33
EXPECTED_BETA = 21

N_LONG = 16
N_LAT = 12
PATCH_DIM = 192

EPS = 1.0e-12


def read_csv(path):
    with path.open(
        "r",
        newline="",
    ) as f:
        return list(
            csv.DictReader(f)
        )


def highres_patch_columns(rows):
    fields = list(
        rows[0].keys()
    )

    cols = [
        name
        for name in fields
        if re.fullmatch(
            r"h_l\d{2}_r\d{2}_m",
            name,
        )
    ]

    def key(name):
        match = re.fullmatch(
            r"h_l(\d{2})_r(\d{2})_m",
            name,
        )

        if match is None:
            raise RuntimeError(
                f"Invalid patch field: {name}"
            )

        return (
            int(
                match.group(1)
            ),
            int(
                match.group(2)
            ),
        )

    cols = sorted(
        cols,
        key=key,
    )

    expected = [
        f"h_l{i:02d}_r{j:02d}_m"
        for i in range(N_LONG)
        for j in range(N_LAT)
    ]

    if cols != expected:
        raise RuntimeError(
            "High-resolution patch schema mismatch.\n"
            f"found={len(cols)}, "
            f"expected={len(expected)}"
        )

    return cols


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


def ranks(x):
    order = np.argsort(
        x,
        kind="stable",
    )

    r = np.empty(
        len(x),
        dtype=np.float64,
    )

    r[order] = np.arange(
        1,
        len(x) + 1,
        dtype=np.float64,
    )

    return r


def main():
    if OUT_DIR.exists():
        raise SystemExit(
            f"REFUSING TO OVERWRITE: "
            f"{OUT_DIR}"
        )

    for path in (
        HIGHRES_CSV,
        HIGHRES_MANIFEST,
        QUALITY_SURFACE,
        QUALITY_MANIFEST,
        LOWRES_SUMMARY,
        LOWRES_MANIFEST,
    ):
        if not path.exists():
            raise FileNotFoundError(
                path
            )


    # ========================================================
    # Frozen contracts
    # ========================================================

    highres_manifest = json.loads(
        HIGHRES_MANIFEST.read_text()
    )

    if highres_manifest.get(
        "status"
    ) != "FREEZE_PASS":
        raise RuntimeError(
            "T6.3a high-resolution patch "
            "is not FREEZE_PASS."
        )

    if bool(
        highres_manifest.get(
            "heldout_used",
            False,
        )
    ):
        raise RuntimeError(
            "T6.3a reports heldout use."
        )

    if list(
        highres_manifest[
            "patch_shape"
        ]
    ) != [16, 12]:
        raise RuntimeError(
            "Expected T6.3a patch shape [16,12]."
        )

    if int(
        highres_manifest[
            "patch_dim"
        ]
    ) != PATCH_DIM:
        raise RuntimeError(
            "Expected T6.3a patch dimension 192."
        )

    if (
        highres_manifest.get(
            "controlled_change"
        )
        != "Sampling resolution only: 8x6 -> 16x12."
    ):
        raise RuntimeError(
            "T6.3a controlled-change contract mismatch."
        )


    quality_manifest = json.loads(
        QUALITY_MANIFEST.read_text()
    )

    if quality_manifest.get(
        "status"
    ) != "FREEZE_PASS":
        raise RuntimeError(
            "T6.1 quality surface "
            "is not FREEZE_PASS."
        )

    if bool(
        quality_manifest.get(
            "heldout_used",
            False,
        )
    ):
        raise RuntimeError(
            "T6.1 reports heldout use."
        )


    lowres_manifest = json.loads(
        LOWRES_MANIFEST.read_text()
    )

    if lowres_manifest.get(
        "status"
    ) != "COMPUTE_PASS":
        raise RuntimeError(
            "T6.2b baseline is not COMPUTE_PASS."
        )


    # ========================================================
    # High-resolution terrain context
    # ========================================================

    terrain_rows = read_csv(
        HIGHRES_CSV
    )

    if len(
        terrain_rows
    ) != EXPECTED_TOTAL:
        raise RuntimeError(
            f"Expected 33 high-res terrain rows; "
            f"got {len(terrain_rows)}."
        )

    patch_cols = highres_patch_columns(
        terrain_rows
    )

    print(
        "high-resolution patch features:",
        len(
            patch_cols
        ),
    )

    original_seeds = [
        int(x)
        for x in highres_manifest[
            "original_seeds"
        ]
    ]

    extension_seeds = [
        int(x)
        for x in highres_manifest[
            "extension_seeds"
        ]
    ]

    if len(
        original_seeds
    ) != EXPECTED_OLD:
        raise RuntimeError(
            "Expected 18 original rough seeds."
        )

    if len(
        extension_seeds
    ) != EXPECTED_EXT:
        raise RuntimeError(
            "Expected 15 extension rough seeds."
        )

    original_contexts = [
        f"rough_seed_{seed}"
        for seed in original_seeds
    ]

    extension_contexts = [
        f"rough_seed_{seed}"
        for seed in extension_seeds
    ]


    row_lookup = {
        row[
            "context_id"
        ]:
            row
        for row in terrain_rows
    }

    expected_contexts = (
        original_contexts
        + extension_contexts
    )

    if set(
        row_lookup
    ) != set(
        expected_contexts
    ):
        raise RuntimeError(
            "High-resolution context set mismatch."
        )


    context_lookup = {}

    for cid in expected_contexts:
        row = row_lookup[
            cid
        ]

        vec = [
            float(
                row[
                    "friction_mu"
                ]
            )
        ]

        vec.extend(
            float(
                row[name]
            )
            for name in patch_cols
        )

        vec = np.asarray(
            vec,
            dtype=np.float64,
        )

        if vec.shape != (193,):
            raise RuntimeError(
                f"{cid}: expected 193D context; "
                f"got {vec.shape}."
            )

        if not np.all(
            np.isfinite(
                vec
            )
        ):
            raise RuntimeError(
                f"{cid}: non-finite context."
            )

        context_lookup[
            cid
        ] = vec


    # ========================================================
    # Ground-truth balanced Q surfaces
    # ========================================================

    q_rows = read_csv(
        QUALITY_SURFACE
    )

    grouped = {}

    for row in q_rows:
        cid = row[
            "context_id"
        ]

        if cid not in context_lookup:
            continue

        grouped.setdefault(
            cid,
            [],
        ).append(
            row
        )


    q_lookup = {}

    beta_order = None

    for cid in expected_contexts:
        if cid not in grouped:
            raise RuntimeError(
                f"Missing Q surface for {cid}."
            )

        local = sorted(
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
            local
        ) != EXPECTED_BETA:
            raise RuntimeError(
                f"{cid}: expected 21 Q rows."
            )

        names = [
            row[
                "beta_name"
            ]
            for row in local
        ]

        if beta_order is None:
            beta_order = names

        elif names != beta_order:
            raise RuntimeError(
                f"{cid}: beta bank/order mismatch."
            )

        q = np.asarray(
            [
                float(
                    row[
                        "quality_Q"
                    ]
                )
                for row in local
            ],
            dtype=np.float64,
        )

        if not np.all(
            np.isfinite(
                q
            )
        ):
            raise RuntimeError(
                f"{cid}: non-finite Q."
            )

        q_lookup[
            cid
        ] = q


    # ========================================================
    # Strict LOCO oracle 1-NN transfer
    #
    # Exactly the same semantics as T6.2b:
    #   held = one original rough terrain
    #   train = remaining 17 original
    #           + 15 extension
    #   feature normalization = TRAIN only
    #   nearest-neighbor metric = standardized Euclidean
    # ========================================================

    results = []

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
                context_lookup[
                    cid
                ]
                for cid in train_contexts
            ],
            axis=0,
        )

        xh = context_lookup[
            held
        ].copy()


        # TRAIN-only normalization.
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


        distances = np.linalg.norm(
            Xn
            - xhn[
                None,
                :
            ],
            axis=1,
        )

        nn_idx = int(
            np.argmin(
                distances
            )
        )

        nn_context = train_contexts[
            nn_idx
        ]


        held_q = q_lookup[
            held
        ]

        nn_q = q_lookup[
            nn_context
        ]


        oracle_idx = int(
            np.argmin(
                held_q
            )
        )

        nn_selected_idx = int(
            np.argmin(
                nn_q
            )
        )


        # Same no-context baseline:
        # minimum mean TRAIN Q.
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

        baseline_idx = int(
            np.argmin(
                mean_train_q
            )
        )


        true_order = np.argsort(
            held_q,
            kind="stable",
        )

        nn_true_rank = (
            int(
                np.where(
                    true_order
                    == nn_selected_idx
                )[0][0]
            )
            + 1
        )


        held_rank = ranks(
            held_q
        )

        nn_rank = ranks(
            nn_q
        )


        q_excess = float(
            held_q[
                nn_selected_idx
            ]
            - held_q[
                oracle_idx
            ]
        )

        baseline_excess = float(
            held_q[
                baseline_idx
            ]
            - held_q[
                oracle_idx
            ]
        )


        row = {
            "held_context":
                held,

            "nearest_context":
                nn_context,

            "nearest_distance":
                float(
                    distances[
                        nn_idx
                    ]
                ),

            "q_surface_pearson":
                corr(
                    held_q,
                    nn_q,
                ),

            "q_surface_spearman":
                corr(
                    held_rank,
                    nn_rank,
                ),

            "oracle_beta":
                beta_order[
                    oracle_idx
                ],

            "nn_selected_beta":
                beta_order[
                    nn_selected_idx
                ],

            "baseline_beta":
                beta_order[
                    baseline_idx
                ],

            "nn_exact":
                int(
                    nn_selected_idx
                    == oracle_idx
                ),

            "nn_true_rank":
                nn_true_rank,

            "nn_top3":
                int(
                    nn_true_rank <= 3
                ),

            "nn_top5":
                int(
                    nn_true_rank <= 5
                ),

            "nn_Q_excess":
                q_excess,

            "baseline_Q_excess":
                baseline_excess,

            "nn_beats_baseline":
                int(
                    q_excess
                    < baseline_excess
                    - EPS
                ),
        }

        results.append(
            row
        )

        print(
            f"{held:<16} "
            f"NN={nn_context:<16} "
            f"d={row['nearest_distance']:.3f} "
            f"rho={row['q_surface_spearman']:+.3f} "
            f"rank={nn_true_rank:<2} "
            f"Qex={q_excess:.4f} "
            f"baseEx={baseline_excess:.4f}"
        )


    def arr(key):
        return np.asarray(
            [
                row[key]
                for row in results
            ],
            dtype=np.float64,
        )


    aggregate = {
        "nn_q_pearson_mean":
            float(
                np.mean(
                    arr(
                        "q_surface_pearson"
                    )
                )
            ),

        "nn_q_spearman_mean":
            float(
                np.mean(
                    arr(
                        "q_surface_spearman"
                    )
                )
            ),

        "nn_exact_fraction":
            float(
                np.mean(
                    arr(
                        "nn_exact"
                    )
                )
            ),

        "nn_top3_fraction":
            float(
                np.mean(
                    arr(
                        "nn_top3"
                    )
                )
            ),

        "nn_top5_fraction":
            float(
                np.mean(
                    arr(
                        "nn_top5"
                    )
                )
            ),

        "nn_true_rank_mean":
            float(
                np.mean(
                    arr(
                        "nn_true_rank"
                    )
                )
            ),

        "nn_true_rank_median":
            float(
                np.median(
                    arr(
                        "nn_true_rank"
                    )
                )
            ),

        "nn_Q_excess_mean":
            float(
                np.mean(
                    arr(
                        "nn_Q_excess"
                    )
                )
            ),

        "nn_Q_excess_median":
            float(
                np.median(
                    arr(
                        "nn_Q_excess"
                    )
                )
            ),

        "baseline_Q_excess_mean":
            float(
                np.mean(
                    arr(
                        "baseline_Q_excess"
                    )
                )
            ),

        "nn_beats_baseline_fraction":
            float(
                np.mean(
                    arr(
                        "nn_beats_baseline"
                    )
                )
            ),
    }


    # ========================================================
    # Paired comparison against frozen T6.2b low-resolution
    # ========================================================

    low_rows = read_csv(
        LOWRES_SUMMARY
    )

    low_lookup = {
        row[
            "held_context"
        ]:
            row
        for row in low_rows
    }

    if set(
        low_lookup
    ) != set(
        original_contexts
    ):
        raise RuntimeError(
            "T6.2b held-context set mismatch."
        )

    comparison_rows = []

    for hi in results:
        cid = hi[
            "held_context"
        ]

        lo = low_lookup[
            cid
        ]

        comparison_rows.append(
            {
                "held_context":
                    cid,

                "lowres_nn_context":
                    lo[
                        "nearest_context"
                    ],

                "highres_nn_context":
                    hi[
                        "nearest_context"
                    ],

                "lowres_spearman":
                    float(
                        lo[
                            "q_surface_spearman"
                        ]
                    ),

                "highres_spearman":
                    float(
                        hi[
                            "q_surface_spearman"
                        ]
                    ),

                "spearman_delta_high_minus_low":
                    float(
                        hi[
                            "q_surface_spearman"
                        ]
                    )
                    - float(
                        lo[
                            "q_surface_spearman"
                        ]
                    ),

                "lowres_true_rank":
                    int(
                        lo[
                            "nn_true_rank"
                        ]
                    ),

                "highres_true_rank":
                    int(
                        hi[
                            "nn_true_rank"
                        ]
                    ),

                "rank_delta_high_minus_low":
                    int(
                        hi[
                            "nn_true_rank"
                        ]
                    )
                    - int(
                        lo[
                            "nn_true_rank"
                        ]
                    ),

                "lowres_Q_excess":
                    float(
                        lo[
                            "nn_Q_excess"
                        ]
                    ),

                "highres_Q_excess":
                    float(
                        hi[
                            "nn_Q_excess"
                        ]
                    ),

                "Q_excess_delta_high_minus_low":
                    float(
                        hi[
                            "nn_Q_excess"
                        ]
                    )
                    - float(
                        lo[
                            "nn_Q_excess"
                        ]
                    ),

                "baseline_Q_excess":
                    float(
                        hi[
                            "baseline_Q_excess"
                        ]
                    ),
            }
        )


    low_agg = lowres_manifest[
        "aggregate"
    ]

    comparison = {
        "lowres_patch_shape":
            [
                8,
                6,
            ],

        "highres_patch_shape":
            [
                16,
                12,
            ],

        "lowres_spearman_mean":
            float(
                low_agg[
                    "nn_q_spearman_mean"
                ]
            ),

        "highres_spearman_mean":
            aggregate[
                "nn_q_spearman_mean"
            ],

        "spearman_mean_delta":
            (
                aggregate[
                    "nn_q_spearman_mean"
                ]
                - float(
                    low_agg[
                        "nn_q_spearman_mean"
                    ]
                )
            ),

        "lowres_Q_excess_mean":
            float(
                low_agg[
                    "nn_Q_excess_mean"
                ]
            ),

        "highres_Q_excess_mean":
            aggregate[
                "nn_Q_excess_mean"
            ],

        "Q_excess_mean_delta":
            (
                aggregate[
                    "nn_Q_excess_mean"
                ]
                - float(
                    low_agg[
                        "nn_Q_excess_mean"
                    ]
                )
            ),

        "lowres_top5_fraction":
            float(
                low_agg[
                    "nn_top5_fraction"
                ]
            ),

        "highres_top5_fraction":
            aggregate[
                "nn_top5_fraction"
            ],

        "lowres_beats_baseline_fraction":
            float(
                low_agg[
                    "nn_beats_baseline_fraction"
                ]
            ),

        "highres_beats_baseline_fraction":
            aggregate[
                "nn_beats_baseline_fraction"
            ],

        "contexts_with_lower_Q_excess_highres":
            int(
                sum(
                    row[
                        "Q_excess_delta_high_minus_low"
                    ]
                    < -EPS
                    for row in comparison_rows
                )
            ),

        "contexts_with_higher_spearman_highres":
            int(
                sum(
                    row[
                        "spearman_delta_high_minus_low"
                    ]
                    > EPS
                    for row in comparison_rows
                )
            ),
    }


    # ========================================================
    # Write only after all checks complete
    # ========================================================

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )


    with OUT_CONTEXT.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                results[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            results
        )


    with OUT_COMPARE.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                comparison_rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            comparison_rows
        )


    manifest = {
        "schema":
            "icra27_os_t6p3b_highres_context_q_identifiability_v0",

        "status":
            "COMPUTE_PASS",

        "heldout_used":
            False,

        "terrain_context":
            (
                "friction_mu + frozen 16x12 "
                "high-resolution relative-height patch"
            ),

        "context_dim":
            193,

        "evaluation":
            (
                "Strict LOCO over the same 18 original "
                "rough contexts. For each fold, nearest "
                "neighbor is selected from the remaining "
                "17 original + 15 frozen extension "
                "TRAIN contexts."
            ),

        "distance_metric":
            (
                "TRAIN-standardized Euclidean distance, "
                "identical procedure to T6.2b"
            ),

        "controlled_comparison":
            (
                "T6.2b 8x6 versus T6.3b 16x12; "
                "spatial domain, terrain seeds, Q surface, "
                "LOCO split and 1-NN algorithm unchanged."
            ),

        "aggregate":
            aggregate,

        "lowres_vs_highres":
            comparison,

        "decision_rule":
            (
                "A substantial improvement in Q-surface "
                "rank correlation and selected-beta "
                "Q excess would support insufficient "
                "spatial resolution as a major bottleneck. "
                "Little or no improvement would argue "
                "against resolution as the primary cause "
                "and motivate testing richer interaction/"
                "state context rather than further "
                "terrain-grid resolution tuning."
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


    print()
    print("=" * 118)
    print(
        "ICRA27 OS-T6.3b HIGH-RES CONTEXT "
        "Q IDENTIFIABILITY AUDIT"
    )
    print("=" * 118)

    print()
    print(
        "HIGH-RES aggregate"
    )

    for key, value in aggregate.items():
        print(
            f"  {key:<38}: {value}"
        )

    print()
    print(
        "LOW-RES -> HIGH-RES controlled comparison"
    )

    for key, value in comparison.items():
        print(
            f"  {key:<38}: {value}"
        )

    print()
    print(
        "[ICRA27] OS-T6.3b high-res context "
        "Q identifiability: COMPUTE PASS"
    )

    print("=" * 118)


if __name__ == "__main__":
    main()
