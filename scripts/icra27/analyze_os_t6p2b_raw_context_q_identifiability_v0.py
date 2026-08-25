from __future__ import annotations

import csv
import json
import re
import runpy
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]

F4_SOURCE = (
    ROOT
    / "scripts/icra27"
    / "train_os_t5p5f4_coverage_expanded_log_regret_selector_loco_v0.py"
)

contract = runpy.run_path(
    str(F4_SOURCE),
    run_name="icra27_t6p2b_contract",
)

PATCH_CSV = Path(
    contract["PATCH_CSV"]
)

EXT_PATCH_CSV = Path(
    contract["EXT_PATCH_CSV"]
)

ATLAS_MANIFEST = Path(
    contract["ATLAS_MANIFEST"]
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

OUT_DIR = (
    ROOT
    / "results/icra27"
    / "os_t6p2b_raw_context_q_identifiability_v0"
)

OUT_CONTEXT = (
    OUT_DIR
    / "context_identifiability_summary.csv"
)

OUT_MANIFEST = (
    OUT_DIR
    / "raw_context_q_identifiability_manifest.json"
)


EXPECTED_OLD = 18
EXPECTED_EXT = 15
EXPECTED_BETA = 21

EPS = 1.0e-12


def read_csv(path):
    with path.open(
        "r",
        newline="",
    ) as f:
        return list(
            csv.DictReader(f)
        )


def patch_columns(rows):
    cols = [
        x
        for x in rows[0].keys()
        if re.fullmatch(
            r"h_l\d{2}_r\d{2}_m",
            x,
        )
    ]

    cols = sorted(
        cols,
        key=lambda x: (
            int(
                re.fullmatch(
                    r"h_l(\d{2})_r(\d{2})_m",
                    x,
                ).group(1)
            ),
            int(
                re.fullmatch(
                    r"h_l(\d{2})_r(\d{2})_m",
                    x,
                ).group(2)
            ),
        ),
    )

    expected = [
        f"h_l{l:02d}_r{r:02d}_m"
        for l in range(8)
        for r in range(6)
    ]

    if cols != expected:
        raise RuntimeError(
            f"Expected frozen 48D patch schema; "
            f"got {len(cols)} columns."
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
            f"REFUSING TO OVERWRITE: {OUT_DIR}"
        )

    for path in (
        PATCH_CSV,
        EXT_PATCH_CSV,
        ATLAS_MANIFEST,
        QUALITY_SURFACE,
        QUALITY_MANIFEST,
    ):
        if not path.exists():
            raise FileNotFoundError(
                path
            )

    qm = json.loads(
        QUALITY_MANIFEST.read_text()
    )

    if qm.get(
        "status"
    ) != "FREEZE_PASS":
        raise RuntimeError(
            "T6.1 is not FREEZE_PASS."
        )

    atlas_manifest = json.loads(
        ATLAS_MANIFEST.read_text()
    )

    ext_seeds = [
        int(x)
        for x in atlas_manifest[
            "full_grid_extension_seeds"
        ]
    ]

    if len(ext_seeds) != EXPECTED_EXT:
        raise RuntimeError(
            "Expected 15 eligible extension seeds."
        )

    extension_contexts = [
        f"rough_seed_{x}"
        for x in ext_seeds
    ]


    # ========================================================
    # Raw terrain context
    # ========================================================

    old_rows_all = read_csv(
        PATCH_CSV
    )

    ext_rows_all = read_csv(
        EXT_PATCH_CSV
    )

    old_rows = [
        row
        for row in old_rows_all
        if row[
            "context_id"
        ].startswith(
            "rough_seed_"
        )
    ]

    if len(old_rows) != EXPECTED_OLD:
        raise RuntimeError(
            "Expected 18 original rough contexts."
        )

    ext_lookup = {
        row[
            "context_id"
        ]:
            row
        for row in ext_rows_all
    }

    ext_rows = [
        ext_lookup[cid]
        for cid in extension_contexts
    ]

    patch_cols = patch_columns(
        old_rows
    )

    context_lookup = {}

    for row in (
        old_rows
        + ext_rows
    ):
        cid = row[
            "context_id"
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
                row[col]
            )
            for col in patch_cols
        )

        context_lookup[
            cid
        ] = np.asarray(
            vec,
            dtype=np.float64,
        )

    if len(context_lookup) != 33:
        raise RuntimeError(
            "Expected 33 rough contexts."
        )


    # ========================================================
    # Ground-truth Q vectors
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
            []
        ).append(
            row
        )

    q_lookup = {}

    beta_order = None

    for cid in context_lookup:
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

        if len(local) != EXPECTED_BETA:
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
                f"{cid}: beta bank mismatch."
            )

        q_lookup[
            cid
        ] = np.asarray(
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


    original_contexts = [
        row[
            "context_id"
        ]
        for row in old_rows
    ]

    results = []


    # ========================================================
    # Strict LOCO 1-NN transfer
    # ========================================================

    for held in original_contexts:
        old_train = [
            cid
            for cid in original_contexts
            if cid != held
        ]

        train_contexts = (
            old_train
            + extension_contexts
        )

        if len(train_contexts) != 32:
            raise RuntimeError(
                "Expected 32 TRAIN contexts."
            )

        X = np.stack(
            [
                context_lookup[cid]
                for cid in train_contexts
            ],
            axis=0,
        )

        xh = context_lookup[
            held
        ].copy()

        # TRAIN-only feature normalization.
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


        # Global no-context baseline.
        mean_train_q = np.mean(
            np.stack(
                [
                    q_lookup[cid]
                    for cid in train_contexts
                ],
                axis=0,
            ),
            axis=0,
        )

        base_idx = int(
            np.argmin(
                mean_train_q
            )
        )


        true_order = np.argsort(
            held_q,
            kind="stable",
        )

        nn_rank = int(
            np.where(
                true_order
                == nn_selected_idx
            )[0][0]
        ) + 1


        true_rank_vec = ranks(
            held_q
        )

        nn_rank_vec = ranks(
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
                base_idx
            ]
            - held_q[
                oracle_idx
            ]
        )


        results.append(
            {
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
                        true_rank_vec,
                        nn_rank_vec,
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
                        base_idx
                    ],

                "nn_exact":
                    int(
                        nn_selected_idx
                        == oracle_idx
                    ),

                "nn_true_rank":
                    nn_rank,

                "nn_top3":
                    int(
                        nn_rank <= 3
                    ),

                "nn_top5":
                    int(
                        nn_rank <= 5
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
        )

        print(
            f"{held:<16} "
            f"NN={nn_context:<16} "
            f"d={distances[nn_idx]:.3f} "
            f"rho={results[-1]['q_surface_spearman']:+.3f} "
            f"rank={nn_rank:<2} "
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


    manifest = {
        "schema":
            "icra27_os_t6p2b_raw_context_q_identifiability_v0",

        "status":
            "COMPUTE_PASS",

        "heldout_used":
            False,

        "terrain_context":
            "friction_mu + frozen raw 48D local height patch",

        "evaluation":
            (
                "Strict LOCO over the same 18 original rough "
                "contexts; nearest neighbor selected from "
                "17 original + 15 extension TRAIN contexts."
            ),

        "purpose":
            (
                "Test whether geometric similarity in the "
                "current terrain context implies similarity "
                "of the ground-truth balanced beta-Q surface."
            ),

        "aggregate":
            aggregate,

        "decision_rule":
            (
                "If oracle 1-NN Q-surface transfer is weak "
                "and does not beat the no-context baseline, "
                "the current context representation is not "
                "sufficiently identifiable/smooth for "
                "c->Q learning. If 1-NN is strong, focus "
                "next on learned ranking/listwise objectives."
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
        "ICRA27 OS-T6.2b RAW-CONTEXT "
        "Q IDENTIFIABILITY AUDIT"
    )
    print("=" * 118)

    for key, value in aggregate.items():
        print(
            f"  {key:<34}: {value}"
        )

    print()
    print(
        "[ICRA27] OS-T6.2b raw-context "
        "Q identifiability: COMPUTE PASS"
    )
    print("=" * 118)


if __name__ == "__main__":
    main()
