from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]

ATLAS = (
    ROOT
    / "results/icra27"
    / "os_t5p5f3_expanded_physical_atlas_v0"
    / "physical_beta_response_atlas.csv"
)

ATLAS_MANIFEST = (
    ROOT
    / "results/icra27"
    / "os_t5p5f3_expanded_physical_atlas_v0"
    / "expanded_physical_atlas_manifest.json"
)

QUALITY = (
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

HIGHRES_NN = (
    ROOT
    / "results/icra27"
    / "os_t6p3b_highres_context_q_identifiability_v0"
    / "highres_context_identifiability_summary.csv"
)

HIGHRES_MANIFEST = (
    ROOT
    / "results/icra27"
    / "os_t6p3b_highres_context_q_identifiability_v0"
    / "highres_context_q_identifiability_manifest.json"
)

OUT_DIR = (
    ROOT
    / "results/icra27"
    / "os_t6p4_physical_response_vs_q_transfer_v0"
)

OUT_CONTEXT = (
    OUT_DIR
    / "physical_vs_q_transfer_by_context.csv"
)

OUT_MANIFEST = (
    OUT_DIR
    / "physical_vs_q_transfer_manifest.json"
)


EXPECTED_CONTEXTS = 18
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


def main():
    if OUT_DIR.exists():
        raise SystemExit(
            f"REFUSING TO OVERWRITE: {OUT_DIR}"
        )

    for path in (
        ATLAS,
        ATLAS_MANIFEST,
        QUALITY,
        QUALITY_MANIFEST,
        HIGHRES_NN,
        HIGHRES_MANIFEST,
    ):
        if not path.exists():
            raise FileNotFoundError(
                path
            )


    atlas_manifest = json.loads(
        ATLAS_MANIFEST.read_text()
    )

    quality_manifest = json.loads(
        QUALITY_MANIFEST.read_text()
    )

    high_manifest = json.loads(
        HIGHRES_MANIFEST.read_text()
    )

    if atlas_manifest.get(
        "status"
    ) != "FREEZE_PASS":
        raise RuntimeError(
            "Physical atlas is not FREEZE_PASS."
        )

    if quality_manifest.get(
        "status"
    ) != "FREEZE_PASS":
        raise RuntimeError(
            "Balanced Q surface is not FREEZE_PASS."
        )

    if high_manifest.get(
        "status"
    ) != "COMPUTE_PASS":
        raise RuntimeError(
            "T6.3b is not COMPUTE_PASS."
        )

    if any(
        bool(
            x.get(
                "heldout_used",
                False,
            )
        )
        for x in (
            atlas_manifest,
            quality_manifest,
            high_manifest,
        )
    ):
        raise RuntimeError(
            "A source artifact reports heldout use."
        )


    # ========================================================
    # Physical beta-response atlas
    # ========================================================

    atlas_rows = read_csv(
        ATLAS
    )

    physical = {}

    for row in atlas_rows:
        cid = row[
            "context_id"
        ]

        beta = row[
            "beta_name"
        ]

        physical[
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


    # ========================================================
    # Balanced Q surface + beta order
    # ========================================================

    quality_rows = read_csv(
        QUALITY
    )

    q_group = {}

    for row in quality_rows:
        q_group.setdefault(
            row[
                "context_id"
            ],
            [],
        ).append(
            row
        )

    q_lookup = {}

    beta_lookup = {}

    for cid, rows in q_group.items():
        rows = sorted(
            rows,
            key=lambda x:
                int(
                    x[
                        "beta_index"
                    ]
                ),
        )

        if len(rows) != EXPECTED_BETA:
            raise RuntimeError(
                f"{cid}: expected 21 beta rows."
            )

        beta_lookup[
            cid
        ] = [
            x[
                "beta_name"
            ]
            for x in rows
        ]

        q_lookup[
            cid
        ] = np.asarray(
            [
                float(
                    x[
                        "quality_Q"
                    ]
                )
                for x in rows
            ],
            dtype=np.float64,
        )


    nn_rows = read_csv(
        HIGHRES_NN
    )

    if len(
        nn_rows
    ) != EXPECTED_CONTEXTS:
        raise RuntimeError(
            f"Expected 18 T6.3b context rows; "
            f"got {len(nn_rows)}."
        )


    results = []

    for row in nn_rows:
        held = row[
            "held_context"
        ]

        near = row[
            "nearest_context"
        ]

        if (
            held not in q_lookup
            or near not in q_lookup
        ):
            raise RuntimeError(
                f"Missing Q surface: "
                f"{held} / {near}"
            )

        beta_order = beta_lookup[
            held
        ]

        if beta_lookup[
            near
        ] != beta_order:
            raise RuntimeError(
                f"Beta order mismatch: {held}/{near}"
            )


        J_held = np.stack(
            [
                physical[
                    (
                        held,
                        beta,
                    )
                ]
                for beta in beta_order
            ],
            axis=0,
        )

        J_near = np.stack(
            [
                physical[
                    (
                        near,
                        beta,
                    )
                ]
                for beta in beta_order
            ],
            axis=0,
        )

        q_held = q_lookup[
            held
        ]

        q_near = q_lookup[
            near
        ]


        rho_m = spearman(
            J_held[:, 0],
            J_near[:, 0],
        )

        rho_s = spearman(
            J_held[:, 1],
            J_near[:, 1],
        )

        rho_e = spearman(
            J_held[:, 2],
            J_near[:, 2],
        )

        rho_q = spearman(
            q_held,
            q_near,
        )


        q_order = np.argsort(
            q_held,
            kind="stable",
        )

        q_best = float(
            q_held[
                q_order[0]
            ]
        )

        q_second = float(
            q_held[
                q_order[1]
            ]
        )

        q_range = float(
            np.max(
                q_held
            )
            - np.min(
                q_held
            )
        )

        best_second_gap = (
            q_second
            - q_best
        )

        relative_gap = (
            best_second_gap
            / q_range
            if q_range > EPS
            else 0.0
        )


        results.append(
            {
                "held_context":
                    held,

                "nearest_context":
                    near,

                "rho_J_motion":
                    rho_m,

                "rho_J_stability":
                    rho_s,

                "rho_J_energy":
                    rho_e,

                "rho_J_mean":
                    float(
                        np.mean(
                            [
                                rho_m,
                                rho_s,
                                rho_e,
                            ]
                        )
                    ),

                "rho_Q":
                    rho_q,

                "rho_Jmean_minus_Q":
                    float(
                        np.mean(
                            [
                                rho_m,
                                rho_s,
                                rho_e,
                            ]
                        )
                        - rho_q
                    ),

                "Q_best_second_gap":
                    best_second_gap,

                "Q_range":
                    q_range,

                "Q_relative_best_second_gap":
                    relative_gap,

                "T6p3b_Q_excess":
                    float(
                        row[
                            "nn_Q_excess"
                        ]
                    ),
            }
        )

        print(
            f"{held:<16} "
            f"NN={near:<16} "
            f"rhoM={rho_m:+.3f} "
            f"rhoS={rho_s:+.3f} "
            f"rhoE={rho_e:+.3f} "
            f"rhoQ={rho_q:+.3f} "
            f"gap={best_second_gap:.5f}"
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
        "rho_J_motion_mean":
            float(
                np.mean(
                    arr(
                        "rho_J_motion"
                    )
                )
            ),

        "rho_J_stability_mean":
            float(
                np.mean(
                    arr(
                        "rho_J_stability"
                    )
                )
            ),

        "rho_J_energy_mean":
            float(
                np.mean(
                    arr(
                        "rho_J_energy"
                    )
                )
            ),

        "rho_J_objective_mean":
            float(
                np.mean(
                    arr(
                        "rho_J_mean"
                    )
                )
            ),

        "rho_Q_mean":
            float(
                np.mean(
                    arr(
                        "rho_Q"
                    )
                )
            ),

        "rho_Jmean_minus_Q_mean":
            float(
                np.mean(
                    arr(
                        "rho_Jmean_minus_Q"
                    )
                )
            ),

        "contexts_Jmean_above_Q":
            int(
                np.sum(
                    arr(
                        "rho_Jmean_minus_Q"
                    )
                    > EPS
                )
            ),

        "Q_best_second_gap_min":
            float(
                np.min(
                    arr(
                        "Q_best_second_gap"
                    )
                )
            ),

        "Q_best_second_gap_median":
            float(
                np.median(
                    arr(
                        "Q_best_second_gap"
                    )
                )
            ),

        "Q_best_second_gap_max":
            float(
                np.max(
                    arr(
                        "Q_best_second_gap"
                    )
                )
            ),

        "Q_relative_gap_median":
            float(
                np.median(
                    arr(
                        "Q_relative_best_second_gap"
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
            "icra27_os_t6p4_physical_response_vs_q_transfer_v0",

        "status":
            "COMPUTE_PASS",

        "heldout_used":
            False,

        "source_nearest_neighbors":
            (
                "Frozen T6.3b 16x12 terrain-context "
                "nearest-neighbor pairs."
            ),

        "purpose":
            (
                "Separate physical beta-response "
                "transferability from conditioning "
                "introduced by context-local balanced "
                "Q construction."
            ),

        "aggregate":
            aggregate,

        "interpretation_rule": {
            "J_high_Q_low":
                (
                    "Physical response surfaces transfer "
                    "but Q construction destroys useful "
                    "ordering; inspect local normalization/"
                    "Pareto scalarization."
                ),

            "J_low_Q_low":
                (
                    "Physical beta-response surfaces "
                    "themselves are weakly transferable "
                    "between geometrically similar terrains; "
                    "do not continue terrain-grid resolution "
                    "or ordinary ranking-model tuning."
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
        "ICRA27 OS-T6.4 PHYSICAL RESPONSE "
        "VS BALANCED-Q TRANSFER"
    )
    print("=" * 118)

    for key, value in aggregate.items():
        print(
            f"  {key:<38}: {value}"
        )

    print()
    print(
        "[ICRA27] OS-T6.4 physical-vs-Q "
        "transfer audit: COMPUTE PASS"
    )
    print("=" * 118)


if __name__ == "__main__":
    main()
