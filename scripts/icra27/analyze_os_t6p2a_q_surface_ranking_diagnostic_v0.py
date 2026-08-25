from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]

PRED = (
    ROOT
    / "results/icra27"
    / "os_t6p2_raw_patch_balanced_q_surface_loco_v0"
    / "q_surface_predictions.csv"
)

CONTEXT = (
    ROOT
    / "results/icra27"
    / "os_t6p2_raw_patch_balanced_q_surface_loco_v0"
    / "context_summary.csv"
)

SRC_MANIFEST = (
    ROOT
    / "results/icra27"
    / "os_t6p2_raw_patch_balanced_q_surface_loco_v0"
    / "raw_patch_q_surface_selector_manifest.json"
)

OUT_DIR = (
    ROOT
    / "results/icra27"
    / "os_t6p2a_q_surface_ranking_diagnostic_v0"
)

OUT_CONTEXT = (
    OUT_DIR
    / "ranking_context_summary.csv"
)

OUT_MANIFEST = (
    OUT_DIR
    / "ranking_diagnostic_manifest.json"
)


EXPECTED_FOLDS = 18
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


def rankdata_average(x):
    x = np.asarray(
        x,
        dtype=np.float64,
    )

    order = np.argsort(
        x,
        kind="stable",
    )

    ranks = np.empty(
        len(x),
        dtype=np.float64,
    )

    i = 0

    while i < len(x):
        j = i + 1

        while (
            j < len(x)
            and abs(
                x[
                    order[j]
                ]
                - x[
                    order[i]
                ]
            )
            <= EPS
        ):
            j += 1

        average_rank = (
            (i + 1)
            + j
        ) / 2.0

        for k in range(
            i,
            j,
        ):
            ranks[
                order[k]
            ] = average_rank

        i = j

    return ranks


def corr(
    a,
    b,
):
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


def affine_aligned_mae(
    true_q,
    pred_q,
):
    X = np.stack(
        [
            pred_q,
            np.ones_like(
                pred_q
            ),
        ],
        axis=1,
    )

    coef, *_ = np.linalg.lstsq(
        X,
        true_q,
        rcond=None,
    )

    aligned = (
        X
        @ coef
    )

    return float(
        np.mean(
            np.abs(
                aligned
                - true_q
            )
        )
    )


def main():
    if OUT_DIR.exists():
        raise SystemExit(
            f"REFUSING TO OVERWRITE: "
            f"{OUT_DIR}"
        )

    for path in (
        PRED,
        CONTEXT,
        SRC_MANIFEST,
    ):
        if not path.exists():
            raise FileNotFoundError(
                path
            )

    src = json.loads(
        SRC_MANIFEST.read_text()
    )

    if src.get(
        "status"
    ) != "COMPUTE_PASS":
        raise RuntimeError(
            "T6.2 source is not COMPUTE_PASS."
        )

    if bool(
        src.get(
            "heldout_used",
            False,
        )
    ):
        raise RuntimeError(
            "T6.2 reports heldout use."
        )

    pred_rows = read_csv(
        PRED
    )

    context_rows_old = read_csv(
        CONTEXT
    )

    grouped = {}

    for row in pred_rows:
        cid = row[
            "held_context"
        ]

        grouped.setdefault(
            cid,
            []
        ).append(
            row
        )

    if len(
        grouped
    ) != EXPECTED_FOLDS:
        raise RuntimeError(
            f"Expected 18 held contexts; "
            f"got {len(grouped)}."
        )

    old_lookup = {
        row[
            "held_context"
        ]:
            row
        for row in context_rows_old
    }

    rows = []

    for cid, local in grouped.items():
        local = sorted(
            local,
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
                f"{cid}: expected 21 rows."
            )

        true_q = np.asarray(
            [
                float(
                    x[
                        "true_Q"
                    ]
                )
                for x in local
            ],
            dtype=np.float64,
        )

        pred_q = np.asarray(
            [
                float(
                    x[
                        "pred_Q"
                    ]
                )
                for x in local
            ],
            dtype=np.float64,
        )

        true_rank = rankdata_average(
            true_q
        )

        pred_rank = rankdata_average(
            pred_q
        )

        oracle_idx = int(
            np.argmin(
                true_q
            )
        )

        pred_idx = int(
            np.argmin(
                pred_q
            )
        )

        selected_true_rank = int(
            np.argsort(
                true_q,
                kind="stable",
            ).tolist().index(
                pred_idx
            )
            + 1
        )

        pearson = corr(
            true_q,
            pred_q,
        )

        spearman = corr(
            true_rank,
            pred_rank,
        )

        raw_mae = float(
            np.mean(
                np.abs(
                    pred_q
                    - true_q
                )
            )
        )

        aligned_mae = affine_aligned_mae(
            true_q,
            pred_q,
        )

        top3 = set(
            np.argsort(
                true_q,
                kind="stable",
            )[:3].tolist()
        )

        top5 = set(
            np.argsort(
                true_q,
                kind="stable",
            )[:5].tolist()
        )

        old = old_lookup[
            cid
        ]

        rows.append(
            {
                "held_context":
                    cid,

                "pearson_Q":
                    pearson,

                "spearman_Q":
                    spearman,

                "raw_Q_mae":
                    raw_mae,

                "affine_aligned_Q_mae":
                    aligned_mae,

                "affine_mae_fraction_of_raw":
                    (
                        aligned_mae
                        / raw_mae
                        if raw_mae > EPS
                        else 0.0
                    ),

                "pred_true_rank":
                    selected_true_rank,

                "top3_hit":
                    int(
                        pred_idx
                        in top3
                    ),

                "top5_hit":
                    int(
                        pred_idx
                        in top5
                    ),

                "selector_exact":
                    int(
                        pred_idx
                        == oracle_idx
                    ),

                "selector_Q_excess":
                    float(
                        old[
                            "selector_Q_excess"
                        ]
                    ),

                "baseline_Q_excess":
                    float(
                        old[
                            "baseline_Q_excess"
                        ]
                    ),
            }
        )

    def arr(key):
        return np.asarray(
            [
                x[key]
                for x in rows
            ],
            dtype=np.float64,
        )

    aggregate = {
        "pearson_mean":
            float(
                np.mean(
                    arr(
                        "pearson_Q"
                    )
                )
            ),

        "spearman_mean":
            float(
                np.mean(
                    arr(
                        "spearman_Q"
                    )
                )
            ),

        "raw_Q_mae_mean":
            float(
                np.mean(
                    arr(
                        "raw_Q_mae"
                    )
                )
            ),

        "affine_aligned_Q_mae_mean":
            float(
                np.mean(
                    arr(
                        "affine_aligned_Q_mae"
                    )
                )
            ),

        "affine_mae_fraction_mean":
            float(
                np.mean(
                    arr(
                        "affine_mae_fraction_of_raw"
                    )
                )
            ),

        "true_rank_median":
            float(
                np.median(
                    arr(
                        "pred_true_rank"
                    )
                )
            ),

        "top3_hit_fraction":
            float(
                np.mean(
                    arr(
                        "top3_hit"
                    )
                )
            ),

        "top5_hit_fraction":
            float(
                np.mean(
                    arr(
                        "top5_hit"
                    )
                )
            ),

        "exact_fraction":
            float(
                np.mean(
                    arr(
                        "selector_exact"
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
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            rows
        )

    manifest = {
        "schema":
            "icra27_os_t6p2a_q_surface_ranking_diagnostic_v0",

        "status":
            "COMPUTE_PASS",

        "source":
            "T6.2 raw-patch balanced Q surface",

        "heldout_used":
            False,

        "purpose":
            (
                "Determine whether absolute Q calibration "
                "error is misaligned with within-context "
                "beta ranking quality."
            ),

        "aggregate":
            aggregate,

        "next_decision":
            (
                "If ranking is materially stronger than "
                "absolute calibration, replace global "
                "absolute-Q regression with context-wise "
                "ranking/listwise training."
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
    print("=" * 112)
    print(
        "ICRA27 OS-T6.2a Q-SURFACE "
        "RANKING DIAGNOSTIC"
    )
    print("=" * 112)

    for key, value in aggregate.items():
        print(
            f"  {key:<34}: {value}"
        )

    print()
    print(
        "[ICRA27] OS-T6.2a ranking "
        "diagnostic: COMPUTE PASS"
    )
    print("=" * 112)


if __name__ == "__main__":
    main()
