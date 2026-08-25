from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm


ROOT = Path(__file__).resolve().parents[2]

SOURCE_DIR = (
    ROOT
    / "results/icra27"
    / "os_t6p12e_heldout_physical_atlas_v0"
)

SOURCE_CSV = (
    SOURCE_DIR
    / "heldout_physical_beta_response_atlas.csv"
)

SOURCE_MANIFEST = (
    SOURCE_DIR
    / "heldout_physical_atlas_manifest.json"
)

OUT_DIR = (
    ROOT
    / "results/icra27"
    / "os_t6p13a_descriptive_beta_tradeoff_heatmap_v0"
)

OUT_CONTEXT = (
    OUT_DIR
    / "per_context_beta_physical_spearman.csv"
)

OUT_SUMMARY = (
    OUT_DIR
    / "beta_physical_tradeoff_summary.csv"
)

OUT_PNG = (
    OUT_DIR
    / "beta_physical_tradeoff_heatmap.png"
)

OUT_PDF = (
    OUT_DIR
    / "beta_physical_tradeoff_heatmap.pdf"
)

OUT_MANIFEST = (
    OUT_DIR
    / "beta_physical_tradeoff_heatmap_manifest.json"
)


EXPECTED_CONTEXTS = 9
EXPECTED_BETAS = 21
EXPECTED_ROWS = (
    EXPECTED_CONTEXTS
    * EXPECTED_BETAS
)


BETA_AXES = (
    (
        "motion",
        "beta_motion",
        r"$\beta_M$",
    ),
    (
        "stability",
        "beta_stability",
        r"$\beta_S$",
    ),
    (
        "energy",
        "beta_energy",
        r"$\beta_E$",
    ),
)


OBJECTIVES = (
    (
        "motion",
        "J_motion_s_per_m",
        r"$J_M$",
    ),
    (
        "stability",
        "J_stability",
        r"$J_S$",
    ),
    (
        "energy",
        "J_energy_j_per_m",
        r"$J_E$",
    ),
)


def sha256(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            chunk = f.read(
                1024 * 1024
            )

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def write_csv(path, rows):
    if not rows:
        raise RuntimeError(
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


def finite(x):
    y = float(x)

    if not math.isfinite(y):
        raise ValueError(
            f"Non-finite value: {x!r}"
        )

    return y


def average_ranks(values):
    """
    NumPy-only average ranks with tie handling.

    Equivalent to rank(method="average").
    """

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    order = np.argsort(
        values,
        kind="mergesort",
    )

    ranks = np.empty(
        len(values),
        dtype=np.float64,
    )

    start = 0

    while start < len(values):

        end = start + 1

        while (
            end < len(values)
            and values[
                order[end]
            ] == values[
                order[start]
            ]
        ):
            end += 1

        # 1-based average rank.
        average_rank = (
            (start + 1)
            + end
        ) / 2.0

        ranks[
            order[
                start:end
            ]
        ] = average_rank

        start = end

    return ranks


def spearman_with_ties(x, y):
    """
    Spearman correlation via average ranks.
    Handles repeated beta values without pandas/scipy.
    """

    x = np.asarray(
        x,
        dtype=np.float64,
    )

    y = np.asarray(
        y,
        dtype=np.float64,
    )

    if x.shape != y.shape:
        raise RuntimeError(
            "Spearman input shape mismatch."
        )

    rx = average_ranks(
        x
    )

    ry = average_ranks(
        y
    )

    sx = float(
        np.std(rx)
    )

    sy = float(
        np.std(ry)
    )

    if (
        sx <= 0.0
        or sy <= 0.0
    ):
        raise RuntimeError(
            "Degenerate Spearman axis."
        )

    rho = float(
        np.corrcoef(
            rx,
            ry,
        )[
            0,
            1
        ]
    )

    if not math.isfinite(rho):
        raise RuntimeError(
            "Non-finite Spearman rho."
        )

    return rho


def main():
    if OUT_DIR.exists():
        raise SystemExit(
            f"REFUSING TO OVERWRITE: "
            f"{OUT_DIR}"
        )

    if not SOURCE_CSV.exists():
        raise FileNotFoundError(
            SOURCE_CSV
        )

    if not SOURCE_MANIFEST.exists():
        raise FileNotFoundError(
            SOURCE_MANIFEST
        )


    manifest = json.loads(
        SOURCE_MANIFEST.read_text()
    )

    if manifest.get(
        "status"
    ) != "FREEZE_PASS":
        raise RuntimeError(
            "T6.12e physical atlas is not "
            "FREEZE_PASS."
        )


    df = pd.read_csv(
        SOURCE_CSV
    )


    # ========================================================
    # Descriptive standard-heldout analysis only.
    #
    # HARD is deliberately excluded because T6.12e showed
    # 0/21 feasible beta candidates in every HARD context.
    # ========================================================

    df = df[
        df[
            "split_group"
        ].isin(
            [
                "val",
                "test",
            ]
        )
    ].copy()


    feasible = (
        pd.to_numeric(
            df[
                "feasible"
            ],
            errors="raise",
        )
        == 1
    )

    if not bool(
        feasible.all()
    ):
        bad = df.loc[
            ~feasible,
            [
                "context_id",
                "beta_name",
                "status",
            ],
        ]

        raise RuntimeError(
            "STANDARD set unexpectedly contains "
            "infeasible beta candidates:\n"
            f"{bad}"
        )


    if len(df) != EXPECTED_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_ROWS} "
            f"STANDARD rows, got {len(df)}."
        )


    context_ids = list(
        dict.fromkeys(
            df[
                "context_id"
            ].tolist()
        )
    )


    if len(
        context_ids
    ) != EXPECTED_CONTEXTS:
        raise RuntimeError(
            f"Expected 9 contexts, "
            f"got {len(context_ids)}."
        )


    for cid in context_ids:
        sub = df[
            df[
                "context_id"
            ]
            == cid
        ]

        if len(sub) != EXPECTED_BETAS:
            raise RuntimeError(
                f"{cid}: expected 21 beta "
                f"rows, got {len(sub)}."
            )

        if sub[
            "beta_name"
        ].nunique() != EXPECTED_BETAS:
            raise RuntimeError(
                f"{cid}: beta-name support drift."
            )


    # ========================================================
    # Per-context 3x3 Spearman matrices.
    # ========================================================

    context_rows = []


    for cid in context_ids:

        sub = (
            df[
                df[
                    "context_id"
                ]
                == cid
            ]
            .copy()
        )


        split_group = str(
            sub[
                "split_group"
            ].iloc[0]
        )


        for (
            beta_axis,
            beta_col,
            _beta_label,
        ) in BETA_AXES:

            x = sub[
                beta_col
            ].to_numpy(
                dtype=np.float64
            )


            for (
                objective,
                objective_col,
                _objective_label,
            ) in OBJECTIVES:

                y = sub[
                    objective_col
                ].to_numpy(
                    dtype=np.float64
                )


                rho = spearman_with_ties(
                    x,
                    y,
                )


                context_rows.append(
                    {
                        "context_id":
                            cid,

                        "split_group":
                            split_group,

                        "beta_axis":
                            beta_axis,

                        "physical_objective":
                            objective,

                        "spearman_rho":
                            rho,

                        "direction":
                            (
                                "improves_cost"
                                if rho < 0.0
                                else
                                "worsens_cost"
                                if rho > 0.0
                                else
                                "neutral"
                            ),
                    }
                )


    # ========================================================
    # Aggregate across the 9 fully-feasible heldout terrains.
    # ========================================================

    summary_rows = []

    matrix = np.zeros(
        (
            3,
            3,
        ),
        dtype=np.float64,
    )

    annotations = [
        [
            ""
            for _ in range(3)
        ]
        for _ in range(3)
    ]


    for i, (
        beta_axis,
        _beta_col,
        _beta_label,
    ) in enumerate(
        BETA_AXES
    ):

        for j, (
            objective,
            _objective_col,
            _objective_label,
        ) in enumerate(
            OBJECTIVES
        ):

            values = np.asarray(
                [
                    float(
                        row[
                            "spearman_rho"
                        ]
                    )
                    for row in context_rows
                    if (
                        row[
                            "beta_axis"
                        ]
                        == beta_axis
                        and row[
                            "physical_objective"
                        ]
                        == objective
                    )
                ],
                dtype=np.float64,
            )


            if len(values) != EXPECTED_CONTEXTS:
                raise RuntimeError(
                    f"{beta_axis}/{objective}: "
                    f"expected 9 rho values."
                )


            median_rho = float(
                np.median(
                    values
                )
            )

            mean_rho = float(
                np.mean(
                    values
                )
            )

            negative_count = int(
                np.sum(
                    values < 0.0
                )
            )

            positive_count = int(
                np.sum(
                    values > 0.0
                )
            )

            zero_count = int(
                np.sum(
                    np.isclose(
                        values,
                        0.0,
                        atol=1e-12,
                    )
                )
            )


            matrix[
                i,
                j
            ] = median_rho


            if median_rho < -0.05:
                direction_symbol = "↓"

                consistency = (
                    negative_count
                )

            elif median_rho > 0.05:
                direction_symbol = "↑"

                consistency = (
                    positive_count
                )

            else:
                direction_symbol = "≈"

                consistency = max(
                    negative_count,
                    positive_count,
                    zero_count,
                )


            annotations[
                i
            ][
                j
            ] = (
                f"ρ={median_rho:+.2f}\n"
                f"{direction_symbol} "
                f"{consistency}/9"
            )


            summary_rows.append(
                {
                    "beta_axis":
                        beta_axis,

                    "physical_objective":
                        objective,

                    "median_spearman_rho":
                        median_rho,

                    "mean_spearman_rho":
                        mean_rho,

                    "negative_contexts":
                        negative_count,

                    "positive_contexts":
                        positive_count,

                    "zero_contexts":
                        zero_count,

                    "abs_median_rho":
                        abs(
                            median_rho
                        ),
                }
            )


    # ========================================================
    # Plot.
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(
            8.4,
            5.8,
        )
    )


    norm = TwoSlopeNorm(
        vmin=-1.0,
        vcenter=0.0,
        vmax=1.0,
    )


    im = ax.imshow(
        matrix,
        cmap="RdBu_r",
        norm=norm,
        aspect="equal",
    )


    ax.set_xticks(
        np.arange(3)
    )

    ax.set_yticks(
        np.arange(3)
    )


    ax.set_xticklabels(
        [
            item[
                2
            ]
            for item in OBJECTIVES
        ],
        fontsize=13,
    )

    ax.set_yticklabels(
        [
            item[
                2
            ]
            for item in BETA_AXES
        ],
        fontsize=13,
    )


    ax.set_xlabel(
        "Observed physical cost",
        fontsize=12,
        labelpad=10,
    )

    ax.set_ylabel(
        "Reward weight varied",
        fontsize=12,
        labelpad=10,
    )


    ax.set_title(
        "Reward weights induce distinct but coupled "
        "physical response patterns",
        fontsize=14,
        pad=14,
    )


    for i in range(3):
        for j in range(3):

            value = matrix[
                i,
                j
            ]

            text_color = (
                "white"
                if abs(
                    value
                ) > 0.55
                else
                "black"
            )


            ax.text(
                j,
                i,
                annotations[
                    i
                ][
                    j
                ],
                ha="center",
                va="center",
                fontsize=12,
                color=text_color,
                fontweight="semibold",
            )


    cbar = fig.colorbar(
        im,
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )

    cbar.set_label(
        (
            "Median within-terrain Spearman ρ\n"
            "(β weight vs. physical cost)"
        ),
        fontsize=10,
    )


    fig.text(
        0.5,
        0.018,
        (
            "9 fully-feasible held-out terrains × "
            "21 β candidates. "
            "ρ < 0: increasing β tends to reduce "
            "the physical cost (improve); "
            "ρ > 0: trade-off / degradation."
        ),
        ha="center",
        fontsize=9,
    )


    fig.tight_layout(
        rect=[
            0,
            0.055,
            1,
            1,
        ]
    )


    # ========================================================
    # Persist immutable descriptive artifacts.
    # ========================================================

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )


    write_csv(
        OUT_CONTEXT,
        context_rows,
    )

    write_csv(
        OUT_SUMMARY,
        summary_rows,
    )


    fig.savefig(
        OUT_PNG,
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        OUT_PDF,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


    out_manifest = {
        "schema":
            "icra27_os_t6p13a_descriptive_beta_tradeoff_heatmap_v0",

        "status":
            "DESCRIPTIVE_ANALYSIS_PASS",

        "confirmatory_primary_metric":
            False,

        "role":
            (
                "Post-hoc descriptive analysis of "
                "the already-frozen T6.12e physical "
                "response atlas; no selector tuning."
            ),

        "source":
            str(
                SOURCE_CSV.relative_to(
                    ROOT
                )
            ),

        "source_sha256":
            sha256(
                SOURCE_CSV
            ),

        "population":
            "STANDARD_VAL_TEST",

        "contexts":
            EXPECTED_CONTEXTS,

        "beta_candidates_per_context":
            EXPECTED_BETAS,

        "rows":
            EXPECTED_ROWS,

        "hard_contexts_excluded":
            True,

        "hard_exclusion_reason":
            (
                "All 3 HARD contexts had 0/21 "
                "feasible beta candidates in the "
                "frozen T6.12e atlas."
            ),

        "cell_definition":
            (
                "Median across contexts of the "
                "within-context Spearman correlation "
                "between one beta weight and one "
                "physical cost over all 21 beta "
                "candidates."
            ),

        "interpretation":
            {
                "negative_rho":
                    (
                        "Increasing the reward weight "
                        "is associated with lower "
                        "physical cost (improvement)."
                    ),

                "positive_rho":
                    (
                        "Increasing the reward weight "
                        "is associated with higher "
                        "physical cost "
                        "(trade-off/degradation)."
                    ),
            },

        "simplex_caveat":
            (
                "Beta lies on a simplex. Each cell "
                "is a directional association toward "
                "one reward emphasis, not an isolated "
                "causal partial derivative with the "
                "other beta components held fixed."
            ),

        "artifact_hashes":
            {
                "per_context_beta_physical_spearman.csv":
                    sha256(
                        OUT_CONTEXT
                    ),

                "beta_physical_tradeoff_summary.csv":
                    sha256(
                        OUT_SUMMARY
                    ),

                "beta_physical_tradeoff_heatmap.png":
                    sha256(
                        OUT_PNG
                    ),

                "beta_physical_tradeoff_heatmap.pdf":
                    sha256(
                        OUT_PDF
                    ),
            },
    }


    OUT_MANIFEST.write_text(
        json.dumps(
            out_manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


    # ========================================================
    # Terminal summary.
    # ========================================================

    print()
    print("=" * 100)

    print(
        "T6.13a DESCRIPTIVE β -> PHYSICAL "
        "TRADE-OFF HEATMAP"
    )

    print("=" * 100)


    print()
    print(
        "Median within-terrain Spearman rho"
    )

    print(
        "             J_M       J_S       J_E"
    )


    for i, (
        beta_axis,
        _,
        beta_label,
    ) in enumerate(
        BETA_AXES
    ):

        print(
            f"{beta_axis:<10} "
            f"{matrix[i,0]:+8.4f} "
            f"{matrix[i,1]:+8.4f} "
            f"{matrix[i,2]:+8.4f}"
        )


    print()
    print(
        "Cell directional consistency:"
    )

    for i, (
        beta_axis,
        _,
        _,
    ) in enumerate(
        BETA_AXES
    ):
        print(
            f"  {beta_axis:<10}: "
            f"{annotations[i][0]} | "
            f"{annotations[i][1]} | "
            f"{annotations[i][2]}"
        )


    print()
    print(
        "NOTE: negative rho = lower physical "
        "cost as beta increases."
    )

    print(
        "NOTE: descriptive held-out analysis; "
        "NOT a new confirmatory selector metric."
    )

    print()
    print(
        "[ICRA27] OS-T6.13a descriptive "
        "beta trade-off heatmap: PASS"
    )

    print()
    print(
        "PNG:",
        OUT_PNG,
    )

    print(
        "PDF:",
        OUT_PDF,
    )

    print("=" * 100)


if __name__ == "__main__":
    main()
