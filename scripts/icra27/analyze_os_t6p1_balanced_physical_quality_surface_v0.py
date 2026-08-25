from __future__ import annotations

import csv
import json
import math
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

OUT_DIR = (
    ROOT
    / "results/icra27"
    / "os_t6p1_balanced_physical_quality_surface_v0"
)

OUT_SURFACE = (
    OUT_DIR
    / "balanced_beta_quality_surface.csv"
)

OUT_CONTEXT = (
    OUT_DIR
    / "context_quality_summary.csv"
)

OUT_MANIFEST = (
    OUT_DIR
    / "balanced_quality_surface_manifest.json"
)


EXPECTED_CONTEXTS = 35
EXPECTED_BETA = 21
EXPECTED_ROWS = 735

RHO = 0.01

ETA = np.asarray(
    [
        1.0 / 3.0,
        1.0 / 3.0,
        1.0 / 3.0,
    ],
    dtype=np.float64,
)

EPS = 1.0e-12


def read_csv(path: Path):
    with path.open(
        "r",
        newline="",
    ) as f:
        return list(
            csv.DictReader(f)
        )


def finite(x):
    y = float(x)

    if not math.isfinite(y):
        raise ValueError(
            f"Non-finite value: {x}"
        )

    return y


def resolve_column(
    fieldnames,
    aliases,
    semantic,
):
    found = [
        name
        for name in aliases
        if name in fieldnames
    ]

    if len(found) != 1:
        raise RuntimeError(
            f"Could not uniquely resolve "
            f"{semantic}. "
            f"Found={found}; "
            f"fields={fieldnames}"
        )

    return found[0]


def pareto_mask(
    J: np.ndarray,
):
    n = J.shape[0]

    keep = np.ones(
        n,
        dtype=bool,
    )

    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            weakly_better = np.all(
                J[j] <= J[i] + EPS
            )

            strictly_better = np.any(
                J[j] < J[i] - EPS
            )

            if (
                weakly_better
                and strictly_better
            ):
                keep[i] = False
                break

    return keep


def main():
    if OUT_DIR.exists():
        raise SystemExit(
            f"REFUSING TO OVERWRITE: "
            f"{OUT_DIR}"
        )

    for path in (
        ATLAS,
        ATLAS_MANIFEST,
    ):
        if not path.exists():
            raise FileNotFoundError(
                path
            )

    manifest = json.loads(
        ATLAS_MANIFEST.read_text()
    )

    if manifest.get(
        "status"
    ) != "FREEZE_PASS":
        raise RuntimeError(
            "T5.5f3 physical atlas is "
            "not FREEZE_PASS."
        )

    if bool(
        manifest.get(
            "heldout_used",
            False,
        )
    ):
        raise RuntimeError(
            "Physical atlas reports "
            "heldout use."
        )

    rows = read_csv(
        ATLAS
    )

    if len(rows) != EXPECTED_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_ROWS} rows; "
            f"got {len(rows)}."
        )

    fieldnames = list(
        rows[0].keys()
    )

    context_col = resolve_column(
        fieldnames,
        [
            "context_id",
        ],
        "context id",
    )

    beta_col = resolve_column(
        fieldnames,
        [
            "beta_name",
        ],
        "beta name",
    )

    jm_col = resolve_column(
        fieldnames,
        [
            "J_motion_s_per_m",
            "J_motion",
            "j_motion_s_per_m",
            "j_motion",
        ],
        "J_M",
    )

    js_col = resolve_column(
        fieldnames,
        [
            "J_stability",
            "j_stability",
        ],
        "J_S",
    )

    je_col = resolve_column(
        fieldnames,
        [
            "J_energy_j_per_m",
            "J_energy",
            "j_energy_j_per_m",
            "j_energy",
        ],
        "J_E",
    )

    print(
        "resolved columns:"
    )
    print(
        "  context:",
        context_col,
    )
    print(
        "  beta   :",
        beta_col,
    )
    print(
        "  J_M    :",
        jm_col,
    )
    print(
        "  J_S    :",
        js_col,
    )
    print(
        "  J_E    :",
        je_col,
    )

    grouped = {}

    context_order = []

    for row in rows:
        cid = row[
            context_col
        ]

        if cid not in grouped:
            grouped[
                cid
            ] = []

            context_order.append(
                cid
            )

        grouped[
            cid
        ].append(
            row
        )

    if len(
        grouped
    ) != EXPECTED_CONTEXTS:
        raise RuntimeError(
            f"Expected {EXPECTED_CONTEXTS} "
            f"contexts; got {len(grouped)}."
        )

    beta_order = [
        row[
            beta_col
        ]
        for row in grouped[
            context_order[0]
        ]
    ]

    if (
        len(beta_order)
        != EXPECTED_BETA
        or len(set(beta_order))
        != EXPECTED_BETA
    ):
        raise RuntimeError(
            "Invalid first-context beta bank."
        )

    surface_rows = []

    context_rows = []

    pareto_counts = []

    best_second_gaps = []

    exact_best_counts = []

    for cid in context_order:
        local = grouped[
            cid
        ]

        if len(local) != EXPECTED_BETA:
            raise RuntimeError(
                f"{cid}: expected "
                f"{EXPECTED_BETA} beta rows; "
                f"got {len(local)}."
            )

        by_beta = {
            row[
                beta_col
            ]:
                row
            for row in local
        }

        if set(
            by_beta
        ) != set(
            beta_order
        ):
            raise RuntimeError(
                f"{cid}: beta bank mismatch."
            )

        local = [
            by_beta[
                beta
            ]
            for beta in beta_order
        ]

        J = np.asarray(
            [
                [
                    finite(
                        row[
                            jm_col
                        ]
                    ),
                    finite(
                        row[
                            js_col
                        ]
                    ),
                    finite(
                        row[
                            je_col
                        ]
                    ),
                ]
                for row in local
            ],
            dtype=np.float64,
        )

        p_mask = pareto_mask(
            J
        )

        pareto = J[
            p_mask
        ]

        if pareto.shape[0] < 1:
            raise RuntimeError(
                f"{cid}: empty Pareto front."
            )

        ideal = np.min(
            pareto,
            axis=0,
        )

        nadir = np.max(
            pareto,
            axis=0,
        )

        span = (
            nadir
            - ideal
        )

        regret = np.zeros_like(
            J
        )

        for k in range(3):
            if span[k] > EPS:
                regret[
                    :,
                    k
                ] = (
                    J[
                        :,
                        k
                    ]
                    - ideal[k]
                ) / span[k]
            else:
                regret[
                    :,
                    k
                ] = 0.0

        weighted = (
            regret
            * ETA[
                None,
                :
            ]
        )

        Q = (
            np.max(
                weighted,
                axis=1,
            )
            + RHO
            * np.sum(
                weighted,
                axis=1,
            )
        )

        q_min = float(
            np.min(
                Q
            )
        )

        q_gap = (
            Q
            - q_min
        )

        order = np.argsort(
            Q,
            kind="stable",
        )

        exact_best = np.flatnonzero(
            np.abs(
                q_gap
            )
            <= EPS
        )

        if len(order) >= 2:
            second_gap = float(
                Q[
                    order[1]
                ]
                - Q[
                    order[0]
                ]
            )
        else:
            second_gap = 0.0

        pareto_count = int(
            np.sum(
                p_mask
            )
        )

        pareto_counts.append(
            pareto_count
        )

        best_second_gaps.append(
            second_gap
        )

        exact_best_counts.append(
            int(
                exact_best.size
            )
        )

        for i, row in enumerate(
            local
        ):
            surface_rows.append(
                {
                    "context_id":
                        cid,

                    "beta_index":
                        i,

                    "beta_name":
                        row[
                            beta_col
                        ],

                    "J_motion_s_per_m":
                        J[i, 0],

                    "J_stability":
                        J[i, 1],

                    "J_energy_j_per_m":
                        J[i, 2],

                    "r_motion":
                        regret[i, 0],

                    "r_stability":
                        regret[i, 1],

                    "r_energy":
                        regret[i, 2],

                    "quality_Q":
                        Q[i],

                    "quality_gap_from_best":
                        q_gap[i],

                    "quality_rank":
                        int(
                            np.where(
                                order == i
                            )[0][0]
                        )
                        + 1,

                    "is_pareto":
                        int(
                            p_mask[i]
                        ),

                    "is_exact_best":
                        int(
                            i
                            in set(
                                exact_best.tolist()
                            )
                        ),
                }
            )

        context_rows.append(
            {
                "context_id":
                    cid,

                "pareto_count":
                    pareto_count,

                "exact_best_count":
                    int(
                        exact_best.size
                    ),

                "best_beta_name":
                    local[
                        int(
                            order[0]
                        )
                    ][
                        beta_col
                    ],

                "second_beta_name":
                    local[
                        int(
                            order[1]
                        )
                    ][
                        beta_col
                    ],

                "best_Q":
                    float(
                        Q[
                            order[0]
                        ]
                    ),

                "second_Q":
                    float(
                        Q[
                            order[1]
                        ]
                    ),

                "best_second_gap":
                    second_gap,

                "J_M_ideal":
                    ideal[0],

                "J_S_ideal":
                    ideal[1],

                "J_E_ideal":
                    ideal[2],

                "J_M_nadir":
                    nadir[0],

                "J_S_nadir":
                    nadir[1],

                "J_E_nadir":
                    nadir[2],

                "J_M_span":
                    span[0],

                "J_S_span":
                    span[1],

                "J_E_span":
                    span[2],
            }
        )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    with OUT_SURFACE.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                surface_rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            surface_rows
        )

    with OUT_CONTEXT.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                context_rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            context_rows
        )

    gaps = np.asarray(
        best_second_gaps,
        dtype=np.float64,
    )

    output_manifest = {
        "schema":
            "icra27_os_t6p1_balanced_physical_quality_surface_v0",

        "status":
            "FREEZE_PASS",

        "source_physical_atlas":
            str(
                ATLAS.relative_to(
                    ROOT
                )
            ),

        "rows":
            len(
                surface_rows
            ),

        "contexts":
            len(
                context_rows
            ),

        "beta_points_per_context":
            EXPECTED_BETA,

        "beta_order":
            beta_order,

        "physical_objective_vector":
            [
                "J_M=T/progress [s/m]",
                "J_S=max(attitude, established-contact slip)",
                "J_E=mechanical_energy/progress [J/m]",
            ],

        "objective_direction":
            "all lower is better",

        "regret":
            (
                "context-local normalization using "
                "Pareto-front ideal/nadir"
            ),

        "balanced_weights":
            ETA.tolist(),

        "rho":
            RHO,

        "quality":
            (
                "Q=max_k(eta_k*r_k)"
                "+rho*sum_k(eta_k*r_k)"
            ),

        "selection":
            "argmin_beta Q(c,beta)",

        "heldout_used":
            False,

        "diagnostics": {
            "pareto_count_min":
                int(
                    min(
                        pareto_counts
                    )
                ),

            "pareto_count_max":
                int(
                    max(
                        pareto_counts
                    )
                ),

            "exact_best_count_max":
                int(
                    max(
                        exact_best_counts
                    )
                ),

            "best_second_gap_min":
                float(
                    np.min(
                        gaps
                    )
                ),

            "best_second_gap_median":
                float(
                    np.median(
                        gaps
                    )
                ),

            "best_second_gap_p10":
                float(
                    np.quantile(
                        gaps,
                        0.10,
                    )
                ),

            "best_second_gap_max":
                float(
                    np.max(
                        gaps
                    )
                ),
        },

        "next_step":
            (
                "Inspect quality-gap structure before "
                "choosing any epsilon-optimal threshold "
                "or learned selector loss."
            ),
    }

    OUT_MANIFEST.write_text(
        json.dumps(
            output_manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print()
    print("=" * 118)
    print(
        "ICRA27 OS-T6.1 BALANCED "
        "PHYSICAL QUALITY SURFACE"
    )
    print("=" * 118)

    print(
        "contexts              :",
        len(
            context_rows
        ),
    )

    print(
        "beta/context          :",
        EXPECTED_BETA,
    )

    print(
        "surface rows          :",
        len(
            surface_rows
        ),
    )

    print(
        "Pareto count range    :",
        [
            min(
                pareto_counts
            ),
            max(
                pareto_counts
            ),
        ],
    )

    print(
        "exact-best count max  :",
        max(
            exact_best_counts
        ),
    )

    print(
        "best-second gap min   :",
        float(
            np.min(
                gaps
            )
        ),
    )

    print(
        "best-second gap p10   :",
        float(
            np.quantile(
                gaps,
                0.10,
            )
        ),
    )

    print(
        "best-second gap median:",
        float(
            np.median(
                gaps
            )
        ),
    )

    print(
        "best-second gap max   :",
        float(
            np.max(
                gaps
            )
        ),
    )

    print()
    print(
        "outputs:"
    )
    print(
        " ",
        OUT_SURFACE,
    )
    print(
        " ",
        OUT_CONTEXT,
    )
    print(
        " ",
        OUT_MANIFEST,
    )

    print()
    print(
        "[ICRA27] OS-T6.1 balanced physical "
        "quality surface: FREEZE PASS"
    )

    print("=" * 118)


if __name__ == "__main__":
    main()
