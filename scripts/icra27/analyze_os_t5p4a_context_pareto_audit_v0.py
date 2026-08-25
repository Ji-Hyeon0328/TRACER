from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]

SOURCE = (
    ROOT
    / "results/icra27"
    / "os_t5p3b_physical_atlas_normalization_v0"
    / "physical_beta_response_atlas.csv"
)

NORMALIZATION = (
    ROOT
    / "results/icra27"
    / "os_t5p3b_physical_atlas_normalization_v0"
    / "normalization_stats.json"
)

OUT_DIR = (
    ROOT
    / "results/icra27"
    / "os_t5p4a_context_pareto_audit_v0"
)

MEMBERSHIP_CSV = (
    OUT_DIR
    / "pareto_membership.csv"
)

FRONT_CSV = (
    OUT_DIR
    / "pareto_front.csv"
)

CONTEXT_CSV = (
    OUT_DIR
    / "context_pareto_summary.csv"
)

MANIFEST = (
    OUT_DIR
    / "pareto_manifest.json"
)

EXPECTED_ROWS = 189
EXPECTED_BETAS = 21
EXPECTED_CONTEXTS = 9

# Numerical dominance tolerance is applied in
# normalized coordinates only for equality noise.
DOM_TOL = 1.0e-9


RAW_METRICS = (
    "J_motion_s_per_m",
    "J_stability",
    "J_energy_j_per_m",
)

NORMALIZED_METRICS = (
    "J_motion_normalized",
    "J_stability_normalized",
    "J_energy_normalized",
)


def finite(
    value: Any,
) -> float:
    value = float(value)

    if not math.isfinite(value):
        raise ValueError(
            f"Non-finite value: {value!r}"
        )

    return value


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    if not rows:
        raise RuntimeError(
            f"Refusing empty CSV: {path}"
        )

    fields: list[str] = []

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


def pareto_mask(
    values: np.ndarray,
    *,
    tol: float,
) -> np.ndarray:
    if (
        values.ndim != 2
        or values.shape[1] != 3
    ):
        raise ValueError(
            f"Expected Nx3 values, got "
            f"{values.shape}"
        )

    n = values.shape[0]

    result = np.ones(
        n,
        dtype=bool,
    )

    for i in range(n):
        candidate = values[i]

        dominates_i = (
            np.all(
                values
                <= candidate[None, :]
                + tol,
                axis=1,
            )
            & np.any(
                values
                < candidate[None, :]
                - tol,
                axis=1,
            )
        )

        dominates_i[i] = False

        if np.any(
            dominates_i
        ):
            result[i] = False

    return result


def dominated_by_count(
    values: np.ndarray,
    index: int,
    *,
    tol: float,
) -> int:
    candidate = values[
        index
    ]

    mask = (
        np.all(
            values
            <= candidate[None, :]
            + tol,
            axis=1,
        )
        & np.any(
            values
            < candidate[None, :]
            - tol,
            axis=1,
        )
    )

    mask[index] = False

    return int(
        np.sum(mask)
    )


def main() -> None:
    if OUT_DIR.exists():
        raise SystemExit(
            f"REFUSING TO OVERWRITE: {OUT_DIR}"
        )

    if not SOURCE.exists():
        raise FileNotFoundError(
            SOURCE
        )

    if not NORMALIZATION.exists():
        raise FileNotFoundError(
            NORMALIZATION
        )

    norm = json.loads(
        NORMALIZATION.read_text()
    )

    if norm.get(
        "status"
    ) != "FROZEN":
        raise RuntimeError(
            "Normalization is not frozen."
        )

    if bool(
        norm.get(
            "heldout_used",
            True,
        )
    ):
        raise RuntimeError(
            "Held-out leakage detected."
        )

    with SOURCE.open(
        newline="",
    ) as f:
        rows = list(
            csv.DictReader(f)
        )

    if len(rows) != EXPECTED_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_ROWS} rows; "
            f"got {len(rows)}"
        )

    beta_names = sorted({
        row["beta_name"]
        for row in rows
    })

    contexts = sorted({
        (
            row["group"],
            int(
                float(
                    row["seed"]
                )
            ),
        )
        for row in rows
    })

    if len(beta_names) != EXPECTED_BETAS:
        raise RuntimeError(
            f"Expected {EXPECTED_BETAS} beta "
            f"points; got {len(beta_names)}"
        )

    if len(contexts) != EXPECTED_CONTEXTS:
        raise RuntimeError(
            f"Expected {EXPECTED_CONTEXTS} "
            f"context slots; got "
            f"{len(contexts)}"
        )

    membership_rows = []
    front_rows = []
    context_rows = []

    front_sets: dict[
        tuple[str, int],
        tuple[str, ...],
    ] = {}

    for group, seed in contexts:
        subset = [
            row
            for row in rows
            if (
                row["group"] == group
                and int(
                    float(
                        row["seed"]
                    )
                ) == seed
            )
        ]

        if len(subset) != EXPECTED_BETAS:
            raise RuntimeError(
                f"{group}/{seed}: expected "
                f"{EXPECTED_BETAS} rows, got "
                f"{len(subset)}"
            )

        # Deterministic beta ordering.
        subset.sort(
            key=lambda row: (
                -finite(
                    row[
                        "lambda_motion"
                    ]
                ),
                -finite(
                    row[
                        "lambda_stability"
                    ]
                ),
                -finite(
                    row[
                        "lambda_energy"
                    ]
                ),
            )
        )

        raw_values = np.asarray(
            [
                [
                    finite(row[key])
                    for key in RAW_METRICS
                ]
                for row in subset
            ],
            dtype=np.float64,
        )

        normalized_values = np.asarray(
            [
                [
                    finite(row[key])
                    for key
                    in NORMALIZED_METRICS
                ]
                for row in subset
            ],
            dtype=np.float64,
        )

        # Exact physical dominance.
        raw_mask = pareto_mask(
            raw_values,
            tol=1.0e-12,
        )

        # Same calculation after frozen affine
        # normalization.
        normalized_mask = pareto_mask(
            normalized_values,
            tol=DOM_TOL,
        )

        if not np.array_equal(
            raw_mask,
            normalized_mask,
        ):
            raise RuntimeError(
                "Raw/normalized Pareto "
                "membership mismatch for "
                f"{group}/{seed}."
            )

        names = tuple(
            subset[i][
                "beta_name"
            ]
            for i in range(
                EXPECTED_BETAS
            )
            if bool(
                raw_mask[i]
            )
        )

        front_sets[
            (
                group,
                seed,
            )
        ] = names

        for i, row in enumerate(
            subset
        ):
            is_pareto = bool(
                raw_mask[i]
            )

            out = dict(
                row
            )

            out[
                "pareto_context_group"
            ] = group

            out[
                "pareto_context_seed"
            ] = seed

            out[
                "is_pareto"
            ] = int(
                is_pareto
            )

            out[
                "dominated_by_count"
            ] = dominated_by_count(
                normalized_values,
                i,
                tol=DOM_TOL,
            )

            membership_rows.append(
                out
            )

            if is_pareto:
                front_rows.append(
                    out
                )

        context_rows.append(
            {
                "group":
                    group,

                "seed":
                    seed,

                "pareto_count":
                    int(
                        np.sum(
                            raw_mask
                        )
                    ),

                "dominated_count":
                    int(
                        EXPECTED_BETAS
                        - np.sum(
                            raw_mask
                        )
                    ),

                "pareto_fraction":
                    float(
                        np.mean(
                            raw_mask
                        )
                    ),

                "pareto_beta_names":
                    "|".join(
                        names
                    ),
            }
        )

    # --------------------------------------------------------
    # Deterministic flat/LF seed slots should have identical
    # front membership. Do not interpret those seed labels
    # as independent stochastic evidence.
    # --------------------------------------------------------

    deterministic_group_checks = {}

    for group in (
        "flat",
        "low_friction",
    ):
        sets = [
            front_sets[
                (
                    g,
                    seed,
                )
            ]
            for g, seed in contexts
            if g == group
        ]

        identical = (
            len(
                set(sets)
            )
            == 1
        )

        deterministic_group_checks[
            group
        ] = {
            "front_sets_identical":
                bool(
                    identical
                ),

            "context_slots":
                len(
                    sets
                ),

            "front":
                list(
                    sets[0]
                )
                if sets
                else [],
        }

        if not identical:
            raise RuntimeError(
                f"{group}: deterministic "
                "context slots produced "
                "different Pareto fronts."
            )

    rough_counts = [
        int(
            row[
                "pareto_count"
            ]
        )
        for row in context_rows
        if row["group"] == "rough"
    ]

    summary_by_group = {}

    for group in (
        "flat",
        "low_friction",
        "rough",
    ):
        subset = [
            row
            for row in context_rows
            if row[
                "group"
            ] == group
        ]

        counts = [
            int(
                row[
                    "pareto_count"
                ]
            )
            for row in subset
        ]

        summary_by_group[
            group
        ] = {
            "context_slots":
                len(
                    subset
                ),

            "pareto_counts":
                counts,

            "pareto_count_min":
                min(
                    counts
                ),

            "pareto_count_max":
                max(
                    counts
                ),

            "pareto_count_mean":
                float(
                    np.mean(
                        counts
                    )
                ),
        }

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    write_csv(
        MEMBERSHIP_CSV,
        membership_rows,
    )

    write_csv(
        FRONT_CSV,
        front_rows,
    )

    write_csv(
        CONTEXT_CSV,
        context_rows,
    )

    manifest = {
        "schema":
            "icra27_os_t5p4a_context_pareto_audit_v0",

        "status":
            "COMPUTE_PASS",

        "source":
            str(
                SOURCE.relative_to(
                    ROOT
                )
            ),

        "source_sha256":
            hashlib.sha256(
                SOURCE.read_bytes()
            ).hexdigest(),

        "normalization":
            str(
                NORMALIZATION.relative_to(
                    ROOT
                )
            ),

        "normalization_sha256":
            hashlib.sha256(
                NORMALIZATION.read_bytes()
            ).hexdigest(),

        "objectives":
            {
                "motion":
                    "J_motion_s_per_m",

                "stability":
                    "J_stability",

                "energy":
                    "J_energy_j_per_m",
            },

        "objective_direction":
            "all_lower_is_better",

        "dominance_definition":
            (
                "a dominates b iff a is no worse "
                "in all three frozen physical "
                "objectives and strictly better "
                "in at least one"
            ),

        "raw_normalized_fronts_match":
            True,

        "episodes":
            len(
                membership_rows
            ),

        "pareto_rows":
            len(
                front_rows
            ),

        "beta_points_per_context":
            EXPECTED_BETAS,

        "context_slots":
            EXPECTED_CONTEXTS,

        "group_summary":
            summary_by_group,

        "deterministic_group_checks":
            deterministic_group_checks,

        "rough_pareto_counts":
            rough_counts,

        "interpretation_guard":
            (
                "Pareto membership is a geometric "
                "property of the frozen physical "
                "response atlas. No external mission "
                "preference w is used or selected "
                "in OS-T5.4a."
            ),

        "next_stage":
            (
                "OS-T5.4b preference scalarization "
                "contract and context-specific "
                "preferred beta extraction"
            ),
    }

    MANIFEST.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print("=" * 100)
    print(
        "ICRA27 OS-T5.4a "
        "CONTEXT-WISE PARETO AUDIT"
    )
    print("=" * 100)

    print()
    print(
        "episodes      :",
        len(
            membership_rows
        ),
    )

    print(
        "pareto rows   :",
        len(
            front_rows
        ),
    )

    print(
        "contexts      :",
        len(
            context_rows
        ),
    )

    print(
        "raw/norm front:",
        "MATCH",
    )

    print()
    print("CONTEXT FRONTS")

    for row in context_rows:
        print(
            f"  {row['group']:<14} "
            f"seed={row['seed']:<6} "
            f"pareto="
            f"{row['pareto_count']:>2}/"
            f"{EXPECTED_BETAS}"
        )

    print()
    print("GROUP SUMMARY")

    for group, item in (
        summary_by_group.items()
    ):
        print(
            f"  {group:<14} "
            f"counts={item['pareto_counts']}"
        )

    print()
    print(
        "flat deterministic front:",
        deterministic_group_checks[
            "flat"
        ][
            "front"
        ],
    )

    print(
        "LF deterministic front  :",
        deterministic_group_checks[
            "low_friction"
        ][
            "front"
        ],
    )

    print()
    print("outputs:")
    print(
        " ",
        MEMBERSHIP_CSV,
    )
    print(
        " ",
        FRONT_CSV,
    )
    print(
        " ",
        CONTEXT_CSV,
    )
    print(
        " ",
        MANIFEST,
    )

    print()
    print(
        "[ICRA27] OS-T5.4a "
        "context-wise Pareto audit: COMPUTE PASS"
    )


if __name__ == "__main__":
    main()
