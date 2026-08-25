from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


SOURCE_DIR = (
    ROOT
    / "results/icra27"
    / "os_t5p5c_expanded_physical_atlas_v0"
)

SOURCE_CSV = (
    SOURCE_DIR
    / "physical_beta_response_atlas.csv"
)

SOURCE_MANIFEST = (
    SOURCE_DIR
    / "expanded_physical_atlas_manifest.json"
)


OUT_DIR = (
    ROOT
    / "results/icra27"
    / "os_t5p5d_expanded_preference_labels_v0"
)

OUT_FRONT = (
    OUT_DIR
    / "pareto_front.csv"
)

OUT_CONTEXT_SUMMARY = (
    OUT_DIR
    / "context_pareto_summary.csv"
)

OUT_REGRET = (
    OUT_DIR
    / "context_regret_reference.csv"
)

OUT_LABELS = (
    OUT_DIR
    / "preference_to_beta_labels.csv"
)

OUT_CANONICAL = (
    OUT_DIR
    / "canonical_preference_summary.csv"
)

OUT_MANIFEST = (
    OUT_DIR
    / "preference_label_manifest.json"
)


EXPECTED_ROWS = 420
EXPECTED_CONTEXTS = 20
EXPECTED_BETAS = 21

EXPECTED_PREFERENCES = 25
EXPECTED_LABELS = (
    EXPECTED_CONTEXTS
    * EXPECTED_PREFERENCES
)

# Independently audited from the frozen 420-row T5.5c atlas.
EXPECTED_PARETO_ROWS = 133

PREFERENCE_GRID_DENOMINATOR = 5

RHO = 0.01
TOL = 1.0e-12


OBJECTIVES = (
    (
        "motion",
        "J_motion_s_per_m",
    ),
    (
        "stability",
        "J_stability",
    ),
    (
        "energy",
        "J_energy_j_per_m",
    ),
)


CONTEXT_FEATURES = (
    "context_friction_mu",
    "context_height_std_m",
    "context_height_relief_p95_p05_m",
    "context_slope_rms",
    "context_slope_q95",
)


# Frozen OS-T5.4a regression points.
OLD_PARETO_COUNTS = {
    "flat":
        1,

    "low_friction":
        6,

    "rough_seed_13":
        9,

    "rough_seed_7":
        12,

    "rough_seed_15":
        11,
}


# Frozen OS-T5.4b canonical preference regression.
OLD_CANONICAL = {
    "flat":
        {
            "balanced":
                "lm100_ls000_le000",

            "motion":
                "lm100_ls000_le000",

            "stability":
                "lm100_ls000_le000",

            "energy":
                "lm100_ls000_le000",
        },

    "low_friction":
        {
            "balanced":
                "lm060_ls040_le000",

            "motion":
                "lm080_ls020_le000",

            "stability":
                "lm000_ls080_le020",

            "energy":
                "lm000_ls080_le020",
        },

    "rough_seed_13":
        {
            "balanced":
                "lm080_ls020_le000",

            "motion":
                "lm080_ls020_le000",

            "stability":
                "lm040_ls060_le000",

            "energy":
                "lm040_ls000_le060",
        },

    "rough_seed_7":
        {
            "balanced":
                "lm000_ls060_le040",

            "motion":
                "lm000_ls100_le000",

            "stability":
                "lm020_ls020_le060",

            "energy":
                "lm020_ls020_le060",
        },

    "rough_seed_15":
        {
            "balanced":
                "lm060_ls040_le000",

            "motion":
                "lm040_ls060_le000",

            "stability":
                "lm000_ls040_le060",

            "energy":
                "lm000_ls080_le020",
        },
}


def finite(
    x: Any,
) -> float:
    y = float(x)

    if not math.isfinite(y):
        raise ValueError(
            f"Non-finite value: {x!r}"
        )

    return y


def read_csv(
    path: Path,
) -> list[dict[str, str]]:
    with path.open(
        newline="",
    ) as f:
        return list(
            csv.DictReader(f)
        )


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    if not rows:
        raise ValueError(
            f"No rows for {path}"
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


def sha256_file(
    path: Path,
) -> str:
    h = hashlib.sha256()

    with path.open(
        "rb",
    ) as f:
        for chunk in iter(
            lambda: f.read(
                1024 * 1024
            ),
            b"",
        ):
            h.update(
                chunk
            )

    return h.hexdigest()


def build_preferences():
    prefs = []

    d = (
        PREFERENCE_GRID_DENOMINATOR
    )

    # Same deterministic simplex order as beta lattice:
    # motion-heavy -> stability-heavy -> energy-heavy.
    for i_m in range(
        d,
        -1,
        -1,
    ):
        remaining = (
            d
            - i_m
        )

        for i_s in range(
            remaining,
            -1,
            -1,
        ):
            i_e = (
                d
                - i_m
                - i_s
            )

            prefs.append(
                {
                    "preference_name":
                        (
                            f"wm{i_m * 20:03d}_"
                            f"ws{i_s * 20:03d}_"
                            f"we{i_e * 20:03d}"
                        ),

                    "preference_kind":
                        "grid",

                    "w_motion":
                        i_m / d,

                    "w_stability":
                        i_s / d,

                    "w_energy":
                        i_e / d,
                }
            )

    if len(prefs) != 21:
        raise RuntimeError(
            "Expected 21 simplex-grid "
            f"preferences; got {len(prefs)}"
        )

    prefs.extend(
        [
            {
                "preference_name":
                    "balanced",

                "preference_kind":
                    "canonical",

                "w_motion":
                    1.0 / 3.0,

                "w_stability":
                    1.0 / 3.0,

                "w_energy":
                    1.0 / 3.0,
            },

            {
                "preference_name":
                    "motion",

                "preference_kind":
                    "canonical",

                "w_motion":
                    0.70,

                "w_stability":
                    0.15,

                "w_energy":
                    0.15,
            },

            {
                "preference_name":
                    "stability",

                "preference_kind":
                    "canonical",

                "w_motion":
                    0.15,

                "w_stability":
                    0.70,

                "w_energy":
                    0.15,
            },

            {
                "preference_name":
                    "energy",

                "preference_kind":
                    "canonical",

                "w_motion":
                    0.15,

                "w_stability":
                    0.15,

                "w_energy":
                    0.70,
            },
        ]
    )

    if len(prefs) != (
        EXPECTED_PREFERENCES
    ):
        raise RuntimeError(
            "Preference-bank count mismatch."
        )

    names = {
        p[
            "preference_name"
        ]
        for p in prefs
    }

    if len(names) != len(
        prefs
    ):
        raise RuntimeError(
            "Duplicate preference names."
        )

    for p in prefs:
        total = (
            finite(
                p["w_motion"]
            )
            + finite(
                p["w_stability"]
            )
            + finite(
                p["w_energy"]
            )
        )

        if not math.isclose(
            total,
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError(
                f"Preference does not sum "
                f"to one: {p}"
            )

    return prefs


def dominates(
    a: dict[str, Any],
    b: dict[str, Any],
) -> bool:
    av = [
        finite(
            a[column]
        )
        for _, column in OBJECTIVES
    ]

    bv = [
        finite(
            b[column]
        )
        for _, column in OBJECTIVES
    ]

    no_worse = all(
        x <= y + TOL
        for x, y in zip(
            av,
            bv,
        )
    )

    strictly_better = any(
        x < y - TOL
        for x, y in zip(
            av,
            bv,
        )
    )

    return (
        no_worse
        and strictly_better
    )


def main() -> None:
    if OUT_DIR.exists():
        raise SystemExit(
            f"REFUSING TO OVERWRITE: "
            f"{OUT_DIR}"
        )

    for path in (
        SOURCE_CSV,
        SOURCE_MANIFEST,
    ):
        if not path.exists():
            raise FileNotFoundError(
                path
            )

    manifest = json.loads(
        SOURCE_MANIFEST.read_text()
    )

    if manifest.get(
        "schema"
    ) != (
        "icra27_os_t5p5c_"
        "expanded_physical_atlas_v0"
    ):
        raise RuntimeError(
            "Unexpected T5.5c schema."
        )

    if manifest.get(
        "status"
    ) != "COMPUTE_PASS":
        raise RuntimeError(
            "T5.5c is not COMPUTE_PASS."
        )

    if bool(
        manifest.get(
            "heldout_used"
        )
    ):
        raise RuntimeError(
            "T5.5c reports held-out use."
        )

    if int(
        manifest[
            "rows"
        ]
    ) != EXPECTED_ROWS:
        raise RuntimeError(
            "T5.5c row-count mismatch."
        )

    if int(
        manifest[
            "contexts"
        ]
    ) != EXPECTED_CONTEXTS:
        raise RuntimeError(
            "T5.5c context-count mismatch."
        )

    if int(
        manifest[
            "beta_points_per_context"
        ]
    ) != EXPECTED_BETAS:
        raise RuntimeError(
            "T5.5c beta-count mismatch."
        )

    context_order = list(
        manifest[
            "context_order"
        ]
    )

    if len(
        context_order
    ) != EXPECTED_CONTEXTS:
        raise RuntimeError(
            "Context-order mismatch."
        )

    rows = read_csv(
        SOURCE_CSV
    )

    if len(rows) != (
        EXPECTED_ROWS
    ):
        raise RuntimeError(
            "Physical atlas CSV "
            f"row mismatch: {len(rows)}"
        )


    # --------------------------------------------------------
    # Group rows by frozen context order.
    # --------------------------------------------------------

    by_context = {
        cid:
            []
        for cid in context_order
    }

    for row in rows:
        cid = str(
            row[
                "context_id"
            ]
        )

        if cid not in (
            by_context
        ):
            raise RuntimeError(
                f"Unexpected context: {cid}"
            )

        by_context[
            cid
        ].append(
            row
        )


    # --------------------------------------------------------
    # Recover exact beta ordering from first context.
    # T5.5c is already context-major / beta-minor.
    # --------------------------------------------------------

    first_context = (
        context_order[0]
    )

    beta_order = [
        row[
            "beta_name"
        ]
        for row in (
            by_context[
                first_context
            ]
        )
    ]

    if (
        len(beta_order)
        != EXPECTED_BETAS
        or len(
            set(beta_order)
        )
        != EXPECTED_BETAS
    ):
        raise RuntimeError(
            "Invalid beta ordering."
        )

    beta_rank = {
        name:
            index
        for index, name in enumerate(
            beta_order
        )
    }


    # --------------------------------------------------------
    # Coverage and 5D-context constancy.
    # --------------------------------------------------------

    for cid in context_order:
        context_rows = (
            by_context[
                cid
            ]
        )

        if len(
            context_rows
        ) != EXPECTED_BETAS:
            raise RuntimeError(
                f"{cid}: expected 21 beta rows; "
                f"got {len(context_rows)}"
            )

        names = {
            row[
                "beta_name"
            ]
            for row in context_rows
        }

        if names != set(
            beta_order
        ):
            raise RuntimeError(
                f"{cid}: beta coverage mismatch."
            )

        context_rows.sort(
            key=lambda row:
                beta_rank[
                    row[
                        "beta_name"
                    ]
                ]
        )

        for feature in (
            CONTEXT_FEATURES
        ):
            values = [
                finite(
                    row[
                        feature
                    ]
                )
                for row in context_rows
            ]

            spread = (
                max(values)
                - min(values)
            )

            if spread > TOL:
                raise RuntimeError(
                    f"{cid}/{feature}: "
                    "context feature varies "
                    f"across beta rows; "
                    f"spread={spread}"
                )


    preferences = (
        build_preferences()
    )


    pareto_rows = []
    context_summary = []
    regret_rows = []
    label_rows = []
    canonical_rows = []

    pareto_counts = {}
    selected_beta_diversity = {}


    # --------------------------------------------------------
    # Context-wise Pareto -> regret -> preference labels.
    # --------------------------------------------------------

    for cid in context_order:
        context_rows = (
            by_context[
                cid
            ]
        )

        front = []

        for i, candidate in enumerate(
            context_rows
        ):
            dominated = any(
                j != i
                and dominates(
                    other,
                    candidate,
                )
                for j, other in enumerate(
                    context_rows
                )
            )

            if not dominated:
                front.append(
                    candidate
                )

        if not front:
            raise RuntimeError(
                f"{cid}: empty Pareto front."
            )

        front.sort(
            key=lambda row:
                beta_rank[
                    row[
                        "beta_name"
                    ]
                ]
        )

        pareto_counts[
            cid
        ] = len(
            front
        )


        values = np.asarray(
            [
                [
                    finite(
                        row[column]
                    )
                    for _, column in (
                        OBJECTIVES
                    )
                ]
                for row in front
            ],
            dtype=np.float64,
        )

        ideal = np.min(
            values,
            axis=0,
        )

        nadir = np.max(
            values,
            axis=0,
        )

        span = (
            nadir
            - ideal
        )

        regrets = np.zeros_like(
            values
        )

        for k in range(3):
            if span[k] > TOL:
                regrets[
                    :,
                    k
                ] = (
                    values[
                        :,
                        k
                    ]
                    - ideal[k]
                ) / span[k]


        # ----------------------------------------------------
        # Save Pareto rows + regret coordinates.
        # ----------------------------------------------------

        for rank, (
            row,
            regret,
        ) in enumerate(
            zip(
                front,
                regrets,
            )
        ):
            out = dict(
                row
            )

            out[
                "pareto_rank"
            ] = rank

            out[
                "regret_motion"
            ] = float(
                regret[0]
            )

            out[
                "regret_stability"
            ] = float(
                regret[1]
            )

            out[
                "regret_energy"
            ] = float(
                regret[2]
            )

            pareto_rows.append(
                out
            )


        first = context_rows[0]

        context_summary.append(
            {
                "context_id":
                    cid,

                "terrain":
                    first[
                        "terrain"
                    ],

                "seed":
                    first[
                        "seed"
                    ],

                "pareto_count":
                    len(
                        front
                    ),

                "pareto_beta_names":
                    ";".join(
                        row[
                            "beta_name"
                        ]
                        for row in front
                    ),

                **{
                    feature:
                        finite(
                            first[
                                feature
                            ]
                        )
                    for feature in (
                        CONTEXT_FEATURES
                    )
                },
            }
        )


        regret_rows.append(
            {
                "context_id":
                    cid,

                "motion_ideal":
                    float(
                        ideal[0]
                    ),

                "motion_nadir":
                    float(
                        nadir[0]
                    ),

                "motion_span":
                    float(
                        span[0]
                    ),

                "stability_ideal":
                    float(
                        ideal[1]
                    ),

                "stability_nadir":
                    float(
                        nadir[1]
                    ),

                "stability_span":
                    float(
                        span[1]
                    ),

                "energy_ideal":
                    float(
                        ideal[2]
                    ),

                "energy_nadir":
                    float(
                        nadir[2]
                    ),

                "energy_span":
                    float(
                        span[2]
                    ),
            }
        )


        selected_names = []


        # ----------------------------------------------------
        # 25 mission preferences.
        # ----------------------------------------------------

        for preference in preferences:
            w = np.asarray(
                [
                    finite(
                        preference[
                            "w_motion"
                        ]
                    ),
                    finite(
                        preference[
                            "w_stability"
                        ]
                    ),
                    finite(
                        preference[
                            "w_energy"
                        ]
                    ),
                ],
                dtype=np.float64,
            )

            weighted = (
                regrets
                * w[
                    None,
                    :
                ]
            )

            max_terms = np.max(
                weighted,
                axis=1,
            )

            aug_terms = (
                RHO
                * np.sum(
                    weighted,
                    axis=1,
                )
            )

            scores = (
                max_terms
                + aug_terms
            )


            # Deterministic tie break:
            # retain frozen beta lattice order.
            selected_index = min(
                range(
                    len(front)
                ),
                key=lambda i: (
                    float(
                        scores[i]
                    ),
                    beta_rank[
                        front[i][
                            "beta_name"
                        ]
                    ],
                ),
            )

            selected = (
                front[
                    selected_index
                ]
            )

            selected_regret = (
                regrets[
                    selected_index
                ]
            )

            selected_names.append(
                selected[
                    "beta_name"
                ]
            )


            out = {
                "context_id":
                    cid,

                "terrain":
                    first[
                        "terrain"
                    ],

                "seed":
                    first[
                        "seed"
                    ],

                **{
                    feature:
                        finite(
                            first[
                                feature
                            ]
                        )
                    for feature in (
                        CONTEXT_FEATURES
                    )
                },

                **preference,

                "selected_beta_name":
                    selected[
                        "beta_name"
                    ],

                "selected_beta_motion":
                    finite(
                        selected[
                            "beta_motion"
                        ]
                    ),

                "selected_beta_stability":
                    finite(
                        selected[
                            "beta_stability"
                        ]
                    ),

                "selected_beta_energy":
                    finite(
                        selected[
                            "beta_energy"
                        ]
                    ),

                "selected_lambda_motion":
                    finite(
                        selected[
                            "lambda_motion"
                        ]
                    ),

                "selected_lambda_stability":
                    finite(
                        selected[
                            "lambda_stability"
                        ]
                    ),

                "selected_lambda_energy":
                    finite(
                        selected[
                            "lambda_energy"
                        ]
                    ),

                "selected_J_motion_s_per_m":
                    finite(
                        selected[
                            "J_motion_s_per_m"
                        ]
                    ),

                "selected_J_stability":
                    finite(
                        selected[
                            "J_stability"
                        ]
                    ),

                "selected_J_energy_j_per_m":
                    finite(
                        selected[
                            "J_energy_j_per_m"
                        ]
                    ),

                "regret_motion":
                    float(
                        selected_regret[
                            0
                        ]
                    ),

                "regret_stability":
                    float(
                        selected_regret[
                            1
                        ]
                    ),

                "regret_energy":
                    float(
                        selected_regret[
                            2
                        ]
                    ),

                "tchebycheff_max_term":
                    float(
                        max_terms[
                            selected_index
                        ]
                    ),

                "tchebycheff_aug_term":
                    float(
                        aug_terms[
                            selected_index
                        ]
                    ),

                "scalarization_score":
                    float(
                        scores[
                            selected_index
                        ]
                    ),

                "pareto_count":
                    len(
                        front
                    ),
            }

            label_rows.append(
                out
            )


            if (
                preference[
                    "preference_kind"
                ]
                == "canonical"
            ):
                canonical_rows.append(
                    {
                        "context_id":
                            cid,

                        "preference_name":
                            preference[
                                "preference_name"
                            ],

                        "selected_beta_name":
                            selected[
                                "beta_name"
                            ],
                    }
                )


        selected_beta_diversity[
            cid
        ] = len(
            set(
                selected_names
            )
        )


    # --------------------------------------------------------
    # Hard coverage gates.
    # --------------------------------------------------------

    if len(
        pareto_rows
    ) != EXPECTED_PARETO_ROWS:
        raise RuntimeError(
            "Pareto-row mismatch: "
            f"{len(pareto_rows)} vs "
            f"{EXPECTED_PARETO_ROWS}"
        )

    if len(
        label_rows
    ) != EXPECTED_LABELS:
        raise RuntimeError(
            "Preference-label mismatch: "
            f"{len(label_rows)} vs "
            f"{EXPECTED_LABELS}"
        )

    if len(
        canonical_rows
    ) != (
        EXPECTED_CONTEXTS
        * 4
    ):
        raise RuntimeError(
            "Canonical-label count mismatch."
        )


    # --------------------------------------------------------
    # Regression against frozen T5.4a.
    # --------------------------------------------------------

    for cid, expected in (
        OLD_PARETO_COUNTS.items()
    ):
        actual = (
            pareto_counts[
                cid
            ]
        )

        if actual != expected:
            raise RuntimeError(
                f"T5.4a Pareto regression "
                f"failed for {cid}: "
                f"{actual} vs {expected}"
            )


    # --------------------------------------------------------
    # Regression against frozen T5.4b canonical mappings.
    # --------------------------------------------------------

    canonical_lookup = {
        (
            row[
                "context_id"
            ],
            row[
                "preference_name"
            ],
        ):
            row[
                "selected_beta_name"
            ]
        for row in canonical_rows
    }

    for cid, expected_map in (
        OLD_CANONICAL.items()
    ):
        for pref_name, expected_beta in (
            expected_map.items()
        ):
            actual = (
                canonical_lookup[
                    (
                        cid,
                        pref_name,
                    )
                ]
            )

            if actual != expected_beta:
                raise RuntimeError(
                    "T5.4b canonical regression "
                    f"failed for "
                    f"{cid}/{pref_name}: "
                    f"{actual} vs "
                    f"{expected_beta}"
                )


    # --------------------------------------------------------
    # Write outputs only after all computation passes.
    # --------------------------------------------------------

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    write_csv(
        OUT_FRONT,
        pareto_rows,
    )

    write_csv(
        OUT_CONTEXT_SUMMARY,
        context_summary,
    )

    write_csv(
        OUT_REGRET,
        regret_rows,
    )

    write_csv(
        OUT_LABELS,
        label_rows,
    )

    write_csv(
        OUT_CANONICAL,
        canonical_rows,
    )


    manifest_out = {
        "schema":
            (
                "icra27_os_t5p5d_"
                "expanded_preference_labels_v0"
            ),

        "status":
            "FREEZE_PASS",

        "source":
            str(
                SOURCE_CSV.relative_to(
                    ROOT
                )
            ),

        "source_sha256":
            sha256_file(
                SOURCE_CSV
            ),

        "source_manifest":
            str(
                SOURCE_MANIFEST.relative_to(
                    ROOT
                )
            ),

        "source_manifest_sha256":
            sha256_file(
                SOURCE_MANIFEST
            ),

        "source_rows":
            len(
                rows
            ),

        "contexts":
            len(
                context_order
            ),

        "beta_points_per_context":
            len(
                beta_order
            ),

        "pareto_rows":
            len(
                pareto_rows
            ),

        "pareto_counts":
            pareto_counts,

        "preference_points":
            len(
                preferences
            ),

        "preference_grid_points":
            21,

        "canonical_preferences":
            [
                "balanced",
                "motion",
                "stability",
                "energy",
            ],

        "labels":
            len(
                label_rows
            ),

        "selected_beta_diversity":
            (
                selected_beta_diversity
            ),

        "objective_direction":
            "all_lower_is_better",

        "objectives":
            {
                name:
                    column
                for name, column in (
                    OBJECTIVES
                )
            },

        "pareto_definition":
            (
                "a dominates b iff a is no worse "
                "in all three frozen physical "
                "objectives and strictly better "
                "in at least one"
            ),

        "regret_definition":
            (
                "(J_k - context Pareto ideal_k) / "
                "(context Pareto nadir_k - ideal_k); "
                "zero when the Pareto span is zero"
            ),

        "scalarization":
            (
                "augmented weighted Tchebycheff: "
                "max_k(w_k r_k) + "
                "rho * sum_k(w_k r_k)"
            ),

        "rho":
            RHO,

        "selection_domain":
            (
                "context-local Pareto front only"
            ),

        "tie_break":
            (
                "frozen 21-point beta-lattice order"
            ),

        "context_features":
            list(
                CONTEXT_FEATURES
            ),

        "selector_input_semantics":
            (
                "[c_OS, w] -> beta*, where c_OS "
                "is the frozen 5D physical local "
                "terrain descriptor and w is an "
                "external physical mission preference"
            ),

        "selector_target_semantics":
            (
                "beta* is the internal reward/planner "
                "conditioning point selected from the "
                "frozen beta-response Pareto atlas"
            ),

        "supervision_type":
            (
                "supervised oracle labels; "
                "not inverse reinforcement learning"
            ),

        "heldout_used":
            False,

        "old_t5p4a_regression":
            "PASS",

        "old_t5p4b_canonical_regression":
            "PASS",

        "next_stage":
            (
                "OS-T5.5e: train and validate the "
                "supervised Objective Selector "
                "[c_OS,w] -> beta inside the frozen "
                "biased-anchor convex hull."
            ),
    }


    OUT_MANIFEST.write_text(
        json.dumps(
            manifest_out,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


    # --------------------------------------------------------
    # Report.
    # --------------------------------------------------------

    print()
    print("=" * 116)
    print(
        "ICRA27 OS-T5.5d EXPANDED "
        "PREFERENCE -> BETA LABELS"
    )
    print("=" * 116)

    print(
        "contexts       :",
        len(
            context_order
        ),
    )

    print(
        "beta/context   :",
        len(
            beta_order
        ),
    )

    print(
        "Pareto rows    :",
        len(
            pareto_rows
        ),
    )

    print(
        "preferences    :",
        len(
            preferences
        ),
    )

    print(
        "labels         :",
        len(
            label_rows
        ),
    )

    print()
    print("PARETO / LABEL DIVERSITY")

    for cid in context_order:
        print(
            f"  {cid:<20} "
            f"front={pareto_counts[cid]:2d} "
            f"selected="
            f"{selected_beta_diversity[cid]:2d}"
        )

    print()
    print(
        "T5.4a Pareto regression       : PASS"
    )

    print(
        "T5.4b canonical regression    : PASS"
    )

    print()
    print("CANONICAL PREFERENCES")

    for cid in context_order:
        entries = {
            row[
                "preference_name"
            ]:
                row[
                    "selected_beta_name"
                ]
            for row in canonical_rows
            if row[
                "context_id"
            ] == cid
        }

        print(
            f"  {cid:<20} "
            f"B={entries['balanced']} "
            f"M={entries['motion']} "
            f"S={entries['stability']} "
            f"E={entries['energy']}"
        )

    print()
    print("outputs:")
    print(" ", OUT_FRONT)
    print(" ", OUT_CONTEXT_SUMMARY)
    print(" ", OUT_REGRET)
    print(" ", OUT_LABELS)
    print(" ", OUT_CANONICAL)
    print(" ", OUT_MANIFEST)

    print()
    print(
        "[ICRA27] OS-T5.5d "
        "expanded preference labels: FREEZE PASS"
    )
    print("=" * 116)


if __name__ == "__main__":
    main()
