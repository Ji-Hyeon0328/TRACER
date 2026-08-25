from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path

import analyze_os_t5p3b_physical_atlas_normalization_v0 as phys


ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# Frozen existing T5.5c atlas
# ============================================================

OLD_ROOT = (
    ROOT
    / "results/icra27"
    / "os_t5p5c_expanded_physical_atlas_v0"
)

OLD_CSV = (
    OLD_ROOT
    / "physical_beta_response_atlas.csv"
)

OLD_MANIFEST = (
    OLD_ROOT
    / "expanded_physical_atlas_manifest.json"
)


# ============================================================
# Frozen recovered extension evidence
# ============================================================

F2R_ROOT = (
    ROOT
    / "results/icra27"
    / "os_t5p5f2r_recovered_selector_extension_atlas_v0"
)

F2R_CSV = (
    F2R_ROOT
    / "episodes.csv"
)

F2R_MANIFEST = (
    F2R_ROOT
    / "recovered_atlas_manifest.json"
)


# ============================================================
# Frozen T5.2b physical contract
# ============================================================

CONTRACT = (
    ROOT
    / "results/icra27"
    / "os_t5p2b_physical_preference_contract_v0"
    / "physical_preference_contract.json"
)


# ============================================================
# Outputs
# ============================================================

OUT_DIR = (
    ROOT
    / "results/icra27"
    / "os_t5p5f3_expanded_physical_atlas_v0"
)

OUT_EXTENSION = (
    OUT_DIR
    / "extension_physical_beta_response_atlas.csv"
)

OUT_SELECTOR = (
    OUT_DIR
    / "physical_beta_response_atlas.csv"
)

OUT_MANIFEST = (
    OUT_DIR
    / "expanded_physical_atlas_manifest.json"
)


EXPECTED_OLD_ROWS = 420
EXPECTED_OLD_CONTEXTS = 20

EXPECTED_EXTENSION_CONTEXTS = 15
EXPECTED_BETAS = 21
EXPECTED_EXTENSION_ROWS = (
    EXPECTED_EXTENSION_CONTEXTS
    * EXPECTED_BETAS
)

EXPECTED_TOTAL_CONTEXTS = (
    EXPECTED_OLD_CONTEXTS
    + EXPECTED_EXTENSION_CONTEXTS
)

EXPECTED_TOTAL_ROWS = (
    EXPECTED_OLD_ROWS
    + EXPECTED_EXTENSION_ROWS
)

EXPECTED_SETTLING_STEPS = 5

TOL = 1.0e-10


def read_csv(path):
    with path.open(
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
            f"No rows: {path}"
        )

    fields = []

    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(
                    key
                )

    with path.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(
            rows
        )


def finite(x):
    y = float(x)

    if not math.isfinite(y):
        raise ValueError(
            f"Non-finite: {x!r}"
        )

    return y


def as_int(x):
    return int(
        float(x)
    )


def context_id_for_seed(
    seed,
):
    return (
        f"rough_seed_{int(seed)}"
    )


def selector_projection(
    row,
    *,
    source_stage,
):
    return {
        "context_id":
            row["context_id"],

        "beta_name":
            row["beta_name"],

        "beta_motion":
            finite(
                row["beta_motion"]
            ),

        "beta_stability":
            finite(
                row["beta_stability"]
            ),

        "beta_energy":
            finite(
                row["beta_energy"]
            ),

        "lambda_motion":
            finite(
                row["lambda_motion"]
            ),

        "lambda_stability":
            finite(
                row["lambda_stability"]
            ),

        "lambda_energy":
            finite(
                row["lambda_energy"]
            ),

        "terrain":
            row.get(
                "terrain",
                "",
            ),

        "seed":
            row.get(
                "seed",
                "",
            ),

        "J_motion_s_per_m":
            finite(
                row[
                    "J_motion_s_per_m"
                ]
            ),

        "J_stability_attitude":
            finite(
                row[
                    "J_stability_attitude"
                ]
            ),

        "J_stability_slip":
            finite(
                row[
                    "J_stability_slip"
                ]
            ),

        "J_stability":
            finite(
                row[
                    "J_stability"
                ]
            ),

        "J_stability_dominant":
            row[
                "J_stability_dominant"
            ],

        "J_energy_j_per_m":
            finite(
                row[
                    "J_energy_j_per_m"
                ]
            ),

        "source_stage":
            source_stage,
    }


def main():
    if OUT_DIR.exists():
        raise SystemExit(
            f"REFUSING TO OVERWRITE: "
            f"{OUT_DIR}"
        )


    for path in (
        OLD_CSV,
        OLD_MANIFEST,
        F2R_CSV,
        F2R_MANIFEST,
        CONTRACT,
    ):
        if not path.exists():
            raise FileNotFoundError(
                path
            )


    # ========================================================
    # Frozen upstream contracts
    # ========================================================

    old_manifest = json.loads(
        OLD_MANIFEST.read_text()
    )

    if old_manifest.get(
        "schema"
    ) != (
        "icra27_os_t5p5c_"
        "expanded_physical_atlas_v0"
    ):
        raise RuntimeError(
            "Unexpected T5.5c schema."
        )

    if bool(
        old_manifest[
            "heldout_used"
        ]
    ):
        raise RuntimeError(
            "T5.5c used heldout."
        )

    if int(
        old_manifest[
            "rows"
        ]
    ) != EXPECTED_OLD_ROWS:
        raise RuntimeError(
            "Unexpected T5.5c row count."
        )

    if int(
        old_manifest[
            "contexts"
        ]
    ) != EXPECTED_OLD_CONTEXTS:
        raise RuntimeError(
            "Unexpected T5.5c context count."
        )


    f2r_manifest = json.loads(
        F2R_MANIFEST.read_text()
    )

    if f2r_manifest.get(
        "status"
    ) != "FREEZE_PASS":
        raise RuntimeError(
            "T5.5f2r is not FREEZE_PASS."
        )

    if bool(
        f2r_manifest[
            "heldout_used"
        ]
    ):
        raise RuntimeError(
            "T5.5f2r used heldout."
        )


    eligible_seeds = [
        int(x)
        for x in (
            f2r_manifest[
                "full_21_beta_eligible_contexts"
            ]
        )
    ]


    if len(
        eligible_seeds
    ) != EXPECTED_EXTENSION_CONTEXTS:
        raise RuntimeError(
            "Expected exactly 15 "
            "full-grid extension contexts."
        )


    if f2r_manifest[
        "partial_response_contexts"
    ] != [46]:
        raise RuntimeError(
            "Unexpected partial context set."
        )

    if f2r_manifest[
        "zero_feasible_contexts"
    ] != [34, 43]:
        raise RuntimeError(
            "Unexpected zero-feasible "
            "context set."
        )


    contract = json.loads(
        CONTRACT.read_text()
    )

    if contract.get(
        "schema"
    ) != (
        "icra27_os_t5p2b_"
        "physical_preference_contract_v0"
    ):
        raise RuntimeError(
            "Unexpected T5.2b contract."
        )


    stability_contract = (
        contract[
            "primary_objectives"
        ][
            "stability"
        ]
    )


    roll_unsafe = finite(
        stability_contract[
            "attitude_component"
        ][
            "roll_unsafe_rad"
        ]
    )

    pitch_unsafe = finite(
        stability_contract[
            "attitude_component"
        ][
            "pitch_unsafe_rad"
        ]
    )

    slip_deadband = finite(
        stability_contract[
            "slip_component"
        ][
            "threshold_mps"
        ]
    )


    if not math.isclose(
        slip_deadband,
        0.002,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError(
            "Unexpected frozen "
            "slip threshold."
        )


    # ========================================================
    # Existing immutable 420-row physical atlas
    # ========================================================

    old_rows = read_csv(
        OLD_CSV
    )

    if len(
        old_rows
    ) != EXPECTED_OLD_ROWS:
        raise RuntimeError(
            "Old atlas row mismatch."
        )


    old_context_order = []

    old_beta_order = []


    for row in old_rows:
        cid = row[
            "context_id"
        ]

        beta_name = row[
            "beta_name"
        ]

        if cid not in old_context_order:
            old_context_order.append(
                cid
            )

        if beta_name not in old_beta_order:
            old_beta_order.append(
                beta_name
            )


    if len(
        old_context_order
    ) != EXPECTED_OLD_CONTEXTS:
        raise RuntimeError(
            "Old context count mismatch."
        )

    if len(
        old_beta_order
    ) != EXPECTED_BETAS:
        raise RuntimeError(
            "Old beta count mismatch."
        )


    old_pairs = {
        (
            row[
                "context_id"
            ],
            row[
                "beta_name"
            ],
        )
        for row in old_rows
    }


    if len(
        old_pairs
    ) != EXPECTED_OLD_ROWS:
        raise RuntimeError(
            "Old atlas contains duplicate "
            "context-beta pairs."
        )


    # ========================================================
    # Recovered 378 compact rows
    # ========================================================

    source_rows = read_csv(
        F2R_CSV
    )

    if len(
        source_rows
    ) != 378:
        raise RuntimeError(
            "Expected 378 f2r rows."
        )


    # ========================================================
    # Pair each compact row to the preserved raw log.
    #
    # Frozen T5.5c convention:
    # preserve source-row ordering inside each
    # beta/group block and zip to sorted episode_*.jsonl.
    #
    # Here source root is additionally part of the key because
    # prefix and recovery-tail raw logs live separately.
    # ========================================================

    source_indices_by_group = {}


    for source_index, row in enumerate(
        source_rows
    ):
        key = (
            row[
                "f2r_source_part"
            ],
            row[
                "f2r_raw_root"
            ],
            row[
                "beta_name"
            ],
            row[
                "group"
            ],
        )

        source_indices_by_group.setdefault(
            key,
            [],
        ).append(
            source_index
        )


    raw_path_by_source_index = {}


    for (
        source_part,
        raw_root_string,
        beta_name,
        group,
    ), source_indices in (
        source_indices_by_group.items()
    ):
        raw_root = (
            ROOT
            / raw_root_string
        )

        raw_dir = (
            raw_root
            / beta_name
            / group
        )


        raw_files = sorted(
            raw_dir.glob(
                "episode_*.jsonl"
            )
        )


        if len(
            raw_files
        ) != len(
            source_indices
        ):
            raise RuntimeError(
                f"{source_part}/"
                f"{beta_name}/{group}: "
                f"{len(raw_files)} raw logs "
                f"vs {len(source_indices)} "
                "compact rows."
            )


        for source_index, raw_path in zip(
            source_indices,
            raw_files,
        ):
            raw_path_by_source_index[
                source_index
            ] = raw_path


    if len(
        raw_path_by_source_index
    ) != len(
        source_rows
    ):
        raise RuntimeError(
            "Raw-log pairing incomplete."
        )


    # ========================================================
    # Reconstruct ONLY the 15 full-grid eligible contexts.
    #
    # The other three contexts remain preserved in T5.5f2r.
    # They are not deleted or resampled.
    # ========================================================

    extension_rows = []

    settling_counts = set()

    max_time_error = 0.0
    max_progress_error = 0.0
    max_energy_error = 0.0

    dominance_counts = Counter()


    for source_index, row in enumerate(
        source_rows
    ):
        seed = as_int(
            row[
                "seed"
            ]
        )


        if seed not in eligible_seeds:
            continue


        if str(
            row[
                "f2r_feasible"
            ]
        ).strip() not in (
            "1",
            "1.0",
            "true",
            "True",
        ):
            raise RuntimeError(
                f"Eligible seed {seed} contains "
                "an infeasible beta row."
            )


        if row[
            "status"
        ] != "success":
            raise RuntimeError(
                f"Eligible seed {seed} has "
                f"status={row['status']!r}."
            )


        if as_int(
            row[
                "m4_interventions"
            ]
        ) != 0:
            raise RuntimeError(
                "Eligible row contains M4."
            )


        raw_path = (
            raw_path_by_source_index[
                source_index
            ]
        )


        steps = phys.load_steps(
            raw_path
        )


        policy_steps = as_int(
            row[
                "policy_steps"
            ]
        )


        settling_steps = (
            len(
                steps
            )
            - policy_steps
        )


        settling_counts.add(
            settling_steps
        )


        if settling_steps != (
            EXPECTED_SETTLING_STEPS
        ):
            raise RuntimeError(
                f"{raw_path}: settling="
                f"{settling_steps}, expected="
                f"{EXPECTED_SETTLING_STEPS}"
            )


        boundary = steps[
            settling_steps - 1
        ]

        policy = steps[
            settling_steps:
        ]


        if len(
            policy
        ) != policy_steps:
            raise RuntimeError(
                "Policy-window slicing "
                "mismatch."
            )


        if not policy:
            raise RuntimeError(
                "Empty policy window."
            )


        final = policy[
            -1
        ]


        # ----------------------------------------------------
        # Time
        # ----------------------------------------------------

        decision_time = sum(
            finite(
                x[
                    "decision_dt_s"
                ]
            )
            for x in policy
        )


        compact_time = finite(
            row[
                "decision_time_s"
            ]
        )


        time_error = abs(
            decision_time
            - compact_time
        )


        max_time_error = max(
            max_time_error,
            time_error,
        )


        if time_error > 1e-7:
            raise RuntimeError(
                "Decision-time reconstruction "
                "mismatch."
            )


        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        progress_m = (
            finite(
                boundary[
                    "goal_distance"
                ]
            )
            - finite(
                final[
                    "goal_distance"
                ]
            )
        )


        compact_progress = finite(
            row[
                "progress_m"
            ]
        )


        progress_error = abs(
            progress_m
            - compact_progress
        )


        max_progress_error = max(
            max_progress_error,
            progress_error,
        )


        if progress_error > 1e-7:
            raise RuntimeError(
                "Progress reconstruction "
                "mismatch."
            )


        progress_integrated = sum(
            finite(
                x[
                    "reward_components"
                ][
                    "progress_rate_mps"
                ]
            )
            * finite(
                x[
                    "decision_dt_s"
                ]
            )
            for x in policy
        )


        if abs(
            progress_m
            - progress_integrated
        ) > 1e-7:
            raise RuntimeError(
                "Integrated progress mismatch."
            )


        if progress_m <= 0.0:
            raise RuntimeError(
                "Physical contract requires "
                "positive progress."
            )


        # ----------------------------------------------------
        # Mechanical work
        # ----------------------------------------------------

        energy_abs_j = sum(
            finite(
                x[
                    "reward_components"
                ][
                    "energy_abs_j"
                ]
            )
            for x in policy
        )


        compact_energy = finite(
            row[
                "energy_abs_j"
            ]
        )


        energy_error = abs(
            energy_abs_j
            - compact_energy
        )


        max_energy_error = max(
            max_energy_error,
            energy_error,
        )


        if energy_error > 1e-6:
            raise RuntimeError(
                "Energy reconstruction mismatch."
            )


        # ----------------------------------------------------
        # Physical attitude
        # ----------------------------------------------------

        abs_roll = []

        abs_pitch = []


        for x in policy:
            rc = x[
                "reward_components"
            ]


            abs_roll.append(
                finite(
                    rc[
                        "roll_fraction_of_unsafe"
                    ]
                )
                * finite(
                    rc[
                        "m4_roll_unsafe_rad"
                    ]
                )
            )


            abs_pitch.append(
                finite(
                    rc[
                        "pitch_fraction_of_unsafe"
                    ]
                )
                * finite(
                    rc[
                        "m4_pitch_unsafe_rad"
                    ]
                )
            )


        roll_rms = phys.rms(
            abs_roll
        )

        pitch_rms = phys.rms(
            abs_pitch
        )


        max_abs_roll = max(
            abs_roll
        )

        max_abs_pitch = max(
            abs_pitch
        )


        # ----------------------------------------------------
        # Established-contact slip
        # ----------------------------------------------------

        boundary_slip = (
            boundary[
                "eval_stance_slip"
            ]
        )

        final_slip = (
            final[
                "eval_stance_slip"
            ]
        )


        slip_fraction = (
            phys.policy_tail_fraction(
                boundary_slip=(
                    boundary_slip
                ),
                final_slip=(
                    final_slip
                ),
                threshold=(
                    slip_deadband
                ),
            )
        )


        slip_tail_005 = (
            phys.policy_tail_fraction(
                boundary_slip=(
                    boundary_slip
                ),
                final_slip=(
                    final_slip
                ),
                threshold=0.05,
            )
        )


        slip_tail_010 = (
            phys.policy_tail_fraction(
                boundary_slip=(
                    boundary_slip
                ),
                final_slip=(
                    final_slip
                ),
                threshold=0.10,
            )
        )


        # ----------------------------------------------------
        # Frozen T5.2b physical objectives
        # ----------------------------------------------------

        j_motion = (
            decision_time
            / progress_m
        )


        j_stability_attitude = max(
            roll_rms
            / roll_unsafe,

            pitch_rms
            / pitch_unsafe,
        )


        j_stability_slip = (
            slip_fraction
        )


        j_stability = max(
            j_stability_attitude,
            j_stability_slip,
        )


        j_energy = (
            energy_abs_j
            / progress_m
        )


        if (
            j_stability_attitude
            > j_stability_slip
            + TOL
        ):
            dominant = "attitude"

        elif (
            j_stability_slip
            > j_stability_attitude
            + TOL
        ):
            dominant = "slip"

        else:
            dominant = "tie"


        dominance_counts[
            dominant
        ] += 1


        stem = (
            raw_path.stem
        )

        prefix = (
            "episode_"
        )


        if not stem.startswith(
            prefix
        ):
            raise RuntimeError(
                f"Unexpected raw filename: "
                f"{raw_path.name}"
            )


        episode_index = int(
            stem[
                len(
                    prefix
                ):
            ]
        )


        out = {
            "context_id":
                context_id_for_seed(
                    seed
                ),

            "beta_name":
                row[
                    "beta_name"
                ],

            "beta_motion":
                finite(
                    row[
                        "beta_motion"
                    ]
                ),

            "beta_stability":
                finite(
                    row[
                        "beta_stability"
                    ]
                ),

            "beta_energy":
                finite(
                    row[
                        "beta_energy"
                    ]
                ),

            "lambda_motion":
                finite(
                    row[
                        "lambda_motion"
                    ]
                ),

            "lambda_stability":
                finite(
                    row[
                        "lambda_stability"
                    ]
                ),

            "lambda_energy":
                finite(
                    row[
                        "lambda_energy"
                    ]
                ),

            "group":
                "rough",

            "terrain":
                "rough_perlin",

            "seed":
                seed,

            "episode_index":
                episode_index,

            "raw_path":
                str(
                    raw_path.relative_to(
                        ROOT
                    )
                ),

            "raw_source_part":
                row[
                    "f2r_source_part"
                ],

            "policy_steps":
                policy_steps,

            "settling_steps":
                settling_steps,

            "decision_time_s":
                decision_time,

            "progress_m":
                progress_m,

            "energy_abs_j":
                energy_abs_j,

            "roll_rms_rad":
                roll_rms,

            "pitch_rms_rad":
                pitch_rms,

            "max_abs_roll_rad":
                max_abs_roll,

            "max_abs_pitch_rad":
                max_abs_pitch,

            "established_slip_fraction_ge_0p002":
                slip_fraction,

            "established_slip_fraction_ge_0p05":
                slip_tail_005,

            "established_slip_fraction_ge_0p10":
                slip_tail_010,

            "J_motion_s_per_m":
                j_motion,

            "J_stability_attitude":
                j_stability_attitude,

            "J_stability_slip":
                j_stability_slip,

            "J_stability":
                j_stability,

            "J_stability_dominant":
                dominant,

            "J_energy_j_per_m":
                j_energy,

            "source_stage":
                "T5.5f3_extension",
        }


        extension_rows.append(
            out
        )


    if len(
        extension_rows
    ) != EXPECTED_EXTENSION_ROWS:
        raise RuntimeError(
            f"Expected "
            f"{EXPECTED_EXTENSION_ROWS} "
            f"extension physical rows; "
            f"got {len(extension_rows)}"
        )


    # ========================================================
    # Extension 15 x 21 exact grid validation
    # ========================================================

    extension_context_order = [
        context_id_for_seed(
            seed
        )
        for seed in eligible_seeds
    ]


    extension_pairs = {
        (
            row[
                "context_id"
            ],
            row[
                "beta_name"
            ],
        )
        for row in extension_rows
    }


    if len(
        extension_pairs
    ) != EXPECTED_EXTENSION_ROWS:
        raise RuntimeError(
            "Duplicate extension "
            "context-beta pair."
        )


    for cid in extension_context_order:
        rows = [
            row
            for row in extension_rows
            if row[
                "context_id"
            ] == cid
        ]

        names = {
            row[
                "beta_name"
            ]
            for row in rows
        }


        if len(
            rows
        ) != EXPECTED_BETAS:
            raise RuntimeError(
                f"{cid}: expected 21 rows."
            )


        if names != set(
            old_beta_order
        ):
            raise RuntimeError(
                f"{cid}: beta lattice "
                "does not match T5.5c."
            )


    # ========================================================
    # Build combined SELECTOR physical atlas.
    #
    # Do not fabricate old v1 context descriptors for new
    # extension terrains.  Keep only the common physical
    # response schema required by downstream Pareto labels.
    # Terrain representation is joined separately in f4.
    # ========================================================

    combined_rows = []


    for row in old_rows:
        combined_rows.append(
            selector_projection(
                row,
                source_stage=(
                    "T5.5c_frozen"
                ),
            )
        )


    for row in extension_rows:
        combined_rows.append(
            selector_projection(
                row,
                source_stage=(
                    "T5.5f3_extension"
                ),
            )
        )


    # Sort deterministically by context then frozen beta order.
    context_order = (
        old_context_order
        + extension_context_order
    )


    context_rank = {
        cid:
            i
        for i, cid in enumerate(
            context_order
        )
    }


    beta_rank = {
        beta:
            i
        for i, beta in enumerate(
            old_beta_order
        )
    }


    combined_rows.sort(
        key=lambda row: (
            context_rank[
                row[
                    "context_id"
                ]
            ],
            beta_rank[
                row[
                    "beta_name"
                ]
            ],
        )
    )


    if len(
        combined_rows
    ) != EXPECTED_TOTAL_ROWS:
        raise RuntimeError(
            "Combined atlas row mismatch."
        )


    combined_pairs = {
        (
            row[
                "context_id"
            ],
            row[
                "beta_name"
            ],
        )
        for row in combined_rows
    }


    if len(
        combined_pairs
    ) != EXPECTED_TOTAL_ROWS:
        raise RuntimeError(
            "Combined atlas contains "
            "duplicate context-beta pairs."
        )


    if len(
        {
            row[
                "context_id"
            ]
            for row in combined_rows
        }
    ) != EXPECTED_TOTAL_CONTEXTS:
        raise RuntimeError(
            "Combined context count mismatch."
        )


    # ========================================================
    # Descriptive ranges only.
    #
    # NOT a new normalization contract.
    # ========================================================

    objective_columns = (
        "J_motion_s_per_m",
        "J_stability",
        "J_energy_j_per_m",
    )


    objective_ranges = {}


    for column in objective_columns:
        values = [
            finite(
                row[
                    column
                ]
            )
            for row in combined_rows
        ]


        objective_ranges[
            column
        ] = [
            min(
                values
            ),
            max(
                values
            ),
        ]


    # ========================================================
    # Write
    # ========================================================

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )


    write_csv(
        OUT_EXTENSION,
        extension_rows,
    )

    write_csv(
        OUT_SELECTOR,
        combined_rows,
    )


    manifest = {
        "schema":
            (
                "icra27_os_t5p5f3_"
                "expanded_physical_atlas_v0"
            ),

        "status":
            "FREEZE_PASS",

        "heldout_used":
            False,

        "original_split_modified":
            False,

        "physical_contract":
            (
                "Frozen OS-T5.2b "
                "physical preference contract"
            ),

        "stability_definition":
            (
                "max(J_stability_attitude, "
                "J_stability_slip)"
            ),

        "objective_direction":
            "all_lower_is_better",

        "normalization_policy":
            (
                "No global normalization is "
                "defined or refrozen. "
                "Downstream preference labels "
                "must use context-local "
                "Pareto ideal/nadir regret."
            ),

        "old_source":
            str(
                OLD_CSV.relative_to(
                    ROOT
                )
            ),

        "old_rows":
            EXPECTED_OLD_ROWS,

        "old_contexts":
            EXPECTED_OLD_CONTEXTS,

        "extension_evidence_source":
            str(
                F2R_CSV.relative_to(
                    ROOT
                )
            ),

        "sampled_extension_contexts":
            18,

        "full_grid_extension_contexts":
            EXPECTED_EXTENSION_CONTEXTS,

        "full_grid_extension_seeds":
            eligible_seeds,

        "preserved_non_full_grid_evidence":
            {
                "partial_response":
                    [46],

                "zero_feasible":
                    [34, 43],
            },

        "extension_rows":
            EXPECTED_EXTENSION_ROWS,

        "combined_contexts":
            EXPECTED_TOTAL_CONTEXTS,

        "combined_rows":
            EXPECTED_TOTAL_ROWS,

        "beta_points_per_context":
            EXPECTED_BETAS,

        "context_beta_pairs_unique":
            True,

        "context_order":
            context_order,

        "reconstruction_validation":
            {
                "settling_steps":
                    sorted(
                        settling_counts
                    ),

                "max_time_error_s":
                    max_time_error,

                "max_progress_error_m":
                    max_progress_error,

                "max_energy_error_j":
                    max_energy_error,
            },

        "extension_stability_dominance":
            dict(
                sorted(
                    dominance_counts.items()
                )
            ),

        "objective_ranges_descriptive_only":
            objective_ranges,

        "selector_row_semantics":
            (
                "Common physical response schema "
                "only: context_id, beta, and "
                "frozen physical J_M/J_S/J_E. "
                "Terrain representation is joined "
                "from frozen oracle height-patch "
                "artifacts downstream."
            ),

        "next_stage":
            (
                "T5.5f3b: recompute context-wise "
                "Pareto fronts and the unchanged "
                "25-preference augmented-"
                "Tchebycheff labels for "
                "35 x 25 = 875 TRAIN-only "
                "(context,w)->beta* examples."
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
    print("=" * 120)

    print(
        "ICRA27 OS-T5.5f3 EXPANDED "
        "PHYSICAL RESPONSE ATLAS FREEZE"
    )

    print("=" * 120)

    print(
        "old rows                :",
        EXPECTED_OLD_ROWS,
    )

    print(
        "extension rows          :",
        len(
            extension_rows
        ),
    )

    print(
        "combined rows           :",
        len(
            combined_rows
        ),
    )

    print(
        "combined contexts       :",
        len(
            context_order
        ),
    )

    print(
        "beta/context            :",
        EXPECTED_BETAS,
    )

    print(
        "eligible extension seeds:",
        eligible_seeds,
    )

    print(
        "preserved non-full-grid :",
        {
            "partial": [46],
            "zero": [34, 43],
        },
    )

    print()
    print("RECONSTRUCTION")

    print(
        "  settling steps        :",
        sorted(
            settling_counts
        ),
    )

    print(
        "  max time error        :",
        max_time_error,
    )

    print(
        "  max progress error    :",
        max_progress_error,
    )

    print(
        "  max energy error      :",
        max_energy_error,
    )

    print(
        "  stability dominance   :",
        dict(
            dominance_counts
        ),
    )

    print()
    print("OBJECTIVE RANGES — descriptive only")

    for key, value in (
        objective_ranges.items()
    ):
        print(
            f"  {key:<24}: "
            f"[{value[0]}, {value[1]}]"
        )

    print()
    print("outputs:")
    print(" ", OUT_EXTENSION)
    print(" ", OUT_SELECTOR)
    print(" ", OUT_MANIFEST)

    print()
    print(
        "[ICRA27] OS-T5.5f3 expanded "
        "physical atlas: FREEZE PASS"
    )

    print("=" * 120)


if __name__ == "__main__":
    main()
