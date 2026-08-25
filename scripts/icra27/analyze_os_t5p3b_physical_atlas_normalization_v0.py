from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]

ATLAS_ROOT = (
    ROOT
    / "results/icra27"
    / "os_t5p3_train_beta_response_atlas_v0"
)

SOURCE_CSV = (
    ATLAS_ROOT
    / "episodes.csv"
)

RAW_ROOT = (
    ATLAS_ROOT
    / "env_logs"
)

CONTRACT_PATH = (
    ROOT
    / "results/icra27"
    / "os_t5p2b_physical_preference_contract_v0"
    / "physical_preference_contract.json"
)

OUT_DIR = (
    ROOT
    / "results/icra27"
    / "os_t5p3b_physical_atlas_normalization_v0"
)

OUT_CSV = (
    OUT_DIR
    / "physical_beta_response_atlas.csv"
)

OUT_STATS = (
    OUT_DIR
    / "normalization_stats.json"
)

OUT_MANIFEST = (
    OUT_DIR
    / "physical_atlas_manifest.json"
)

EXPECTED_EPISODES = 189
EXPECTED_BETAS = 21
EXPECTED_CONTEXTS = 9
EXPECTED_SETTLING_STEPS = 5

EPS = 1.0e-12
TOL = 1.0e-10


def finite(x: Any) -> float:
    y = float(x)

    if not math.isfinite(y):
        raise ValueError(
            f"Non-finite value: {x!r}"
        )

    return y


def as_bool(x: Any) -> bool:
    return str(x).strip().lower() in {
        "1",
        "true",
        "yes",
        "success",
    }


def load_steps(
    path: Path,
) -> list[dict[str, Any]]:
    rows = []

    with path.open() as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            obj = json.loads(line)

            if obj.get("event") == "step":
                rows.append(obj)

    return rows


def histogram_tail_time(
    histogram: list[dict[str, Any]],
    threshold: float,
) -> float:
    total = 0.0

    for item in histogram:
        low = item.get(
            "low_mps"
        )

        if low is None:
            continue

        if finite(low) + TOL >= threshold:
            total += finite(
                item.get(
                    "contact_time_s",
                    0.0,
                )
            )

    return total


def policy_tail_fraction(
    *,
    boundary_slip: dict[str, Any],
    final_slip: dict[str, Any],
    threshold: float,
) -> float:
    contact_dt = (
        finite(
            final_slip[
                "established_contact_time_s"
            ]
        )
        - finite(
            boundary_slip[
                "established_contact_time_s"
            ]
        )
    )

    if contact_dt <= 0.0:
        raise RuntimeError(
            "Non-positive established "
            "policy contact time."
        )

    final_tail = histogram_tail_time(
        final_slip[
            "established_slip_histogram"
        ],
        threshold,
    )

    boundary_tail = histogram_tail_time(
        boundary_slip[
            "established_slip_histogram"
        ],
        threshold,
    )

    tail_dt = (
        final_tail
        - boundary_tail
    )

    fraction = (
        tail_dt
        / contact_dt
    )

    if (
        fraction < -1e-9
        or fraction > 1.0 + 1e-9
    ):
        raise RuntimeError(
            "Slip fraction outside [0,1]: "
            f"{fraction}"
        )

    return min(
        1.0,
        max(
            0.0,
            fraction,
        ),
    )


def rms(
    values: list[float],
) -> float:
    if not values:
        raise ValueError(
            "Empty RMS input."
        )

    return math.sqrt(
        sum(
            x * x
            for x in values
        )
        / len(values)
    )


def row_group(
    row: dict[str, str],
) -> str:
    for key in (
        "group",
        "terrain_label",
    ):
        value = row.get(key)

        if value:
            return str(value)

    raise KeyError(
        "Could not determine atlas group."
    )


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
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


def main() -> None:
    if OUT_DIR.exists():
        raise SystemExit(
            f"REFUSING TO OVERWRITE: {OUT_DIR}"
        )

    if not SOURCE_CSV.exists():
        raise FileNotFoundError(
            SOURCE_CSV
        )

    if not CONTRACT_PATH.exists():
        raise FileNotFoundError(
            CONTRACT_PATH
        )

    contract = json.loads(
        CONTRACT_PATH.read_text()
    )

    if contract.get("schema") != (
        "icra27_os_t5p2b_physical_preference_contract_v0"
    ):
        raise RuntimeError(
            "Unexpected T5.2b contract schema."
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
            "Unexpected frozen slip threshold: "
            f"{slip_deadband}"
        )

    with SOURCE_CSV.open(
        newline="",
    ) as f:
        source_rows = list(
            csv.DictReader(f)
        )

    if len(source_rows) != EXPECTED_EPISODES:
        raise RuntimeError(
            f"Expected {EXPECTED_EPISODES} "
            f"atlas rows; got "
            f"{len(source_rows)}"
        )

    if any(
        not as_bool(
            row.get(
                "success",
                False,
            )
        )
        for row in source_rows
    ):
        raise RuntimeError(
            "T5.3b requires all atlas "
            "trajectories feasible/successful."
        )

    beta_names = {
        row["beta_name"]
        for row in source_rows
    }

    contexts = {
        (
            row_group(row),
            int(
                float(
                    row["seed"]
                )
            ),
        )
        for row in source_rows
    }

    if len(beta_names) != EXPECTED_BETAS:
        raise RuntimeError(
            f"Expected {EXPECTED_BETAS} betas; "
            f"got {len(beta_names)}"
        )

    if len(contexts) != EXPECTED_CONTEXTS:
        raise RuntimeError(
            f"Expected {EXPECTED_CONTEXTS} "
            f"context slots; got "
            f"{len(contexts)}"
        )

    grouped: dict[
        tuple[str, str],
        list[dict[str, str]],
    ] = {}

    for row in source_rows:
        key = (
            row["beta_name"],
            row_group(row),
        )

        grouped.setdefault(
            key,
            [],
        ).append(row)

    output_rows: list[
        dict[str, Any]
    ] = []

    settling_counts = set()

    max_time_error = 0.0
    max_energy_error = 0.0
    max_progress_error = 0.0

    dominance_counts = {
        "attitude":
            0,

        "slip":
            0,

        "tie":
            0,
    }

    for (
        beta_name,
        group,
    ), rows in grouped.items():
        raw_dir = (
            RAW_ROOT
            / beta_name
            / group
        )

        raw_files = sorted(
            raw_dir.glob(
                "episode_*.jsonl"
            )
        )

        if len(raw_files) != len(rows):
            raise RuntimeError(
                f"{beta_name}/{group}: "
                f"{len(raw_files)} raw files "
                f"vs {len(rows)} source rows."
            )

        for episode_index, (
            row,
            raw_path,
        ) in enumerate(
            zip(
                rows,
                raw_files,
            )
        ):
            steps = load_steps(
                raw_path
            )

            policy_steps = int(
                float(
                    row[
                        "policy_steps"
                    ]
                )
            )

            settling_steps = (
                len(steps)
                - policy_steps
            )

            settling_counts.add(
                settling_steps
            )

            if settling_steps <= 0:
                raise RuntimeError(
                    "Missing pre-policy "
                    "settling boundary."
                )

            boundary = steps[
                settling_steps - 1
            ]

            policy = steps[
                settling_steps:
            ]

            if len(policy) != policy_steps:
                raise RuntimeError(
                    "Policy-window slicing mismatch."
                )

            final = policy[-1]

            # -----------------------------------------------
            # Time / progress.
            # -----------------------------------------------

            decision_time = sum(
                finite(
                    x["decision_dt_s"]
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
                    "Decision-time "
                    "reconstruction mismatch."
                )

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

            progress_integrated = sum(
                finite(
                    x[
                        "reward_components"
                    ][
                        "progress_rate_mps"
                    ]
                )
                * finite(
                    x["decision_dt_s"]
                )
                for x in policy
            )

            progress_error = abs(
                progress_m
                - progress_integrated
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

            if progress_m <= 0.0:
                raise RuntimeError(
                    "Physical contract requires "
                    "positive progress."
                )

            # -----------------------------------------------
            # Mechanical work.
            # -----------------------------------------------

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
                    "Energy reconstruction "
                    "mismatch."
                )

            # -----------------------------------------------
            # Physical attitude.
            # -----------------------------------------------

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

            roll_rms = rms(
                abs_roll
            )

            pitch_rms = rms(
                abs_pitch
            )

            max_abs_roll = max(
                abs_roll
            )

            max_abs_pitch = max(
                abs_pitch
            )

            # -----------------------------------------------
            # Established-contact physical slip.
            # -----------------------------------------------

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
                policy_tail_fraction(
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
                policy_tail_fraction(
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
                policy_tail_fraction(
                    boundary_slip=(
                        boundary_slip
                    ),
                    final_slip=(
                        final_slip
                    ),
                    threshold=0.10,
                )
            )

            # -----------------------------------------------
            # Frozen OS-T5.2b physical preference contract.
            # -----------------------------------------------

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

            out = {
                "beta_name":
                    beta_name,

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
                    group,

                "terrain":
                    row[
                        "terrain"
                    ],

                "seed":
                    int(
                        float(
                            row["seed"]
                        )
                    ),

                "episode_index":
                    episode_index,

                "raw_path":
                    str(
                        raw_path.relative_to(
                            ROOT
                        )
                    ),

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
            }

            output_rows.append(
                out
            )

    if len(output_rows) != EXPECTED_EPISODES:
        raise RuntimeError(
            f"Expected {EXPECTED_EPISODES} "
            f"physical rows; got "
            f"{len(output_rows)}"
        )

    if settling_counts != {
        EXPECTED_SETTLING_STEPS
    }:
        raise RuntimeError(
            "Unexpected settling-step counts: "
            f"{sorted(settling_counts)}"
        )

    # -------------------------------------------------------
    # Global TRAIN-only robust normalization.
    #
    # Important:
    #   - one common normalization for all contexts;
    #   - otherwise the semantic meaning of w would change
    #     with terrain.
    #
    # The protocol has 63 rows per terrain group:
    # 21 beta x 3 context slots.
    #
    # Flat / low-friction seed labels are deterministic
    # replicates, not independent stochastic trials.
    # Keeping all 189 rows provides equal protocol weight
    # to flat, low-friction, and rough terrain groups.
    # -------------------------------------------------------

    metric_columns = {
        "motion":
            "J_motion_s_per_m",

        "stability":
            "J_stability",

        "energy":
            "J_energy_j_per_m",
    }

    stats = {}

    for objective, column in (
        metric_columns.items()
    ):
        values = np.asarray(
            [
                finite(row[column])
                for row in output_rows
            ],
            dtype=np.float64,
        )

        q25, median, q75 = (
            np.quantile(
                values,
                [
                    0.25,
                    0.50,
                    0.75,
                ],
                method="linear",
            )
        )

        iqr = float(
            q75 - q25
        )

        if iqr <= EPS:
            raise RuntimeError(
                f"{objective}: "
                f"non-positive IQR={iqr}"
            )

        stats[
            objective
        ] = {
            "source_column":
                column,

            "q25":
                float(q25),

            "median":
                float(median),

            "q75":
                float(q75),

            "iqr":
                iqr,

            "epsilon":
                EPS,

            "denominator":
                iqr + EPS,

            "minimum":
                float(
                    np.min(values)
                ),

            "maximum":
                float(
                    np.max(values)
                ),

            "count":
                int(
                    values.size
                ),
        }

    for row in output_rows:
        for objective, column in (
            metric_columns.items()
        ):
            s = stats[
                objective
            ]

            row[
                f"J_{objective}_normalized"
            ] = (
                finite(
                    row[column]
                )
                - s["median"]
            ) / s["denominator"]

    # -------------------------------------------------------
    # Per-terrain physical spread diagnostic.
    # This does NOT alter normalization.
    # -------------------------------------------------------

    group_summary = {}

    for group in (
        "flat",
        "low_friction",
        "rough",
    ):
        rows = [
            row
            for row in output_rows
            if row["group"] == group
        ]

        group_summary[
            group
        ] = {
            "rows":
                len(rows),

            "unique_seeds":
                sorted({
                    int(row["seed"])
                    for row in rows
                }),

            "J_motion_range":
                [
                    min(
                        finite(
                            x[
                                "J_motion_s_per_m"
                            ]
                        )
                        for x in rows
                    ),
                    max(
                        finite(
                            x[
                                "J_motion_s_per_m"
                            ]
                        )
                        for x in rows
                    ),
                ],

            "J_stability_range":
                [
                    min(
                        finite(
                            x[
                                "J_stability"
                            ]
                        )
                        for x in rows
                    ),
                    max(
                        finite(
                            x[
                                "J_stability"
                            ]
                        )
                        for x in rows
                    ),
                ],

            "J_energy_range":
                [
                    min(
                        finite(
                            x[
                                "J_energy_j_per_m"
                            ]
                        )
                        for x in rows
                    ),
                    max(
                        finite(
                            x[
                                "J_energy_j_per_m"
                            ]
                        )
                        for x in rows
                    ),
                ],
        }

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    write_csv(
        OUT_CSV,
        output_rows,
    )

    normalization = {
        "schema":
            "icra27_os_t5p3b_train_normalization_v0",

        "status":
            "FROZEN",

        "scope":
            "TRAIN-only beta-response atlas",

        "population_rows":
            len(output_rows),

        "method":
            "global median / IQR",

        "formula":
            (
                "(J_k - median_train(J_k)) / "
                "(IQR_train(J_k) + epsilon)"
            ),

        "lower_is_better":
            True,

        "context_specific_normalization":
            False,

        "heldout_used":
            False,

        "protocol_weighting_note":
            (
                "All 189 TRAIN protocol rows are retained. "
                "There are 63 rows per terrain group "
                "(21 beta x 3 context slots). Flat and "
                "low-friction seed slots are deterministic "
                "replicates and are not interpreted as "
                "independent stochastic trials."
            ),

        "statistics":
            stats,
    }

    OUT_STATS.write_text(
        json.dumps(
            normalization,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    manifest = {
        "schema":
            "icra27_os_t5p3b_physical_atlas_v0",

        "status":
            "FREEZE_PASS",

        "source_atlas_csv":
            str(
                SOURCE_CSV.relative_to(
                    ROOT
                )
            ),

        "source_atlas_sha256":
            hashlib.sha256(
                SOURCE_CSV.read_bytes()
            ).hexdigest(),

        "physical_contract":
            str(
                CONTRACT_PATH.relative_to(
                    ROOT
                )
            ),

        "physical_contract_sha256":
            hashlib.sha256(
                CONTRACT_PATH.read_bytes()
            ).hexdigest(),

        "episodes":
            len(output_rows),

        "beta_points":
            len(beta_names),

        "context_slots":
            len(contexts),

        "settling_steps":
            sorted(
                settling_counts
            ),

        "validation":
            {
                "max_time_reconstruction_error_s":
                    max_time_error,

                "max_energy_reconstruction_error_j":
                    max_energy_error,

                "max_progress_reconstruction_error_m":
                    max_progress_error,
            },

        "stability_dominance":
            dominance_counts,

        "terrain_group_summary":
            group_summary,

        "normalization_file":
            str(
                OUT_STATS.relative_to(
                    ROOT
                )
            ),

        "next_stage":
            (
                "OS-T5.4: fixed external mission "
                "preferences w -> context-specific "
                "preferred beta extraction / Pareto audit"
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

    print("=" * 104)
    print(
        "ICRA27 OS-T5.3b "
        "PHYSICAL ATLAS + NORMALIZATION FREEZE"
    )
    print("=" * 104)

    print()
    print(
        "episodes       :",
        len(output_rows),
    )
    print(
        "beta points    :",
        len(beta_names),
    )
    print(
        "context slots  :",
        len(contexts),
    )
    print(
        "settling steps :",
        sorted(settling_counts),
    )

    print()
    print("RECONSTRUCTION")
    print(
        "  max time error    :",
        max_time_error,
    )
    print(
        "  max energy error  :",
        max_energy_error,
    )
    print(
        "  max progress error:",
        max_progress_error,
    )

    print()
    print("GLOBAL TRAIN NORMALIZATION")

    for objective in (
        "motion",
        "stability",
        "energy",
    ):
        s = stats[
            objective
        ]

        print(
            f"  {objective:<10} "
            f"median={s['median']:.9f} "
            f"IQR={s['iqr']:.9f} "
            f"range=["
            f"{s['minimum']:.9f}, "
            f"{s['maximum']:.9f}]"
        )

    print()
    print(
        "J_S dominance:",
        dominance_counts,
    )

    print()
    print("PER-TERRAIN PHYSICAL SPREAD")

    for group, item in (
        group_summary.items()
    ):
        print(
            f"  {group:<14} "
            f"J_M={item['J_motion_range']} "
            f"J_S={item['J_stability_range']} "
            f"J_E={item['J_energy_range']}"
        )

    print()
    print("outputs:")
    print(" ", OUT_CSV)
    print(" ", OUT_STATS)
    print(" ", OUT_MANIFEST)

    print()
    print(
        "[ICRA27] OS-T5.3b "
        "physical atlas normalization: FREEZE PASS"
    )


if __name__ == "__main__":
    main()
