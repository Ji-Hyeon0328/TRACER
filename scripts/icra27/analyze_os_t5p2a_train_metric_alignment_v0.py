from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]

SOURCE_CSV = (
    ROOT
    / "results/icra27"
    / "os_t4p8d_v4_lockstep_checkpoint_sweep_v0"
    / "train9"
    / "episodes.csv"
)

OUT_DIR = (
    ROOT
    / "results/icra27"
    / "os_t5p2a_train_metric_alignment_v0"
)

OUT_CSV = OUT_DIR / "paired_metric_alignment.csv"
OUT_JSON = OUT_DIR / "metric_alignment_manifest.json"

TOL = 1.0e-12


def as_float(row: dict[str, str], key: str) -> float:
    value = row.get(key)

    if value is None or value == "":
        raise KeyError(
            f"Missing required numeric field {key!r}"
        )

    result = float(value)

    if not math.isfinite(result):
        raise ValueError(
            f"Non-finite value for {key!r}: {value!r}"
        )

    return result


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value

    text = str(value).strip().lower()

    return text in {
        "1",
        "true",
        "yes",
        "success",
    }


def first_existing(
    fieldnames: list[str],
    candidates: list[str],
) -> str | None:
    for key in candidates:
        if key in fieldnames:
            return key

    return None


def is_u30(row: dict[str, str]) -> bool:
    checkpoint = str(
        row.get("checkpoint", "")
    ).lower()

    if (
        "update_0030" in checkpoint
        or "update0030" in checkpoint
        or "u30" in checkpoint
    ):
        return True

    for key in (
        "update",
        "checkpoint_update",
        "update_index",
    ):
        value = row.get(key)

        if value not in (None, ""):
            try:
                if int(float(value)) == 30:
                    return True
            except ValueError:
                pass

    return False


def context_key(
    row: dict[str, str],
) -> tuple[str, int]:
    group = str(
        row.get(
            "group",
            row.get(
                "terrain",
                "",
            ),
        )
    )

    seed = int(
        float(
            row["seed"]
        )
    )

    return group, seed


def safe_div(
    numerator: float,
    denominator: float,
) -> float:
    if abs(denominator) <= TOL:
        raise ZeroDivisionError(
            "Refusing division by near-zero progress."
        )

    return numerator / denominator


def pearson(
    xs: list[float],
    ys: list[float],
) -> float | None:
    if len(xs) != len(ys):
        raise ValueError(
            "Pearson inputs must have equal length."
        )

    if len(xs) < 2:
        return None

    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)

    dx = [
        x - mx
        for x in xs
    ]

    dy = [
        y - my
        for y in ys
    ]

    sx = math.sqrt(
        sum(x * x for x in dx)
    )

    sy = math.sqrt(
        sum(y * y for y in dy)
    )

    if sx <= TOL or sy <= TOL:
        return None

    return (
        sum(
            x * y
            for x, y in zip(dx, dy)
        )
        / (sx * sy)
    )


def sign_class(value: float) -> int:
    if value < -TOL:
        return -1

    if value > TOL:
        return +1

    return 0


def metric_value(
    row: dict[str, str],
    metric_name: str,
    source_key: str | None,
) -> float:
    if metric_name == "seconds_per_meter":
        return safe_div(
            as_float(
                row,
                "decision_time_s",
            ),
            as_float(
                row,
                "progress_m",
            ),
        )

    if metric_name == "energy_j_per_meter":
        return safe_div(
            as_float(
                row,
                "energy_abs_j",
            ),
            as_float(
                row,
                "progress_m",
            ),
        )

    if metric_name == "attitude_rms_norm":
        pitch = as_float(
            row,
            "pitch_rms_rad",
        )

        roll = as_float(
            row,
            "roll_rms_rad",
        )

        return math.sqrt(
            pitch * pitch
            + roll * roll
        )

    if metric_name == "attitude_peak_max":
        pitch = as_float(
            row,
            "max_abs_pitch_rad",
        )

        roll = as_float(
            row,
            "max_abs_roll_rad",
        )

        return max(
            pitch,
            roll,
        )

    if source_key is None:
        raise RuntimeError(
            f"No source field for {metric_name!r}"
        )

    return as_float(
        row,
        source_key,
    )


def main() -> None:
    if not SOURCE_CSV.exists():
        raise SystemExit(
            "Missing source CSV: "
            + str(SOURCE_CSV)
        )

    with SOURCE_CSV.open(
        newline="",
    ) as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise SystemExit(
                "Source CSV has no header."
            )

        fieldnames = list(
            reader.fieldnames
        )

        all_rows = list(reader)

    u30_rows = [
        row
        for row in all_rows
        if is_u30(row)
    ]

    if len(u30_rows) != 36:
        raise SystemExit(
            "Expected exactly 36 u30 TRAIN rows "
            "(9 contexts x 4 beta settings); "
            f"found {len(u30_rows)}."
        )

    beta_names = {
        str(row["beta_name"])
        for row in u30_rows
    }

    expected_beta_names = {
        "balanced",
        "motion",
        "stability",
        "energy",
    }

    if beta_names != expected_beta_names:
        raise SystemExit(
            "Unexpected beta_name set: "
            f"{sorted(beta_names)}"
        )

    failed_rows = [
        row
        for row in u30_rows
        if not as_bool(
            row.get(
                "success",
                row.get(
                    "status",
                    "",
                ),
            )
        )
    ]

    if failed_rows:
        raise SystemExit(
            "Expected all selected u30 TRAIN rows "
            "to be successful; found "
            f"{len(failed_rows)} failures."
        )

    posture_key = first_existing(
        fieldnames,
        [
            "mean_cost_posture",
            "mean_posture_cost",
            "mean_stability_posture_cost",
            "mean_cost_stability_posture",
            "cost_posture_mean",
        ],
    )

    traction_key = first_existing(
        fieldnames,
        [
            "mean_cost_traction",
            "mean_traction_cost",
            "mean_stability_traction_cost",
            "mean_cost_stability_traction",
            "cost_traction_mean",
        ],
    )

    indexed: dict[
        tuple[str, int],
        dict[str, dict[str, str]],
    ] = {}

    for row in u30_rows:
        key = context_key(row)
        beta_name = str(
            row["beta_name"]
        )

        indexed.setdefault(
            key,
            {},
        )

        if beta_name in indexed[key]:
            raise SystemExit(
                "Duplicate context/beta row: "
                f"{key} / {beta_name}"
            )

        indexed[key][
            beta_name
        ] = row

    if len(indexed) != 9:
        raise SystemExit(
            "Expected 9 TRAIN contexts; "
            f"found {len(indexed)}."
        )

    for key, beta_rows in indexed.items():
        if set(beta_rows) != expected_beta_names:
            raise SystemExit(
                "Incomplete beta bank for "
                f"context {key}: "
                f"{sorted(beta_rows)}"
            )

    metric_specs: dict[
        str,
        dict[str, Any],
    ] = {
        "motion": {
            "candidate":
                "motion",

            "target_key":
                "mean_cost_motion",

            "physical": [
                {
                    "name":
                        "seconds_per_meter",
                    "source_key":
                        None,
                    "lower_is_better":
                        True,
                },
                {
                    "name":
                        "decision_time_s",
                    "source_key":
                        "decision_time_s",
                    "lower_is_better":
                        True,
                },
                {
                    "name":
                        "progress_rate_mps",
                    "source_key":
                        "mean_progress_rate_mps",
                    "lower_is_better":
                        False,
                },
                {
                    "name":
                        "applied_vx_mps",
                    "source_key":
                        "mean_applied_vx_mps",
                    "lower_is_better":
                        False,
                },
                {
                    "name":
                        "final_goal_distance_m",
                    "source_key":
                        "final_goal_distance_m",
                    "lower_is_better":
                        True,
                },
            ],
        },

        "stability": {
            "candidate":
                "stability",

            "target_key":
                "mean_cost_stability",

            "physical": [],
        },

        "energy": {
            "candidate":
                "energy",

            "target_key":
                "mean_cost_energy",

            "physical": [
                {
                    "name":
                        "energy_abs_j",
                    "source_key":
                        "energy_abs_j",
                    "lower_is_better":
                        True,
                },
                {
                    "name":
                        "energy_j_per_meter",
                    "source_key":
                        None,
                    "lower_is_better":
                        True,
                },
                {
                    "name":
                        "energy_power_ratio",
                    "source_key":
                        "mean_energy_power_ratio",
                    "lower_is_better":
                        True,
                },
            ],
        },
    }

    stability_physical = [
        {
            "name":
                "pitch_rms_rad",
            "source_key":
                "pitch_rms_rad",
            "lower_is_better":
                True,
        },
        {
            "name":
                "roll_rms_rad",
            "source_key":
                "roll_rms_rad",
            "lower_is_better":
                True,
        },
        {
            "name":
                "attitude_rms_norm",
            "source_key":
                None,
            "lower_is_better":
                True,
        },
        {
            "name":
                "max_abs_pitch_rad",
            "source_key":
                "max_abs_pitch_rad",
            "lower_is_better":
                True,
        },
        {
            "name":
                "max_abs_roll_rad",
            "source_key":
                "max_abs_roll_rad",
            "lower_is_better":
                True,
        },
        {
            "name":
                "attitude_peak_max",
            "source_key":
                None,
            "lower_is_better":
                True,
        },
    ]

    if posture_key is not None:
        stability_physical.insert(
            0,
            {
                "name":
                    "reward_posture_component",
                "source_key":
                    posture_key,
                "lower_is_better":
                    True,
            },
        )

    if traction_key is not None:
        insert_index = (
            1
            if posture_key is not None
            else 0
        )

        stability_physical.insert(
            insert_index,
            {
                "name":
                    "reward_traction_component",
                "source_key":
                    traction_key,
                "lower_is_better":
                    True,
            },
        )

    metric_specs[
        "stability"
    ][
        "physical"
    ] = stability_physical

    # --------------------------------------------------------
    # Source-schema-aware metric availability.
    #
    # The compact T4.8d TRAIN checkpoint-sweep CSV intentionally
    # contains only a subset of the richer trajectory diagnostics.
    # Do not infer or fabricate missing physical quantities.
    #
    # Instead:
    #   1. activate metrics whose required source columns exist,
    #   2. skip unavailable metrics explicitly,
    #   3. record the missing requirements in the manifest.
    #
    # This keeps the analyzer reusable with a richer source CSV
    # later without changing the semantic contract.
    # --------------------------------------------------------

    derived_requirements: dict[
        str,
        set[str],
    ] = {
        "seconds_per_meter": {
            "decision_time_s",
            "progress_m",
        },

        "energy_j_per_meter": {
            "energy_abs_j",
            "progress_m",
        },

        "attitude_rms_norm": {
            "pitch_rms_rad",
            "roll_rms_rad",
        },

        "attitude_peak_max": {
            "max_abs_pitch_rad",
            "max_abs_roll_rad",
        },
    }

    skipped_metrics: dict[
        str,
        list[dict[str, Any]],
    ] = {}

    for objective, spec in metric_specs.items():
        available_physical: list[
            dict[str, Any]
        ] = []

        skipped: list[
            dict[str, Any]
        ] = []

        for physical in spec["physical"]:
            name = str(
                physical["name"]
            )

            source_key = physical[
                "source_key"
            ]

            if name in derived_requirements:
                required = set(
                    derived_requirements[
                        name
                    ]
                )

            elif source_key is not None:
                required = {
                    str(source_key)
                }

            else:
                required = set()

            missing = sorted(
                key
                for key in required
                if key not in fieldnames
            )

            if missing:
                skipped.append(
                    {
                        "metric":
                            name,

                        "missing_columns":
                            missing,
                    }
                )

                continue

            available_physical.append(
                physical
            )

        spec["physical"] = (
            available_physical
        )

        skipped_metrics[
            objective
        ] = skipped

    print()
    print(
        "metric availability:"
    )

    for objective_name, spec in metric_specs.items():
        active_names = [
            str(item["name"])
            for item in spec["physical"]
        ]

        skipped_names = [
            str(item["metric"])
            for item in skipped_metrics[
                objective_name
            ]
        ]

        print(
            f"  {objective_name:<10} "
            f"active={active_names}"
        )

        print(
            f"  {'':<10} "
            f"skipped={skipped_names}"
        )

    paired_rows: list[
        dict[str, Any]
    ] = []

    summaries: dict[
        str,
        Any,
    ] = {}

    print(
        "=" * 96
    )
    print(
        "ICRA27 OS-T5.2a "
        "TRAIN-ONLY METRIC ALIGNMENT AUDIT"
    )
    print(
        "=" * 96
    )

    print()
    print(
        f"source       : {SOURCE_CSV}"
    )
    print(
        "checkpoint   : u30 "
        "(checkpoint_update_0030)"
    )
    print(
        "scope        : TRAIN-only 9 contexts"
    )
    print(
        "comparison   : objective beta "
        "vs balanced beta"
    )
    print(
        "interpretation: negative cost delta "
        "= improvement"
    )

    print()
    print(
        "stability component fields:"
    )
    print(
        "  posture :",
        posture_key
        if posture_key is not None
        else "NOT FOUND",
    )
    print(
        "  traction:",
        traction_key
        if traction_key is not None
        else "NOT FOUND",
    )

    for objective, spec in metric_specs.items():
        candidate_name = str(
            spec["candidate"]
        )

        target_key = str(
            spec["target_key"]
        )

        objective_rows: list[
            dict[str, Any]
        ] = []

        print()
        print(
            "-" * 96
        )
        print(
            objective.upper()
        )
        print(
            "-" * 96
        )

        for (
            group,
            seed,
        ), beta_rows in sorted(
            indexed.items()
        ):
            reference = beta_rows[
                "balanced"
            ]

            candidate = beta_rows[
                candidate_name
            ]

            target_reference = as_float(
                reference,
                target_key,
            )

            target_candidate = as_float(
                candidate,
                target_key,
            )

            delta_target = (
                target_candidate
                - target_reference
            )

            out_row: dict[str, Any] = {
                "objective":
                    objective,

                "group":
                    group,

                "seed":
                    seed,

                "candidate_beta":
                    candidate_name,

                "reference_beta":
                    "balanced",

                "target_metric":
                    target_key,

                "target_reference":
                    target_reference,

                "target_candidate":
                    target_candidate,

                "delta_target_cost":
                    delta_target,

                "target_improved":
                    int(
                        delta_target
                        < -TOL
                    ),
            }

            for physical in spec[
                "physical"
            ]:
                name = str(
                    physical["name"]
                )

                source_key = (
                    physical[
                        "source_key"
                    ]
                )

                lower_is_better = bool(
                    physical[
                        "lower_is_better"
                    ]
                )

                reference_value = (
                    metric_value(
                        reference,
                        name,
                        source_key,
                    )
                )

                candidate_value = (
                    metric_value(
                        candidate,
                        name,
                        source_key,
                    )
                )

                delta_raw = (
                    candidate_value
                    - reference_value
                )

                # Convert every physical metric to
                # a common "cost-like" orientation:
                #
                #   negative = physically better
                #   positive = physically worse
                #
                delta_physical_cost = (
                    delta_raw
                    if lower_is_better
                    else -delta_raw
                )

                out_row[
                    f"{name}__reference"
                ] = reference_value

                out_row[
                    f"{name}__candidate"
                ] = candidate_value

                out_row[
                    f"{name}__delta_raw"
                ] = delta_raw

                out_row[
                    f"{name}__delta_cost_oriented"
                ] = delta_physical_cost

                out_row[
                    f"{name}__improved"
                ] = int(
                    delta_physical_cost
                    < -TOL
                )

            objective_rows.append(
                out_row
            )

            paired_rows.append(
                out_row
            )

        target_deltas = [
            float(
                row[
                    "delta_target_cost"
                ]
            )
            for row in objective_rows
        ]

        objective_summary: dict[
            str,
            Any,
        ] = {
            "contexts":
                len(objective_rows),

            "target_metric":
                target_key,

            "target_improved":
                sum(
                    int(
                        d < -TOL
                    )
                    for d in target_deltas
                ),

            "target_worsened":
                sum(
                    int(
                        d > TOL
                    )
                    for d in target_deltas
                ),

            "target_tied":
                sum(
                    int(
                        abs(d) <= TOL
                    )
                    for d in target_deltas
                ),

            "mean_delta_target_cost":
                (
                    sum(target_deltas)
                    / len(
                        target_deltas
                    )
                ),

            "physical_metrics":
                {},
        }

        print(
            "target cost: "
            f"{objective_summary['target_improved']}"
            "/9 improved, "
            f"{objective_summary['target_worsened']}"
            "/9 worsened"
        )

        for physical in spec[
            "physical"
        ]:
            name = str(
                physical["name"]
            )

            physical_deltas = [
                float(
                    row[
                        f"{name}"
                        "__delta_cost_oriented"
                    ]
                )
                for row in objective_rows
            ]

            physical_wins = sum(
                int(
                    d < -TOL
                )
                for d in physical_deltas
            )

            physical_losses = sum(
                int(
                    d > TOL
                )
                for d in physical_deltas
            )

            physical_ties = (
                len(physical_deltas)
                - physical_wins
                - physical_losses
            )

            both_improve = sum(
                int(
                    td < -TOL
                    and pd < -TOL
                )
                for td, pd in zip(
                    target_deltas,
                    physical_deltas,
                )
            )

            semantic_conflicts = sum(
                int(
                    td < -TOL
                    and pd > TOL
                )
                for td, pd in zip(
                    target_deltas,
                    physical_deltas,
                )
            )

            sign_pairs = [
                (
                    sign_class(td),
                    sign_class(pd),
                )
                for td, pd in zip(
                    target_deltas,
                    physical_deltas,
                )
                if (
                    sign_class(td) != 0
                    and sign_class(pd) != 0
                )
            ]

            sign_agree = sum(
                int(a == b)
                for a, b in sign_pairs
            )

            corr = pearson(
                target_deltas,
                physical_deltas,
            )

            metric_summary = {
                "physical_improved":
                    physical_wins,

                "physical_worsened":
                    physical_losses,

                "physical_tied":
                    physical_ties,

                "both_target_and_physical_improved":
                    both_improve,

                "target_improved_but_physical_worsened":
                    semantic_conflicts,

                "sign_agreement":
                    sign_agree,

                "sign_comparable_contexts":
                    len(
                        sign_pairs
                    ),

                "mean_delta_physical_cost_oriented":
                    (
                        sum(
                            physical_deltas
                        )
                        / len(
                            physical_deltas
                        )
                    ),

                "pearson_delta_alignment":
                    corr,

                "note":
                    (
                        "All physical deltas are "
                        "cost-oriented: negative "
                        "means physically better."
                    ),
            }

            objective_summary[
                "physical_metrics"
            ][
                name
            ] = metric_summary

            corr_text = (
                "n/a"
                if corr is None
                else f"{corr:+.3f}"
            )

            print(
                f"  {name:<30} "
                f"physical="
                f"{physical_wins}/"
                f"{physical_losses}/"
                f"{physical_ties} "
                f"both_improve="
                f"{both_improve}/9 "
                f"conflict="
                f"{semantic_conflicts}/9 "
                f"corr="
                f"{corr_text}"
            )

        summaries[
            objective
        ] = objective_summary

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    all_fieldnames: list[str] = []

    for row in paired_rows:
        for key in row:
            if key not in all_fieldnames:
                all_fieldnames.append(
                    key
                )

    with OUT_CSV.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=all_fieldnames,
        )

        writer.writeheader()

        for row in paired_rows:
            writer.writerow(
                row
            )

    source_bytes = (
        SOURCE_CSV.read_bytes()
    )

    manifest = {
        "schema":
            "icra27_os_t5p2a_train_metric_alignment_v0",

        "purpose":
            (
                "Audit alignment between "
                "reward-side objective costs "
                "and physical trajectory metrics "
                "before defining external "
                "preference vector w."
            ),

        "development_scope":
            "TRAIN-only",

        "heldout_used_for_metric_design":
            False,

        "checkpoint":
            "update_0030",

        "comparison":
            (
                "same frozen u30 policy; "
                "objective-biased beta "
                "vs balanced beta"
            ),

        "source_csv":
            str(
                SOURCE_CSV.relative_to(
                    ROOT
                )
            ),

        "source_sha256":
            hashlib.sha256(
                source_bytes
            ).hexdigest(),

        "u30_rows":
            len(
                u30_rows
            ),

        "contexts":
            len(
                indexed
            ),

        "beta_names":
            sorted(
                beta_names
            ),

        "stability_component_columns":
            {
                "posture":
                    posture_key,

                "traction":
                    traction_key,
            },

        "source_fieldnames":
            fieldnames,

        "skipped_metrics_missing_from_compact_source":
            skipped_metrics,

        "metric_scope_note":
            (
                "The compact TRAIN checkpoint-sweep CSV "
                "does not contain every richer physical "
                "trajectory diagnostic. Missing metrics "
                "are skipped rather than inferred."
            ),

        "reward_component_metrics_are_not_independent_physical_evidence":
            True,

        "derived_physical_metrics":
            {
                "seconds_per_meter":
                    "decision_time_s / progress_m",

                "energy_j_per_meter":
                    "energy_abs_j / progress_m",

                "attitude_rms_norm":
                    (
                        "sqrt("
                        "pitch_rms_rad^2 + "
                        "roll_rms_rad^2)"
                    ),

                "attitude_peak_max":
                    (
                        "max("
                        "max_abs_pitch_rad, "
                        "max_abs_roll_rad)"
                    ),
            },

        "interpretation_contract":
            {
                "target_delta":
                    (
                        "candidate objective-beta "
                        "cost minus balanced cost; "
                        "negative is improvement"
                    ),

                "physical_delta":
                    (
                        "converted to cost-like "
                        "orientation; negative "
                        "always means physically "
                        "better"
                    ),

                "pearson":
                    (
                        "descriptive only; "
                        "n=9 and not used as a "
                        "statistical pass/fail gate"
                    ),
            },

        "summary":
            summaries,
    }

    with OUT_JSON.open(
        "w",
    ) as f:
        json.dump(
            manifest,
            f,
            indent=2,
            sort_keys=True,
        )

        f.write("\n")

    print()
    print(
        "=" * 96
    )
    print(
        "outputs:"
    )
    print(
        " ",
        OUT_CSV,
    )
    print(
        " ",
        OUT_JSON,
    )

    print()
    print(
        "[ICRA27] OS-T5.2a "
        "TRAIN-only metric alignment audit: "
        "COMPUTE PASS"
    )


if __name__ == "__main__":
    main()
