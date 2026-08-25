from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]

SOURCE = (
    ROOT
    / "results/icra27"
    / "os_t5p2a2_train_raw_physical_alignment_v0"
    / "episode_physical_metrics.csv"
)

OUT_DIR = (
    ROOT
    / "results/icra27"
    / "os_t5p2b_physical_preference_contract_v0"
)

OUT_CSV = (
    OUT_DIR
    / "train_contract_metrics.csv"
)

OUT_JSON = (
    OUT_DIR
    / "physical_preference_contract.json"
)

ROLL_UNSAFE_RAD = math.radians(15.0)
PITCH_UNSAFE_RAD = math.radians(15.0)

SLIP_DEADBAND_MPS = 0.002

EPS = 1.0e-12


def as_float(
    row: dict[str, str],
    key: str,
) -> float:
    if key not in row:
        raise KeyError(
            f"Missing required field: {key}"
        )

    value = float(row[key])

    if not math.isfinite(value):
        raise ValueError(
            f"Non-finite {key}: {value}"
        )

    return value


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    keys: list[str] = []

    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)

    with path.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=keys,
        )

        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if not SOURCE.exists():
        raise SystemExit(
            f"Missing source: {SOURCE}"
        )

    if OUT_DIR.exists():
        raise SystemExit(
            f"REFUSING TO OVERWRITE: {OUT_DIR}"
        )

    with SOURCE.open(
        newline="",
    ) as f:
        rows = list(
            csv.DictReader(f)
        )

    if len(rows) != 36:
        raise SystemExit(
            "Expected 36 TRAIN u30 trajectories, "
            f"found {len(rows)}."
        )

    required = {
        "beta_name",
        "terrain_label",
        "terrain",
        "seed",
        "decision_time_s",
        "progress_m",
        "seconds_per_meter",
        "energy_abs_j",
        "energy_j_per_meter",
        "roll_rms_rad",
        "pitch_rms_rad",
        "max_abs_roll_rad",
        "max_abs_pitch_rad",
        "established_slip_fraction_ge_0p002",
        "established_slip_fraction_ge_0p05",
        "established_slip_fraction_ge_0p1",
        "mean_abs_mechanical_power_w",
    }

    fieldnames = set(
        rows[0].keys()
    )

    missing = sorted(
        required - fieldnames
    )

    if missing:
        raise SystemExit(
            "Missing required source fields: "
            f"{missing}"
        )

    output_rows: list[
        dict[str, Any]
    ] = []

    max_motion_reconstruction_error = 0.0
    max_energy_reconstruction_error = 0.0

    for row in rows:
        decision_time = as_float(
            row,
            "decision_time_s",
        )

        progress_m = as_float(
            row,
            "progress_m",
        )

        if progress_m <= EPS:
            raise RuntimeError(
                "Physical preference contract requires "
                "positive forward progress; "
                f"got {progress_m}."
            )

        energy_abs_j = as_float(
            row,
            "energy_abs_j",
        )

        roll_rms = as_float(
            row,
            "roll_rms_rad",
        )

        pitch_rms = as_float(
            row,
            "pitch_rms_rad",
        )

        max_roll = as_float(
            row,
            "max_abs_roll_rad",
        )

        max_pitch = as_float(
            row,
            "max_abs_pitch_rad",
        )

        slip_fraction = as_float(
            row,
            "established_slip_fraction_ge_0p002",
        )

        slip_tail_005 = as_float(
            row,
            "established_slip_fraction_ge_0p05",
        )

        slip_tail_010 = as_float(
            row,
            "established_slip_fraction_ge_0p1",
        )

        if not (
            -EPS <= slip_fraction <= 1.0 + EPS
        ):
            raise RuntimeError(
                "Slip fraction outside [0,1]: "
                f"{slip_fraction}"
            )

        # ----------------------------------------------------
        # Physical preference contract.
        # All J quantities are costs:
        # lower is better.
        # ----------------------------------------------------

        j_motion = (
            decision_time
            / progress_m
        )

        j_stability_attitude = max(
            roll_rms
            / ROLL_UNSAFE_RAD,
            pitch_rms
            / PITCH_UNSAFE_RAD,
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

        # ----------------------------------------------------
        # Reconstruction checks against T5.2a-2 outputs.
        # ----------------------------------------------------

        source_seconds_per_meter = (
            as_float(
                row,
                "seconds_per_meter",
            )
        )

        source_energy_j_per_meter = (
            as_float(
                row,
                "energy_j_per_meter",
            )
        )

        motion_error = abs(
            j_motion
            - source_seconds_per_meter
        )

        energy_error = abs(
            j_energy
            - source_energy_j_per_meter
        )

        max_motion_reconstruction_error = max(
            max_motion_reconstruction_error,
            motion_error,
        )

        max_energy_reconstruction_error = max(
            max_energy_reconstruction_error,
            energy_error,
        )

        out = {
            "beta_name":
                row["beta_name"],

            "terrain_label":
                row["terrain_label"],

            "terrain":
                row["terrain"],

            "seed":
                int(
                    float(
                        row["seed"]
                    )
                ),

            # Primary physical preference metrics.
            "J_motion_s_per_m":
                j_motion,

            "J_stability":
                j_stability,

            "J_stability_attitude":
                j_stability_attitude,

            "J_stability_slip":
                j_stability_slip,

            "J_energy_j_per_m":
                j_energy,

            # Raw physical quantities retained for audit.
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
                max_roll,

            "max_abs_pitch_rad":
                max_pitch,

            "mean_abs_mechanical_power_w":
                as_float(
                    row,
                    "mean_abs_mechanical_power_w",
                ),

            # Tail-risk diagnostics.
            "tail_slip_fraction_ge_0p05":
                slip_tail_005,

            "tail_slip_fraction_ge_0p10":
                slip_tail_010,

            "tail_attitude_peak_max_rad":
                max(
                    max_roll,
                    max_pitch,
                ),
        }

        output_rows.append(
            out
        )

    # --------------------------------------------------------
    # Contract itself.
    #
    # IMPORTANT:
    # No normalization statistics are frozen here.
    # Those are computed later from the complete TRAIN-only
    # beta-response atlas.
    # --------------------------------------------------------

    contract = {
        "schema":
            "icra27_os_t5p2b_physical_preference_contract_v0",

        "status":
            "FROZEN_BEFORE_BETA_RESPONSE_ATLAS",

        "scope":
            (
                "External physical/task preference "
                "semantics for Objective Selector "
                "development."
            ),

        "principle":
            (
                "External mission preference w is "
                "defined over physical trajectory "
                "outcomes and is intentionally "
                "distinct from internal RL reward "
                "conditioning beta."
            ),

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

        "development_data_policy":
            {
                "metric_design":
                    "TRAIN-only",

                "heldout_used_for_metric_design":
                    False,

                "heldout_policy":
                    (
                        "Held-out contexts remain "
                        "evaluation-only and must not "
                        "be used to tune metric "
                        "definitions or normalization."
                    ),
            },

        "feasibility_contract":
            {
                "role":
                    (
                        "Feasibility is evaluated "
                        "before preference ranking."
                    ),

                "requirements":
                    [
                        "successful task completion",
                        "no unsafe/M4 terminal event",
                        "positive task progress",
                    ],

                "semantic_note":
                    (
                        "Infeasible trajectories are "
                        "not rescued by favorable "
                        "motion, stability, or energy "
                        "trade-offs."
                    ),
            },

        "primary_objectives":
            {
                "motion": {
                    "symbol":
                        "J_M",

                    "formula":
                        "decision_time_s / progress_m",

                    "unit":
                        "s/m",

                    "direction":
                        "lower_is_better",

                    "semantic":
                        (
                            "time required per meter "
                            "of task progress"
                        ),
                },

                "stability": {
                    "symbol":
                        "J_S",

                    "formula":
                        (
                            "max("
                            "J_S_attitude,"
                            "J_S_slip"
                            ")"
                        ),

                    "unit":
                        "dimensionless",

                    "direction":
                        "lower_is_better",

                    "attitude_component": {
                        "symbol":
                            "J_S_attitude",

                        "formula":
                            (
                                "max("
                                "roll_rms_rad/"
                                "roll_unsafe_rad,"
                                "pitch_rms_rad/"
                                "pitch_unsafe_rad"
                                ")"
                            ),

                        "roll_unsafe_rad":
                            ROLL_UNSAFE_RAD,

                        "pitch_unsafe_rad":
                            PITCH_UNSAFE_RAD,

                        "semantic":
                            (
                                "RMS body-attitude "
                                "burden relative to "
                                "the frozen unsafe "
                                "attitude boundary"
                            ),
                    },

                    "slip_component": {
                        "symbol":
                            "J_S_slip",

                        "formula":
                            (
                                "established-contact "
                                "time fraction with "
                                "|foot slip speed| "
                                ">= 0.002 m/s"
                            ),

                        "threshold_mps":
                            SLIP_DEADBAND_MPS,

                        "semantic":
                            (
                                "meaningful established-"
                                "stance slip burden"
                            ),
                    },

                    "composition_reason":
                        (
                            "max composition avoids "
                            "introducing an arbitrary "
                            "attitude-vs-slip weight "
                            "before preference learning "
                            "and preserves the dominant "
                            "physical stability burden."
                        ),
                },

                "energy": {
                    "symbol":
                        "J_E",

                    "formula":
                        "energy_abs_j / progress_m",

                    "unit":
                        "J/m",

                    "direction":
                        "lower_is_better",

                    "semantic":
                        (
                            "absolute mechanical work "
                            "required per meter of task "
                            "progress"
                        ),

                    "note":
                        (
                            "Mean mechanical power is "
                            "retained as a diagnostic "
                            "but is not the external "
                            "mission-level energy "
                            "objective."
                        ),
                },
            },

        "tail_risk_diagnostics":
            {
                "role":
                    (
                        "Safety/risk diagnostics are "
                        "reported separately from "
                        "nominal J_S and are not "
                        "silently averaged into the "
                        "primary preference scalar."
                    ),

                "metrics":
                    [
                        {
                            "name":
                                "max_abs_roll_rad",

                            "direction":
                                "lower_is_better",
                        },
                        {
                            "name":
                                "max_abs_pitch_rad",

                            "direction":
                                "lower_is_better",
                        },
                        {
                            "name":
                                "established_slip_fraction_ge_0p05",

                            "threshold_mps":
                                0.05,

                            "direction":
                                "lower_is_better",
                        },
                        {
                            "name":
                                "established_slip_fraction_ge_0p10",

                            "threshold_mps":
                                0.10,

                            "direction":
                                "lower_is_better",
                        },
                    ],
            },

        "preference_vector": {
            "symbol":
                "w",

            "components":
                [
                    "w_M",
                    "w_S",
                    "w_E",
                ],

            "semantic":
                (
                    "External mission preference over "
                    "normalized physical motion, "
                    "stability, and energy outcomes."
                ),

            "simplex_constraint":
                (
                    "w_k >= 0 and "
                    "w_M + w_S + w_E = 1"
                ),

            "important_distinction":
                (
                    "w is not beta. "
                    "w specifies what physical "
                    "outcome is desired; beta is "
                    "the internal planner/reward "
                    "conditioning variable used to "
                    "realize that preference."
                ),
        },

        "normalization_contract": {
            "status":
                "DEFINITION_FROZEN_VALUES_PENDING",

            "method":
                (
                    "TRAIN-only robust centering/scaling "
                    "using the complete feasible "
                    "beta-response atlas."
                ),

            "candidate_formula":
                (
                    "(J_k - median_train(J_k)) / "
                    "(IQR_train(J_k) + epsilon)"
                ),

            "statistics_freeze_time":
                "after OS-T5.3 TRAIN beta-response atlas",

            "heldout_leakage_prohibited":
                True,

            "note":
                (
                    "Metric definitions are frozen "
                    "before the beta sweep; only "
                    "normalization constants remain "
                    "to be estimated."
                ),
        },

        "next_stage":
            {
                "id":
                    "OS-T5.3",

                "name":
                    "TRAIN beta-response atlas",

                "purpose":
                    (
                        "Map context and beta to "
                        "physical trajectory outcomes "
                        "(J_M, J_S, J_E) before "
                        "learning the Objective Selector."
                    ),
            },

        "validation":
            {
                "trajectories_checked":
                    len(
                        output_rows
                    ),

                "max_J_motion_reconstruction_error":
                    max_motion_reconstruction_error,

                "max_J_energy_reconstruction_error":
                    max_energy_reconstruction_error,
            },
    }

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    write_csv(
        OUT_CSV,
        output_rows,
    )

    with OUT_JSON.open(
        "w",
    ) as f:
        json.dump(
            contract,
            f,
            indent=2,
            sort_keys=True,
        )
        f.write("\n")

    print("=" * 100)
    print(
        "ICRA27 OS-T5.2b "
        "PHYSICAL PREFERENCE CONTRACT FREEZE"
    )
    print("=" * 100)

    print()
    print("trajectories:", len(output_rows))
    print()
    print("PRIMARY CONTRACT")
    print("  J_M = T / progress                         [s/m]")
    print(
        "  J_S = max(attitude RMS burden, "
        "established slip burden)"
    )
    print("  J_E = mechanical work / progress           [J/m]")
    print()
    print(
        "normalization: definition frozen; "
        "TRAIN atlas statistics pending"
    )
    print(
        "held-out used for metric design: NO"
    )
    print()
    print(
        "max J_M reconstruction error:",
        max_motion_reconstruction_error,
    )
    print(
        "max J_E reconstruction error:",
        max_energy_reconstruction_error,
    )

    print()
    print("outputs:")
    print(" ", OUT_CSV)
    print(" ", OUT_JSON)
    print()
    print(
        "[ICRA27] OS-T5.2b "
        "physical preference contract: FREEZE PASS"
    )


if __name__ == "__main__":
    main()
