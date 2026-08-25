#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
import statistics
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


PHASE2B_PATH = (
    ROOT
    / "scripts"
    / "icra27"
    / "evaluate_phase2b_beta_train_archive_v0.py"
)

DEFAULT_CHECKPOINT = (
    ROOT
    / "results"
    / "icra27"
    / "phase1a_beta_conditioned_v2_seed27027"
    / "phase1_beta_identifiable_selected_u20.pt"
)

DEFAULT_OUT_DIR = (
    ROOT
    / "results"
    / "icra27"
    / "os_t3p4_balanced_slip_v0"
    / "u0020_train9"
)


# TRAIN-only diagnostic seeds.
#
# Do NOT replace these with VAL/TEST seeds.
GROUPS = (
    {
        "name": "flat",
        "terrain": "flat",
        "seeds": (
            27100,
            27101,
            27102,
        ),
    },
    {
        "name": "low_friction",
        "terrain": "low_friction",
        "seeds": (
            27200,
            27201,
            27202,
        ),
    },
    {
        "name": "rough",
        "terrain": "rough_perlin",
        "seeds": (
            13,
            7,
            15,
        ),
    },
)


def load_phase2b():
    spec = importlib.util.spec_from_file_location(
        "os_t3p4_phase2b",
        PHASE2B_PATH,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Could not load {PHASE2B_PATH}"
        )

    mod = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        mod
    )

    return mod


def balanced_beta(mod):
    bank = {
        name: tuple(
            float(x)
            for x in beta
        )
        for name, beta
        in mod.BETA_BANK
    }

    if "balanced" not in bank:
        raise RuntimeError(
            "Phase-2B beta bank has no "
            "'balanced' entry."
        )

    return bank["balanced"]


def finite_float(
    value,
    *,
    name,
):
    value = float(value)

    if not (
        value == value
        and abs(value) != float("inf")
    ):
        raise RuntimeError(
            f"Non-finite {name}: {value!r}"
        )

    return value


def extract_slip(
    env,
):
    base_env = env.unwrapped

    info = getattr(
        base_env,
        "last_info",
        None,
    )

    if info is None:
        raise RuntimeError(
            "env.unwrapped.last_info is missing"
        )

    slip = info.get(
        "eval_stance_slip"
    )

    if slip is None:
        raise RuntimeError(
            "Final env info did not expose "
            "eval_stance_slip"
        )

    role = slip.get(
        "measurement_role"
    )

    if (
        role
        != "evaluation_only_contact_slip_v0"
    ):
        raise RuntimeError(
            "Unexpected slip measurement role: "
            f"{role!r}"
        )

    contact_samples = int(
        slip[
            "contact_samples"
        ]
    )

    contact_time_s = finite_float(
        slip[
            "contact_time_s"
        ],
        name="contact_time_s",
    )

    if (
        contact_samples <= 0
        or contact_time_s <= 0.0
    ):
        raise RuntimeError(
            "Episode produced no valid stance "
            "contact samples"
        )

    rms = slip.get(
        "speed_rms_mps"
    )

    mean = slip.get(
        "speed_mean_mps"
    )

    if rms is None or mean is None:
        raise RuntimeError(
            "Slip mean/RMS missing despite "
            "positive contact duration"
        )

    return {
        "slip_speed_mean_mps":
            finite_float(
                mean,
                name="slip_speed_mean_mps",
            ),

        "slip_speed_rms_mps":
            finite_float(
                rms,
                name="slip_speed_rms_mps",
            ),

        "slip_speed_max_mps":
            finite_float(
                slip[
                    "speed_max_mps"
                ],
                name="slip_speed_max_mps",
            ),

        "slip_contact_time_s":
            contact_time_s,

        "slip_contact_samples":
            contact_samples,

        "slip_physics_steps_with_contact":
            int(
                slip[
                    "physics_steps_with_contact"
                ]
            ),

        "slip_per_foot_contact_samples":
            json.dumps(
                slip[
                    "per_foot_contact_samples"
                ],
                sort_keys=True,
            ),

        "slip_foot_geom_names":
            json.dumps(
                slip.get(
                    "foot_geom_names"
                ),
                sort_keys=True,
            ),
    }


def median(values):
    values = [
        float(x)
        for x in values
    ]

    if not values:
        raise RuntimeError(
            "Cannot compute median of empty set"
        )

    return float(
        statistics.median(
            values
        )
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )

    ap.add_argument(
        "--base-port",
        type=int,
        default=59110,
    )

    ap.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
    )

    args = ap.parse_args()

    checkpoint = (
        args.checkpoint.resolve()
    )

    if not checkpoint.exists():
        raise FileNotFoundError(
            checkpoint
        )

    mod = load_phase2b()

    beta = balanced_beta(
        mod
    )

    policy = mod.base.load_policy(
        name="beta_conditioned",
        path=checkpoint,
        expected_reward_mode=(
            "tracer_cost_v2"
        ),
    )

    mod.validate_checkpoint_beta_bank(
        policy[
            "payload"
        ]
    )

    out_dir = (
        args.out_dir.resolve()
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 96)
    print(
        "ICRA27 OS-T3.4 BALANCED TERRAIN "
        "STANCE-SLIP DIAGNOSTIC"
    )
    print("=" * 96)

    print(
        "checkpoint :",
        checkpoint,
    )

    print(
        "beta       :",
        beta,
    )

    print(
        "scope      : TRAIN-only, 9 rollouts"
    )

    print(
        "primary    : episode cumulative "
        "stance-slip RMS"
    )

    print(
        "output     :",
        out_dir,
    )

    print()

    rows = []

    for group_index, group in enumerate(
        GROUPS
    ):
        name = group[
            "name"
        ]

        terrain = group[
            "terrain"
        ]

        seeds = group[
            "seeds"
        ]

        command_port = (
            int(args.base_port)
            + 10 * group_index
        )

        log_dir = (
            out_dir
            / "env_logs"
            / name
        )

        env = mod.make_env(
            terrain=terrain,
            beta=beta,
            command_port=command_port,
            log_dir=log_dir,
        )

        print(
            f"[{name}] "
            f"terrain={terrain} "
            f"seeds={list(seeds)}"
        )

        try:
            for seed in seeds:
                row = mod.base.run_episode(
                    env,
                    policy_name=(
                        "beta_conditioned"
                    ),
                    checkpoint=(
                        policy["path"]
                    ),
                    trained_reward_mode=(
                        policy[
                            "trained_reward_mode"
                        ]
                    ),
                    model=policy["model"],
                    group=(
                        f"os_t3p4_{name}"
                    ),
                    terrain=terrain,
                    seed=seed,
                )

                slip = extract_slip(
                    env
                )

                diagnostic_row = {
                    "group":
                        name,

                    "terrain":
                        terrain,

                    "seed":
                        int(seed),

                    "beta_name":
                        "balanced",

                    "beta_motion":
                        float(beta[0]),

                    "beta_stability":
                        float(beta[1]),

                    "beta_energy":
                        float(beta[2]),

                    "status":
                        row.get(
                            "status"
                        ),

                    "success":
                        bool(
                            row.get(
                                "success",
                                False,
                            )
                        ),

                    "m4_terminal":
                        bool(
                            row.get(
                                "m4_terminal",
                                False,
                            )
                        ),

                    "m4_interventions":
                        row.get(
                            "m4_interventions"
                        ),

                    "policy_steps":
                        row.get(
                            "policy_steps"
                        ),

                    "decision_time_s":
                        row.get(
                            "decision_time_s"
                        ),

                    "energy_abs_j":
                        row.get(
                            "energy_abs_j"
                        ),

                    "mean_cost_motion":
                        row.get(
                            "mean_cost_motion"
                        ),

                    "mean_cost_stability":
                        row.get(
                            "mean_cost_stability"
                        ),

                    "mean_cost_energy":
                        row.get(
                            "mean_cost_energy"
                        ),

                    "roll_rms_rad":
                        row.get(
                            "roll_rms_rad"
                        ),

                    "pitch_rms_rad":
                        row.get(
                            "pitch_rms_rad"
                        ),

                    "mean_applied_vx_mps":
                        row.get(
                            "mean_applied_vx_mps"
                        ),

                    **slip,
                }

                rows.append(
                    diagnostic_row
                )

                print(
                    f"  seed={seed:<6} "
                    f"status="
                    f"{diagnostic_row['status']!s:<18} "
                    f"success="
                    f"{diagnostic_row['success']} "
                    f"slip_rms="
                    f"{diagnostic_row['slip_speed_rms_mps']:.6f} "
                    f"slip_mean="
                    f"{diagnostic_row['slip_speed_mean_mps']:.6f} "
                    f"slip_max="
                    f"{diagnostic_row['slip_speed_max_mps']:.6f}"
                )

        finally:
            env.close()

        print()

    if len(rows) != 9:
        raise RuntimeError(
            f"Expected 9 rows, got {len(rows)}"
        )

    csv_path = (
        out_dir
        / "episodes.csv"
    )

    with csv_path.open(
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

    summary = {
        "schema":
            "icra27_os_t3p4_balanced_slip_v0",

        "checkpoint":
            str(checkpoint),

        "beta_name":
            "balanced",

        "beta": [
            float(x)
            for x in beta
        ],

        "scope":
            "TRAIN-only diagnostic",

        "primary_metric":
            "slip_speed_rms_mps",

        "secondary_metric":
            "slip_speed_mean_mps",

        "max_metric_role":
            "diagnostic_only_touchdown_sensitive",

        "groups": {},
    }

    for group in GROUPS:
        name = group[
            "name"
        ]

        subset = [
            row
            for row in rows
            if row["group"] == name
        ]

        rms_values = [
            row[
                "slip_speed_rms_mps"
            ]
            for row in subset
        ]

        mean_values = [
            row[
                "slip_speed_mean_mps"
            ]
            for row in subset
        ]

        max_values = [
            row[
                "slip_speed_max_mps"
            ]
            for row in subset
        ]

        success_count = sum(
            int(
                bool(
                    row["success"]
                )
            )
            for row in subset
        )

        summary[
            "groups"
        ][name] = {
            "terrain":
                group["terrain"],

            "seeds":
                list(
                    group["seeds"]
                ),

            "episodes":
                len(subset),

            "success_count":
                int(success_count),

            "all_success":
                bool(
                    success_count
                    == len(subset)
                ),

            "slip_rms_mps":
                {
                    "values":
                        rms_values,

                    "median":
                        median(
                            rms_values
                        ),
                },

            "slip_mean_mps":
                {
                    "values":
                        mean_values,

                    "median":
                        median(
                            mean_values
                        ),
                },

            "slip_max_mps":
                {
                    "values":
                        max_values,

                    "median":
                        median(
                            max_values
                        ),
                },
        }

    flat = summary[
        "groups"
    ][
        "flat"
    ][
        "slip_rms_mps"
    ][
        "median"
    ]

    low = summary[
        "groups"
    ][
        "low_friction"
    ][
        "slip_rms_mps"
    ][
        "median"
    ]

    rough = summary[
        "groups"
    ][
        "rough"
    ][
        "slip_rms_mps"
    ][
        "median"
    ]

    all_success = all(
        group_summary[
            "all_success"
        ]
        for group_summary
        in summary[
            "groups"
        ].values()
    )

    if not all_success:
        verdict = (
            "INCONCLUSIVE_EPISODE_FAILURE"
        )

    elif low > flat:
        verdict = (
            "LOW_FRICTION_GT_FLAT"
        )

    else:
        verdict = (
            "LOW_FRICTION_NOT_GT_FLAT"
        )

    summary[
        "diagnostic_gate"
    ] = {
        "question":
            "Is median low-friction "
            "stance-slip RMS greater "
            "than flat?",

        "flat_median_rms_mps":
            float(flat),

        "low_friction_median_rms_mps":
            float(low),

        "rough_median_rms_mps":
            float(rough),

        "low_minus_flat_mps":
            float(
                low - flat
            ),

        "low_over_flat":
            (
                None
                if flat <= 0.0
                else float(
                    low / flat
                )
            ),

        "rough_over_flat":
            (
                None
                if flat <= 0.0
                else float(
                    rough / flat
                )
            ),

        "all_episodes_successful":
            bool(
                all_success
            ),

        "verdict":
            verdict,

        "note":
            (
                "Signal-existence diagnostic only. "
                "No significance threshold, "
                "reward modification, or "
                "stability-cost redefinition."
            ),
    }

    summary_path = (
        out_dir
        / "summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print("=" * 96)
    print("OS-T3.4 SUMMARY")
    print("=" * 96)

    print(
        f"flat median RMS : "
        f"{flat:.6f} m/s"
    )

    print(
        f"low  median RMS : "
        f"{low:.6f} m/s"
    )

    print(
        f"rough median RMS: "
        f"{rough:.6f} m/s"
    )

    print(
        f"low - flat      : "
        f"{low - flat:+.6f} m/s"
    )

    if flat > 0.0:
        print(
            f"low / flat      : "
            f"{low / flat:.3f}x"
        )

        print(
            f"rough / flat    : "
            f"{rough / flat:.3f}x"
        )

    print(
        "all success     :",
        all_success,
    )

    print(
        "diagnostic gate :",
        verdict,
    )

    print()
    print(
        "episodes.csv :",
        csv_path,
    )

    print(
        "summary.json :",
        summary_path,
    )


if __name__ == "__main__":
    main()
