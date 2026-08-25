#!/usr/bin/env python3

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import socket
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

CHECKPOINT = (
    ROOT
    / "results"
    / "icra27"
    / "phase1a_beta_conditioned_v2_seed27027"
    / "phase1_beta_identifiable_selected_u20.pt"
)

OUT_DIR = (
    ROOT
    / "results"
    / "icra27"
    / "os_t4p3c_train9_cost_basis_v0"
    / "u0020_train9_rerun1"
)


TERRAIN_CASES = (
    (
        "flat",
        "flat",
        (27100, 27101, 27102),
    ),
    (
        "low_friction",
        "low_friction",
        (27200, 27201, 27202),
    ),
    (
        "rough",
        "rough_perlin",
        (13, 7, 15),
    ),
)


EXPECTED_REWARD_SCHEMA = (
    "icra27_simplified_tracer_cost_v3"
)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Could not load {path}"
        )

    mod = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        mod
    )

    return mod


def udp_bindable(port):
    sock = socket.socket(
        socket.AF_INET,
        socket.SOCK_DGRAM,
    )

    try:
        sock.bind(
            ("127.0.0.1", int(port))
        )
        return True

    except OSError:
        return False

    finally:
        sock.close()


def choose_base_port():
    offsets = (
        0, 1, 2,
        10, 11, 12,
        20, 21, 22,
    )

    for base in range(
        63000,
        64501,
        100,
    ):
        if all(
            udp_bindable(
                base + offset
            )
            for offset in offsets
        ):
            return base

    raise RuntimeError(
        "No free TRAIN9 UDP layout."
    )


def mean(values):
    values = [
        float(x)
        for x in values
    ]

    if not values:
        return None

    return float(
        sum(values)
        / len(values)
    )


def median(values):
    values = [
        float(x)
        for x in values
    ]

    if not values:
        return None

    return float(
        statistics.median(values)
    )


def make_env(
    phase2b,
    *,
    terrain,
    beta,
    port,
    log_dir,
):
    base_env = phase2b.PyMPCM7Env(
        terrain=terrain,

        reward_mode="tracer_cost_v3",

        tracer_beta=beta,

        terminate_on_m4_unsafe=True,

        goal_distance_m=(
            phase2b.base.GOAL_DISTANCE_M
        ),

        success_radius_m=(
            phase2b.base.SUCCESS_RADIUS_M
        ),

        decision_dt_s=(
            phase2b.base.DECISION_DT_S
        ),

        max_episode_steps=(
            phase2b.base.POLICY_HORIZON
            + phase2b.base.SETTLING_STEPS
        ),

        command_port=port,
        telemetry_port=port + 1,
        state_port=port + 2,

        command_repeat_hz=20.0,
        telemetry_hz=100.0,
        state_hz=100.0,

        log_dir=log_dir,
    )

    env = (
        phase2b.M7BetaConditioningWrapper(
            phase2b.M7FixedClearanceActionWrapper(
                base_env
            ),
            beta=beta,
        )
    )

    return (
        env,
        base_env,
    )


def read_reward_steps(
    path,
):
    if not path.exists():
        raise FileNotFoundError(
            path
        )

    result = []

    for line in path.read_text().splitlines():
        if not line.strip():
            continue

        row = json.loads(line)

        if row.get("event") != "step":
            continue

        rc = row.get(
            "reward_components"
        )

        if not isinstance(
            rc,
            dict,
        ):
            continue

        if (
            rc.get(
                "reward_schema"
            )
            != EXPECTED_REWARD_SCHEMA
        ):
            raise RuntimeError(
                "Unexpected reward schema in "
                f"{path}: "
                f"{rc.get('reward_schema')!r}"
            )

        required = (
            "cost_motion",
            "cost_posture",
            "cost_traction",
            "cost_stability",
            "cost_energy",
            "objective_cost",
        )

        missing = [
            key
            for key in required
            if key not in rc
        ]

        if missing:
            raise RuntimeError(
                f"Missing v3 reward keys "
                f"in {path}: {missing}"
            )

        result.append(
            {
                "cost_motion":
                    float(
                        rc["cost_motion"]
                    ),

                "cost_posture":
                    float(
                        rc["cost_posture"]
                    ),

                "cost_traction":
                    float(
                        rc["cost_traction"]
                    ),

                "cost_stability":
                    float(
                        rc["cost_stability"]
                    ),

                "cost_energy":
                    float(
                        rc["cost_energy"]
                    ),

                "objective_cost":
                    float(
                        rc["objective_cost"]
                    ),

                "traction_contact_dt_s":
                    float(
                        rc.get(
                            "traction_interval_contact_dt_s",
                            0.0,
                        )
                    ),
            }
        )

    if not result:
        raise RuntimeError(
            "No tracer_cost_v3 step rows in "
            f"{path}"
        )

    return result


def select_policy_steps(
    steps,
    base_row,
):
    """
    Prefer the evaluator's reported policy-step count so that
    fixed settling transitions do not dominate the diagnostic.

    If that metadata is unavailable, retain all v3 reward steps.
    """

    policy_steps = (
        base_row.get(
            "policy_steps"
        )
    )

    if policy_steps is None:
        return (
            steps,
            "all_reward_steps",
        )

    count = int(
        policy_steps
    )

    if (
        count <= 0
        or count > len(steps)
    ):
        return (
            steps,
            "all_reward_steps_fallback",
        )

    return (
        steps[-count:],
        "last_policy_steps",
    )


def summarize_steps(
    steps,
):
    for step in steps:
        expected = max(
            step[
                "cost_posture"
            ],
            step[
                "cost_traction"
            ],
        )

        if abs(
            step[
                "cost_stability"
            ]
            - expected
        ) > 1e-10:
            raise RuntimeError(
                "C_stability != "
                "max(C_posture, C_traction)"
            )

        for key in (
            "cost_motion",
            "cost_posture",
            "cost_traction",
            "cost_stability",
            "cost_energy",
        ):
            if step[key] < 0.0:
                raise RuntimeError(
                    f"Negative objective cost: "
                    f"{key}={step[key]}"
                )

    n = len(
        steps
    )

    traction_dominant = sum(
        int(
            step[
                "cost_traction"
            ]
            >
            step[
                "cost_posture"
            ]
        )
        for step in steps
    )

    ties = sum(
        int(
            abs(
                step[
                    "cost_traction"
                ]
                - step[
                    "cost_posture"
                ]
            )
            <= 1e-12
        )
        for step in steps
    )

    return {
        "reward_steps":
            n,

        "mean_cost_motion":
            mean(
                x["cost_motion"]
                for x in steps
            ),

        "mean_cost_posture":
            mean(
                x["cost_posture"]
                for x in steps
            ),

        "mean_cost_traction":
            mean(
                x["cost_traction"]
                for x in steps
            ),

        "mean_cost_stability":
            mean(
                x["cost_stability"]
                for x in steps
            ),

        "mean_cost_energy":
            mean(
                x["cost_energy"]
                for x in steps
            ),

        "mean_objective_cost":
            mean(
                x["objective_cost"]
                for x in steps
            ),

        "median_cost_posture":
            median(
                x["cost_posture"]
                for x in steps
            ),

        "median_cost_traction":
            median(
                x["cost_traction"]
                for x in steps
            ),

        "traction_dominant_steps":
            traction_dominant,

        "traction_dominant_fraction":
            float(
                traction_dominant
                / n
            ),

        "posture_dominant_fraction":
            float(
                (
                    n
                    - traction_dominant
                    - ties
                )
                / n
            ),

        "tie_fraction":
            float(
                ties / n
            ),

        "mean_established_contact_dt_s":
            mean(
                x[
                    "traction_contact_dt_s"
                ]
                for x in steps
            ),
    }


def write_csv(
    path,
    rows,
):
    if not rows:
        return

    with path.open(
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


def main():
    if OUT_DIR.exists():
        raise RuntimeError(
            "Output directory already exists; "
            "use a fresh version/rerun path: "
            f"{OUT_DIR}"
        )

    OUT_DIR.mkdir(
        parents=True
    )

    phase2b = load_module(
        PHASE2B_PATH,
        "os_t4p3c_phase2b",
    )

    # --------------------------------------------------------
    # Diagnostic-only evaluator compatibility.
    #
    # phase2b.base is the historical common evaluator module.
    # Do NOT edit that source file merely to teach it a new
    # reward-mode name. For this process only, point its
    # contract guard at tracer_cost_v3.
    #
    # v3 intentionally preserves all legacy evaluator fields:
    #   cost_motion
    #   cost_stability
    #   cost_energy
    #   progress_rate_mps
    #   progress_normalized
    #   energy_power_ratio
    #   roll/pitch fractions
    #   energy diagnostics
    #
    # and only adds posture/traction decomposition.
    # --------------------------------------------------------

    historical_eval_mode = (
        phase2b.base.EVAL_REWARD_MODE
    )

    if (
        historical_eval_mode
        != "tracer_cost_v2"
    ):
        raise RuntimeError(
            "Unexpected historical evaluator mode: "
            f"{historical_eval_mode!r}; "
            "refusing compatibility override."
        )

    phase2b.base.EVAL_REWARD_MODE = (
        "tracer_cost_v3"
    )

    print(
        "[OS-T4.3c] common evaluator contract: "
        f"{historical_eval_mode} -> tracer_cost_v3 "
        "(process-local only)"
    )

    beta_bank = {
        str(name): tuple(
            float(x)
            for x in beta
        )
        for name, beta
        in phase2b.BETA_BANK
    }

    beta = beta_bank[
        "balanced"
    ]

    policy = (
        phase2b.base.load_policy(
            name="beta_conditioned",
            path=CHECKPOINT,
            expected_reward_mode=(
                "tracer_cost_v2"
            ),
        )
    )

    phase2b.validate_checkpoint_beta_bank(
        policy["payload"]
    )

    base_port = (
        choose_base_port()
    )

    print("=" * 112)
    print(
        "ICRA27 OS-T4.3c TRAIN9 "
        "TRACER-COST-V3 TERRAIN BASIS SANITY"
    )
    print("=" * 112)

    print(
        "checkpoint :",
        CHECKPOINT,
    )

    print(
        "policy trained reward : tracer_cost_v2"
    )

    print(
        "evaluation reward     : tracer_cost_v3"
    )

    print(
        "beta                  :",
        beta,
    )

    print(
        "scope                 : TRAIN only"
    )

    print(
        "base port             :",
        base_port,
    )

    print(
        "output                :",
        OUT_DIR,
    )

    print()

    episode_rows = []

    for terrain_index, (
        label,
        terrain,
        seeds,
    ) in enumerate(
        TERRAIN_CASES
    ):
        log_dir = (
            OUT_DIR
            / "env_logs"
            / label
        )

        (
            env,
            base_env,
        ) = make_env(
            phase2b,
            terrain=terrain,
            beta=beta,
            port=(
                base_port
                + 10 * terrain_index
            ),
            log_dir=log_dir,
        )

        print(
            f"[{label}] "
            f"terrain={terrain} "
            f"seeds={list(seeds)}"
        )

        try:
            for seed in seeds:
                base_row = (
                    phase2b.base.run_episode(
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

                        model=(
                            policy["model"]
                        ),

                        group=(
                            "os_t4p3c_"
                            f"{label}"
                        ),

                        terrain=terrain,

                        seed=seed,
                    )
                )

                episode_index = int(
                    base_env.episode_index
                )

                log_path = (
                    log_dir
                    / (
                        f"episode_"
                        f"{episode_index:04d}"
                        f".jsonl"
                    )
                )

                all_steps = (
                    read_reward_steps(
                        log_path
                    )
                )

                (
                    selected_steps,
                    aggregation_rule,
                ) = select_policy_steps(
                    all_steps,
                    base_row,
                )

                summary = (
                    summarize_steps(
                        selected_steps
                    )
                )

                row = {
                    "terrain_label":
                        label,

                    "terrain":
                        terrain,

                    "seed":
                        int(seed),

                    "success":
                        bool(
                            base_row.get(
                                "success",
                                False,
                            )
                        ),

                    "status":
                        base_row.get(
                            "status"
                        ),

                    "policy_steps_reported":
                        base_row.get(
                            "policy_steps"
                        ),

                    "logged_reward_steps":
                        len(all_steps),

                    "aggregation_rule":
                        aggregation_rule,

                    **summary,
                }

                episode_rows.append(
                    row
                )

                print(
                    f"  seed={seed:<6} "
                    f"success={row['success']} "
                    f"Cpost="
                    f"{row['mean_cost_posture']:.6f} "
                    f"Ctr="
                    f"{row['mean_cost_traction']:.6f} "
                    f"Cs="
                    f"{row['mean_cost_stability']:.6f} "
                    f"T-dom="
                    f"{row['traction_dominant_fraction']:.3f}"
                )

        finally:
            env.close()

        print()

    if len(episode_rows) != 9:
        raise RuntimeError(
            "Expected exactly 9 TRAIN episodes, "
            f"got {len(episode_rows)}"
        )

    if not all(
        bool(row["success"])
        for row in episode_rows
    ):
        print(
            "WARNING: not all TRAIN9 episodes "
            "were successful."
        )

    terrain_summary = {}

    for label, _, _ in TERRAIN_CASES:
        subset = [
            row
            for row in episode_rows
            if row[
                "terrain_label"
            ] == label
        ]

        terrain_summary[
            label
        ] = {
            "episodes":
                len(subset),

            "success_count":
                sum(
                    int(
                        row["success"]
                    )
                    for row in subset
                ),

            "median_mean_cost_posture":
                median(
                    row[
                        "mean_cost_posture"
                    ]
                    for row in subset
                ),

            "median_mean_cost_traction":
                median(
                    row[
                        "mean_cost_traction"
                    ]
                    for row in subset
                ),

            "median_mean_cost_stability":
                median(
                    row[
                        "mean_cost_stability"
                    ]
                    for row in subset
                ),

            "median_mean_cost_motion":
                median(
                    row[
                        "mean_cost_motion"
                    ]
                    for row in subset
                ),

            "median_mean_cost_energy":
                median(
                    row[
                        "mean_cost_energy"
                    ]
                    for row in subset
                ),

            "median_traction_dominant_fraction":
                median(
                    row[
                        "traction_dominant_fraction"
                    ]
                    for row in subset
                ),

            "max_traction_dominant_fraction":
                max(
                    float(
                        row[
                            "traction_dominant_fraction"
                        ]
                    )
                    for row in subset
                ),
        }

    summary = {
        "schema":
            "icra27_os_t4p3c_train9_cost_basis_v0",

        "scope":
            "TRAIN-only pre-retraining reward-basis sanity",

        "checkpoint":
            str(CHECKPOINT),

        "policy_training_reward":
            "tracer_cost_v2",

        "evaluation_reward":
            "tracer_cost_v3",

        "beta":
            list(beta),

        "stability_definition":
            "max(endpoint_posture, interval_mean_traction)",

        "terrain_summary":
            terrain_summary,

        "episodes":
            episode_rows,
    }

    summary_path = (
        OUT_DIR
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

    csv_path = (
        OUT_DIR
        / "episodes.csv"
    )

    write_csv(
        csv_path,
        episode_rows,
    )

    print("=" * 112)
    print(
        "OS-T4.3c TERRAIN BASIS SUMMARY"
    )
    print("=" * 112)

    print(
        f"{'terrain':<14} "
        f"{'succ':>6} "
        f"{'Cpost':>12} "
        f"{'Ctr':>12} "
        f"{'Cs':>12} "
        f"{'Cm':>12} "
        f"{'Ce':>12} "
        f"{'T-dom':>10}"
    )

    print("-" * 104)

    for label, _, _ in TERRAIN_CASES:
        item = terrain_summary[
            label
        ]

        print(
            f"{label:<14} "
            f"{item['success_count']:>3}/3 "
            f"{item['median_mean_cost_posture']:12.6f} "
            f"{item['median_mean_cost_traction']:12.6f} "
            f"{item['median_mean_cost_stability']:12.6f} "
            f"{item['median_mean_cost_motion']:12.6f} "
            f"{item['median_mean_cost_energy']:12.6f} "
            f"{item['median_traction_dominant_fraction']:10.3f}"
        )

    flat = terrain_summary[
        "flat"
    ]

    low = terrain_summary[
        "low_friction"
    ]

    rough = terrain_summary[
        "rough"
    ]

    flat_ctr = float(
        flat[
            "median_mean_cost_traction"
        ]
    )

    print()
    print("TRACTION TERRAIN RATIOS")
    print("-" * 48)

    if flat_ctr > 0.0:
        print(
            "low / flat   : "
            f"{float(low['median_mean_cost_traction']) / flat_ctr:.3f}"
        )

        print(
            "rough / flat : "
            f"{float(rough['median_mean_cost_traction']) / flat_ctr:.3f}"
        )

    else:
        print(
            "flat traction median is zero; "
            "ratio undefined."
        )

    print()
    print(
        "episodes:",
        csv_path,
    )

    print(
        "summary :",
        summary_path,
    )


if __name__ == "__main__":
    main()
