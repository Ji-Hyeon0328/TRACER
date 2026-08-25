#!/usr/bin/env python3

from __future__ import annotations

import argparse
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

T4P1B_PATH = (
    ROOT
    / "scripts"
    / "icra27"
    / "evaluate_os_t4p1b_contact_age_train9_v0.py"
)

T4P1D_PATH = (
    ROOT
    / "scripts"
    / "icra27"
    / "evaluate_os_t4p1d_established_slip_distribution_train9_v0.py"
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
    / "os_t4p1d3_friction_boundary_v0"
    / "u0020_train27"
)


# Previous characterization placed the viability cliff
# near mu ~= 0.152 / 0.150. Re-characterize around it with
# the new established-stance contact metric.
FRICTION_VALUES = (
    0.200,
    0.180,
    0.170,
    0.160,
    0.155,
    0.152,
    0.151,
    0.150,
    0.148,
)

TRAIN_SEEDS = (
    27200,
    27201,
    27202,
)

ESTABLISHED_GATE_S = 0.040
SLIP_DEADBAND_MPS = 0.002
Q95 = 0.95


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
    offsets = []

    for index in range(
        len(FRICTION_VALUES)
    ):
        port = 10 * index
        offsets.extend(
            (
                port,
                port + 1,
                port + 2,
            )
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
        "No free friction-sweep UDP layout found."
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


def make_env_with_friction(
    phase2b,
    *,
    beta,
    friction,
    command_port,
    log_dir,
):
    # Keep the low-friction terrain semantic context fixed.
    #
    # Only the simulator friction coefficient changes.
    base_env = phase2b.PyMPCM7Env(
        terrain="low_friction",
        terrain_friction=float(
            friction
        ),

        reward_mode=(
            phase2b.EVAL_REWARD_MODE
        ),

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

        command_port=command_port,

        telemetry_port=(
            command_port + 1
        ),

        state_port=(
            command_port + 2
        ),

        command_repeat_hz=20.0,
        telemetry_hz=100.0,
        state_hz=100.0,

        log_dir=log_dir,
    )

    action_env = (
        phase2b.M7FixedClearanceActionWrapper(
            base_env
        )
    )

    env = (
        phase2b.M7BetaConditioningWrapper(
            action_env,
            beta=beta,
        )
    )

    if (
        base_env.observation_space.shape
        != (phase2b.BASE_OBS_DIM,)
    ):
        raise RuntimeError(
            "Unexpected base observation shape: "
            f"{base_env.observation_space.shape}"
        )

    if (
        env.observation_space.shape
        != (phase2b.OBS_DIM,)
    ):
        raise RuntimeError(
            "Unexpected conditioned observation shape: "
            f"{env.observation_space.shape}"
        )

    if (
        env.action_space.shape
        != (phase2b.POLICY_ACTION_DIM,)
    ):
        raise RuntimeError(
            "Unexpected policy action shape: "
            f"{env.action_space.shape}"
        )

    return env


def write_csv(path, rows):
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
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
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

    out_dir = (
        args.out_dir.resolve()
    )

    if out_dir.exists():
        raise RuntimeError(
            "Output directory already exists; "
            "use a fresh --out-dir: "
            f"{out_dir}"
        )

    out_dir.mkdir(
        parents=True
    )

    phase2b = load_module(
        PHASE2B_PATH,
        "os_t4p1d3_phase2b",
    )

    t4p1b = load_module(
        T4P1B_PATH,
        "os_t4p1d3_t4p1b",
    )

    t4p1d = load_module(
        T4P1D_PATH,
        "os_t4p1d3_t4p1d",
    )

    from tracer_core.highlevel_rl.terrain import (
        get_terrain_preset,
    )

    flat_preset = (
        get_terrain_preset(
            "flat"
        )
    )

    low_preset = (
        get_terrain_preset(
            "low_friction"
        )
    )

    print("=" * 104)
    print(
        "ICRA27 OS-T4.1d3 FRICTION-BOUNDARY "
        "ESTABLISHED-SLIP CHARACTERIZATION"
    )
    print("=" * 104)

    print(
        "flat scene        :",
        flat_preset.scene,
    )

    print(
        "low-friction scene:",
        low_preset.scene,
    )

    print(
        "nominal low mu    :",
        low_preset.friction_coeff,
    )

    print(
        "low context       :",
        low_preset.oracle_context,
    )

    # We want a friction-only physics sweep.
    if (
        flat_preset.scene
        != low_preset.scene
    ):
        raise RuntimeError(
            "flat and low_friction presets use "
            "different scenes. Refusing to call "
            "this a friction-only sweep."
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
            path=checkpoint,
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

    print(
        "checkpoint        :",
        checkpoint,
    )

    print(
        "beta              :",
        beta,
    )

    print(
        "friction grid     :",
        FRICTION_VALUES,
    )

    print(
        "TRAIN seeds       :",
        TRAIN_SEEDS,
    )

    print(
        "established gate  :",
        ESTABLISHED_GATE_S,
    )

    print(
        "slip deadband     :",
        SLIP_DEADBAND_MPS,
    )

    print(
        "base port         :",
        base_port,
    )

    print(
        "output            :",
        out_dir,
    )

    print()

    rows = []

    csv_path = (
        out_dir
        / "episodes.csv"
    )

    for friction_index, mu in enumerate(
        FRICTION_VALUES
    ):
        command_port = (
            base_port
            + 10 * friction_index
        )

        log_dir = (
            out_dir
            / "env_logs"
            / f"mu_{mu:.3f}"
        )

        env = make_env_with_friction(
            phase2b,
            beta=beta,
            friction=mu,
            command_port=command_port,
            log_dir=log_dir,
        )

        print(
            f"[mu={mu:.3f}] "
            f"ports={command_port}:"
            f"{command_port + 2}"
        )

        try:
            for seed in TRAIN_SEEDS:
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
                            f"os_t4p1d3_mu_"
                            f"{mu:.3f}"
                        ),

                        terrain=(
                            "low_friction"
                        ),

                        seed=seed,
                    )
                )

                established = None
                post40 = None

                try:
                    established = (
                        t4p1d
                        .extract_established_histogram(
                            env
                        )
                    )

                    profile = (
                        t4p1b.extract_profile(
                            env
                        )
                    )

                    post40 = (
                        t4p1b
                        .aggregate_after_gate(
                            profile["bins"],
                            ESTABLISHED_GATE_S,
                        )
                    )

                except RuntimeError:
                    if bool(
                        base_row.get(
                            "success",
                            False,
                        )
                    ):
                        raise

                row = {
                    "friction_mu":
                        float(mu),

                    "seed":
                        int(seed),

                    "status":
                        base_row.get(
                            "status"
                        ),

                    "success":
                        bool(
                            base_row.get(
                                "success",
                                False,
                            )
                        ),

                    "m4_terminal":
                        bool(
                            base_row.get(
                                "m4_terminal",
                                False,
                            )
                        ),

                    "m4_interventions":
                        base_row.get(
                            "m4_interventions"
                        ),

                    "policy_steps":
                        base_row.get(
                            "policy_steps"
                        ),

                    "progress_m":
                        base_row.get(
                            "progress_m"
                        ),

                    "decision_time_s":
                        base_row.get(
                            "decision_time_s"
                        ),

                    "mean_applied_vx_mps":
                        base_row.get(
                            "mean_applied_vx_mps"
                        ),

                    "attitude_cost":
                        base_row.get(
                            "mean_cost_stability"
                        ),

                    "established_profile_available":
                        bool(
                            established is not None
                        ),

                    "established_contact_time_s":
                        (
                            None
                            if established is None
                            else established[
                                "total_time_s"
                            ]
                        ),

                    "established_rms_mps":
                        (
                            None
                            if post40 is None
                            else post40[
                                "rms_mps"
                            ]
                        ),

                    "established_mean_mps":
                        (
                            None
                            if post40 is None
                            else post40[
                                "mean_mps"
                            ]
                        ),
                }

                # Keep the histogram in a side JSON field,
                # not flattened into CSV columns.
                row[
                    "_established_histogram"
                ] = (
                    None
                    if established is None
                    else established[
                        "histogram"
                    ]
                )

                rows.append(row)

                csv_rows = []

                for item in rows:
                    csv_rows.append(
                        {
                            key: value
                            for key, value
                            in item.items()
                            if not key.startswith("_")
                        }
                    )

                # Incremental preservation.
                write_csv(
                    csv_path,
                    csv_rows,
                )

                print(
                    f"  seed={seed:<6} "
                    f"status="
                    f"{str(row['status']):<18} "
                    f"success="
                    f"{row['success']} "
                    f"progress="
                    f"{row['progress_m']} "
                    f"estRMS="
                    f"{row['established_rms_mps']}"
                )

        finally:
            env.close()

        print()

    expected = (
        len(FRICTION_VALUES)
        * len(TRAIN_SEEDS)
    )

    if len(rows) != expected:
        raise RuntimeError(
            f"Expected {expected} episodes, "
            f"got {len(rows)}"
        )

    summary = {
        "schema":
            "icra27_os_t4p1d3_friction_boundary_v0",

        "scope":
            "TRAIN-only low-friction-context "
            "friction-boundary characterization",

        "checkpoint":
            str(checkpoint),

        "beta":
            list(beta),

        "terrain_preset":
            "low_friction",

        "scene":
            str(low_preset.scene),

        "nominal_low_friction_mu":
            float(
                low_preset.friction_coeff
            ),

        "friction_values":
            list(
                FRICTION_VALUES
            ),

        "train_seeds":
            list(
                TRAIN_SEEDS
            ),

        "established_stance_gate_s":
            ESTABLISHED_GATE_S,

        "slip_deadband_mps":
            SLIP_DEADBAND_MPS,

        "frictions":
            {},
    }

    mu_summaries = []

    for mu in FRICTION_VALUES:
        subset = [
            row
            for row in rows
            if abs(
                float(row["friction_mu"])
                - float(mu)
            ) < 1e-12
        ]

        success_count = sum(
            int(
                bool(
                    row["success"]
                )
            )
            for row in subset
        )

        hist_episodes = [
            {
                "established": {
                    "histogram":
                        row[
                            "_established_histogram"
                        ]
                }
            }
            for row in subset
            if row[
                "_established_histogram"
            ] is not None
        ]

        pooled = None
        q95 = None
        exceed_deadband = None

        if hist_episodes:
            pooled = (
                t4p1d.pool_histograms(
                    hist_episodes
                )
            )

            q95 = (
                t4p1d.conservative_quantile(
                    pooled,
                    Q95,
                )
            )

            exceed_deadband = (
                t4p1d.exceedance_fraction(
                    pooled,
                    SLIP_DEADBAND_MPS,
                )
            )

        established_rms_values = [
            row[
                "established_rms_mps"
            ]
            for row in subset
            if row[
                "established_rms_mps"
            ] is not None
        ]

        mu_summary = {
            "friction_mu":
                float(mu),

            "episodes":
                len(subset),

            "success_count":
                int(success_count),

            "all_success":
                bool(
                    success_count
                    == len(subset)
                ),

            "median_progress_m":
                median(
                    row["progress_m"]
                    for row in subset
                    if row[
                        "progress_m"
                    ] is not None
                ),

            "median_m4_interventions":
                median(
                    row["m4_interventions"]
                    for row in subset
                    if row[
                        "m4_interventions"
                    ] is not None
                ),

            "median_applied_vx_mps":
                median(
                    row["mean_applied_vx_mps"]
                    for row in subset
                    if row[
                        "mean_applied_vx_mps"
                    ] is not None
                ),

            "median_established_rms_mps":
                median(
                    established_rms_values
                ),

            "pooled_q95":
                q95,

            "pooled_fraction_above_deadband":
                exceed_deadband,

            "pooled_histogram":
                pooled,
        }

        summary[
            "frictions"
        ][
            f"{mu:.3f}"
        ] = mu_summary

        mu_summaries.append(
            mu_summary
        )

    # --------------------------------------------------------
    # Marginally viable boundary candidate.
    #
    # Definition:
    # smallest tested mu with 3/3 success for which at least
    # one lower tested mu is not 3/3 successful.
    # --------------------------------------------------------
    edge_candidates = []

    for item in mu_summaries:
        if not item[
            "all_success"
        ]:
            continue

        mu = float(
            item[
                "friction_mu"
            ]
        )

        harder = [
            other
            for other in mu_summaries
            if (
                float(
                    other[
                        "friction_mu"
                    ]
                ) < mu
                and not other[
                    "all_success"
                ]
            )
        ]

        if harder:
            edge_candidates.append(
                item
            )

    edge = (
        None
        if not edge_candidates
        else min(
            edge_candidates,
            key=lambda x: float(
                x["friction_mu"]
            ),
        )
    )

    boundary = {
        "definition":
            "smallest tested mu with 3/3 success "
            "and at least one lower tested mu "
            "without 3/3 success",

        "marginal_viable_mu":
            None,

        "nearest_lower_nonfull_mu":
            None,

        "candidate_s_unsafe_mps":
            None,

        "candidate_s_unsafe_rule":
            (
                "conservative pooled established-stance "
                "Q95 at marginal viable mu"
            ),

        "status":
            "NOT_BRACKETED",
    }

    if edge is not None:
        edge_mu = float(
            edge[
                "friction_mu"
            ]
        )

        lower_nonfull = [
            item
            for item in mu_summaries
            if (
                float(
                    item[
                        "friction_mu"
                    ]
                ) < edge_mu
                and not item[
                    "all_success"
                ]
            )
        ]

        nearest_lower = max(
            lower_nonfull,
            key=lambda x: float(
                x[
                    "friction_mu"
                ]
            ),
        )

        q95_item = edge.get(
            "pooled_q95"
        )

        candidate_unsafe = (
            None
            if q95_item is None
            else q95_item[
                "conservative_upper_mps"
            ]
        )

        boundary.update(
            {
                "marginal_viable_mu":
                    edge_mu,

                "nearest_lower_nonfull_mu":
                    float(
                        nearest_lower[
                            "friction_mu"
                        ]
                    ),

                "candidate_s_unsafe_mps":
                    candidate_unsafe,

                "status":
                    (
                        "BRACKETED"
                        if candidate_unsafe
                        is not None
                        else
                        "BRACKETED_NO_Q95"
                    ),
            }
        )

    summary[
        "boundary_candidate"
    ] = boundary

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

    print("=" * 104)
    print(
        "OS-T4.1d3 FRICTION BOUNDARY SUMMARY"
    )
    print("=" * 104)

    print(
        f"{'mu':>7} "
        f"{'success':>9} "
        f"{'progress':>10} "
        f"{'est RMS':>11} "
        f"{'Q95 upper':>11} "
        f"{'P(s>dead)':>11}"
    )

    print("-" * 68)

    for item in mu_summaries:
        q95 = item[
            "pooled_q95"
        ]

        q95_upper = (
            None
            if q95 is None
            else q95[
                "conservative_upper_mps"
            ]
        )

        def fmt(value):
            if value is None:
                return "None"
            return f"{float(value):.5f}"

        print(
            f"{item['friction_mu']:7.3f} "
            f"{item['success_count']:>6}/3 "
            f"{fmt(item['median_progress_m']):>10} "
            f"{fmt(item['median_established_rms_mps']):>11} "
            f"{fmt(q95_upper):>11} "
            f"{fmt(item['pooled_fraction_above_deadband']):>11}"
        )

    print()
    print(
        "boundary status :",
        boundary[
            "status"
        ],
    )

    print(
        "marginal mu     :",
        boundary[
            "marginal_viable_mu"
        ],
    )

    print(
        "lower nonfull   :",
        boundary[
            "nearest_lower_nonfull_mu"
        ],
    )

    print(
        "s_unsafe cand.  :",
        boundary[
            "candidate_s_unsafe_mps"
        ],
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
