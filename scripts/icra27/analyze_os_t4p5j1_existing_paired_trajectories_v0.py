#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
from pathlib import Path
import statistics

import numpy as np


ROOT = Path(__file__).resolve().parents[2]

SWEEP_ROOT = (
    ROOT
    / "results"
    / "icra27"
    / "os_t4p5_checkpoint_sweep_v0"
    / "train9"
)

EPISODES_CSV = (
    SWEEP_ROOT
    / "episodes.csv"
)

OUT_DIR = (
    ROOT
    / "results"
    / "icra27"
    / "os_t4p5j1_existing_paired_trajectories_v0"
)

UPDATES = (
    30,
    40,
)

BETA_NAMES = (
    "balanced",
    "motion",
    "stability",
    "energy",
)

OBJECTIVES = (
    "motion",
    "stability",
    "energy",
)

TERRAINS = (
    "flat",
    "low_friction",
    "rough",
)

COST_KEY = {
    "motion": "cost_motion",
    "stability": "cost_stability",
    "energy": "cost_energy",
}

CSV_COST_KEY = {
    "motion": "mean_cost_motion",
    "stability": "mean_cost_stability",
    "energy": "mean_cost_energy",
}

REQUIRED_RC = (
    "cost_motion",
    "cost_posture",
    "cost_traction",
    "cost_stability",
    "cost_energy",
)


def finite_float(x, name):
    value = float(x)

    if not np.isfinite(value):
        raise RuntimeError(
            f"Non-finite {name}: {value}"
        )

    return value


def mean(values):
    values = list(values)

    if not values:
        raise RuntimeError(
            "mean() received no values"
        )

    return float(
        statistics.fmean(values)
    )


def read_episode_csv():
    if not EPISODES_CSV.exists():
        raise FileNotFoundError(
            EPISODES_CSV
        )

    rows = []

    with EPISODES_CSV.open(
        newline=""
    ) as f:
        reader = csv.DictReader(f)

        for index, row in enumerate(
            reader
        ):
            update = int(
                row["update"]
            )

            if update not in UPDATES:
                continue

            copied = dict(row)
            copied["_csv_index"] = index

            rows.append(copied)

    if len(rows) != 72:
        raise RuntimeError(
            "Expected exactly 72 u30/u40 "
            f"episode rows; got {len(rows)}"
        )

    return rows


def attach_log_paths(rows):
    grouped = {}

    for row in rows:
        key = (
            int(row["update"]),
            row["beta_name"],
            row["terrain_label"],
        )

        grouped.setdefault(
            key,
            [],
        ).append(row)

    result = {}

    for key, group_rows in grouped.items():
        update, beta, terrain = key

        group_rows = sorted(
            group_rows,
            key=lambda x: x["_csv_index"],
        )

        log_dir = (
            SWEEP_ROOT
            / f"u{update:02d}"
            / "env_logs"
            / beta
            / terrain
        )

        paths = sorted(
            log_dir.glob(
                "episode_*.jsonl"
            )
        )

        if len(group_rows) != 3:
            raise RuntimeError(
                f"{key}: expected 3 CSV rows, "
                f"got {len(group_rows)}"
            )

        if len(paths) != 3:
            raise RuntimeError(
                f"{key}: expected 3 JSONLs, "
                f"got {len(paths)}"
            )

        for row, path in zip(
            group_rows,
            paths,
            strict=True,
        ):
            episode_key = (
                update,
                beta,
                terrain,
                int(row["seed"]),
            )

            if episode_key in result:
                raise RuntimeError(
                    "Duplicate episode key: "
                    f"{episode_key}"
                )

            row = dict(row)
            row["log_path"] = str(path)

            result[episode_key] = row

    if len(result) != 72:
        raise RuntimeError(
            "Expected 72 attached episodes; "
            f"got {len(result)}"
        )

    return result


def read_trace(row):
    path = Path(
        row["log_path"]
    )

    raw_steps = []

    for line in path.read_text().splitlines():
        if not line.strip():
            continue

        item = json.loads(line)

        if item.get("event") != "step":
            continue

        rc = item.get(
            "reward_components"
        )

        if not isinstance(
            rc,
            dict,
        ):
            continue

        missing = [
            key
            for key in REQUIRED_RC
            if key not in rc
        ]

        if missing:
            raise RuntimeError(
                f"{path}: missing reward keys "
                f"{missing}"
            )

        raw_steps.append(
            item
        )

    policy_steps = int(
        row["policy_steps"]
    )

    if (
        policy_steps <= 0
        or policy_steps > len(raw_steps)
    ):
        raise RuntimeError(
            f"{path}: invalid policy_steps="
            f"{policy_steps}, "
            f"reward_steps={len(raw_steps)}"
        )

    selected = raw_steps[
        -policy_steps:
    ]

    # Observation immediately before the first policy
    # decision is the final settling transition output.
    pre_index = (
        len(raw_steps)
        - policy_steps
        - 1
    )

    pre_policy_observation = None

    if pre_index >= 0:
        candidate = raw_steps[
            pre_index
        ].get(
            "observation"
        )

        if candidate is not None:
            pre_policy_observation = (
                np.asarray(
                    candidate,
                    dtype=np.float64,
                )
            )

    trace = []

    for k, item in enumerate(
        selected
    ):
        rc = item[
            "reward_components"
        ]

        observation = np.asarray(
            item["observation"],
            dtype=np.float64,
        )

        requested = np.asarray(
            item["requested_physical"],
            dtype=np.float64,
        )

        applied = np.asarray(
            item["applied_physical"],
            dtype=np.float64,
        )

        if observation.ndim != 1:
            raise RuntimeError(
                f"{path}: observation must be 1D"
            )

        # Frozen M7 physical transport contract:
        #
        # [vx, yaw_rate, body_height, swing_clearance,
        #  gait_period, duty_factor]
        #
        # The learned policy is still action3 [vx,yaw,h].
        # Clearance is supplied by the fixed-clearance wrapper,
        # and period/duty are appended by the downstream mapper.
        if requested.shape != (6,):
            raise RuntimeError(
                f"{path}: unexpected requested "
                f"shape {requested.shape}; expected (6,)"
            )

        if applied.shape != (6,):
            raise RuntimeError(
                f"{path}: unexpected applied "
                f"shape {applied.shape}; expected (6,)"
            )

        c_posture = finite_float(
            rc["cost_posture"],
            "cost_posture",
        )

        c_traction = finite_float(
            rc["cost_traction"],
            "cost_traction",
        )

        c_stability = finite_float(
            rc["cost_stability"],
            "cost_stability",
        )

        expected_stability = max(
            c_posture,
            c_traction,
        )

        if abs(
            c_stability
            - expected_stability
        ) > 1e-10:
            raise RuntimeError(
                f"{path}: "
                "C_stability != "
                "max(C_posture,C_traction)"
            )

        trace.append(
            {
                "k":
                    int(k),

                # This is the observation produced by
                # env.step(action_k), i.e. o_{k+1}.
                "next_observation":
                    observation,

                "requested_physical":
                    requested,

                "applied_physical":
                    applied,

                "cost_motion":
                    finite_float(
                        rc["cost_motion"],
                        "cost_motion",
                    ),

                "cost_posture":
                    c_posture,

                "cost_traction":
                    c_traction,

                "cost_stability":
                    c_stability,

                "cost_energy":
                    finite_float(
                        rc["cost_energy"],
                        "cost_energy",
                    ),

                "traction_dominant":
                    bool(
                        c_traction
                        > c_posture
                    ),

                "decision_dt_s":
                    finite_float(
                        item["decision_dt_s"],
                        "decision_dt_s",
                    ),

                "state_sim_time_s":
                    finite_float(
                        item["state_sim_time_s"],
                        "state_sim_time_s",
                    ),

                "m4_intervention":
                    bool(
                        item.get(
                            "m4_intervention",
                            False,
                        )
                    ),

                "override_active":
                    bool(
                        item.get(
                            "override_active",
                            False,
                        )
                    ),

                "safety_state":
                    item.get(
                        "safety_state"
                    ),
            }
        )

    return {
        "trace":
            trace,

        "pre_policy_observation":
            pre_policy_observation,
    }


def validate_episode_means(
    row,
    trace,
):
    for objective in OBJECTIVES:
        cost_key = COST_KEY[
            objective
        ]

        csv_key = CSV_COST_KEY[
            objective
        ]

        reconstructed = mean(
            step[cost_key]
            for step in trace
        )

        expected = finite_float(
            row[csv_key],
            csv_key,
        )

        if abs(
            reconstructed
            - expected
        ) > 1e-9:
            raise RuntimeError(
                f"{row['log_path']}: "
                f"{csv_key} mismatch: "
                f"jsonl={reconstructed:.12g}, "
                f"csv={expected:.12g}"
            )

    for key in (
        "mean_cost_posture",
        "mean_cost_traction",
    ):
        trace_key = key.replace(
            "mean_",
            "",
        )

        reconstructed = mean(
            step[trace_key]
            for step in trace
        )

        expected = finite_float(
            row[key],
            key,
        )

        if abs(
            reconstructed
            - expected
        ) > 1e-9:
            raise RuntimeError(
                f"{row['log_path']}: "
                f"{key} mismatch: "
                f"jsonl={reconstructed:.12g}, "
                f"csv={expected:.12g}"
            )


def episode_mean(
    episode,
    key,
):
    return mean(
        x[key]
        for x in episode["trace"]
    )


def episode_advantage(
    balanced,
    specialized,
    key,
):
    # Positive means specialized beta has LOWER cost
    # than balanced and therefore has the intended
    # semantic direction.
    return (
        episode_mean(
            balanced,
            key,
        )
        - episode_mean(
            specialized,
            key,
        )
    )


def vector_norm(a, b):
    return float(
        np.linalg.norm(
            np.asarray(a)
            - np.asarray(b)
        )
    )


def json_ready_step(step):
    out = dict(step)

    out["next_observation"] = (
        step[
            "next_observation"
        ].tolist()
    )

    out["requested_physical"] = (
        step[
            "requested_physical"
        ].tolist()
    )

    out["applied_physical"] = (
        step[
            "applied_physical"
        ].tolist()
    )

    return out


def main():
    if OUT_DIR.exists():
        raise RuntimeError(
            "Output directory already exists; "
            f"refusing overwrite: {OUT_DIR}"
        )

    OUT_DIR.mkdir(
        parents=True
    )

    rows = read_episode_csv()

    indexed_rows = attach_log_paths(
        rows
    )

    episodes = {}

    obs_dims = set()

    for key, row in indexed_rows.items():
        parsed = read_trace(
            row
        )

        validate_episode_means(
            row,
            parsed["trace"],
        )

        for step in parsed["trace"]:
            obs_dims.add(
                int(
                    len(
                        step[
                            "next_observation"
                        ]
                    )
                )
            )

        episodes[key] = {
            **parsed,
            "row":
                row,
        }

    print("=" * 122)
    print(
        "ICRA27 OS-T4.5j1 EXISTING "
        "PAIRED TRAJECTORY EXTRACTION"
    )
    print("=" * 122)
    print(
        "source      : existing OS-T4.5 "
        "u30/u40 JSONL only"
    )
    print(
        "simulation  : NONE"
    )
    print(
        "episodes    :",
        len(episodes),
    )
    print(
        "obs dims    :",
        sorted(obs_dims),
    )
    print()

    # ----------------------------------------------------------
    # Reproduce the original full-episode semantic signs.
    # ----------------------------------------------------------
    gate_counts = {
        update: {
            objective: 0
            for objective
            in OBJECTIVES
        }
        for update
        in UPDATES
    }

    summary_rows = []
    paired_trace_path = (
        OUT_DIR
        / "paired_step_trace.jsonl"
    )

    with paired_trace_path.open(
        "w"
    ) as trace_file:

        for terrain in TERRAINS:
            for seed in sorted(
                {
                    int(row["seed"])
                    for row in rows
                    if (
                        row["terrain_label"]
                        == terrain
                    )
                }
            ):
                for objective in OBJECTIVES:
                    specialized_beta = (
                        objective
                    )

                    quartet = {}

                    for update in UPDATES:
                        for beta in (
                            "balanced",
                            specialized_beta,
                        ):
                            key = (
                                update,
                                beta,
                                terrain,
                                seed,
                            )

                            if key not in episodes:
                                raise RuntimeError(
                                    "Missing paired episode: "
                                    f"{key}"
                                )

                            quartet[
                                (update, beta)
                            ] = episodes[
                                key
                            ]

                    # Full-episode semantic advantage.
                    advantage = {}

                    for update in UPDATES:
                        bal = quartet[
                            (
                                update,
                                "balanced",
                            )
                        ]

                        spec = quartet[
                            (
                                update,
                                specialized_beta,
                            )
                        ]

                        cost_key = COST_KEY[
                            objective
                        ]

                        advantage[
                            update
                        ] = episode_advantage(
                            bal,
                            spec,
                            cost_key,
                        )

                        if advantage[
                            update
                        ] > 0.0:
                            gate_counts[
                                update
                            ][
                                objective
                            ] += 1

                    bal30 = quartet[
                        (30, "balanced")
                    ]
                    spec30 = quartet[
                        (
                            30,
                            specialized_beta,
                        )
                    ]
                    bal40 = quartet[
                        (40, "balanced")
                    ]
                    spec40 = quartet[
                        (
                            40,
                            specialized_beta,
                        )
                    ]

                    shared_steps = min(
                        len(
                            bal30["trace"]
                        ),
                        len(
                            spec30["trace"]
                        ),
                        len(
                            bal40["trace"]
                        ),
                        len(
                            spec40["trace"]
                        ),
                    )

                    # Stability decomposition is still useful
                    # for all objectives as a diagnostic.
                    posture_adv30 = (
                        episode_advantage(
                            bal30,
                            spec30,
                            "cost_posture",
                        )
                    )

                    posture_adv40 = (
                        episode_advantage(
                            bal40,
                            spec40,
                            "cost_posture",
                        )
                    )

                    traction_adv30 = (
                        episode_advantage(
                            bal30,
                            spec30,
                            "cost_traction",
                        )
                    )

                    traction_adv40 = (
                        episode_advantage(
                            bal40,
                            spec40,
                            "cost_traction",
                        )
                    )

                    summary_rows.append(
                        {
                            "terrain":
                                terrain,

                            "seed":
                                seed,

                            "objective":
                                objective,

                            "shared_steps":
                                shared_steps,

                            "advantage_u30":
                                advantage[30],

                            "advantage_u40":
                                advantage[40],

                            "advantage_change_u40_minus_u30":
                                (
                                    advantage[40]
                                    - advantage[30]
                                ),

                            "posture_advantage_u30":
                                posture_adv30,

                            "posture_advantage_u40":
                                posture_adv40,

                            "traction_advantage_u30":
                                traction_adv30,

                            "traction_advantage_u40":
                                traction_adv40,

                            "u30_balanced_steps":
                                len(
                                    bal30["trace"]
                                ),

                            "u30_specialized_steps":
                                len(
                                    spec30["trace"]
                                ),

                            "u40_balanced_steps":
                                len(
                                    bal40["trace"]
                                ),

                            "u40_specialized_steps":
                                len(
                                    spec40["trace"]
                                ),
                        }
                    )

                    for k in range(
                        shared_steps
                    ):
                        b30 = bal30[
                            "trace"
                        ][k]

                        s30 = spec30[
                            "trace"
                        ][k]

                        b40 = bal40[
                            "trace"
                        ][k]

                        s40 = spec40[
                            "trace"
                        ][k]

                        key_cost = COST_KEY[
                            objective
                        ]

                        step_adv30 = (
                            b30[key_cost]
                            - s30[key_cost]
                        )

                        step_adv40 = (
                            b40[key_cost]
                            - s40[key_cost]
                        )

                        payload = {
                            "terrain":
                                terrain,

                            "seed":
                                seed,

                            "objective":
                                objective,

                            "k":
                                k,

                            "u30_balanced":
                                json_ready_step(
                                    b30
                                ),

                            "u30_specialized":
                                json_ready_step(
                                    s30
                                ),

                            "u40_balanced":
                                json_ready_step(
                                    b40
                                ),

                            "u40_specialized":
                                json_ready_step(
                                    s40
                                ),

                            "semantic_advantage_u30":
                                step_adv30,

                            "semantic_advantage_u40":
                                step_adv40,

                            "semantic_advantage_change":
                                (
                                    step_adv40
                                    - step_adv30
                                ),

                            "posture_advantage_u30":
                                (
                                    b30[
                                        "cost_posture"
                                    ]
                                    - s30[
                                        "cost_posture"
                                    ]
                                ),

                            "posture_advantage_u40":
                                (
                                    b40[
                                        "cost_posture"
                                    ]
                                    - s40[
                                        "cost_posture"
                                    ]
                                ),

                            "traction_advantage_u30":
                                (
                                    b30[
                                        "cost_traction"
                                    ]
                                    - s30[
                                        "cost_traction"
                                    ]
                                ),

                            "traction_advantage_u40":
                                (
                                    b40[
                                        "cost_traction"
                                    ]
                                    - s40[
                                        "cost_traction"
                                    ]
                                ),

                            # Realized beta-induced closed-loop
                            # command divergence from balanced.
                            "requested_divergence_u30":
                                vector_norm(
                                    b30[
                                        "requested_physical"
                                    ],
                                    s30[
                                        "requested_physical"
                                    ],
                                ),

                            "requested_divergence_u40":
                                vector_norm(
                                    b40[
                                        "requested_physical"
                                    ],
                                    s40[
                                        "requested_physical"
                                    ],
                                ),

                            "applied_divergence_u30":
                                vector_norm(
                                    b30[
                                        "applied_physical"
                                    ],
                                    s30[
                                        "applied_physical"
                                    ],
                                ),

                            "applied_divergence_u40":
                                vector_norm(
                                    b40[
                                        "applied_physical"
                                    ],
                                    s40[
                                        "applied_physical"
                                    ],
                                ),

                            # Raw obs norm is retained only as
                            # descriptive data. j2 will define
                            # the normalized state metric.
                            "next_obs_raw_divergence_u30":
                                vector_norm(
                                    b30[
                                        "next_observation"
                                    ],
                                    s30[
                                        "next_observation"
                                    ],
                                ),

                            "next_obs_raw_divergence_u40":
                                vector_norm(
                                    b40[
                                        "next_observation"
                                    ],
                                    s40[
                                        "next_observation"
                                    ],
                                ),
                        }

                        trace_file.write(
                            json.dumps(
                                payload,
                                sort_keys=True,
                            )
                            + "\n"
                        )

    # ----------------------------------------------------------
    # Write episode-pair summary.
    # ----------------------------------------------------------
    summary_path = (
        OUT_DIR
        / "paired_episode_summary.csv"
    )

    with summary_path.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                summary_rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            summary_rows
        )

    print(
        "FULL-EPISODE SEMANTIC SIGN REPRODUCTION"
    )

    for objective in OBJECTIVES:
        print(
            f"  {objective:<10} "
            f"u30="
            f"{gate_counts[30][objective]}/9 "
            f"u40="
            f"{gate_counts[40][objective]}/9"
        )

    print()
    print(
        "STABILITY DECOMPOSITION "
        "(positive = stability beta improves over balanced)"
    )

    for terrain in TERRAINS:
        selected = [
            row
            for row in summary_rows
            if (
                row["terrain"] == terrain
                and row["objective"]
                == "stability"
            )
        ]

        def med(key):
            return float(
                np.median(
                    [
                        x[key]
                        for x in selected
                    ]
                )
            )

        print(
            f"  {terrain:<13} "
            f"CsAdv30={med('advantage_u30'):+.6f} "
            f"CsAdv40={med('advantage_u40'):+.6f} "
            f"CpostAdv30="
            f"{med('posture_advantage_u30'):+.6f} "
            f"CpostAdv40="
            f"{med('posture_advantage_u40'):+.6f} "
            f"CtrAdv30="
            f"{med('traction_advantage_u30'):+.6f} "
            f"CtrAdv40="
            f"{med('traction_advantage_u40'):+.6f}"
        )

    manifest = {
        "schema":
            "icra27_os_t4p5j1_existing_paired_trajectories_v0",

        "source":
            str(SWEEP_ROOT),

        "simulation":
            False,

        "updates":
            list(UPDATES),

        "episode_count":
            len(episodes),

        "quadruple_pair_count":
            len(summary_rows),

        "observation_dims":
            sorted(obs_dims),

        "transport_physical_dim":
            6,

        "semantic_gate_counts":
            {
                str(update):
                    gate_counts[update]
                for update
                in UPDATES
            },

        "pairing":
            (
                "same terrain + same TRAIN seed; "
                "balanced vs specialized within checkpoint; "
                "u30 vs u40 difference-in-differences"
            ),

        "temporal_contract":
            (
                "JSONL observation is next observation "
                "o_{k+1} generated by action/request at k"
            ),

        "shared_horizon":
            (
                "minimum policy-step count across "
                "u30-balanced/u30-specialized/"
                "u40-balanced/u40-specialized"
            ),
    }

    manifest_path = (
        OUT_DIR
        / "manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print()
    print(
        "episode summary:",
        summary_path,
    )
    print(
        "paired trace   :",
        paired_trace_path,
    )
    print(
        "manifest       :",
        manifest_path,
    )

    print()
    print(
        "[ICRA27] OS-T4.5j1 existing paired "
        "trajectory extraction: PASS"
    )


if __name__ == "__main__":
    main()
