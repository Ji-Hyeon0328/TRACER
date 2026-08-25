#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]

K5_ROOT = (
    ROOT
    / "results"
    / "icra27"
    / "os_t4p5k5_lockstep_checkpoint_sweep_v0"
    / "train9"
)

OUT_DIR = (
    ROOT
    / "results"
    / "icra27"
    / "os_t4p5k6_lowfriction_action_mechanism_v0"
)


UPDATES = (
    10,
    20,
    30,
    40,
)

BETAS = (
    "balanced",
    "stability",
    "energy",
)

TARGET_TERRAIN = "low_friction"
TARGET_SEED = 27200

SETTLING_STEPS = 5


# Frozen obs21 semantics:
#
#  7 : vx
#  8 : vy
#  9 : body yaw rate
# 10 : base z
# 11 : roll
# 12 : pitch
#
OBS_VX = 7
OBS_VY = 8
OBS_YAW_RATE = 9
OBS_Z = 10
OBS_ROLL = 11
OBS_PITCH = 12


PHYS_VX = 0
PHYS_YAW = 1
PHYS_HEIGHT = 2
PHYS_CLEARANCE = 3
PHYS_PERIOD = 4
PHYS_DUTY = 5


def finite_or_nan(
    value,
):
    if value is None:
        return float("nan")

    value = float(value)

    if not math.isfinite(value):
        return float("nan")

    return value


def read_jsonl(
    path: Path,
):
    rows = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line_no, line in enumerate(
            f,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                row = json.loads(line)

            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"{path}:{line_no}: "
                    "invalid JSON"
                ) from exc

            rows.append(row)

    return rows


def episode_seed(
    rows,
):
    for row in rows:
        if (
            row.get("event")
            == "reset"
            and row.get("seed")
            is not None
        ):
            return int(
                row["seed"]
            )

    # Some historical log variants expose seed
    # outside the reset row. Accept that only as
    # a fallback.
    for row in rows:
        if row.get("seed") is not None:
            return int(
                row["seed"]
            )

    return None


def find_episode(
    *,
    update,
    beta,
):
    log_dir = (
        K5_ROOT
        / f"u{update:02d}"
        / "env_logs"
        / beta
        / TARGET_TERRAIN
    )

    if not log_dir.exists():
        raise FileNotFoundError(
            log_dir
        )

    matches = []

    for path in sorted(
        log_dir.glob(
            "episode_*.jsonl"
        )
    ):
        rows = read_jsonl(
            path
        )

        seed = episode_seed(
            rows
        )

        if seed == TARGET_SEED:
            matches.append(
                (
                    path,
                    rows,
                )
            )

    if len(matches) != 1:
        raise RuntimeError(
            f"u{update:02d}/{beta}: "
            f"expected exactly one episode "
            f"for seed={TARGET_SEED}, "
            f"found={len(matches)}"
        )

    return matches[0]


def require_vector(
    row,
    key,
    expected_dim,
):
    value = row.get(key)

    if value is None:
        raise RuntimeError(
            f"Missing {key!r} "
            f"in episode_step="
            f"{row.get('episode_step')}"
        )

    arr = np.asarray(
        value,
        dtype=np.float64,
    ).reshape(-1)

    if arr.shape != (
        expected_dim,
    ):
        raise RuntimeError(
            f"{key}: expected "
            f"shape=({expected_dim},), "
            f"got={arr.shape}"
        )

    if not np.all(
        np.isfinite(arr)
    ):
        raise RuntimeError(
            f"{key} contains "
            "non-finite values"
        )

    return arr


def reward_components(
    row,
):
    rc = row.get(
        "reward_components"
    )

    if not isinstance(
        rc,
        dict,
    ):
        raise RuntimeError(
            "Missing reward_components "
            f"at episode_step="
            f"{row.get('episode_step')}"
        )

    required = (
        "cost_motion",
        "cost_stability",
        "cost_energy",
        "cost_posture",
        "cost_traction",
    )

    missing = [
        key
        for key in required
        if key not in rc
    ]

    if missing:
        raise RuntimeError(
            "Missing tracer_cost_v3 fields: "
            + ", ".join(missing)
        )

    return rc


def extract_trace(
    *,
    update,
    beta,
):
    path, rows = find_episode(
        update=update,
        beta=beta,
    )

    step_rows = [
        row
        for row in rows
        if row.get("event")
        == "step"
    ]

    if len(step_rows) <= SETTLING_STEPS:
        raise RuntimeError(
            f"{path}: only "
            f"{len(step_rows)} step rows; "
            "not enough after settling"
        )

    # Settling is exactly five zero-HL commands.
    policy_rows = [
        row
        for row in step_rows
        if int(
            row["episode_step"]
        ) > SETTLING_STEPS
    ]

    if not policy_rows:
        raise RuntimeError(
            f"{path}: no policy rows"
        )

    first_episode_step = int(
        policy_rows[0][
            "episode_step"
        ]
    )

    if first_episode_step != (
        SETTLING_STEPS + 1
    ):
        raise RuntimeError(
            f"{path}: expected first policy "
            f"episode_step="
            f"{SETTLING_STEPS + 1}, "
            f"got={first_episode_step}"
        )

    trace = []

    for row in policy_rows:
        episode_step = int(
            row["episode_step"]
        )

        policy_step = (
            episode_step
            - SETTLING_STEPS
            - 1
        )

        obs = require_vector(
            row,
            "observation",
            21,
        )

        requested = require_vector(
            row,
            "requested_physical",
            6,
        )

        applied = require_vector(
            row,
            "applied_physical",
            6,
        )

        requested_norm = (
            np.asarray(
                row.get(
                    "requested_normalized",
                    [],
                ),
                dtype=np.float64,
            ).reshape(-1)
        )

        rc = reward_components(
            row
        )

        slip = row.get(
            "eval_stance_slip"
        )

        if not isinstance(
            slip,
            dict,
        ):
            slip = {}

        trace.append(
            {
                "update":
                    int(update),

                "beta":
                    beta,

                "terrain":
                    TARGET_TERRAIN,

                "seed":
                    TARGET_SEED,

                "source_log":
                    str(
                        path.relative_to(
                            ROOT
                        )
                    ),

                "episode_step":
                    episode_step,

                "policy_step":
                    policy_step,

                "command_seq":
                    int(
                        row.get(
                            "command_seq",
                            -1,
                        )
                    ),

                "state_sim_time_s":
                    finite_or_nan(
                        row.get(
                            "state_sim_time_s"
                        )
                    ),

                "decision_dt_s":
                    finite_or_nan(
                        row.get(
                            "decision_dt_s"
                        )
                    ),

                # --------------------------
                # HL requested reference
                # --------------------------
                "req_vx_mps":
                    float(
                        requested[
                            PHYS_VX
                        ]
                    ),

                "req_yaw_rate_rps":
                    float(
                        requested[
                            PHYS_YAW
                        ]
                    ),

                "req_body_height_m":
                    float(
                        requested[
                            PHYS_HEIGHT
                        ]
                    ),

                "req_clearance_m":
                    float(
                        requested[
                            PHYS_CLEARANCE
                        ]
                    ),

                "req_gait_period_s":
                    float(
                        requested[
                            PHYS_PERIOD
                        ]
                    ),

                "req_duty_factor":
                    float(
                        requested[
                            PHYS_DUTY
                        ]
                    ),

                "req_norm_0":
                    (
                        float(
                            requested_norm[0]
                        )
                        if requested_norm.size
                        >= 1
                        else float("nan")
                    ),

                "req_norm_1":
                    (
                        float(
                            requested_norm[1]
                        )
                        if requested_norm.size
                        >= 2
                        else float("nan")
                    ),

                "req_norm_2":
                    (
                        float(
                            requested_norm[2]
                        )
                        if requested_norm.size
                        >= 3
                        else float("nan")
                    ),

                # --------------------------
                # Applied LL reference
                # --------------------------
                "app_vx_mps":
                    float(
                        applied[
                            PHYS_VX
                        ]
                    ),

                "app_yaw_rate_rps":
                    float(
                        applied[
                            PHYS_YAW
                        ]
                    ),

                "app_body_height_m":
                    float(
                        applied[
                            PHYS_HEIGHT
                        ]
                    ),

                "app_clearance_m":
                    float(
                        applied[
                            PHYS_CLEARANCE
                        ]
                    ),

                # --------------------------
                # Closed-loop physical state
                # --------------------------
                "state_vx_mps":
                    float(
                        obs[
                            OBS_VX
                        ]
                    ),

                "state_vy_mps":
                    float(
                        obs[
                            OBS_VY
                        ]
                    ),

                "state_yaw_rate_rps":
                    float(
                        obs[
                            OBS_YAW_RATE
                        ]
                    ),

                "state_z_m":
                    float(
                        obs[
                            OBS_Z
                        ]
                    ),

                "roll_rad":
                    float(
                        obs[
                            OBS_ROLL
                        ]
                    ),

                "pitch_rad":
                    float(
                        obs[
                            OBS_PITCH
                        ]
                    ),

                # --------------------------
                # Objective basis
                # --------------------------
                "cost_motion":
                    finite_or_nan(
                        rc.get(
                            "cost_motion"
                        )
                    ),

                "cost_stability":
                    finite_or_nan(
                        rc.get(
                            "cost_stability"
                        )
                    ),

                "cost_posture":
                    finite_or_nan(
                        rc.get(
                            "cost_posture"
                        )
                    ),

                "cost_traction":
                    finite_or_nan(
                        rc.get(
                            "cost_traction"
                        )
                    ),

                "cost_energy":
                    finite_or_nan(
                        rc.get(
                            "cost_energy"
                        )
                    ),

                "progress_rate_mps":
                    finite_or_nan(
                        rc.get(
                            "progress_rate_mps"
                        )
                    ),

                "energy_power_ratio":
                    finite_or_nan(
                        rc.get(
                            "energy_power_ratio"
                        )
                    ),

                "energy_abs_j":
                    finite_or_nan(
                        rc.get(
                            "energy_abs_j"
                        )
                    ),

                "energy_dt_s":
                    finite_or_nan(
                        rc.get(
                            "energy_dt_s"
                        )
                    ),

                # Cumulative slip diagnostics.
                # These are intentionally labelled
                # cumulative; cost_traction above is
                # the reward-semantic interval quantity.
                "slip_cum_rms_mps":
                    finite_or_nan(
                        slip.get(
                            "speed_rms_mps"
                        )
                    ),

                "slip_cum_mean_mps":
                    finite_or_nan(
                        slip.get(
                            "speed_mean_mps"
                        )
                    ),

                "slip_cum_max_mps":
                    finite_or_nan(
                        slip.get(
                            "speed_max_mps"
                        )
                    ),

                "goal_distance_m":
                    finite_or_nan(
                        row.get(
                            "goal_distance"
                        )
                    ),

                "safety_state":
                    str(
                        row.get(
                            "safety_state",
                            "",
                        )
                    ),

                "override_active":
                    int(
                        bool(
                            row.get(
                                "override_active",
                                False,
                            )
                        )
                    ),

                "terminated":
                    int(
                        bool(
                            row.get(
                                "terminated",
                                False,
                            )
                        )
                    ),

                "truncated":
                    int(
                        bool(
                            row.get(
                                "truncated",
                                False,
                            )
                        )
                    ),
            }
        )

    return trace


def mean(
    rows,
    key,
):
    values = np.asarray(
        [
            float(row[key])
            for row in rows
        ],
        dtype=np.float64,
    )

    values = values[
        np.isfinite(values)
    ]

    if values.size == 0:
        return float("nan")

    return float(
        np.mean(values)
    )


def mean_abs(
    rows,
    key,
):
    values = np.asarray(
        [
            float(row[key])
            for row in rows
        ],
        dtype=np.float64,
    )

    values = values[
        np.isfinite(values)
    ]

    if values.size == 0:
        return float("nan")

    return float(
        np.mean(
            np.abs(values)
        )
    )


def maximum(
    rows,
    key,
):
    values = np.asarray(
        [
            float(row[key])
            for row in rows
        ],
        dtype=np.float64,
    )

    values = values[
        np.isfinite(values)
    ]

    if values.size == 0:
        return float("nan")

    return float(
        np.max(values)
    )


def write_csv(
    path,
    rows,
):
    if not rows:
        raise RuntimeError(
            f"No rows for {path}"
        )

    fieldnames = list(
        rows[0].keys()
    )

    for row in rows:
        if list(
            row.keys()
        ) != fieldnames:
            raise RuntimeError(
                "CSV schema/order mismatch "
                f"for {path}"
            )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            rows
        )


def build_trace_summary(
    trace,
):
    return {
        "update":
            int(
                trace[0][
                    "update"
                ]
            ),

        "beta":
            trace[0]["beta"],

        "policy_steps":
            len(trace),

        "mean_req_vx_mps":
            mean(
                trace,
                "req_vx_mps",
            ),

        "mean_abs_req_yaw_rate_rps":
            mean_abs(
                trace,
                "req_yaw_rate_rps",
            ),

        "mean_req_body_height_m":
            mean(
                trace,
                "req_body_height_m",
            ),

        "mean_app_vx_mps":
            mean(
                trace,
                "app_vx_mps",
            ),

        "mean_app_body_height_m":
            mean(
                trace,
                "app_body_height_m",
            ),

        "mean_state_vx_mps":
            mean(
                trace,
                "state_vx_mps",
            ),

        "mean_abs_state_vy_mps":
            mean_abs(
                trace,
                "state_vy_mps",
            ),

        "mean_abs_state_yaw_rate_rps":
            mean_abs(
                trace,
                "state_yaw_rate_rps",
            ),

        "mean_state_z_m":
            mean(
                trace,
                "state_z_m",
            ),

        "mean_abs_roll_rad":
            mean_abs(
                trace,
                "roll_rad",
            ),

        "mean_abs_pitch_rad":
            mean_abs(
                trace,
                "pitch_rad",
            ),

        "mean_cost_motion":
            mean(
                trace,
                "cost_motion",
            ),

        "mean_cost_stability":
            mean(
                trace,
                "cost_stability",
            ),

        "mean_cost_posture":
            mean(
                trace,
                "cost_posture",
            ),

        "mean_cost_traction":
            mean(
                trace,
                "cost_traction",
            ),

        "mean_cost_energy":
            mean(
                trace,
                "cost_energy",
            ),

        "max_cost_traction":
            maximum(
                trace,
                "cost_traction",
            ),

        "final_goal_distance_m":
            float(
                trace[-1][
                    "goal_distance_m"
                ]
            ),

        "any_override":
            int(
                any(
                    bool(
                        row[
                            "override_active"
                        ]
                    )
                    for row in trace
                )
            ),
    }


PAIR_FIELDS = (
    "req_vx_mps",
    "req_yaw_rate_rps",
    "req_body_height_m",
    "app_vx_mps",
    "app_yaw_rate_rps",
    "app_body_height_m",
    "state_vx_mps",
    "state_vy_mps",
    "state_yaw_rate_rps",
    "state_z_m",
    "roll_rad",
    "pitch_rad",
    "cost_motion",
    "cost_stability",
    "cost_posture",
    "cost_traction",
    "cost_energy",
)


def pair_trace(
    balanced,
    specialized,
    specialized_name,
):
    by_step_b = {
        int(row["policy_step"]):
            row
        for row in balanced
    }

    by_step_x = {
        int(row["policy_step"]):
            row
        for row in specialized
    }

    common = sorted(
        set(by_step_b)
        & set(by_step_x)
    )

    if not common:
        raise RuntimeError(
            "No common policy steps "
            f"for {specialized_name}"
        )

    rows = []

    for step in common:
        b = by_step_b[step]
        x = by_step_x[step]

        row = {
            "update":
                int(
                    b["update"]
                ),

            "specialized_beta":
                specialized_name,

            "policy_step":
                step,

            "balanced_time_s":
                float(
                    b[
                        "state_sim_time_s"
                    ]
                ),

            "specialized_time_s":
                float(
                    x[
                        "state_sim_time_s"
                    ]
                ),
        }

        for key in PAIR_FIELDS:
            bv = float(
                b[key]
            )

            xv = float(
                x[key]
            )

            row[
                f"balanced_{key}"
            ] = bv

            row[
                f"specialized_{key}"
            ] = xv

            row[
                f"delta_{key}"
            ] = (
                xv - bv
            )

        rows.append(
            row
        )

    return rows


def safe_corr(
    x,
    y,
):
    x = np.asarray(
        x,
        dtype=np.float64,
    )

    y = np.asarray(
        y,
        dtype=np.float64,
    )

    mask = (
        np.isfinite(x)
        & np.isfinite(y)
    )

    x = x[mask]
    y = y[mask]

    if x.size < 3:
        return float("nan")

    if (
        np.std(x) < 1e-12
        or np.std(y) < 1e-12
    ):
        return float("nan")

    return float(
        np.corrcoef(
            x,
            y,
        )[0, 1]
    )


def pair_summary(
    pair_rows,
):
    name = (
        pair_rows[0][
            "specialized_beta"
        ]
    )

    def delta(
        key,
    ):
        return np.asarray(
            [
                float(
                    row[
                        f"delta_{key}"
                    ]
                )
                for row
                in pair_rows
            ],
            dtype=np.float64,
        )

    dctr = delta(
        "cost_traction"
    )

    return {
        "update":
            int(
                pair_rows[0][
                    "update"
                ]
            ),

        "specialized_beta":
            name,

        "paired_steps":
            len(pair_rows),

        # Positive delta cost = specialized beta worse.
        "mean_delta_cost_traction":
            float(
                np.mean(
                    dctr
                )
            ),

        "mean_delta_cost_posture":
            float(
                np.mean(
                    delta(
                        "cost_posture"
                    )
                )
            ),

        "mean_delta_cost_stability":
            float(
                np.mean(
                    delta(
                        "cost_stability"
                    )
                )
            ),

        "mean_delta_cost_energy":
            float(
                np.mean(
                    delta(
                        "cost_energy"
                    )
                )
            ),

        # HL mechanism.
        "mean_delta_req_vx_mps":
            float(
                np.mean(
                    delta(
                        "req_vx_mps"
                    )
                )
            ),

        "mean_delta_req_yaw_rate_rps":
            float(
                np.mean(
                    delta(
                        "req_yaw_rate_rps"
                    )
                )
            ),

        "mean_delta_req_body_height_m":
            float(
                np.mean(
                    delta(
                        "req_body_height_m"
                    )
                )
            ),

        # Applied LL references.
        "mean_delta_app_vx_mps":
            float(
                np.mean(
                    delta(
                        "app_vx_mps"
                    )
                )
            ),

        "mean_delta_app_yaw_rate_rps":
            float(
                np.mean(
                    delta(
                        "app_yaw_rate_rps"
                    )
                )
            ),

        "mean_delta_app_body_height_m":
            float(
                np.mean(
                    delta(
                        "app_body_height_m"
                    )
                )
            ),

        # Physical closed-loop response.
        "mean_delta_state_vx_mps":
            float(
                np.mean(
                    delta(
                        "state_vx_mps"
                    )
                )
            ),

        "mean_delta_abs_vy_mps":
            float(
                np.mean(
                    np.abs(
                        np.asarray(
                            [
                                float(
                                    row[
                                        "specialized_state_vy_mps"
                                    ]
                                )
                                for row
                                in pair_rows
                            ]
                        )
                    )
                    - np.abs(
                        np.asarray(
                            [
                                float(
                                    row[
                                        "balanced_state_vy_mps"
                                    ]
                                )
                                for row
                                in pair_rows
                            ]
                        )
                    )
                )
            ),

        "mean_delta_state_z_m":
            float(
                np.mean(
                    delta(
                        "state_z_m"
                    )
                )
            ),

        # Correlation only, NOT causal attribution.
        "corr_dCtr_dReqVx":
            safe_corr(
                dctr,
                delta(
                    "req_vx_mps"
                ),
            ),

        "corr_dCtr_dReqYaw":
            safe_corr(
                dctr,
                delta(
                    "req_yaw_rate_rps"
                ),
            ),

        "corr_dCtr_dReqHeight":
            safe_corr(
                dctr,
                delta(
                    "req_body_height_m"
                ),
            ),

        "traction_better_steps":
            int(
                np.sum(
                    dctr < 0.0
                )
            ),

        "traction_worse_steps":
            int(
                np.sum(
                    dctr > 0.0
                )
            ),

        "traction_equal_steps":
            int(
                np.sum(
                    dctr == 0.0
                )
            ),
    }


def main():
    if not K5_ROOT.exists():
        raise FileNotFoundError(
            K5_ROOT
        )

    if OUT_DIR.exists():
        raise RuntimeError(
            "Output directory already exists; "
            "refusing overwrite: "
            f"{OUT_DIR}"
        )

    OUT_DIR.mkdir(
        parents=True
    )

    print("=" * 108)
    print(
        "ICRA27 OS-T4.5k6 "
        "LOW-FRICTION ACTION-MECHANISM "
        "DECOMPOSITION"
    )
    print("=" * 108)

    print(
        "source  : OS-T4.5k5 clean "
        "100-tick lockstep trajectories"
    )

    print(
        "terrain :",
        TARGET_TERRAIN,
    )

    print(
        "seed    :",
        TARGET_SEED,
    )

    print(
        "updates :",
        list(
            UPDATES
        ),
    )

    print(
        "betas   :",
        list(
            BETAS
        ),
    )

    print(
        "scope   : offline analysis only; "
        "NO retraining / NO new simulation"
    )

    print()

    traces = {}
    all_trace_rows = []
    trace_summaries = []

    for update in UPDATES:
        for beta in BETAS:
            trace = extract_trace(
                update=update,
                beta=beta,
            )

            traces[
                (
                    update,
                    beta,
                )
            ] = trace

            all_trace_rows.extend(
                trace
            )

            trace_summaries.append(
                build_trace_summary(
                    trace
                )
            )

            print(
                f"u{update:02d} "
                f"{beta:<9} "
                f"steps={len(trace):>3} "
                f"<vx_ref>="
                f"{mean(trace, 'req_vx_mps'):+.5f} "
                f"<h_ref>="
                f"{mean(trace, 'req_body_height_m'):+.5f} "
                f"<Ctr>="
                f"{mean(trace, 'cost_traction'):.6f} "
                f"<Cpost>="
                f"{mean(trace, 'cost_posture'):.6f}"
            )

    pair_rows_all = []
    pair_summaries = []

    print()
    print("=" * 108)
    print(
        "PAIRED MECHANISM VS BALANCED"
    )
    print("=" * 108)

    for update in UPDATES:
        balanced = traces[
            (
                update,
                "balanced",
            )
        ]

        for name in (
            "stability",
            "energy",
        ):
            specialized = traces[
                (
                    update,
                    name,
                )
            ]

            pair_rows = pair_trace(
                balanced,
                specialized,
                name,
            )

            pair_rows_all.extend(
                pair_rows
            )

            summary = pair_summary(
                pair_rows
            )

            pair_summaries.append(
                summary
            )

            print()
            print(
                f"u{update:02d} "
                f"{name.upper()} - BALANCED"
            )

            print(
                "  traction "
                f"dCtr="
                f"{summary['mean_delta_cost_traction']:+.6f} "
                f"better/worse="
                f"{summary['traction_better_steps']}/"
                f"{summary['traction_worse_steps']}"
            )

            print(
                "  HL refs  "
                f"dVx="
                f"{summary['mean_delta_req_vx_mps']:+.6f} "
                f"dYaw="
                f"{summary['mean_delta_req_yaw_rate_rps']:+.6f} "
                f"dH="
                f"{summary['mean_delta_req_body_height_m']:+.6f}"
            )

            print(
                "  LL refs  "
                f"dVx="
                f"{summary['mean_delta_app_vx_mps']:+.6f} "
                f"dYaw="
                f"{summary['mean_delta_app_yaw_rate_rps']:+.6f} "
                f"dH="
                f"{summary['mean_delta_app_body_height_m']:+.6f}"
            )

            print(
                "  state    "
                f"dVx="
                f"{summary['mean_delta_state_vx_mps']:+.6f} "
                f"d|Vy|="
                f"{summary['mean_delta_abs_vy_mps']:+.6f} "
                f"dZ="
                f"{summary['mean_delta_state_z_m']:+.6f}"
            )

            print(
                "  costs    "
                f"dPost="
                f"{summary['mean_delta_cost_posture']:+.6f} "
                f"dStab="
                f"{summary['mean_delta_cost_stability']:+.6f} "
                f"dEnergy="
                f"{summary['mean_delta_cost_energy']:+.6f}"
            )

            print(
                "  corr     "
                f"dCtr~dVx="
                f"{summary['corr_dCtr_dReqVx']:+.3f} "
                f"dCtr~dYaw="
                f"{summary['corr_dCtr_dReqYaw']:+.3f} "
                f"dCtr~dH="
                f"{summary['corr_dCtr_dReqHeight']:+.3f}"
            )

    write_csv(
        OUT_DIR
        / "step_trace.csv",
        all_trace_rows,
    )

    write_csv(
        OUT_DIR
        / "trace_summary.csv",
        trace_summaries,
    )

    write_csv(
        OUT_DIR
        / "pairwise_deltas.csv",
        pair_rows_all,
    )

    write_csv(
        OUT_DIR
        / "pairwise_summary.csv",
        pair_summaries,
    )

    manifest = {
        "schema":
            "icra27_os_t4p5k6_"
            "lowfriction_action_mechanism_v0",

        "source":
            str(
                K5_ROOT.relative_to(
                    ROOT
                )
            ),

        "analysis":
            "offline_clean_lockstep_"
            "closed_loop_trajectory_decomposition",

        "terrain":
            TARGET_TERRAIN,

        "seed":
            TARGET_SEED,

        "updates":
            list(
                UPDATES
            ),

        "betas":
            list(
                BETAS
            ),

        "settling_steps_excluded":
            SETTLING_STEPS,

        "observation_indices":
            {
                "vx": OBS_VX,
                "vy": OBS_VY,
                "yaw_rate":
                    OBS_YAW_RATE,
                "z": OBS_Z,
                "roll": OBS_ROLL,
                "pitch": OBS_PITCH,
            },

        "interpretation_note":
            (
                "Pairwise correlations are "
                "descriptive, not causal. "
                "Positive delta cost_traction "
                "means specialized beta is worse "
                "than balanced."
            ),
    }

    with (
        OUT_DIR
        / "manifest.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            manifest,
            f,
            indent=2,
            sort_keys=True,
        )

        f.write("\n")

    print()
    print("=" * 108)
    print(
        "[ICRA27] OS-T4.5k6 "
        "low-friction action-mechanism "
        "decomposition: COMPUTE PASS"
    )
    print("=" * 108)

    print(
        "step trace :",
        OUT_DIR
        / "step_trace.csv",
    )

    print(
        "pair delta :",
        OUT_DIR
        / "pairwise_deltas.csv",
    )

    print(
        "summary    :",
        OUT_DIR
        / "pairwise_summary.csv",
    )


if __name__ == "__main__":
    main()
