#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


from tracer_core.highlevel_rl.ppo import (
    load_ppo_checkpoint,
)

from tracer_core.highlevel_rl.pympc_env import (
    PyMPCM7Env,
)

from tracer_core.highlevel_rl.fixed_clearance_wrapper import (
    M7FixedClearanceActionWrapper,
)


# ============================================================
# Frozen Phase-1.5 evaluation contract
# ============================================================

OBS_DIM = 21
POLICY_ACTION_DIM = 3

# Frozen downstream MetaGaitCommand physical payload:
#
#   [vx, yaw_rate, body_height, swing_clearance,
#    gait_period, duty_factor]
#
# The Phase-1.5 learned policy controls only the first
# three normalized dimensions. The fixed-clearance wrapper
# supplies the fourth learned physical reference, while the
# final two fields remain frozen structural gait quantities.
TRANSPORT_PHYSICAL_DIM = 6
LEARNED_PHYSICAL_DIM = 4

SETTLING_STEPS = 5
POLICY_HORIZON = 50

GOAL_DISTANCE_M = 2.0
SUCCESS_RADIUS_M = 0.15
DECISION_DT_S = 0.20

EVAL_REWARD_MODE = "tracer_cost_v2"

# Frozen 21-D observation:
#
# [0:3]   oracle terrain context
# [3:7]   goal dx, dy, distance, heading
# [7:13]  vx, vy, yaw-rate, z, roll, pitch
# [13:17] applied vx, yaw, height, clearance
# [17:21] previous normalized 4-D downstream action
ROLL_INDEX = 11
PITCH_INDEX = 12
APPLIED_SLICE = slice(13, 16)

DEFAULT_SPLIT_PATH = (
    ROOT
    / "configs"
    / "icra27"
    / "m7_perlin_seed_split_v0.json"
)

DEFAULT_TRACER_CHECKPOINT = (
    ROOT
    / "results"
    / "icra27"
    / "phase1p5_paired_v3"
    / "tracer_cost_v2_seed27027"
    / "m7_ppo_multiterrain_final.pt"
)

DEFAULT_SLR_CHECKPOINT = (
    ROOT
    / "results"
    / "icra27"
    / "phase1p5_paired_v3"
    / "slr_hl_v2_seed27027"
    / "m7_ppo_multiterrain_final.pt"
)


def _finite(
    name: str,
    value,
) -> float:
    value = float(value)

    if not math.isfinite(value):
        raise RuntimeError(
            f"{name} is not finite: {value!r}"
        )

    return value


def _mean(
    values,
):
    if not values:
        return None

    return float(
        np.mean(
            np.asarray(
                values,
                dtype=np.float64,
            )
        )
    )


def _sum(
    values,
):
    if not values:
        return 0.0

    return float(
        np.sum(
            np.asarray(
                values,
                dtype=np.float64,
            )
        )
    )


def _rms(
    values,
):
    if not values:
        return None

    x = np.asarray(
        values,
        dtype=np.float64,
    )

    return float(
        np.sqrt(
            np.mean(
                x * x
            )
        )
    )


def _max_abs(
    values,
):
    if not values:
        return None

    return float(
        np.max(
            np.abs(
                np.asarray(
                    values,
                    dtype=np.float64,
                )
            )
        )
    )


def load_seed_split(
    path: Path,
):
    if not path.exists():
        raise FileNotFoundError(path)

    data = json.loads(
        path.read_text()
    )

    if (
        data.get("schema")
        != "icra27_m7_perlin_seed_split_v0"
    ):
        raise RuntimeError(
            "Unexpected Perlin split schema: "
            f"{data.get('schema')!r}"
        )

    splits = data["splits"]
    stress = data["stress_banks"]

    train = [
        int(x)
        for x in splits["train"]
    ]

    validation = [
        int(x)
        for x in splits["validation"]
    ]

    test = [
        int(x)
        for x in splits["test"]
    ]

    hard = [
        int(x)
        for x in stress["hard"]
    ]

    sets = {
        "train": set(train),
        "validation": set(validation),
        "test": set(test),
        "hard": set(hard),
    }

    names = list(sets)

    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if sets[a] & sets[b]:
                raise RuntimeError(
                    "Perlin split overlap: "
                    f"{a} vs {b}: "
                    f"{sorted(sets[a] & sets[b])}"
                )

    return {
        "raw": data,
        "train": train,
        "validation": validation,
        "test": test,
        "hard": hard,
    }


def build_groups(
    *,
    mode: str,
    split,
    eval_seed_base: int,
):
    if mode == "smoke":
        return [
            {
                "name": "flat",
                "terrain": "flat",
                "seeds": [
                    int(eval_seed_base)
                ],
            },
            {
                "name": "low_friction",
                "terrain": "low_friction",
                "seeds": [
                    int(eval_seed_base)
                ],
            },
            {
                "name": "rough_validation",
                "terrain": "rough_perlin",
                "seeds": list(
                    split["validation"]
                ),
            },
        ]

    if mode == "primary":
        common_seeds = [
            int(eval_seed_base + i)
            for i in range(10)
        ]

        return [
            {
                "name": "flat",
                "terrain": "flat",
                "seeds": common_seeds,
            },
            {
                "name": "low_friction",
                "terrain": "low_friction",
                "seeds": common_seeds,
            },
            {
                "name": "rough_validation",
                "terrain": "rough_perlin",
                "seeds": list(
                    split["validation"]
                ),
            },
            {
                "name": "rough_test",
                "terrain": "rough_perlin",
                "seeds": list(
                    split["test"]
                ),
            },
        ]

    if mode == "hard":
        return [
            {
                "name": "rough_hard",
                "terrain": "rough_perlin",
                "seeds": list(
                    split["hard"]
                ),
            },
        ]

    raise ValueError(
        f"Unsupported mode: {mode!r}"
    )


def load_policy(
    *,
    name,
    path: Path,
    expected_reward_mode: str,
):
    if not path.exists():
        raise FileNotFoundError(path)

    (
        model,
        cfg,
        payload,
    ) = load_ppo_checkpoint(
        path
    )

    if int(payload["obs_dim"]) != OBS_DIM:
        raise RuntimeError(
            f"{name}: expected obs_dim={OBS_DIM}, "
            f"got {payload['obs_dim']}"
        )

    if (
        int(payload["act_dim"])
        != POLICY_ACTION_DIM
    ):
        raise RuntimeError(
            f"{name}: expected act_dim="
            f"{POLICY_ACTION_DIM}, "
            f"got {payload['act_dim']}"
        )

    extra = payload.get(
        "extra",
        {},
    )

    trained_reward_mode = (
        extra.get("reward_mode")
    )

    if (
        trained_reward_mode
        != expected_reward_mode
    ):
        raise RuntimeError(
            f"{name}: checkpoint reward-mode "
            f"lineage mismatch: expected "
            f"{expected_reward_mode!r}, "
            f"got {trained_reward_mode!r}"
        )

    model.eval()

    return {
        "name": name,
        "path": path,
        "model": model,
        "cfg": cfg,
        "payload": payload,
        "trained_reward_mode":
            trained_reward_mode,
    }


def verify_paired_architecture(
    policies,
):
    if len(policies) != 2:
        raise RuntimeError(
            "Expected exactly two paired policies."
        )

    a = policies[0]["payload"]
    b = policies[1]["payload"]

    for key in (
        "obs_dim",
        "act_dim",
        "hidden_sizes",
        "ppo_config",
    ):
        if a[key] != b[key]:
            raise RuntimeError(
                "Paired checkpoint contract "
                f"differs for {key}: "
                f"{a[key]!r} vs {b[key]!r}"
            )


def make_env(
    *,
    terrain,
    command_port,
    log_dir,
):
    base_env = PyMPCM7Env(
        terrain=terrain,

        # Common measurement backend for BOTH policies.
        reward_mode=EVAL_REWARD_MODE,

        terminate_on_m4_unsafe=True,

        goal_distance_m=GOAL_DISTANCE_M,
        success_radius_m=SUCCESS_RADIUS_M,

        decision_dt_s=DECISION_DT_S,

        max_episode_steps=(
            POLICY_HORIZON
            + SETTLING_STEPS
        ),

        command_port=command_port,
        telemetry_port=command_port + 1,
        state_port=command_port + 2,

        command_repeat_hz=20.0,
        telemetry_hz=100.0,
        state_hz=100.0,

        log_dir=log_dir,
    )

    env = M7FixedClearanceActionWrapper(
        base_env
    )

    if env.observation_space.shape != (
        OBS_DIM,
    ):
        raise RuntimeError(
            "Unexpected observation shape: "
            f"{env.observation_space.shape}"
        )

    if env.action_space.shape != (
        POLICY_ACTION_DIM,
    ):
        raise RuntimeError(
            "Unexpected policy action shape: "
            f"{env.action_space.shape}"
        )

    return env


def reset_with_settling(
    env,
    *,
    seed,
):
    obs, info = env.reset(
        seed=int(seed)
    )

    zero = np.zeros(
        env.action_space.shape,
        dtype=np.float32,
    )

    for settling_step in range(
        SETTLING_STEPS
    ):
        (
            obs,
            _reward,
            terminated,
            truncated,
            info,
        ) = env.step(
            zero
        )

        if terminated or truncated:
            return (
                obs,
                info,
                False,
                settling_step + 1,
            )

    return (
        obs,
        info,
        True,
        SETTLING_STEPS,
    )


def classify(
    *,
    info,
    terminated,
    truncated,
):
    if bool(
        info.get(
            "success",
            False,
        )
    ):
        return "success"

    if bool(
        info.get(
            "m4_intervention",
            False,
        )
    ):
        return "m4"

    if truncated:
        return "time_limit"

    if terminated:
        return "terminal"

    return "running"


def reward_components(
    info,
):
    if (
        info.get("reward_mode")
        != EVAL_REWARD_MODE
    ):
        raise RuntimeError(
            "Common evaluator reward-mode "
            "contract violated: "
            f"{info.get('reward_mode')!r}"
        )

    rc = info.get(
        "reward_components"
    )

    if not isinstance(rc, dict):
        raise RuntimeError(
            "Missing reward_components dict."
        )

    expected = (
        "cost_motion",
        "cost_stability",
        "cost_energy",
        "progress_rate_mps",
        "progress_normalized",
        "energy_power_ratio",
        "roll_fraction_of_unsafe",
        "pitch_fraction_of_unsafe",
        "energy_abs_j",
        "energy_dt_s",
        "m4_roll_unsafe_rad",
        "m4_pitch_unsafe_rad",
    )

    missing = [
        key
        for key in expected
        if key not in rc
    ]

    if missing:
        raise RuntimeError(
            "Missing TRACER evaluator metrics: "
            f"{missing}"
        )

    return rc


def verify_step_contract(
    *,
    obs,
    info,
    rc,
):
    obs = np.asarray(
        obs,
        dtype=np.float64,
    )

    if obs.shape != (OBS_DIM,):
        raise RuntimeError(
            "Unexpected step observation shape: "
            f"{obs.shape}"
        )

    applied = np.asarray(
        info["applied_physical"],
        dtype=np.float64,
    )

    if applied.shape != (
        TRANSPORT_PHYSICAL_DIM,
    ):
        raise RuntimeError(
            "Unexpected applied physical "
            "transport shape: "
            f"expected=({TRANSPORT_PHYSICAL_DIM},) "
            f"got={applied.shape}"
        )

    if not np.all(
        np.isfinite(applied)
    ):
        raise RuntimeError(
            "Applied physical transport payload "
            "contains non-finite values."
        )

    # Verify frozen observation indices against the
    # independently exposed applied-command telemetry.
    if not np.allclose(
        obs[APPLIED_SLICE],
        applied[:3],
        rtol=0.0,
        atol=2e-5,
    ):
        raise RuntimeError(
            "Frozen observation applied-command "
            "indices disagree with info telemetry: "
            f"obs={obs[APPLIED_SLICE]} "
            f"info={applied[:3]}"
        )

    roll = float(
        obs[ROLL_INDEX]
    )

    pitch = float(
        obs[PITCH_INDEX]
    )

    roll_limit = _finite(
        "m4_roll_unsafe_rad",
        rc["m4_roll_unsafe_rad"],
    )

    pitch_limit = _finite(
        "m4_pitch_unsafe_rad",
        rc["m4_pitch_unsafe_rad"],
    )

    if (
        roll_limit <= 0.0
        or pitch_limit <= 0.0
    ):
        raise RuntimeError(
            "M4 attitude thresholds must be positive."
        )

    expected_roll_fraction = (
        abs(roll)
        / roll_limit
    )

    expected_pitch_fraction = (
        abs(pitch)
        / pitch_limit
    )

    if not math.isclose(
        expected_roll_fraction,
        _finite(
            "roll_fraction_of_unsafe",
            rc["roll_fraction_of_unsafe"],
        ),
        rel_tol=0.0,
        abs_tol=2e-4,
    ):
        raise RuntimeError(
            "Roll observation index contract "
            "does not match reward telemetry."
        )

    if not math.isclose(
        expected_pitch_fraction,
        _finite(
            "pitch_fraction_of_unsafe",
            rc["pitch_fraction_of_unsafe"],
        ),
        rel_tol=0.0,
        abs_tol=2e-4,
    ):
        raise RuntimeError(
            "Pitch observation index contract "
            "does not match reward telemetry."
        )


def settling_failure_row(
    *,
    policy,
    checkpoint,
    trained_reward_mode,
    group,
    terrain,
    seed,
    settling_steps,
):
    return {
        "policy": policy,
        "checkpoint": str(checkpoint),
        "trained_reward_mode":
            trained_reward_mode,
        "eval_reward_mode":
            EVAL_REWARD_MODE,

        "group": group,
        "terrain": terrain,
        "seed": int(seed),

        "status": "settling_fail",
        "settled": False,
        "settling_steps": int(
            settling_steps
        ),

        "success": False,
        "m4_terminal": False,
        "m4_interventions": 0,

        "policy_steps": 0,

        "initial_goal_distance_m": None,
        "final_goal_distance_m": None,
        "progress_m": None,

        "decision_time_s": None,
        "energy_abs_j": None,
        "energy_bracket_time_s": None,

        "mean_cost_motion": None,
        "mean_cost_stability": None,
        "mean_cost_energy": None,

        "sum_cost_motion": None,
        "sum_cost_stability": None,
        "sum_cost_energy": None,

        "mean_progress_rate_mps": None,
        "mean_energy_power_ratio": None,

        "roll_rms_rad": None,
        "pitch_rms_rad": None,
        "max_abs_roll_rad": None,
        "max_abs_pitch_rad": None,

        "mean_action_vx_norm": None,
        "mean_abs_action_yaw_norm": None,
        "mean_action_height_norm": None,

        "mean_requested_vx_mps": None,
        "mean_abs_requested_yaw_rps": None,
        "mean_requested_height_m": None,
        "mean_requested_clearance_m": None,

        "mean_applied_vx_mps": None,
        "mean_abs_applied_yaw_rps": None,
        "mean_applied_height_m": None,
        "mean_applied_clearance_m": None,
    }


def run_episode(
    env,
    *,
    policy_name,
    checkpoint,
    trained_reward_mode,
    model,
    group,
    terrain,
    seed,
):
    (
        obs,
        info,
        settled,
        settling_steps,
    ) = reset_with_settling(
        env,
        seed=seed,
    )

    if not settled:
        return settling_failure_row(
            policy=policy_name,
            checkpoint=checkpoint,
            trained_reward_mode=(
                trained_reward_mode
            ),
            group=group,
            terrain=terrain,
            seed=seed,
            settling_steps=settling_steps,
        )

    initial_goal_distance = _finite(
        "initial_goal_distance",
        info["goal_distance"],
    )

    actions = []
    requested = []
    applied = []

    decision_dts = []
    energy_abs = []
    energy_dts = []

    c_motion = []
    c_stability = []
    c_energy = []

    progress_rates = []
    energy_power_ratios = []

    rolls = []
    pitches = []

    m4_interventions = 0

    status = "horizon_exhausted"
    final_info = info

    for step in range(
        POLICY_HORIZON
    ):
        (
            action,
            _log_prob,
            _value,
        ) = model.act(
            obs,
            deterministic=True,
        )

        action = np.asarray(
            action,
            dtype=np.float32,
        )

        if action.shape != (
            POLICY_ACTION_DIM,
        ):
            raise RuntimeError(
                "Unexpected deterministic policy "
                f"action shape: {action.shape}"
            )

        (
            next_obs,
            _reward,
            terminated,
            truncated,
            info,
        ) = env.step(
            action
        )

        next_obs = np.asarray(
            next_obs,
            dtype=np.float32,
        )

        rc = reward_components(
            info
        )

        verify_step_contract(
            obs=next_obs,
            info=info,
            rc=rc,
        )

        requested_physical = np.asarray(
            info["requested_physical"],
            dtype=np.float64,
        )

        applied_physical = np.asarray(
            info["applied_physical"],
            dtype=np.float64,
        )

        if requested_physical.shape != (
            TRANSPORT_PHYSICAL_DIM,
        ):
            raise RuntimeError(
                "Unexpected requested physical "
                "transport shape: "
                f"expected=({TRANSPORT_PHYSICAL_DIM},) "
                f"got={requested_physical.shape}"
            )

        if applied_physical.shape != (
            TRANSPORT_PHYSICAL_DIM,
        ):
            raise RuntimeError(
                "Unexpected applied physical "
                "transport shape: "
                f"expected=({TRANSPORT_PHYSICAL_DIM},) "
                f"got={applied_physical.shape}"
            )

        if (
            not np.all(
                np.isfinite(
                    requested_physical
                )
            )
            or not np.all(
                np.isfinite(
                    applied_physical
                )
            )
        ):
            raise RuntimeError(
                "Physical transport payload contains "
                "non-finite values."
            )

        actions.append(
            action.astype(
                np.float64,
                copy=True,
            )
        )

        requested.append(
            requested_physical.copy()
        )

        applied.append(
            applied_physical.copy()
        )

        decision_dt = _finite(
            "decision_dt_s",
            info["decision_dt_s"],
        )

        if decision_dt <= 0.0:
            raise RuntimeError(
                "Evaluator received non-positive "
                f"decision dt: {decision_dt}"
            )

        energy_dt = _finite(
            "energy_dt_s",
            rc["energy_dt_s"],
        )

        if energy_dt <= 0.0:
            raise RuntimeError(
                "Evaluator received non-positive "
                f"energy bracket dt: {energy_dt}"
            )

        decision_dts.append(
            decision_dt
        )

        energy_dts.append(
            energy_dt
        )

        energy_abs.append(
            _finite(
                "energy_abs_j",
                rc["energy_abs_j"],
            )
        )

        c_motion.append(
            _finite(
                "cost_motion",
                rc["cost_motion"],
            )
        )

        c_stability.append(
            _finite(
                "cost_stability",
                rc["cost_stability"],
            )
        )

        c_energy.append(
            _finite(
                "cost_energy",
                rc["cost_energy"],
            )
        )

        progress_rates.append(
            _finite(
                "progress_rate_mps",
                rc["progress_rate_mps"],
            )
        )

        energy_power_ratios.append(
            _finite(
                "energy_power_ratio",
                rc["energy_power_ratio"],
            )
        )

        rolls.append(
            _finite(
                "roll",
                next_obs[ROLL_INDEX],
            )
        )

        pitches.append(
            _finite(
                "pitch",
                next_obs[PITCH_INDEX],
            )
        )

        m4_interventions += int(
            bool(
                info.get(
                    "m4_intervention",
                    False,
                )
            )
        )

        obs = next_obs
        final_info = info

        if terminated or truncated:
            status = classify(
                info=info,
                terminated=terminated,
                truncated=truncated,
            )
            break

    final_goal_distance = _finite(
        "final_goal_distance",
        final_info["goal_distance"],
    )

    action_matrix = np.stack(
        actions,
        axis=0,
    )

    requested_matrix = np.stack(
        requested,
        axis=0,
    )

    applied_matrix = np.stack(
        applied,
        axis=0,
    )

    return {
        "policy": policy_name,
        "checkpoint": str(checkpoint),
        "trained_reward_mode":
            trained_reward_mode,
        "eval_reward_mode":
            EVAL_REWARD_MODE,

        "group": group,
        "terrain": terrain,
        "seed": int(seed),

        "status": status,
        "settled": True,
        "settling_steps":
            int(settling_steps),

        "success":
            bool(status == "success"),

        "m4_terminal":
            bool(
                final_info.get(
                    "m4_terminal",
                    False,
                )
            ),

        "m4_interventions":
            int(m4_interventions),

        "policy_steps":
            int(len(actions)),

        "initial_goal_distance_m":
            float(initial_goal_distance),

        "final_goal_distance_m":
            float(final_goal_distance),

        "progress_m":
            float(
                initial_goal_distance
                - final_goal_distance
            ),

        # Policy decision time uses the transport's
        # terminal-aware partial transition duration.
        "decision_time_s":
            _sum(decision_dts),

        # Mechanical-energy measurement time is kept
        # separate because sparse terminal bracketing may
        # legitimately differ from policy decision time.
        "energy_abs_j":
            _sum(energy_abs),

        "energy_bracket_time_s":
            _sum(energy_dts),

        "mean_cost_motion":
            _mean(c_motion),

        "mean_cost_stability":
            _mean(c_stability),

        "mean_cost_energy":
            _mean(c_energy),

        "sum_cost_motion":
            _sum(c_motion),

        "sum_cost_stability":
            _sum(c_stability),

        "sum_cost_energy":
            _sum(c_energy),

        "mean_progress_rate_mps":
            _mean(progress_rates),

        "mean_energy_power_ratio":
            _mean(
                energy_power_ratios
            ),

        "roll_rms_rad":
            _rms(rolls),

        "pitch_rms_rad":
            _rms(pitches),

        "max_abs_roll_rad":
            _max_abs(rolls),

        "max_abs_pitch_rad":
            _max_abs(pitches),

        "mean_action_vx_norm":
            float(
                np.mean(
                    action_matrix[:, 0]
                )
            ),

        "mean_abs_action_yaw_norm":
            float(
                np.mean(
                    np.abs(
                        action_matrix[:, 1]
                    )
                )
            ),

        "mean_action_height_norm":
            float(
                np.mean(
                    action_matrix[:, 2]
                )
            ),

        "mean_requested_vx_mps":
            float(
                np.mean(
                    requested_matrix[:, 0]
                )
            ),

        "mean_abs_requested_yaw_rps":
            float(
                np.mean(
                    np.abs(
                        requested_matrix[:, 1]
                    )
                )
            ),

        "mean_requested_height_m":
            float(
                np.mean(
                    requested_matrix[:, 2]
                )
            ),

        "mean_requested_clearance_m":
            float(
                np.mean(
                    requested_matrix[:, 3]
                )
            ),

        "mean_applied_vx_mps":
            float(
                np.mean(
                    applied_matrix[:, 0]
                )
            ),

        "mean_abs_applied_yaw_rps":
            float(
                np.mean(
                    np.abs(
                        applied_matrix[:, 1]
                    )
                )
            ),

        "mean_applied_height_m":
            float(
                np.mean(
                    applied_matrix[:, 2]
                )
            ),

        "mean_applied_clearance_m":
            float(
                np.mean(
                    applied_matrix[:, 3]
                )
            ),
    }


AGGREGATE_METRICS = (
    "policy_steps",
    "progress_m",
    "decision_time_s",
    "energy_abs_j",
    "energy_bracket_time_s",

    "mean_cost_motion",
    "mean_cost_stability",
    "mean_cost_energy",

    "sum_cost_motion",
    "sum_cost_stability",
    "sum_cost_energy",

    "mean_progress_rate_mps",
    "mean_energy_power_ratio",

    "roll_rms_rad",
    "pitch_rms_rad",
    "max_abs_roll_rad",
    "max_abs_pitch_rad",

    "mean_action_vx_norm",
    "mean_abs_action_yaw_norm",
    "mean_action_height_norm",

    "mean_requested_vx_mps",
    "mean_abs_requested_yaw_rps",
    "mean_requested_height_m",
    "mean_requested_clearance_m",

    "mean_applied_vx_mps",
    "mean_abs_applied_yaw_rps",
    "mean_applied_height_m",
    "mean_applied_clearance_m",
)


def aggregate_rows(
    rows,
):
    total = len(rows)

    valid = [
        row
        for row in rows
        if row["settled"]
    ]

    success = [
        row
        for row in valid
        if row["status"] == "success"
    ]

    result = {
        "episodes": int(total),
        "valid_episodes": int(len(valid)),

        "success": int(
            sum(
                row["status"] == "success"
                for row in rows
            )
        ),

        "m4": int(
            sum(
                row["status"] == "m4"
                for row in rows
            )
        ),

        "time_limit": int(
            sum(
                row["status"] == "time_limit"
                for row in rows
            )
        ),

        "terminal": int(
            sum(
                row["status"] == "terminal"
                for row in rows
            )
        ),

        "horizon_exhausted": int(
            sum(
                row["status"]
                == "horizon_exhausted"
                for row in rows
            )
        ),

        "settling_fail": int(
            sum(
                not row["settled"]
                for row in rows
            )
        ),
    }

    result[
        "success_rate_all"
    ] = (
        float(result["success"] / total)
        if total > 0
        else None
    )

    result[
        "success_rate_valid"
    ] = (
        float(
            result["success"]
            / len(valid)
        )
        if valid
        else None
    )

    result[
        "mean_success_time_s"
    ] = _mean(
        [
            row["decision_time_s"]
            for row in success
        ]
    )

    result[
        "mean_success_energy_abs_j"
    ] = _mean(
        [
            row["energy_abs_j"]
            for row in success
        ]
    )

    for metric in AGGREGATE_METRICS:
        result[
            f"mean_{metric}"
        ] = _mean(
            [
                row[metric]
                for row in valid
                if row[metric] is not None
            ]
        )

    return result


def write_csv(
    path: Path,
    rows,
):
    if not rows:
        raise RuntimeError(
            "Cannot write empty episode CSV."
        )

    fieldnames = list(
        rows[0].keys()
    )

    for row in rows:
        if list(row.keys()) != fieldnames:
            raise RuntimeError(
                "Episode-row schema changed "
                "during evaluation."
            )

    with path.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--mode",
        choices=(
            "smoke",
            "primary",
            "hard",
        ),
        default="smoke",
    )

    ap.add_argument(
        "--tracer-checkpoint",
        type=Path,
        default=DEFAULT_TRACER_CHECKPOINT,
    )

    ap.add_argument(
        "--slr-checkpoint",
        type=Path,
        default=DEFAULT_SLR_CHECKPOINT,
    )

    ap.add_argument(
        "--perlin-seed-split",
        type=Path,
        default=DEFAULT_SPLIT_PATH,
    )

    ap.add_argument(
        "--eval-seed-base",
        type=int,
        default=27027,
    )

    ap.add_argument(
        "--base-port",
        type=int,
        default=53310,
    )

    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
    )

    args = ap.parse_args()

    split = load_seed_split(
        args.perlin_seed_split
    )

    groups = build_groups(
        mode=args.mode,
        split=split,
        eval_seed_base=(
            args.eval_seed_base
        ),
    )

    policies = [
        load_policy(
            name="tracer_cost_v2",
            path=args.tracer_checkpoint,
            expected_reward_mode=(
                "tracer_cost_v2"
            ),
        ),
        load_policy(
            name="slr_hl_v2",
            path=args.slr_checkpoint,
            expected_reward_mode=(
                "slr_hl_v2"
            ),
        ),
    ]

    verify_paired_architecture(
        policies
    )

    out_dir = (
        args.out_dir
        if args.out_dir is not None
        else (
            ROOT
            / "results"
            / "icra27"
            / "phase1p5_eval_v0"
            / args.mode
        )
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 90)
    print(
        "ICRA27 PHASE-1.5 PAIRED "
        "DETERMINISTIC EVALUATION"
    )
    print("=" * 90)

    print(
        "mode                :",
        args.mode,
    )

    print(
        "evaluation reward   :",
        EVAL_REWARD_MODE,
    )

    print(
        "policy action       : "
        "deterministic tanh(mu)"
    )

    print(
        "goal / success      : "
        f"{GOAL_DISTANCE_M:.2f} m / "
        f"{SUCCESS_RADIUS_M:.2f} m"
    )

    print(
        "policy horizon      :",
        POLICY_HORIZON,
    )

    print(
        "settling            :",
        SETTLING_STEPS,
    )

    print(
        "output              :",
        out_dir,
    )

    print()

    all_rows = []
    aggregates = {}

    for policy_index, policy in enumerate(
        policies
    ):
        policy_name = policy["name"]

        aggregates[
            policy_name
        ] = {}

        print(
            f"POLICY: {policy_name}"
        )

        print(
            "  checkpoint:",
            policy["path"],
        )

        print(
            "  trained reward:",
            policy[
                "trained_reward_mode"
            ],
        )

        for group_index, group in enumerate(
            groups
        ):
            group_name = group["name"]
            terrain = group["terrain"]
            seeds = group["seeds"]

            command_port = (
                int(args.base_port)
                + 100 * policy_index
                + 10 * group_index
            )

            log_dir = (
                out_dir
                / "env_logs"
                / policy_name
                / group_name
            )

            print()
            print(
                f"  GROUP: {group_name} "
                f"terrain={terrain} "
                f"seeds={seeds}"
            )

            env = make_env(
                terrain=terrain,
                command_port=command_port,
                log_dir=log_dir,
            )

            rows = []

            try:
                for seed in seeds:
                    row = run_episode(
                        env,
                        policy_name=policy_name,
                        checkpoint=(
                            policy["path"]
                        ),
                        trained_reward_mode=(
                            policy[
                                "trained_reward_mode"
                            ]
                        ),
                        model=policy["model"],
                        group=group_name,
                        terrain=terrain,
                        seed=seed,
                    )

                    rows.append(row)
                    all_rows.append(row)

                    energy_text = (
                        "n/a"
                        if (
                            row["energy_abs_j"]
                            is None
                        )
                        else (
                            f"{row['energy_abs_j']:.3f}J"
                        )
                    )

                    time_text = (
                        "n/a"
                        if (
                            row["decision_time_s"]
                            is None
                        )
                        else (
                            f"{row['decision_time_s']:.3f}s"
                        )
                    )

                    print(
                        f"    seed={seed:05d} "
                        f"status={row['status']:<17s} "
                        f"steps={row['policy_steps']:02d} "
                        f"time={time_text:<10s} "
                        f"Eabs={energy_text:<12s}"
                    )

            finally:
                env.close()

            a = aggregate_rows(
                rows
            )

            aggregates[
                policy_name
            ][
                group_name
            ] = a

            success_rate = (
                a[
                    "success_rate_valid"
                ]
            )

            success_text = (
                "n/a"
                if success_rate is None
                else (
                    f"{100.0 * success_rate:.1f}%"
                )
            )

            print(
                "    -> "
                f"success={a['success']}/"
                f"{a['valid_episodes']} "
                f"({success_text}) "
                f"m4={a['m4']} "
                f"settle_fail="
                f"{a['settling_fail']}"
            )

    episode_json = (
        out_dir
        / "episodes.json"
    )

    episode_csv = (
        out_dir
        / "episodes.csv"
    )

    summary_json = (
        out_dir
        / "summary.json"
    )

    episode_json.write_text(
        json.dumps(
            all_rows,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    write_csv(
        episode_csv,
        all_rows,
    )

    summary_payload = {
        "schema":
            "icra27_phase1p5_paired_eval_v0",

        "mode":
            args.mode,

        "evaluation_reward_mode":
            EVAL_REWARD_MODE,

        "deterministic_policy":
            "PPOActorCritic.act("
            "deterministic=True)",

        "goal_distance_m":
            GOAL_DISTANCE_M,

        "success_radius_m":
            SUCCESS_RADIUS_M,

        "decision_dt_nominal_s":
            DECISION_DT_S,

        "settling_steps":
            SETTLING_STEPS,

        "policy_horizon":
            POLICY_HORIZON,

        "eval_seed_base":
            int(args.eval_seed_base),

        "perlin_seed_split":
            str(
                args.perlin_seed_split
            ),

        "groups":
            groups,

        "policies": {
            policy["name"]: {
                "checkpoint":
                    str(policy["path"]),

                "trained_reward_mode":
                    policy[
                        "trained_reward_mode"
                    ],

                "obs_dim":
                    int(
                        policy[
                            "payload"
                        ][
                            "obs_dim"
                        ]
                    ),

                "act_dim":
                    int(
                        policy[
                            "payload"
                        ][
                            "act_dim"
                        ]
                    ),

                "hidden_sizes":
                    policy[
                        "payload"
                    ][
                        "hidden_sizes"
                    ],

                "ppo_config":
                    policy[
                        "payload"
                    ][
                        "ppo_config"
                    ],
            }
            for policy in policies
        },

        "aggregates":
            aggregates,
    }

    summary_json.write_text(
        json.dumps(
            summary_payload,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print()
    print("=" * 90)
    print(
        "[ICRA27] Phase-1.5 paired "
        "deterministic evaluation: PASS"
    )
    print("=" * 90)

    print(
        "episodes JSON:",
        episode_json,
    )

    print(
        "episodes CSV :",
        episode_csv,
    )

    print(
        "summary JSON :",
        summary_json,
    )


if __name__ == "__main__":
    main()
