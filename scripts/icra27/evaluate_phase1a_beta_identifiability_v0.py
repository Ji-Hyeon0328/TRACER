#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


from tracer_core.highlevel_rl.pympc_env import (
    PyMPCM7Env,
)

from tracer_core.highlevel_rl.fixed_clearance_wrapper import (
    M7FixedClearanceActionWrapper,
)

from tracer_core.highlevel_rl.beta_conditioning_wrapper import (
    M7BetaConditioningWrapper,
)


# ============================================================
# Reuse the already validated Phase-1.5 evaluator utilities.
# ============================================================

BASE_EVAL_PATH = (
    ROOT
    / "scripts"
    / "icra27"
    / "evaluate_phase1p5_paired_v0.py"
)

spec = importlib.util.spec_from_file_location(
    "phase1p5_eval_base",
    BASE_EVAL_PATH,
)

if spec is None or spec.loader is None:
    raise RuntimeError(
        f"Could not load evaluator: {BASE_EVAL_PATH}"
    )

base = importlib.util.module_from_spec(
    spec
)

spec.loader.exec_module(
    base
)


# ============================================================
# Phase-1A frozen evaluation contract.
# ============================================================

BASE_OBS_DIM = 21
BETA_DIM = 3
OBS_DIM = 24

POLICY_ACTION_DIM = 3

EVAL_REWARD_MODE = "tracer_cost_v2"

DEFAULT_CHECKPOINT = (
    ROOT
    / "results"
    / "icra27"
    / "phase1a_beta_conditioned_v1_seed27027"
    / "m7_ppo_beta_conditioned_final.pt"
)

DEFAULT_SPLIT_PATH = (
    ROOT
    / "configs"
    / "icra27"
    / "m7_perlin_seed_split_v0.json"
)

BETA_BANK = (
    (
        "balanced",
        (
            1.0 / 3.0,
            1.0 / 3.0,
            1.0 / 3.0,
        ),
    ),
    (
        "motion",
        (
            0.70,
            0.15,
            0.15,
        ),
    ),
    (
        "stability",
        (
            0.15,
            0.70,
            0.15,
        ),
    ),
    (
        "energy",
        (
            0.15,
            0.15,
            0.70,
        ),
    ),
)


# The reused evaluator's run_episode() and contract checker
# refer to OBS_DIM through their module globals.
#
# The first 21 indices remain exactly frozen; beta is appended.
base.OBS_DIM = OBS_DIM


def beta_dict():
    return {
        name: [
            float(x)
            for x in beta
        ]
        for name, beta
        in BETA_BANK
    }


def build_groups(
    *,
    mode,
    split,
    eval_seed_base,
):
    if mode == "smoke":
        # Controlled identifiability smoke:
        # same unseen TEST terrain, only beta changes.
        if 2 not in set(
            split["test"]
        ):
            raise RuntimeError(
                "Expected rough TEST seed 2 "
                "is absent from the frozen split."
            )

        return [
            {
                "name": "rough_test_seed2",
                "terrain": "rough_perlin",
                "seeds": [2],
            },
        ]

    if mode == "primary":
        common_seeds = [
            int(
                eval_seed_base + i
            )
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


def validate_checkpoint_beta_bank(
    payload,
):
    extra = payload.get(
        "extra",
        {},
    )

    found = extra.get(
        "beta_bank"
    )

    if not isinstance(
        found,
        dict,
    ):
        raise RuntimeError(
            "Checkpoint does not expose "
            "Phase-1A beta_bank metadata."
        )

    expected = beta_dict()

    if set(found) != set(expected):
        raise RuntimeError(
            "Checkpoint beta-bank names mismatch: "
            f"expected={sorted(expected)}, "
            f"found={sorted(found)}"
        )

    for name in expected:
        np.testing.assert_allclose(
            np.asarray(
                found[name],
                dtype=np.float64,
            ),
            np.asarray(
                expected[name],
                dtype=np.float64,
            ),
            rtol=0.0,
            atol=1e-8,
        )


def make_env(
    *,
    terrain,
    beta,
    command_port,
    log_dir,
):
    base_env = PyMPCM7Env(
        terrain=terrain,

        # Same semantic cost-vector backend.
        # Scalarization uses the beta under test.
        reward_mode=EVAL_REWARD_MODE,
        tracer_beta=beta,

        terminate_on_m4_unsafe=True,

        goal_distance_m=(
            base.GOAL_DISTANCE_M
        ),
        success_radius_m=(
            base.SUCCESS_RADIUS_M
        ),

        decision_dt_s=(
            base.DECISION_DT_S
        ),

        max_episode_steps=(
            base.POLICY_HORIZON
            + base.SETTLING_STEPS
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
        M7FixedClearanceActionWrapper(
            base_env
        )
    )

    env = M7BetaConditioningWrapper(
        action_env,
        beta=beta,
    )

    if (
        base_env.observation_space.shape
        != (BASE_OBS_DIM,)
    ):
        raise RuntimeError(
            "Unexpected base observation shape: "
            f"{base_env.observation_space.shape}"
        )

    if env.observation_space.shape != (
        OBS_DIM,
    ):
        raise RuntimeError(
            "Unexpected conditioned observation shape: "
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


def annotate_row(
    row,
    *,
    beta_name,
    beta,
):
    # Add fields in a deterministic order for CSV schema.
    row[
        "beta_name"
    ] = beta_name

    row[
        "beta_motion"
    ] = float(beta[0])

    row[
        "beta_stability"
    ] = float(beta[1])

    row[
        "beta_energy"
    ] = float(beta[2])

    return row


def print_aggregate(
    *,
    beta_name,
    group_name,
    agg,
):
    def f(
        key,
        digits=4,
    ):
        value = agg.get(key)

        if value is None:
            return "n/a"

        return (
            f"{float(value):.{digits}f}"
        )

    print(
        f"    {beta_name:<10} "
        f"{group_name:<18} "
        f"succ={agg['success']}/"
        f"{agg['episodes']} "
        f"time={f('mean_success_time_s', 3)} "
        f"E={f('mean_success_energy_abs_j', 2)} "
        f"Cm={f('mean_mean_cost_motion')} "
        f"Cs={f('mean_mean_cost_stability')} "
        f"CE={f('mean_mean_cost_energy')} "
        f"vx={f('mean_mean_applied_vx_mps')} "
        f"h={f('mean_mean_applied_height_m')}"
    )


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
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
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
        default=58110,
    )

    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
    )

    args = ap.parse_args()

    split = base.load_seed_split(
        args.perlin_seed_split
    )

    groups = build_groups(
        mode=args.mode,
        split=split,
        eval_seed_base=(
            args.eval_seed_base
        ),
    )

    policy = base.load_policy(
        name="beta_conditioned",
        path=args.checkpoint,
        expected_reward_mode=(
            "tracer_cost_v2"
        ),
    )

    validate_checkpoint_beta_bank(
        policy["payload"]
    )

    out_dir = (
        args.out_dir
        if args.out_dir is not None
        else (
            ROOT
            / "results"
            / "icra27"
            / "phase1a_beta_identifiability_v0"
            / args.mode
        )
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 96)
    print(
        "ICRA27 PHASE-1A BETA IDENTIFIABILITY "
        "DETERMINISTIC EVALUATION"
    )
    print("=" * 96)

    print(
        "mode              :",
        args.mode,
    )

    print(
        "checkpoint        :",
        args.checkpoint,
    )

    print(
        "observation       : "
        "24D = obs21 + beta3"
    )

    print(
        "policy action     : "
        "deterministic tanh(mu)"
    )

    print(
        "beta anchors      : "
        + ", ".join(
            (
                f"{name}="
                f"[{beta[0]:.2f},"
                f"{beta[1]:.2f},"
                f"{beta[2]:.2f}]"
            )
            for name, beta
            in BETA_BANK
        )
    )

    print(
        "output            :",
        out_dir,
    )

    print()

    all_rows = []

    aggregates = {
        name: {}
        for name, _beta
        in BETA_BANK
    }

    for beta_index, (
        beta_name,
        beta,
    ) in enumerate(
        BETA_BANK
    ):
        print(
            f"BETA: {beta_name} "
            f"[{beta[0]:.3f}, "
            f"{beta[1]:.3f}, "
            f"{beta[2]:.3f}]"
        )

        for group_index, group in enumerate(
            groups
        ):
            group_name = group[
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
                + 100 * beta_index
                + 10 * group_index
            )

            log_dir = (
                out_dir
                / "env_logs"
                / beta_name
                / group_name
            )

            env = make_env(
                terrain=terrain,
                beta=beta,
                command_port=command_port,
                log_dir=log_dir,
            )

            rows = []

            try:
                for seed in seeds:
                    row = base.run_episode(
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
                        group=group_name,
                        terrain=terrain,
                        seed=seed,
                    )

                    row = annotate_row(
                        row,
                        beta_name=beta_name,
                        beta=beta,
                    )

                    rows.append(
                        row
                    )

                    all_rows.append(
                        row
                    )

                    print(
                        f"      seed={seed:<6} "
                        f"status={row['status']:<18} "
                        f"steps={row['policy_steps']:<3} "
                        f"time="
                        f"{row['decision_time_s']} "
                        f"E="
                        f"{row['energy_abs_j']} "
                        f"vx="
                        f"{row['mean_applied_vx_mps']}"
                    )

            finally:
                env.close()

            agg = base.aggregate_rows(
                rows
            )

            aggregates[
                beta_name
            ][
                group_name
            ] = agg

            print_aggregate(
                beta_name=beta_name,
                group_name=group_name,
                agg=agg,
            )

        print()

    # --------------------------------------------------------
    # Combined held-out rough aggregate for primary mode.
    # --------------------------------------------------------

    if args.mode == "primary":
        for beta_name, _beta in BETA_BANK:
            rough_rows = [
                row
                for row in all_rows
                if (
                    row["beta_name"]
                    == beta_name
                    and row["group"]
                    in (
                        "rough_validation",
                        "rough_test",
                    )
                )
            ]

            aggregates[
                beta_name
            ][
                "rough_heldout"
            ] = base.aggregate_rows(
                rough_rows
            )

    episodes_json = (
        out_dir
        / "episodes.json"
    )

    episodes_csv = (
        out_dir
        / "episodes.csv"
    )

    summary_path = (
        out_dir
        / "summary.json"
    )

    with episodes_json.open(
        "w"
    ) as f:
        json.dump(
            all_rows,
            f,
            indent=2,
            sort_keys=True,
        )

    base.write_csv(
        episodes_csv,
        all_rows,
    )

    summary = {
        "schema":
            "icra27_phase1a_beta_identifiability_v0",

        "mode":
            args.mode,

        "checkpoint":
            str(args.checkpoint),

        "trained_reward_mode":
            policy[
                "trained_reward_mode"
            ],

        "evaluation_cost_backend":
            EVAL_REWARD_MODE,

        "policy_observation_dim":
            OBS_DIM,

        "policy_action_dim":
            POLICY_ACTION_DIM,

        "deterministic_policy":
            True,

        "beta_bank":
            beta_dict(),

        "groups":
            groups,

        "episodes":
            int(
                len(all_rows)
            ),

        "aggregates":
            aggregates,
    }

    with summary_path.open(
        "w"
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
            sort_keys=True,
        )

    print("=" * 96)
    print(
        "[ICRA27] Phase-1A beta-identifiability "
        "deterministic evaluation: PASS"
    )
    print("=" * 96)

    print(
        "episodes:",
        len(all_rows),
    )

    print(
        "summary :",
        summary_path,
    )

    print(
        "csv     :",
        episodes_csv,
    )


if __name__ == "__main__":
    main()
