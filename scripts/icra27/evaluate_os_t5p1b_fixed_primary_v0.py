#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


T4_PATH = (
    ROOT
    / "scripts"
    / "icra27"
    / "evaluate_os_t4p9b_v4_primary_heldout_v0.py"
)

DEFAULT_FIXED_MANIFEST = (
    ROOT
    / "results"
    / "icra27"
    / "os_t5p1a_fixed_command_train_freeze_v0"
    / "fixed_command_manifest.json"
)

DEFAULT_OUT = (
    ROOT
    / "results"
    / "icra27"
    / "os_t5p1b_fixed_primary_v0"
)


def load_module(
    path: Path,
    name: str,
):
    spec = (
        importlib.util.spec_from_file_location(
            name,
            path,
        )
    )

    if (
        spec is None
        or spec.loader is None
    ):
        raise RuntimeError(
            f"Could not load module: {path}"
        )

    module = (
        importlib.util.module_from_spec(
            spec
        )
    )

    sys.modules[name] = module
    spec.loader.exec_module(module)

    return module


class FixedModel:
    """
    Minimal run_episode-compatible deterministic model.

    The returned 3D action is frozen before held-out
    evaluation using TRAIN-only balanced-policy data.
    """

    def __init__(
        self,
        action,
    ):
        self.action = np.asarray(
            action,
            dtype=np.float32,
        )

        if self.action.shape != (3,):
            raise ValueError(
                "Fixed policy action must be 3D; "
                f"got {self.action.shape}"
            )

        if not np.all(
            np.isfinite(self.action)
        ):
            raise ValueError(
                "Fixed action contains "
                "non-finite values."
            )

        if np.any(
            np.abs(self.action) > 1.0
        ):
            raise ValueError(
                "Fixed normalized action "
                "lies outside [-1, 1]."
            )

    def act(
        self,
        obs,
        *,
        deterministic=False,
    ):
        del obs
        del deterministic

        return (
            self.action.copy(),
            0.0,
            0.0,
        )


def write_csv(
    path,
    rows,
):
    if not rows:
        raise RuntimeError(
            "No rows to write."
        )

    fields = list(
        rows[0].keys()
    )

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


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--fixed-manifest",
        type=Path,
        default=DEFAULT_FIXED_MANIFEST,
    )

    ap.add_argument(
        "--perlin-seed-split",
        type=Path,
        default=None,
    )

    ap.add_argument(
        "--eval-seed-base",
        type=int,
        default=27027,
    )

    ap.add_argument(
        "--base-port",
        type=int,
        default=62010,
    )

    ap.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT,
    )

    args = ap.parse_args()

    if args.out_dir.exists():
        raise RuntimeError(
            "Refusing to overwrite output: "
            f"{args.out_dir}"
        )

    # --------------------------------------------------------
    # Reuse the already-validated T4.9b v4/lockstep evaluator.
    # --------------------------------------------------------

    t4 = load_module(
        T4_PATH,
        "os_t5p1b_t4p9b_base",
    )

    # T4.9b main() normally performs this evaluator-local
    # promotion. Since we import its helpers without main(),
    # reproduce only that runtime-global contract here.
    historical_mode = (
        t4.base.EVAL_REWARD_MODE
    )

    if historical_mode != (
        "tracer_cost_v2"
    ):
        raise RuntimeError(
            "Unexpected historical base reward mode: "
            f"{historical_mode!r}"
        )

    t4.base.EVAL_REWARD_MODE = (
        "tracer_cost_v4"
    )

    if (
        t4.EVAL_REWARD_MODE
        != "tracer_cost_v4"
        or t4.base.EVAL_REWARD_MODE
        != "tracer_cost_v4"
    ):
        raise RuntimeError(
            "OS-T5.1b reward-mode contract failed."
        )

    if int(
        t4.POLICY_ACTION_DIM
    ) != 3:
        raise RuntimeError(
            "OS-T5.1b requires 3D "
            "fixed policy action."
        )

    # --------------------------------------------------------
    # Frozen TRAIN-only fixed action.
    # --------------------------------------------------------

    manifest = json.loads(
        args.fixed_manifest.read_text()
    )

    if (
        manifest.get(
            "schema"
        )
        != "icra27_os_t5p1a_fixed_command_train_freeze_v0"
    ):
        raise RuntimeError(
            "Unexpected fixed-command manifest schema."
        )

    if (
        manifest.get(
            "selection_scope"
        )
        != "TRAIN-only"
    ):
        raise RuntimeError(
            "Fixed baseline is not TRAIN-only."
        )

    if int(
        manifest.get(
            "source_checkpoint_update",
            -1,
        )
    ) != 30:
        raise RuntimeError(
            "Fixed baseline must derive "
            "from frozen u30."
        )

    norm = manifest[
        "fixed_policy_action_normalized"
    ]

    fixed_action = np.asarray(
        [
            float(norm["vx"]),
            float(norm["yaw_rate"]),
            float(norm["body_height"]),
        ],
        dtype=np.float32,
    )

    model = FixedModel(
        fixed_action
    )

    # --------------------------------------------------------
    # EXACT same primary groups as T4.9b.
    # --------------------------------------------------------

    split_path = (
        args.perlin_seed_split
        if args.perlin_seed_split
        is not None
        else t4.DEFAULT_SPLIT_PATH
    )

    split = t4.base.load_seed_split(
        split_path
    )

    groups = t4.build_groups(
        mode="primary",
        split=split,
        eval_seed_base=(
            args.eval_seed_base
        ),
    )

    balanced_beta = dict(
        t4.BETA_BANK
    )["balanced"]

    args.out_dir.mkdir(
        parents=True
    )

    all_rows = []
    aggregates = {}

    print("=" * 104)
    print(
        "ICRA27 OS-T5.1b "
        "FIXED TRAIN-MEAN PRIMARY BASELINE"
    )
    print("=" * 104)

    print(
        "reward       : tracer_cost_v4"
    )
    print(
        "transition   : exact lockstep "
        "100 x 0.002 s = 0.200 s"
    )
    print(
        "fixed action :",
        [
            float(x)
            for x in fixed_action
        ],
    )
    print(
        "source       : TRAIN-only "
        "balanced u30"
    )
    print(
        "groups       :",
        [
            (
                g["name"],
                list(g["seeds"]),
            )
            for g in groups
        ],
    )
    print(
        "output       :",
        args.out_dir,
    )
    print()

    for group_index, group in enumerate(
        groups
    ):
        name = group["name"]
        terrain = group["terrain"]
        seeds = group["seeds"]

        command_port = (
            int(args.base_port)
            + 10 * group_index
        )

        log_dir = (
            args.out_dir
            / "env_logs"
            / "fixed_train_mean"
            / name
        )

        env = t4.make_env(
            terrain=terrain,
            beta=balanced_beta,
            command_port=command_port,
            log_dir=log_dir,
        )

        rows = []

        try:
            for seed in seeds:
                row = t4.base.run_episode(
                    env,
                    policy_name=(
                        "fixed_train_mean"
                    ),
                    checkpoint=(
                        "TRAIN-only balanced-u30 "
                        "mean command"
                    ),
                    trained_reward_mode=(
                        "fixed_baseline_no_training"
                    ),
                    model=model,
                    group=name,
                    terrain=terrain,
                    seed=seed,
                )

                row[
                    "comparison_method"
                ] = "B0_fixed"

                row[
                    "beta_name"
                ] = "fixed"

                row[
                    "fixed_action_norm_vx"
                ] = float(
                    fixed_action[0]
                )

                row[
                    "fixed_action_norm_yaw"
                ] = float(
                    fixed_action[1]
                )

                row[
                    "fixed_action_norm_height"
                ] = float(
                    fixed_action[2]
                )

                rows.append(row)
                all_rows.append(row)

                print(
                    f"  {name:<18} "
                    f"seed={seed:<6} "
                    f"status={row['status']:<18} "
                    f"time={row['decision_time_s']:.3f} "
                    f"Cm={row['mean_cost_motion']:.4f} "
                    f"Cs={row['mean_cost_stability']:.4f} "
                    f"CE={row['mean_cost_energy']:.4f} "
                    f"E={row['energy_abs_j']:.2f} "
                    f"vx={row['mean_applied_vx_mps']:.4f}"
                )

        finally:
            env.close()

        aggregates[
            name
        ] = t4.base.aggregate_rows(
            rows
        )

        agg = aggregates[name]

        print(
            f"    AGG {name:<16} "
            f"succ={agg['success']}/"
            f"{agg['episodes']} "
            f"Cm={agg['mean_mean_cost_motion']:.4f} "
            f"Cs={agg['mean_mean_cost_stability']:.4f} "
            f"CE={agg['mean_mean_cost_energy']:.4f} "
            f"E={agg['mean_energy_abs_j']:.2f}"
        )
        print()

    expected = sum(
        len(g["seeds"])
        for g in groups
    )

    if len(all_rows) != expected:
        raise RuntimeError(
            "Unexpected episode count: "
            f"{len(all_rows)} != {expected}"
        )

    if expected != 29:
        raise RuntimeError(
            "Primary comparison contract "
            f"expected 29 contexts, got {expected}"
        )

    episodes_json = (
        args.out_dir
        / "episodes.json"
    )

    episodes_csv = (
        args.out_dir
        / "episodes.csv"
    )

    summary_path = (
        args.out_dir
        / "summary.json"
    )

    episodes_json.write_text(
        json.dumps(
            all_rows,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    write_csv(
        episodes_csv,
        all_rows,
    )

    summary = {
        "schema":
            "icra27_os_t5p1b_fixed_primary_v0",

        "comparison_method":
            "B0_fixed",

        "source_selection_scope":
            "TRAIN-only",

        "source_checkpoint_update":
            30,

        "fixed_manifest":
            str(args.fixed_manifest),

        "fixed_policy_action_normalized":
            [
                float(x)
                for x in fixed_action
            ],

        "evaluation_cost_backend":
            "tracer_cost_v4",

        "transition_contract":
            {
                "lockstep_eval": True,
                "physics_dt_s": 0.002,
                "physics_steps_per_decision":
                    100,
                "decision_dt_s": 0.200,
            },

        "episodes":
            len(all_rows),

        "groups":
            groups,

        "aggregates":
            aggregates,
    }

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
        "[ICRA27] OS-T5.1b "
        "fixed primary baseline: PASS"
    )
    print("=" * 104)

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
