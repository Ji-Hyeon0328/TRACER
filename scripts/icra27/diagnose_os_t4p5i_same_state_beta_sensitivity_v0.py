#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import statistics
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


TRAINER_PATH = (
    ROOT
    / "scripts"
    / "icra27"
    / "train_m7_ppo_beta_conditioned_v3.py"
)

GRAD_HELPER_PATH = (
    ROOT
    / "scripts"
    / "icra27"
    / "diagnose_os_t4p5g_beta_gradient_interference_v0.py"
)


EXPECTED_TERRAINS = (
    "flat",
    "low_friction",
    "rough_perlin",
)


EXPECTED_BETA_NAMES = (
    "balanced",
    "motion",
    "stability",
    "energy",
)


REFERENCE_CHECKPOINT = "base_u30"


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Could not load module: {path}"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    return module


def beta_dict():
    # Preserve the trainer's canonical Python-float beta
    # values for environment/reward validation.
    #
    # In particular, converting [1/3,1/3,1/3] to float32
    # before set_tracer_beta() yields a sum slightly above 1.
    # Conversion to float32 is done only when constructing
    # the neural-network observation.
    return {
        name:
            tuple(
                float(x)
                for x in beta
            )

        for name, beta
        in trainer.BETA_BANK
    }


@torch.no_grad()
def deterministic_mean_actions(
    model,
    base_obs,
    beta,
):
    base_obs = np.asarray(
        base_obs,
        dtype=np.float32,
    )

    beta = np.asarray(
        beta,
        dtype=np.float32,
    )

    if (
        base_obs.ndim != 2
        or base_obs.shape[1] != 21
    ):
        raise RuntimeError(
            "Expected base state bank shape "
            f"(N,21), got {base_obs.shape}"
        )

    beta_batch = np.repeat(
        beta.reshape(1, 3),
        base_obs.shape[0],
        axis=0,
    )

    conditioned = np.concatenate(
        (
            base_obs,
            beta_batch,
        ),
        axis=1,
    )

    if conditioned.shape[1] != 24:
        raise RuntimeError(
            "Unexpected conditioned shape: "
            f"{conditioned.shape}"
        )

    obs_t = torch.as_tensor(
        conditioned,
        dtype=torch.float32,
    )

    dist, _value = (
        model.distribution_and_value(
            obs_t
        )
    )

    action = torch.tanh(
        dist.mean
    )

    return (
        action
        .cpu()
        .numpy()
        .astype(
            np.float32,
            copy=False,
        )
    )


def choose_episode_seed(
    terrain,
    reset_index,
):
    reset_index = int(
        reset_index
    )

    if terrain == "flat":
        return 27100 + reset_index

    if terrain == "low_friction":
        return 27200 + reset_index

    if terrain == "rough_perlin":
        seeds = (
            13,
            7,
            15,
        )

        return int(
            seeds[
                reset_index
                % len(seeds)
            ]
        )

    raise ValueError(
        f"Unexpected terrain: {terrain}"
    )


def collect_common_state_bank(
    *,
    model,
    states_per_terrain,
    policy_episode_steps,
    base_port,
    out_dir,
):
    """
    Collect one common TRAIN-only physical-state bank.

    Source policy:
      base_u30 + balanced beta.

    Stored state:
      obs21 only.

    The beta tail is removed before later counterfactual
    same-state evaluation.
    """

    betas = beta_dict()

    balanced = betas[
        "balanced"
    ]

    bank = {}

    rollout_stats = {}

    for terrain_index, terrain in enumerate(
        EXPECTED_TERRAINS
    ):
        command_port = (
            int(base_port)
            + 10
            * terrain_index
        )

        log_dir = (
            out_dir
            / "state_bank_source"
            / terrain
        )

        env, _base_env = (
            trainer.make_env(
                terrain=terrain,
                beta=balanced,

                policy_episode_steps=(
                    policy_episode_steps
                ),

                command_port=(
                    command_port
                ),

                log_dir=log_dir,

                reward_mode=(
                    "tracer_cost_v3"
                ),
            )
        )

        states = []

        reset_index = 0
        completed = 0
        successes = 0
        terminals = 0

        obs = None

        try:
            while len(states) < int(
                states_per_terrain
            ):
                if obs is None:
                    while True:
                        seed = choose_episode_seed(
                            terrain,
                            reset_index,
                        )

                        reset_index += 1

                        (
                            candidate_obs,
                            _info,
                            settled,
                            _settling_steps,
                        ) = trainer.reset_with_settling(
                            env,
                            seed=seed,
                        )

                        if settled:
                            obs = (
                                candidate_obs
                            )
                            break

                    obs = np.asarray(
                        obs,
                        dtype=np.float32,
                    )

                if obs.shape != (24,):
                    raise RuntimeError(
                        "Expected conditioned obs24, "
                        f"got {obs.shape}"
                    )

                if not np.allclose(
                    obs[21:24],
                    balanced,
                    rtol=0.0,
                    atol=1e-7,
                ):
                    raise RuntimeError(
                        "State-bank source beta tail "
                        "does not match balanced beta."
                    )

                # Store only the physical/base policy state.
                states.append(
                    obs[:21].copy()
                )

                action, _log_prob, _value = (
                    model.act(
                        obs,
                        deterministic=True,
                    )
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

                if terminated or truncated:
                    completed += 1

                    successes += int(
                        bool(
                            info.get(
                                "success",
                                False,
                            )
                        )
                    )

                    terminals += int(
                        bool(
                            terminated
                        )
                    )

                    obs = None

                else:
                    obs = np.asarray(
                        next_obs,
                        dtype=np.float32,
                    )

        finally:
            env.close()

        state_array = np.stack(
            states,
            axis=0,
        ).astype(
            np.float32
        )

        if state_array.shape != (
            int(states_per_terrain),
            21,
        ):
            raise RuntimeError(
                "Unexpected state-bank shape: "
                f"{state_array.shape}"
            )

        bank[
            terrain
        ] = state_array

        rollout_stats[
            terrain
        ] = {
            "states":
                int(
                    state_array.shape[0]
                ),

            "reset_attempts":
                int(
                    reset_index
                ),

            "completed_episodes":
                int(
                    completed
                ),

            "success_episodes":
                int(
                    successes
                ),

            "terminated_episodes":
                int(
                    terminals
                ),
        }

        print(
            f"  state bank {terrain:<13} "
            f"N={state_array.shape[0]:4d} "
            f"episodes={completed:3d} "
            f"success={successes:3d}"
        )

    return (
        bank,
        rollout_stats,
    )


def quantile(
    values,
    q,
):
    return float(
        np.quantile(
            np.asarray(
                values,
                dtype=np.float64,
            ),
            q,
        )
    )


def summarize_delta(
    delta,
):
    delta = np.asarray(
        delta,
        dtype=np.float64,
    )

    norm = np.linalg.norm(
        delta,
        axis=1,
    )

    abs_delta = np.abs(
        delta
    )

    return {
        "delta_norm": {
            "mean":
                float(
                    np.mean(norm)
                ),

            "median":
                float(
                    np.median(norm)
                ),

            "q25":
                quantile(
                    norm,
                    0.25,
                ),

            "q75":
                quantile(
                    norm,
                    0.75,
                ),

            "max":
                float(
                    np.max(norm)
                ),
        },

        "signed_delta_median":
            [
                float(x)
                for x
                in np.median(
                    delta,
                    axis=0,
                )
            ],

        "signed_delta_mean":
            [
                float(x)
                for x
                in np.mean(
                    delta,
                    axis=0,
                )
            ],

        "mean_abs_delta":
            [
                float(x)
                for x
                in np.mean(
                    abs_delta,
                    axis=0,
                )
            ],
    }


def pairwise_distance_summary(
    actions,
):
    names = tuple(
        EXPECTED_BETA_NAMES
    )

    result = {}

    for i, a in enumerate(
        names
    ):
        for b in names[
            i + 1:
        ]:
            delta = (
                actions[b]
                - actions[a]
            )

            result[
                f"{a}__{b}"
            ] = summarize_delta(
                delta
            )[
                "delta_norm"
            ]

    return result


def evaluate_checkpoint(
    *,
    checkpoint_name,
    checkpoint_path,
    state_bank,
):
    model, cfg, payload = (
        grad_helper.load_checkpoint(
            checkpoint_path
        )
    )

    betas = beta_dict()

    print()
    print("=" * 112)
    print(
        checkpoint_name
    )
    print("=" * 112)

    print(
        "checkpoint:",
        checkpoint_path,
    )

    print(
        "update    :",
        payload.get(
            "extra",
            {},
        ).get(
            "update"
        ),
    )

    result = {
        "checkpoint":
            str(
                checkpoint_path
            ),

        "checkpoint_update":
            payload.get(
                "extra",
                {},
            ).get(
                "update"
            ),

        "learning_rate":
            float(
                cfg.learning_rate
            ),

        "terrains": {},
    }

    all_deltas = {
        "motion": [],
        "stability": [],
        "energy": [],
    }

    for terrain in EXPECTED_TERRAINS:
        states = state_bank[
            terrain
        ]

        actions = {}

        for beta_name in (
            EXPECTED_BETA_NAMES
        ):
            actions[
                beta_name
            ] = deterministic_mean_actions(
                model,
                states,
                betas[
                    beta_name
                ],
            )

        balanced_action = actions[
            "balanced"
        ]

        terrain_result = {
            "state_count":
                int(
                    states.shape[0]
                ),

            "balanced_action": {
                "mean":
                    [
                        float(x)
                        for x
                        in np.mean(
                            balanced_action,
                            axis=0,
                        )
                    ],

                "std":
                    [
                        float(x)
                        for x
                        in np.std(
                            balanced_action,
                            axis=0,
                        )
                    ],

                "max_abs":
                    float(
                        np.max(
                            np.abs(
                                balanced_action
                            )
                        )
                    ),
            },

            "specialized_vs_balanced":
                {},

            "pairwise_beta_distance":
                pairwise_distance_summary(
                    actions
                ),
        }

        for beta_name in (
            "motion",
            "stability",
            "energy",
        ):
            delta = (
                actions[
                    beta_name
                ]
                - balanced_action
            )

            all_deltas[
                beta_name
            ].append(
                delta
            )

            terrain_result[
                "specialized_vs_balanced"
            ][
                beta_name
            ] = summarize_delta(
                delta
            )

        result[
            "terrains"
        ][
            terrain
        ] = terrain_result

        print(
            f"  {terrain:<13}"
        )

        for beta_name in (
            "motion",
            "stability",
            "energy",
        ):
            row = (
                terrain_result[
                    "specialized_vs_balanced"
                ][
                    beta_name
                ]
            )

            med = row[
                "delta_norm"
            ][
                "median"
            ]

            signed = row[
                "signed_delta_median"
            ]

            print(
                f"    {beta_name:<10} "
                f"|Δa|med={med:.6f} "
                f"Δ[vx,yaw,h]_med="
                f"[{signed[0]:+.5f},"
                f"{signed[1]:+.5f},"
                f"{signed[2]:+.5f}]"
            )

    aggregate = {}

    for beta_name in (
        "motion",
        "stability",
        "energy",
    ):
        delta = np.concatenate(
            all_deltas[
                beta_name
            ],
            axis=0,
        )

        aggregate[
            beta_name
        ] = summarize_delta(
            delta
        )

    result[
        "aggregate_all_terrains"
    ] = aggregate

    print(
        "  aggregate:"
    )

    for beta_name in (
        "motion",
        "stability",
        "energy",
    ):
        row = aggregate[
            beta_name
        ]

        print(
            f"    {beta_name:<10} "
            f"|Δa|med="
            f"{row['delta_norm']['median']:.6f} "
            f"q25="
            f"{row['delta_norm']['q25']:.6f} "
            f"q75="
            f"{row['delta_norm']['q75']:.6f}"
        )

    return result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoints",
        nargs="+",
        choices=tuple(
            grad_helper.CHECKPOINTS.keys()
        ),
        default=[
            "base_u30",
            "base_u40",
            "lrhalf_u20",
        ],
    )

    parser.add_argument(
        "--states-per-terrain",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--policy-episode-steps",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--out-dir",
        default=(
            "results/icra27/"
            "os_t4p5i_same_state_beta_sensitivity_v0"
        ),
    )

    args = parser.parse_args()

    if args.states_per_terrain <= 0:
        raise SystemExit(
            "--states-per-terrain must be > 0"
        )

    for checkpoint_name in (
        args.checkpoints
    ):
        checkpoint = (
            grad_helper.CHECKPOINTS[
                checkpoint_name
            ]
        )

        if not checkpoint.exists():
            raise RuntimeError(
                f"Missing checkpoint: {checkpoint}"
            )

    reference_path = (
        grad_helper.CHECKPOINTS[
            REFERENCE_CHECKPOINT
        ]
    )

    reference_model, _cfg, _payload = (
        grad_helper.load_checkpoint(
            reference_path
        )
    )

    out_dir = (
        ROOT
        / args.out_dir
    )

    if out_dir.exists():
        raise RuntimeError(
            "Output directory already exists; "
            "refusing overwrite: "
            f"{out_dir}"
        )

    out_dir.mkdir(
        parents=True
    )

    base_port = (
        grad_helper.choose_base_port()
    )

    print("=" * 112)
    print(
        "ICRA27 OS-T4.5i "
        "SAME-STATE BETA SENSITIVITY"
    )
    print("=" * 112)

    print(
        "state source : "
        "base_u30 / balanced / deterministic"
    )

    print(
        "scope        : TRAIN-only"
    )

    print(
        "state bank   : "
        f"{args.states_per_terrain} "
        "obs21 per terrain"
    )

    print(
        "comparison   : "
        "same obs21, beta tail changed only"
    )

    print(
        "policy output: "
        "deterministic normalized action3 "
        "[vx,yaw,h]"
    )

    (
        state_bank,
        state_bank_stats,
    ) = collect_common_state_bank(
        model=reference_model,

        states_per_terrain=(
            args.states_per_terrain
        ),

        policy_episode_steps=(
            args.policy_episode_steps
        ),

        base_port=base_port,

        out_dir=out_dir,
    )

    bank_path = (
        out_dir
        / "common_state_bank.npz"
    )

    np.savez_compressed(
        bank_path,
        **{
            terrain:
                state_bank[
                    terrain
                ]
            for terrain
            in EXPECTED_TERRAINS
        },
    )

    checkpoint_results = {}

    for checkpoint_name in (
        args.checkpoints
    ):
        checkpoint_results[
            checkpoint_name
        ] = evaluate_checkpoint(
            checkpoint_name=(
                checkpoint_name
            ),

            checkpoint_path=(
                grad_helper.CHECKPOINTS[
                    checkpoint_name
                ]
            ),

            state_bank=(
                state_bank
            ),
        )

    manifest = {
        "schema":
            "icra27_os_t4p5i_same_state_beta_sensitivity_v0",

        "scope":
            "TRAIN-only counterfactual same-state policy diagnostic",

        "state_source": {
            "checkpoint":
                REFERENCE_CHECKPOINT,

            "beta":
                "balanced",

            "policy":
                "deterministic",

            "states_per_terrain":
                int(
                    args.states_per_terrain
                ),

            "terrain_stats":
                state_bank_stats,
        },

        "state_bank":
            str(
                bank_path
            ),

        "observation_contract":
            (
                "same obs21; replace only appended "
                "beta_motion,beta_stability,beta_energy"
            ),

        "action_contract":
            (
                "deterministic normalized policy action3 "
                "[vx,yaw,h] = tanh(mu)"
            ),

        "beta_bank":
            {
                name:
                    [
                        float(x)
                        for x in beta
                    ]

                for name, beta
                in trainer.BETA_BANK
            },

        "checkpoints":
            checkpoint_results,
    }

    manifest_path = (
        out_dir
        / "same_state_beta_sensitivity_manifest.json"
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
        "state bank:",
        bank_path,
    )

    print(
        "manifest  :",
        manifest_path,
    )

    print()
    print(
        "[ICRA27] OS-T4.5i same-state "
        "beta sensitivity: PASS"
    )


if __name__ == "__main__":
    global trainer
    global grad_helper
    global ppo

    from tracer_core.highlevel_rl import (
        ppo as ppo_module
    )

    ppo = ppo_module

    trainer = load_module(
        TRAINER_PATH,
        "os_t4p5i_trainer",
    )

    grad_helper = load_module(
        GRAD_HELPER_PATH,
        "os_t4p5i_grad_helper",
    )

    grad_helper.ppo = ppo

    if tuple(
        trainer.TERRAINS
    ) != EXPECTED_TERRAINS:
        raise RuntimeError(
            "Unexpected terrain order: "
            f"{trainer.TERRAINS}"
        )

    beta_names = tuple(
        name
        for name, _beta
        in trainer.BETA_BANK
    )

    if beta_names != EXPECTED_BETA_NAMES:
        raise RuntimeError(
            "Unexpected beta order: "
            f"{beta_names}"
        )

    main()
