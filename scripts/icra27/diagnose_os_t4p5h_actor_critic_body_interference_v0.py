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
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


GRAD_HELPER_PATH = (
    ROOT
    / "scripts"
    / "icra27"
    / "diagnose_os_t4p5g_beta_gradient_interference_v0.py"
)

TRAINER_PATH = (
    ROOT
    / "scripts"
    / "icra27"
    / "train_m7_ppo_beta_conditioned_v3.py"
)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Could not load module: {path}"
        )

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    return mod


def flatten_grads(
    grads,
    params,
):
    chunks = []

    for grad, param in zip(
        grads,
        params,
    ):
        if grad is None:
            chunks.append(
                torch.zeros_like(
                    param
                ).reshape(-1)
            )
        else:
            chunks.append(
                grad.detach().reshape(-1)
            )

    return torch.cat(
        chunks
    ).cpu()


def vector_norm(x):
    return float(
        torch.linalg.vector_norm(
            x
        )
    )


def cosine(a, b):
    a = torch.as_tensor(
        a,
        dtype=torch.float64,
    )

    b = torch.as_tensor(
        b,
        dtype=torch.float64,
    )

    na = torch.linalg.vector_norm(a)
    nb = torch.linalg.vector_norm(b)

    if (
        float(na) <= 1e-14
        or float(nb) <= 1e-14
    ):
        return None

    return float(
        torch.dot(a, b)
        / (na * nb)
    )


def interaction_metrics(
    actor_grad,
    critic_grad,
):
    actor = torch.as_tensor(
        actor_grad,
        dtype=torch.float64,
    )

    critic = torch.as_tensor(
        critic_grad,
        dtype=torch.float64,
    )

    actor_norm = torch.linalg.vector_norm(
        actor
    )

    critic_norm = torch.linalg.vector_norm(
        critic
    )

    if float(actor_norm) <= 1e-14:
        return {
            "cosine": None,
            "critic_actor_norm_ratio": None,
            "opposing_projection_ratio": None,
            "net_actor_alignment": None,
            "net_actor_norm_ratio": None,
        }

    cos = cosine(
        actor,
        critic,
    )

    rho = float(
        critic_norm
        / actor_norm
    )

    dot = torch.dot(
        actor,
        critic,
    )

    actor_sq = torch.dot(
        actor,
        actor,
    )

    opposing = max(
        0.0,
        float(
            -dot / actor_sq
        ),
    )

    net = (
        actor
        + critic
    )

    net_alignment = cosine(
        actor,
        net,
    )

    net_norm_ratio = float(
        torch.linalg.vector_norm(net)
        / actor_norm
    )

    return {
        "cosine":
            cos,

        "critic_actor_norm_ratio":
            rho,

        "opposing_projection_ratio":
            opposing,

        "net_actor_alignment":
            net_alignment,

        "net_actor_norm_ratio":
            net_norm_ratio,
    }


def gradient_for_slice(
    model,
    cfg,
    *,
    obs,
    actions,
    old_log_probs,
    advantages,
    returns,
    start,
    end,
    body_params,
):
    idx = slice(
        int(start),
        int(end),
    )

    (
        new_log_prob,
        _entropy_proxy,
        value,
    ) = model.evaluate_actions(
        obs[idx],
        actions[idx],
    )

    log_ratio = (
        new_log_prob
        - old_log_probs[idx]
    )

    ratio = torch.exp(
        log_ratio
    )

    adv = advantages[idx]

    unclipped = (
        ratio
        * adv
    )

    clipped = (
        torch.clamp(
            ratio,
            1.0
            - float(
                cfg.clip_ratio
            ),
            1.0
            + float(
                cfg.clip_ratio
            ),
        )
        * adv
    )

    policy_loss = -torch.min(
        unclipped,
        clipped,
    ).mean()

    value_loss = F.mse_loss(
        value,
        returns[idx],
    )

    weighted_value_loss = (
        float(
            cfg.value_coef
        )
        * value_loss
    )

    actor_grads = torch.autograd.grad(
        policy_loss,
        body_params,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )

    critic_grads = torch.autograd.grad(
        weighted_value_loss,
        body_params,
        retain_graph=False,
        create_graph=False,
        allow_unused=True,
    )

    actor_vector = flatten_grads(
        actor_grads,
        body_params,
    )

    critic_vector = flatten_grads(
        critic_grads,
        body_params,
    )

    metrics = interaction_metrics(
        actor_vector,
        critic_vector,
    )

    ratio_max_abs_dev = float(
        torch.max(
            torch.abs(
                ratio.detach()
                - 1.0
            )
        ).cpu()
    )

    return {
        "actor_gradient":
            actor_vector,

        "critic_gradient":
            critic_vector,

        "actor_gradient_norm":
            vector_norm(
                actor_vector
            ),

        "critic_gradient_norm":
            vector_norm(
                critic_vector
            ),

        "policy_loss":
            float(
                policy_loss.detach()
                .cpu()
                .item()
            ),

        "value_loss":
            float(
                value_loss.detach()
                .cpu()
                .item()
            ),

        "weighted_value_loss":
            float(
                weighted_value_loss.detach()
                .cpu()
                .item()
            ),

        "ratio_max_abs_dev":
            ratio_max_abs_dev,

        **metrics,
    }


def summarize_scalar(values):
    clean = [
        float(x)
        for x in values
        if x is not None
    ]

    if not clean:
        return {
            "values": [],
            "mean": None,
            "median": None,
            "negative_fraction": None,
        }

    return {
        "values":
            clean,

        "mean":
            float(
                statistics.mean(
                    clean
                )
            ),

        "median":
            float(
                statistics.median(
                    clean
                )
            ),

        "negative_fraction":
            float(
                sum(
                    x < 0.0
                    for x in clean
                )
                / len(clean)
            ),
    }


def aggregate_vector(
    condition_results,
    *,
    beta_name,
    key,
):
    vectors = []

    for terrain in EXPECTED_TERRAINS:
        vectors.append(
            condition_results[
                (
                    terrain,
                    beta_name,
                )
            ][
                key
            ]
        )

    return torch.stack(
        vectors,
        dim=0,
    ).mean(
        dim=0
    )


def run_probe(
    *,
    checkpoint_name,
    checkpoint_path,
    repeat_index,
    steps_per_condition,
    policy_episode_steps,
    base_port,
    out_dir,
):
    model, cfg, payload = (
        grad_helper.load_checkpoint(
            checkpoint_path
        )
    )

    if abs(
        float(
            cfg.entropy_coef
        )
    ) > 1e-12:
        raise RuntimeError(
            "This diagnostic assumes frozen "
            "entropy_coef=0 training."
        )

    body_params = list(
        model.body.parameters()
    )

    body_numel = sum(
        int(p.numel())
        for p in body_params
    )

    print()
    print("=" * 112)
    print(
        f"{checkpoint_name} "
        f"repeat={repeat_index}"
    )
    print("=" * 112)

    print(
        "checkpoint :",
        checkpoint_path,
    )

    print(
        "update     :",
        payload.get(
            "extra",
            {},
        ).get(
            "update"
        ),
    )

    print(
        "lr         :",
        cfg.learning_rate,
    )

    print(
        "value coef :",
        cfg.value_coef,
    )

    print(
        "body params:",
        body_numel,
    )

    buffer = ppo.PPORolloutBuffer()

    condition_stats = {}

    condition_count = (
        len(EXPECTED_TERRAINS)
        * len(EXPECTED_BETA_NAMES)
    )

    expected_steps = (
        condition_count
        * int(
            steps_per_condition
        )
    )

    for terrain_index, terrain in enumerate(
        trainer.TERRAINS
    ):
        condition_stats[
            terrain
        ] = {}

        for beta_index, (
            beta_name,
            beta,
        ) in enumerate(
            trainer.BETA_BANK
        ):
            condition_index = (
                terrain_index
                * len(
                    trainer.BETA_BANK
                )
                + beta_index
            )

            command_port = (
                base_port
                + 10
                * condition_index
            )

            log_dir = (
                out_dir
                / checkpoint_name
                / (
                    f"repeat_"
                    f"{repeat_index:02d}"
                )
                / terrain
                / beta_name
            )

            # Same action-noise stream across beta anchors
            # within the same terrain/repeat.
            noise_seed = (
                91027
                + 100000
                * int(
                    repeat_index
                )
                + 1000
                * terrain_index
            )

            torch.manual_seed(
                noise_seed
            )

            np.random.seed(
                noise_seed
            )

            env, _base_env = (
                trainer.make_env(
                    terrain=terrain,
                    beta=beta,

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

            try:
                seed_base = (
                    27027
                    + 100000
                    * int(
                        repeat_index
                    )
                    + 1000
                    * terrain_index
                )

                if terrain == "rough_perlin":
                    seed_pool = (
                        grad_helper
                        .ROUGH_TRAIN_SEEDS
                    )

                    seed_pool_offset = (
                        5
                        * int(
                            repeat_index
                        )
                    ) % len(
                        seed_pool
                    )

                else:
                    seed_pool = None
                    seed_pool_offset = 0

                stats = (
                    trainer.collect_terrain_segment(
                        env,

                        terrain=terrain,
                        model=model,
                        buffer=buffer,

                        num_steps=(
                            steps_per_condition
                        ),

                        seed_base=(
                            seed_base
                        ),

                        seed_pool=(
                            seed_pool
                        ),

                        seed_pool_offset=(
                            seed_pool_offset
                        ),
                    )
                )

            finally:
                env.close()

            condition_stats[
                terrain
            ][
                beta_name
            ] = stats

            print(
                f"  {terrain:<13} "
                f"{beta_name:<10} "
                f"steps={stats['steps']:4d} "
                f"episodes="
                f"{stats['completed_episodes']:3d} "
                f"success="
                f"{stats['success_episodes']:3d}"
            )

    if len(buffer) != expected_steps:
        raise RuntimeError(
            "Buffer length mismatch: "
            f"expected={expected_steps}, "
            f"actual={len(buffer)}"
        )

    batch = buffer.as_batch(
        cfg=cfg
    )

    batch, advantage_stats = (
        trainer.normalize_condition_advantages(
            batch,

            steps_per_condition=(
                steps_per_condition
            ),

            condition_count=(
                condition_count
            ),
        )
    )

    # Historical second/global advantage normalization.
    advantages = np.asarray(
        batch["advantages"],
        dtype=np.float32,
    ).copy()

    advantages = (
        advantages
        - advantages.mean()
    ) / (
        advantages.std()
        + 1e-8
    )

    obs = torch.as_tensor(
        batch["obs"],
        dtype=torch.float32,
    )

    actions = torch.as_tensor(
        batch["actions"],
        dtype=torch.float32,
    )

    old_log_probs = torch.as_tensor(
        batch["old_log_probs"],
        dtype=torch.float32,
    )

    advantages_t = torch.as_tensor(
        advantages,
        dtype=torch.float32,
    )

    returns_t = torch.as_tensor(
        batch["returns"],
        dtype=torch.float32,
    )

    condition_results = {}

    for terrain_index, terrain in enumerate(
        trainer.TERRAINS
    ):
        for beta_index, (
            beta_name,
            _beta,
        ) in enumerate(
            trainer.BETA_BANK
        ):
            condition_index = (
                terrain_index
                * len(
                    trainer.BETA_BANK
                )
                + beta_index
            )

            start = (
                condition_index
                * int(
                    steps_per_condition
                )
            )

            end = (
                start
                + int(
                    steps_per_condition
                )
            )

            result = gradient_for_slice(
                model,
                cfg,

                obs=obs,
                actions=actions,
                old_log_probs=(
                    old_log_probs
                ),
                advantages=(
                    advantages_t
                ),
                returns=(
                    returns_t
                ),

                start=start,
                end=end,

                body_params=(
                    body_params
                ),
            )

            condition_results[
                (
                    terrain,
                    beta_name,
                )
            ] = result

    beta_results = {}

    for beta_name in (
        EXPECTED_BETA_NAMES
    ):
        actor = aggregate_vector(
            condition_results,
            beta_name=beta_name,
            key="actor_gradient",
        )

        critic = aggregate_vector(
            condition_results,
            beta_name=beta_name,
            key="critic_gradient",
        )

        beta_results[
            beta_name
        ] = {
            **interaction_metrics(
                actor,
                critic,
            ),

            "actor_gradient_norm":
                vector_norm(
                    actor
                ),

            "critic_gradient_norm":
                vector_norm(
                    critic
                ),
        }

    serial_conditions = {}

    for (
        terrain,
        beta_name,
    ), result in (
        condition_results.items()
    ):
        serial_conditions[
            f"{terrain}/{beta_name}"
        ] = {
            key:
                value
            for key, value
            in result.items()
            if key not in (
                "actor_gradient",
                "critic_gradient",
            )
        }

    max_ratio_dev = max(
        result[
            "ratio_max_abs_dev"
        ]
        for result
        in condition_results.values()
    )

    print()
    print(
        "beta-aggregated actor/critic body geometry:"
    )

    for beta_name in (
        EXPECTED_BETA_NAMES
    ):
        row = beta_results[
            beta_name
        ]

        print(
            f"  {beta_name:<10} "
            f"cos={row['cosine']:+.6f} "
            f"rho={row['critic_actor_norm_ratio']:.3f} "
            f"kappa={row['opposing_projection_ratio']:.3f} "
            f"net_align={row['net_actor_alignment']:+.6f}"
        )

    print(
        "max |ratio-1|:",
        f"{max_ratio_dev:.3e}",
    )

    return {
        "checkpoint_name":
            checkpoint_name,

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

        "value_coef":
            float(
                cfg.value_coef
            ),

        "repeat_index":
            int(
                repeat_index
            ),

        "steps_per_condition":
            int(
                steps_per_condition
            ),

        "beta_aggregate":
            beta_results,

        "condition_geometry":
            serial_conditions,

        "condition_rollout_stats":
            condition_stats,

        "condition_advantage_stats":
            advantage_stats,

        "max_ratio_abs_deviation":
            max_ratio_dev,
    }


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
        "--steps-per-condition",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--policy-episode-steps",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--out-dir",
        default=(
            "results/icra27/"
            "os_t4p5h_actor_critic_body_interference_v0"
        ),
    )

    args = parser.parse_args()

    if args.steps_per_condition <= 0:
        raise SystemExit(
            "--steps-per-condition must be > 0"
        )

    if args.repeats <= 0:
        raise SystemExit(
            "--repeats must be > 0"
        )

    for name in args.checkpoints:
        path = (
            grad_helper.CHECKPOINTS[
                name
            ]
        )

        if not path.exists():
            raise RuntimeError(
                f"Missing checkpoint: {path}"
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
        "ICRA27 OS-T4.5h "
        "ACTOR-CRITIC SHARED-BODY INTERFERENCE"
    )
    print("=" * 112)

    print(
        "checkpoints :",
        args.checkpoints,
    )

    print(
        "steps/cond  :",
        args.steps_per_condition,
    )

    print(
        "repeats     :",
        args.repeats,
    )

    print(
        "gradient    : "
        "actor surrogate vs value_coef * critic MSE"
    )

    print(
        "parameters  : shared body only"
    )

    print(
        "scope       : TRAIN-only"
    )

    runs = []

    for checkpoint_name in (
        args.checkpoints
    ):
        for repeat_index in range(
            args.repeats
        ):
            runs.append(
                run_probe(
                    checkpoint_name=(
                        checkpoint_name
                    ),

                    checkpoint_path=(
                        grad_helper
                        .CHECKPOINTS[
                            checkpoint_name
                        ]
                    ),

                    repeat_index=(
                        repeat_index
                    ),

                    steps_per_condition=(
                        args.steps_per_condition
                    ),

                    policy_episode_steps=(
                        args.policy_episode_steps
                    ),

                    base_port=(
                        base_port
                    ),

                    out_dir=(
                        out_dir
                    ),
                )
            )

    summary = {}

    for checkpoint_name in (
        args.checkpoints
    ):
        selected = [
            row
            for row in runs
            if (
                row[
                    "checkpoint_name"
                ]
                == checkpoint_name
            )
        ]

        beta_summary = {}

        for beta_name in (
            EXPECTED_BETA_NAMES
        ):
            cosine_values = [
                row[
                    "beta_aggregate"
                ][
                    beta_name
                ][
                    "cosine"
                ]

                for row
                in selected
            ]

            rho_values = [
                row[
                    "beta_aggregate"
                ][
                    beta_name
                ][
                    "critic_actor_norm_ratio"
                ]

                for row
                in selected
            ]

            opposing_values = [
                row[
                    "beta_aggregate"
                ][
                    beta_name
                ][
                    "opposing_projection_ratio"
                ]

                for row
                in selected
            ]

            net_alignment_values = [
                row[
                    "beta_aggregate"
                ][
                    beta_name
                ][
                    "net_actor_alignment"
                ]

                for row
                in selected
            ]

            cos_summary = summarize_scalar(
                cosine_values
            )

            conflict_flag = (
                cos_summary["median"] is not None
                and cos_summary["median"] < 0.0
                and cos_summary[
                    "negative_fraction"
                ] >= (
                    2.0 / 3.0
                )
            )

            beta_summary[
                beta_name
            ] = {
                "cosine":
                    cos_summary,

                "critic_actor_norm_ratio":
                    summarize_scalar(
                        rho_values
                    ),

                "opposing_projection_ratio":
                    summarize_scalar(
                        opposing_values
                    ),

                "net_actor_alignment":
                    summarize_scalar(
                        net_alignment_values
                    ),

                "conflict_flag":
                    bool(
                        conflict_flag
                    ),
            }

        summary[
            checkpoint_name
        ] = beta_summary

    manifest = {
        "schema":
            "icra27_os_t4p5h_actor_critic_body_interference_v0",

        "scope":
            "TRAIN-only read-only actor-critic shared-body diagnostic",

        "actor_gradient":
            "PPO clipped actor surrogate gradient on shared body",

        "critic_gradient":
            (
                "value_coef * critic MSE gradient "
                "on the same shared body"
            ),

        "opposing_projection_ratio":
            (
                "max(0, -dot(g_actor,g_critic)"
                "/||g_actor||^2)"
            ),

        "descriptive_conflict_rule":
            (
                "median cosine < 0 and "
                "negative fraction >= 2/3 "
                "across repeats"
            ),

        "checkpoints":
            {
                name:
                    str(
                        grad_helper
                        .CHECKPOINTS[
                            name
                        ]
                    )
                for name
                in args.checkpoints
            },

        "steps_per_condition":
            int(
                args.steps_per_condition
            ),

        "repeats":
            int(
                args.repeats
            ),

        "runs":
            runs,

        "summary":
            summary,
    }

    manifest_path = (
        out_dir
        / "actor_critic_body_interference_manifest.json"
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
    print("=" * 112)
    print(
        "ACTOR-CRITIC INTERFERENCE SUMMARY"
    )
    print("=" * 112)

    for checkpoint_name in (
        args.checkpoints
    ):
        print()
        print(
            checkpoint_name
        )

        for beta_name in (
            EXPECTED_BETA_NAMES
        ):
            row = summary[
                checkpoint_name
            ][
                beta_name
            ]

            print(
                f"  {beta_name:<10} "
                f"cos_med="
                f"{row['cosine']['median']:+.6f} "
                f"neg_frac="
                f"{row['cosine']['negative_fraction']:.3f} "
                f"rho_med="
                f"{row['critic_actor_norm_ratio']['median']:.3f} "
                f"kappa_med="
                f"{row['opposing_projection_ratio']['median']:.3f} "
                f"net_align_med="
                f"{row['net_actor_alignment']['median']:+.6f} "
                f"conflict="
                f"{row['conflict_flag']}"
            )

    print()
    print(
        "manifest:",
        manifest_path,
    )


if __name__ == "__main__":
    global trainer
    global ppo
    global grad_helper
    global EXPECTED_TERRAINS
    global EXPECTED_BETA_NAMES

    from tracer_core.highlevel_rl import (
        ppo as ppo_module
    )

    ppo = ppo_module

    grad_helper = load_module(
        GRAD_HELPER_PATH,
        "os_t4p5h_grad_helper",
    )

    # grad-helper load_checkpoint() expects its own
    # module-global ppo symbol.
    grad_helper.ppo = ppo

    trainer = load_module(
        TRAINER_PATH,
        "os_t4p5h_trainer",
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

    if tuple(
        trainer.TERRAINS
    ) != EXPECTED_TERRAINS:
        raise RuntimeError(
            "Unexpected terrain order: "
            f"{trainer.TERRAINS}"
        )

    if tuple(
        name
        for name, _beta
        in trainer.BETA_BANK
    ) != EXPECTED_BETA_NAMES:
        raise RuntimeError(
            "Unexpected beta-bank order."
        )

    main()
