#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import socket
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


CHECKPOINTS = {
    "base_u30": (
        ROOT
        / "results"
        / "icra27"
        / "phase1a_beta_conditioned_v3_seed27027"
        / "checkpoint_update_0030.pt"
    ),

    "base_u40": (
        ROOT
        / "results"
        / "icra27"
        / "phase1a_beta_conditioned_v3_seed27027"
        / "checkpoint_update_0040.pt"
    ),

    "lrhalf_u20": (
        ROOT
        / "results"
        / "icra27"
        / "phase1a_beta_conditioned_v3_lr1p5e4_seed27027"
        / "checkpoint_update_0020.pt"
    ),
}


ROUGH_TRAIN_SEEDS = (
    13, 7, 15, 0, 5, 11,
    4, 25, 12, 6, 29, 18,
    10, 8, 22, 20, 28, 17,
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


def load_checkpoint(path):
    try:
        payload = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        payload = torch.load(
            path,
            map_location="cpu",
        )

    if payload.get("schema") != "icra27_m7_online_ppo_v0":
        raise RuntimeError(
            f"Unexpected checkpoint schema: {path}"
        )

    if int(payload["obs_dim"]) != 24:
        raise RuntimeError(
            f"Unexpected obs_dim in {path}"
        )

    if int(payload["act_dim"]) != 3:
        raise RuntimeError(
            f"Unexpected act_dim in {path}"
        )

    extra = payload.get(
        "extra",
        {},
    )

    if extra.get("reward_mode") != "tracer_cost_v3":
        raise RuntimeError(
            f"Wrong reward mode in {path}: "
            f"{extra.get('reward_mode')!r}"
        )

    cfg = ppo.PPOConfig(
        **payload["ppo_config"]
    )

    model = ppo.PPOActorCritic(
        obs_dim=int(
            payload["obs_dim"]
        ),
        act_dim=int(
            payload["act_dim"]
        ),
        hidden_sizes=tuple(
            int(x)
            for x
            in payload["hidden_sizes"]
        ),
        initial_std=0.15,
    )

    model.load_state_dict(
        payload["model_state_dict"],
        strict=True,
    )

    model.eval()

    return (
        model,
        cfg,
        payload,
    )


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
    # Each condition receives a separate 10-port slot.
    # Check the first three ports in each slot.
    offsets = []

    for condition_index in range(12):
        base = 10 * condition_index

        offsets.extend(
            (
                base,
                base + 1,
                base + 2,
            )
        )

    for candidate in range(
        60000,
        64001,
        500,
    ):
        if all(
            udp_bindable(candidate + x)
            for x in offsets
        ):
            return candidate

    raise RuntimeError(
        "No free UDP layout found."
    )


def actor_parameter_groups(model):
    body = list(
        model.body.parameters()
    )

    mu = list(
        model.mu_head.parameters()
    )

    actor = (
        body
        + mu
    )

    if not actor:
        raise RuntimeError(
            "Empty actor parameter list."
        )

    body_numel = sum(
        int(p.numel())
        for p in body
    )

    actor_numel = sum(
        int(p.numel())
        for p in actor
    )

    return (
        actor,
        body_numel,
        actor_numel,
    )


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


def policy_gradient_for_slice(
    model,
    cfg,
    *,
    obs,
    actions,
    old_log_probs,
    advantages,
    start,
    end,
    actor_params,
    body_numel,
):
    idx = slice(
        int(start),
        int(end),
    )

    new_log_prob, _, _ = (
        model.evaluate_actions(
            obs[idx],
            actions[idx],
        )
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
            - float(cfg.clip_ratio),
            1.0
            + float(cfg.clip_ratio),
        )
        * adv
    )

    policy_loss = -torch.min(
        unclipped,
        clipped,
    ).mean()

    grads = torch.autograd.grad(
        policy_loss,
        actor_params,
        retain_graph=False,
        create_graph=False,
        allow_unused=True,
    )

    vector = flatten_grads(
        grads,
        actor_params,
    )

    body_vector = vector[
        :body_numel
    ].clone()

    ratio_deviation = float(
        torch.max(
            torch.abs(
                ratio.detach()
                - 1.0
            )
        ).cpu()
    )

    return {
        "policy_loss":
            float(
                policy_loss.detach()
                .cpu()
                .item()
            ),

        "gradient":
            vector,

        "body_gradient":
            body_vector,

        "gradient_norm":
            float(
                torch.linalg.vector_norm(
                    vector
                )
            ),

        "body_gradient_norm":
            float(
                torch.linalg.vector_norm(
                    body_vector
                )
            ),

        "ratio_max_abs_dev":
            ratio_deviation,
    }


def aggregate_vectors(
    condition_results,
    *,
    beta_name,
    key,
):
    vectors = []

    for terrain in EXPECTED_TERRAINS:
        row = condition_results[
            (terrain, beta_name)
        ]

        vectors.append(
            row[key]
        )

    return torch.stack(
        vectors,
        dim=0,
    ).mean(
        dim=0
    )


def pair_dict(vectors):
    pairs = (
        ("motion", "stability"),
        ("motion", "energy"),
        ("stability", "energy"),
    )

    result = {}

    for a, b in pairs:
        result[
            f"{a}__{b}"
        ] = cosine(
            vectors[a],
            vectors[b],
        )

    return result


def summarize_values(values):
    clean = [
        float(x)
        for x in values
        if x is not None
    ]

    if not clean:
        return {
            "values": [],
            "median": None,
            "mean": None,
            "negative_fraction": None,
        }

    negative = sum(
        x < 0.0
        for x in clean
    )

    return {
        "values":
            clean,

        "median":
            float(
                statistics.median(
                    clean
                )
            ),

        "mean":
            float(
                statistics.mean(
                    clean
                )
            ),

        "negative_fraction":
            float(
                negative
                / len(clean)
            ),
    }


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
        load_checkpoint(
            checkpoint_path
        )
    )

    actor_params, body_numel, actor_numel = (
        actor_parameter_groups(
            model
        )
    )

    std = torch.exp(
        model.log_std.detach()
    ).cpu().numpy()

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
        "stored std :",
        [
            float(x)
            for x in std
        ],
    )
    print(
        "actor params:",
        actor_numel,
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

            condition_log_dir = (
                out_dir
                / checkpoint_name
                / (
                    f"repeat_"
                    f"{repeat_index:02d}"
                )
                / terrain
                / beta_name
            )

            # Same action-noise stream for all beta anchors
            # within a terrain/repeat.
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

                    log_dir=(
                        condition_log_dir
                    ),

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
                        ROUGH_TRAIN_SEEDS
                    )

                    seed_pool_offset = (
                        5
                        * int(
                            repeat_index
                        )
                    ) % len(
                        ROUGH_TRAIN_SEEDS
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
                f"{stats['success_episodes']:3d} "
                f"m4="
                f"{stats['m4_episodes']:3d}"
            )

    if len(buffer) != expected_steps:
        raise RuntimeError(
            "Probe buffer length mismatch: "
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

    # Replicate the historical second/global PPO normalization.
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

            result = (
                policy_gradient_for_slice(
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

                    start=start,
                    end=end,

                    actor_params=(
                        actor_params
                    ),

                    body_numel=(
                        body_numel
                    ),
                )
            )

            condition_results[
                (
                    terrain,
                    beta_name,
                )
            ] = result

    full_aggregate = {}
    body_aggregate = {}

    for beta_name in (
        "motion",
        "stability",
        "energy",
    ):
        full_aggregate[
            beta_name
        ] = aggregate_vectors(
            condition_results,
            beta_name=beta_name,
            key="gradient",
        )

        body_aggregate[
            beta_name
        ] = aggregate_vectors(
            condition_results,
            beta_name=beta_name,
            key="body_gradient",
        )

    aggregate_cos = pair_dict(
        full_aggregate
    )

    body_cos = pair_dict(
        body_aggregate
    )

    terrain_cos = {}

    for terrain in EXPECTED_TERRAINS:
        local_full = {
            beta_name:
                condition_results[
                    (
                        terrain,
                        beta_name,
                    )
                ][
                    "gradient"
                ]

            for beta_name in (
                "motion",
                "stability",
                "energy",
            )
        }

        local_body = {
            beta_name:
                condition_results[
                    (
                        terrain,
                        beta_name,
                    )
                ][
                    "body_gradient"
                ]

            for beta_name in (
                "motion",
                "stability",
                "energy",
            )
        }

        terrain_cos[
            terrain
        ] = {
            "full_actor":
                pair_dict(
                    local_full
                ),

            "body_only":
                pair_dict(
                    local_body
                ),
        }

    max_ratio_dev = max(
        row[
            "ratio_max_abs_dev"
        ]
        for row
        in condition_results.values()
    )

    print()
    print("aggregate actor cosine:")
    for key, value in (
        aggregate_cos.items()
    ):
        print(
            f"  {key:<24} "
            f"{value:+.6f}"
        )

    print(
        "aggregate body cosine:"
    )

    for key, value in (
        body_cos.items()
    ):
        print(
            f"  {key:<24} "
            f"{value:+.6f}"
        )

    print(
        "max |ratio-1|:",
        f"{max_ratio_dev:.3e}",
    )

    serial_condition = {}

    for (
        terrain,
        beta_name,
    ), row in (
        condition_results.items()
    ):
        serial_condition[
            f"{terrain}/{beta_name}"
        ] = {
            "policy_loss":
                row[
                    "policy_loss"
                ],

            "gradient_norm":
                row[
                    "gradient_norm"
                ],

            "body_gradient_norm":
                row[
                    "body_gradient_norm"
                ],

            "ratio_max_abs_dev":
                row[
                    "ratio_max_abs_dev"
                ],
        }

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

        "repeat_index":
            int(
                repeat_index
            ),

        "steps_per_condition":
            int(
                steps_per_condition
            ),

        "stored_policy_std":
            [
                float(x)
                for x in std
            ],

        "aggregate_cosine": {
            "full_actor":
                aggregate_cos,

            "body_only":
                body_cos,
        },

        "terrain_cosine":
            terrain_cos,

        "condition_gradient_diagnostics":
            serial_condition,

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
            CHECKPOINTS.keys()
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
            "os_t4p5g_beta_gradient_interference_v0"
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

    global trainer
    global ppo

    trainer = load_module(
        TRAINER_PATH,
        "os_t4p5g_trainer",
    )

    from tracer_core.highlevel_rl import ppo as ppo_module
    ppo = ppo_module

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

    for name in args.checkpoints:
        if not CHECKPOINTS[name].exists():
            raise RuntimeError(
                "Missing checkpoint: "
                f"{CHECKPOINTS[name]}"
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

    base_port = choose_base_port()

    print("=" * 112)
    print(
        "ICRA27 OS-T4.5g "
        "BETA ACTOR-GRADIENT INTERFERENCE DIAGNOSTIC"
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
        "scope       : TRAIN-only"
    )

    print(
        "gradient    : PPO actor surrogate only"
    )

    print(
        "parameters  : shared body + mu_head; "
        "value_head/log_std excluded"
    )

    print(
        "base port   :",
        base_port,
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
                        CHECKPOINTS[
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

    pair_names = (
        "motion__stability",
        "motion__energy",
        "stability__energy",
    )

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

        checkpoint_summary = {
            "full_actor": {},
            "body_only": {},
        }

        for group in (
            "full_actor",
            "body_only",
        ):
            for pair_name in (
                pair_names
            ):
                values = [
                    row[
                        "aggregate_cosine"
                    ][
                        group
                    ][
                        pair_name
                    ]

                    for row
                    in selected
                ]

                checkpoint_summary[
                    group
                ][
                    pair_name
                ] = summarize_values(
                    values
                )

        se_full = (
            checkpoint_summary[
                "full_actor"
            ][
                "stability__energy"
            ]
        )

        se_body = (
            checkpoint_summary[
                "body_only"
            ][
                "stability__energy"
            ]
        )

        # Predeclared descriptive conflict criterion.
        # This is not a formal statistical hypothesis test.
        full_conflict = (
            se_full["median"] is not None
            and se_full["median"] < 0.0
            and se_full[
                "negative_fraction"
            ] >= (2.0 / 3.0)
        )

        body_conflict = (
            se_body["median"] is not None
            and se_body["median"] < 0.0
            and se_body[
                "negative_fraction"
            ] >= (2.0 / 3.0)
        )

        checkpoint_summary[
            "stability_energy_conflict_flag"
        ] = {
            "full_actor":
                bool(
                    full_conflict
                ),

            "body_only":
                bool(
                    body_conflict
                ),
        }

        summary[
            checkpoint_name
        ] = checkpoint_summary

    manifest = {
        "schema":
            "icra27_os_t4p5g_beta_gradient_interference_v0",

        "scope":
            "TRAIN-only read-only actor-gradient diagnostic",

        "gradient_definition":
            (
                "PPO clipped actor surrogate gradient "
                "over shared body + mu_head; "
                "value_head and fixed exploration log_std excluded."
            ),

        "condition_order":
            [
                f"{terrain}/{beta}"
                for terrain
                in EXPECTED_TERRAINS
                for beta
                in EXPECTED_BETA_NAMES
            ],

        "steps_per_condition":
            int(
                args.steps_per_condition
            ),

        "repeats":
            int(
                args.repeats
            ),

        "checkpoints":
            {
                name:
                    str(
                        CHECKPOINTS[
                            name
                        ]
                    )
                for name
                in args.checkpoints
            },

        "descriptive_conflict_rule":
            (
                "median cosine < 0 and "
                "negative fraction >= 2/3 "
                "across repeats"
            ),

        "runs":
            runs,

        "summary":
            summary,
    }

    manifest_path = (
        out_dir
        / "gradient_interference_manifest.json"
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
        "GRADIENT INTERFERENCE SUMMARY"
    )
    print("=" * 112)

    for checkpoint_name in (
        args.checkpoints
    ):
        row = summary[
            checkpoint_name
        ]

        print()
        print(
            checkpoint_name
        )

        for group in (
            "full_actor",
            "body_only",
        ):
            print(
                f"  {group}:"
            )

            for pair_name in (
                pair_names
            ):
                x = row[
                    group
                ][
                    pair_name
                ]

                print(
                    f"    {pair_name:<22} "
                    f"median={x['median']:+.6f} "
                    f"neg_frac="
                    f"{x['negative_fraction']:.3f} "
                    f"values={x['values']}"
                )

        print(
            "  S/E conflict flag:",
            row[
                "stability_energy_conflict_flag"
            ],
        )

    print()
    print(
        "manifest:",
        manifest_path,
    )


if __name__ == "__main__":
    main()
