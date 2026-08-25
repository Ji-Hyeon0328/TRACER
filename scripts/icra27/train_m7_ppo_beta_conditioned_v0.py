#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


from tracer_core.highlevel_rl.ppo import (
    PPOActorCritic,
    PPOConfig,
    PPORolloutBuffer,
    ppo_update,
    save_ppo_checkpoint,
)

from tracer_core.highlevel_rl.pympc_env import (
    PyMPCM7Env,
)

from tracer_core.highlevel_rl.fixed_clearance_wrapper import (
    M7FixedClearanceActionWrapper,
)

from tracer_core.highlevel_rl.beta_conditioning_wrapper import (
    M7BetaConditioningWrapper,
)


TERRAINS = (
    "flat",
    "low_friction",
    "rough_perlin",
)


# Phase-1A semantic preference anchors.
#
# Every beta is trained on every terrain. Terrain and beta
# are intentionally NOT confounded.
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

SETTLING_STEPS = 5

BASE_OBS_DIM = 21
BETA_DIM = 3
OBS_DIM = 24

POLICY_ACTION_DIM = 3
DOWNSTREAM_ACTION_DIM = 4


DEFAULT_PERLIN_SPLIT_PATH = (
    ROOT
    / "configs"
    / "icra27"
    / "m7_perlin_seed_split_v0.json"
)


def load_perlin_train_seeds(
    path: Path,
):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing Perlin seed split: {path}"
        )

    data = json.loads(
        path.read_text()
    )

    schema = data.get(
        "schema"
    )

    if schema != (
        "icra27_m7_perlin_seed_split_v0"
    ):
        raise ValueError(
            "Unexpected Perlin split schema: "
            f"{schema!r}"
        )

    train = [
        int(x)
        for x in (
            data[
                "splits"
            ][
                "train"
            ]
        )
    ]

    validation = {
        int(x)
        for x in (
            data[
                "splits"
            ][
                "validation"
            ]
        )
    }

    test = {
        int(x)
        for x in (
            data[
                "splits"
            ][
                "test"
            ]
        )
    }

    hard = {
        int(x)
        for x in (
            data[
                "stress_banks"
            ][
                "hard"
            ]
        )
    }

    if not train:
        raise ValueError(
            "Perlin train seed bank is empty."
        )

    train_set = set(
        train
    )

    if len(train_set) != len(train):
        raise ValueError(
            "Duplicate Perlin training seeds."
        )

    if (
        train_set & validation
        or train_set & test
        or train_set & hard
    ):
        raise ValueError(
            "Perlin train seeds overlap "
            "validation/test/hard banks."
        )

    return train, data


def make_env(
    *,
    terrain,
    beta,
    policy_episode_steps,
    command_port,
    log_dir,
    reward_mode,
):
    base_env = PyMPCM7Env(
        terrain=terrain,
        reward_mode=reward_mode,
        tracer_beta=beta,
        terminate_on_m4_unsafe=True,

        goal_distance_m=2.0,
        success_radius_m=0.15,

        decision_dt_s=0.20,

        max_episode_steps=(
            policy_episode_steps
            + SETTLING_STEPS
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

    action_env = M7FixedClearanceActionWrapper(
        base_env
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

    if (
        env.observation_space.shape
        != (OBS_DIM,)
    ):
        raise RuntimeError(
            "Unexpected conditioned observation shape: "
            f"{env.observation_space.shape}"
        )

    if (
        env.action_space.shape
        != (POLICY_ACTION_DIM,)
    ):
        raise RuntimeError(
            "Unexpected policy action shape: "
            f"{env.action_space.shape}"
        )

    if (
        base_env.action_space.shape
        != (DOWNSTREAM_ACTION_DIM,)
    ):
        raise RuntimeError(
            "Unexpected downstream action shape: "
            f"{base_env.action_space.shape}"
        )

    return env, base_env


def reset_with_settling(
    env,
    *,
    seed,
):
    """
    Reset physical simulator and execute the characterized
    five nominal high-level settling decisions.

    These transitions are never inserted into PPO storage.
    """

    obs, info = env.reset(
        seed=seed
    )

    nominal_action = np.zeros(
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
            nominal_action
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


def collect_terrain_segment(
    env,
    *,
    terrain,
    model,
    buffer,
    num_steps,
    seed_base,
    seed_pool=None,
    seed_pool_offset=0,
):
    """
    Append exactly num_steps policy transitions from one terrain
    into a shared PPO buffer.

    At the end of the terrain segment, GAE recursion is explicitly
    closed before another terrain is appended.
    """

    segment_start = len(buffer)

    target_len = (
        segment_start
        + int(num_steps)
    )

    reset_attempts = 0
    settling_failures = 0
    consecutive_settling_failures = 0

    interventions = 0

    completed_returns = []

    completed_episodes = 0
    success_episodes = 0
    m4_episodes = 0
    truncated_episodes = 0
    other_terminal_episodes = 0

    max_abs_action = 0.0

    obs = None
    episode_return = 0.0

    while len(buffer) < target_len:

        # ----------------------------------------------------
        # Start a physical episode.
        #
        # If nominal settling itself fails on a particular
        # geometry, discard that reset completely and try the
        # next seed. No PPO sample is created.
        # ----------------------------------------------------
        if obs is None:
            while True:
                if seed_pool is None:
                    episode_seed = (
                        int(seed_base)
                        + reset_attempts
                    )

                else:
                    if not seed_pool:
                        raise RuntimeError(
                            "Empty seed_pool supplied "
                            f"for terrain={terrain}"
                        )

                    seed_index = (
                        int(seed_pool_offset)
                        + reset_attempts
                    ) % len(seed_pool)

                    episode_seed = int(
                        seed_pool[
                            seed_index
                        ]
                    )

                reset_attempts += 1

                (
                    candidate_obs,
                    info,
                    settled,
                    _settling_steps,
                ) = reset_with_settling(
                    env,
                    seed=episode_seed,
                )

                if settled:
                    obs = candidate_obs
                    consecutive_settling_failures = 0
                    break

                settling_failures += 1
                consecutive_settling_failures += 1

                if (
                    consecutive_settling_failures
                    >= 20
                ):
                    raise RuntimeError(
                        "Too many consecutive "
                        "settling failures: "
                        f"terrain={terrain}, "
                        f"seed={episode_seed}"
                    )

            episode_return = 0.0

        # ----------------------------------------------------
        # Learned 3-D policy transition.
        # ----------------------------------------------------
        (
            action,
            log_prob,
            value,
        ) = model.act(
            obs,
            deterministic=False,
        )

        action = np.asarray(
            action,
            dtype=np.float32,
        )

        if action.shape != (
            POLICY_ACTION_DIM,
        ):
            raise RuntimeError(
                "Unexpected PPO action shape: "
                f"{action.shape}"
            )

        max_abs_action = max(
            max_abs_action,
            float(
                np.max(
                    np.abs(action)
                )
            ),
        )

        (
            next_obs,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(
            action
        )

        interventions += int(
            bool(
                info.get(
                    "m4_intervention",
                    False,
                )
            )
        )

        # True terminal:
        #   no value bootstrap.
        #
        # Time limit or artificial rollout boundary:
        #   bootstrap from V(s').
        if terminated:
            next_value = 0.0

        else:
            next_value = model.value(
                next_obs
            )

        continuation = (
            0.0
            if (
                terminated
                or truncated
            )
            else 1.0
        )

        buffer.add(
            obs=obs,
            action=action,
            reward=reward,
            value=value,
            log_prob=log_prob,
            next_value=next_value,
            continuation=continuation,
        )

        episode_return += float(
            reward
        )

        obs = next_obs

        # ----------------------------------------------------
        # Native episode boundary.
        # ----------------------------------------------------
        if terminated or truncated:
            completed_episodes += 1

            completed_returns.append(
                float(
                    episode_return
                )
            )

            if bool(
                info.get(
                    "success",
                    False,
                )
            ):
                success_episodes += 1

            elif bool(
                info.get(
                    "m4_intervention",
                    False,
                )
            ):
                m4_episodes += 1

            elif truncated:
                truncated_episodes += 1

            else:
                other_terminal_episodes += 1

            obs = None
            episode_return = 0.0

    # --------------------------------------------------------
    # Critical:
    # prevent GAE recursion from this terrain into the next
    # terrain while retaining the already-computed next_value
    # bootstrap for the final transition.
    # --------------------------------------------------------
    buffer.close_rollout_boundary()

    if not buffer.continuations:
        raise RuntimeError(
            "Buffer unexpectedly empty "
            "after terrain segment."
        )

    if not np.isclose(
        buffer.continuations[-1],
        0.0,
    ):
        raise RuntimeError(
            "Terrain rollout boundary "
            "was not closed."
        )

    segment_steps = (
        len(buffer)
        - segment_start
    )

    if segment_steps != num_steps:
        raise RuntimeError(
            "Terrain segment length mismatch: "
            f"terrain={terrain}, "
            f"expected={num_steps}, "
            f"actual={segment_steps}"
        )

    mean_completed_return = (
        float(
            np.mean(
                completed_returns
            )
        )
        if completed_returns
        else None
    )

    return {
        "terrain":
            terrain,

        "steps":
            int(segment_steps),

        "reset_attempts":
            int(reset_attempts),

        "settling_failures":
            int(settling_failures),

        "interventions":
            int(interventions),

        "completed_episodes":
            int(completed_episodes),

        "success_episodes":
            int(success_episodes),

        "m4_episodes":
            int(m4_episodes),

        "truncated_episodes":
            int(truncated_episodes),

        "other_terminal_episodes":
            int(other_terminal_episodes),

        "mean_completed_return":
            mean_completed_return,

        "max_abs_action":
            float(max_abs_action),
    }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--updates",
        type=int,
        default=2,
    )

    ap.add_argument(
        "--steps-per-condition",
        type=int,
        default=64,
    )

    ap.add_argument(
        "--policy-episode-steps",
        type=int,
        default=50,
    )

    ap.add_argument(
        "--minibatch-size",
        type=int,
        default=64,
    )

    ap.add_argument(
        "--initial-std",
        type=float,
        default=0.15,
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    ap.add_argument(
        "--base-port",
        type=int,
        default=51010,
    )

    ap.add_argument(
        "--perlin-seed-split",
        default=str(
            DEFAULT_PERLIN_SPLIT_PATH
        ),
    )

    ap.add_argument(
        "--reward-mode",
        choices=(
            "tracer_cost_v2",
        ),
        default="tracer_cost_v2",
    )

    ap.add_argument(
        "--out-dir",
        default=(
            "results/icra27/"
            "m7_ppo_beta_conditioned_v0"
        ),
    )

    args = ap.parse_args()

    if args.updates <= 0:
        raise ValueError(
            "--updates must be positive"
        )

    if args.steps_per_condition <= 0:
        raise ValueError(
            "--steps-per-condition "
            "must be positive"
        )

    if args.policy_episode_steps <= 0:
        raise ValueError(
            "--policy-episode-steps "
            "must be positive"
        )

    if args.minibatch_size <= 0:
        raise ValueError(
            "--minibatch-size "
            "must be positive"
        )

    perlin_split_path = Path(
        args.perlin_seed_split
    )

    (
        perlin_train_seeds,
        perlin_split_data,
    ) = load_perlin_train_seeds(
        perlin_split_path
    )

    np.random.seed(
        args.seed
    )

    torch.manual_seed(
        args.seed
    )

    out_dir = Path(
        args.out_dir
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cfg = PPOConfig(
        gamma=0.97,
        gae_lambda=0.95,

        clip_ratio=0.20,

        learning_rate=3e-4,

        value_coef=0.50,
        entropy_coef=0.01,

        update_epochs=4,
        minibatch_size=(
            args.minibatch_size
        ),

        max_grad_norm=0.50,
    )

    model = PPOActorCritic(
        obs_dim=OBS_DIM,
        act_dim=POLICY_ACTION_DIM,

        hidden_sizes=(
            128,
            128,
        ),

        initial_std=(
            args.initial_std
        ),
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.learning_rate,
    )

    steps_per_update = (
        len(TERRAINS)
        * len(BETA_BANK)
        * args.steps_per_condition
    )

    summary = {
        "schema":
            "icra27_m7_ppo_beta_conditioned_v0",

        "ppo":
            "pure_pytorch_continuous_ppo",

        "terrain_order":
            list(TERRAINS),

        "beta_bank":
            {
                name: [
                    float(x)
                    for x in beta
                ]
                for name, beta
                in BETA_BANK
            },

        "condition_order":
            [
                f"{terrain}/{name}"
                for terrain in TERRAINS
                for name, _beta in BETA_BANK
            ],

        "reward_mode":
            args.reward_mode,

        "perlin_seed_split":
            str(
                perlin_split_path
            ),

        "perlin_train_seeds":
            list(
                perlin_train_seeds
            ),

        "perlin_split_schema":
            perlin_split_data.get(
                "schema"
            ),

        "obs_dim":
            OBS_DIM,

        "policy_action_dim":
            POLICY_ACTION_DIM,

        "downstream_action_dim":
            DOWNSTREAM_ACTION_DIM,

        "fixed_swing_clearance_m":
            0.060,

        "settling_steps":
            SETTLING_STEPS,

        "initial_std":
            float(
                args.initial_std
            ),

        "updates":
            int(args.updates),

        "steps_per_condition":
            int(
                args.steps_per_condition
            ),

        "steps_per_update":
            int(
                steps_per_update
            ),

        "policy_episode_steps":
            int(
                args.policy_episode_steps
            ),

        "update_metrics":
            [],
    }

    print("=" * 88)
    print(
        "ICRA27 M7 BETA-CONDITIONED MULTI-TERRAIN PPO"
    )
    print("=" * 88)

    print(
        "terrains           : "
        + ", ".join(TERRAINS)
    )

    print(
        "beta anchors       : "
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
        "observation        : "
        f"{OBS_DIM}D = obs21 + beta3"
    )

    print(
        "policy action      : "
        "3D [vx, yaw_rate, body_height]"
    )

    print(
        "fixed clearance    : "
        "0.060 m"
    )

    print(
        "initial std        : "
        f"{args.initial_std:.3f}"
    )

    print(
        "settling           : "
        f"{SETTLING_STEPS} "
        "policy-excluded steps"
    )

    print(
        "reward mode        : "
        f"{args.reward_mode}"
    )

    print(
        "Perlin train seeds : "
        f"{perlin_train_seeds}"
    )

    print(
        "Perlin train count : "
        f"{len(perlin_train_seeds)}"
    )

    print(
        "steps / condition  : "
        f"{args.steps_per_condition}"
    )

    print(
        "steps / update     : "
        f"{steps_per_update}"
    )

    print()

    for update_i in range(
        args.updates
    ):
        buffer = PPORolloutBuffer()

        condition_stats = {}

        print(
            f"update "
            f"{update_i + 1}/"
            f"{args.updates}"
        )

        # ----------------------------------------------------
        # Sequential collection is intentional:
        # rough_perlin currently uses a shared heightfield PNG.
        # ----------------------------------------------------
        for terrain_index, terrain in enumerate(
            TERRAINS
        ):
            condition_stats[
                terrain
            ] = {}

            for beta_index, (
                beta_name,
                beta,
            ) in enumerate(
                BETA_BANK
            ):
                condition_index = (
                    terrain_index
                    * len(BETA_BANK)
                    + beta_index
                )

                command_port = (
                    args.base_port
                    + 10 * condition_index
                )

                log_dir = (
                    out_dir
                    / "env_logs"
                    / (
                        f"update_"
                        f"{update_i + 1:04d}"
                    )
                    / terrain
                    / beta_name
                )

                env, base_env = make_env(
                    terrain=terrain,
                    beta=beta,

                    policy_episode_steps=(
                        args.policy_episode_steps
                    ),

                    command_port=command_port,
                    log_dir=log_dir,

                    reward_mode=(
                        args.reward_mode
                    ),
                )

                try:
                    # Paired seed schedule:
                    # beta does NOT enter this expression.
                    seed_base = (
                        int(args.seed)
                        + 10000 * update_i
                        + 1000 * terrain_index
                    )

                    if terrain == "rough_perlin":
                        seed_pool = (
                            perlin_train_seeds
                        )

                        # Same rough realization schedule
                        # for all beta anchors in an update.
                        seed_pool_offset = (
                            5 * update_i
                        ) % len(
                            perlin_train_seeds
                        )

                    else:
                        seed_pool = None
                        seed_pool_offset = 0

                    stats = collect_terrain_segment(
                        env,
                        terrain=terrain,

                        model=model,
                        buffer=buffer,

                        num_steps=(
                            args.steps_per_condition
                        ),

                        seed_base=seed_base,

                        seed_pool=seed_pool,
                        seed_pool_offset=(
                            seed_pool_offset
                        ),
                    )

                finally:
                    env.close()

                stats[
                    "beta_name"
                ] = beta_name

                stats[
                    "beta"
                ] = [
                    float(x)
                    for x in beta
                ]

                condition_stats[
                    terrain
                ][
                    beta_name
                ] = stats

                mean_return = (
                    "n/a"
                    if (
                        stats[
                            "mean_completed_return"
                        ]
                        is None
                    )
                    else (
                        f"{stats['mean_completed_return']:+.4f}"
                    )
                )

                print(
                    f"  {terrain:<13} "
                    f"{beta_name:<10} "
                    f"beta="
                    f"[{beta[0]:.2f},"
                    f"{beta[1]:.2f},"
                    f"{beta[2]:.2f}] "
                    f"steps="
                    f"{stats['steps']:4d} "
                    f"episodes="
                    f"{stats['completed_episodes']:3d} "
                    f"success="
                    f"{stats['success_episodes']:3d} "
                    f"m4="
                    f"{stats['m4_episodes']:3d} "
                    f"settle_fail="
                    f"{stats['settling_failures']:3d} "
                    f"mean_return="
                    f"{mean_return} "
                    f"amax="
                    f"{stats['max_abs_action']:.3f}"
                )

        if len(buffer) != steps_per_update:
            raise RuntimeError(
                "Shared rollout buffer length "
                "mismatch: "
                f"expected={steps_per_update}, "
                f"actual={len(buffer)}"
            )

        batch = buffer.as_batch(
            cfg=cfg
        )

        if batch["obs"].shape != (
            steps_per_update,
            OBS_DIM,
        ):
            raise RuntimeError(
                "Unexpected observation batch "
                f"shape: {batch['obs'].shape}"
            )

        if batch["actions"].shape != (
            steps_per_update,
            POLICY_ACTION_DIM,
        ):
            raise RuntimeError(
                "Unexpected action batch "
                f"shape: {batch['actions'].shape}"
            )

        metrics = ppo_update(
            model=model,
            optimizer=optimizer,
            batch=batch,
            cfg=cfg,
        )

        row = {
            "update":
                int(update_i + 1),

            "steps":
                int(len(buffer)),

            "batch_obs_shape":
                list(
                    batch["obs"].shape
                ),

            "batch_action_shape":
                list(
                    batch["actions"].shape
                ),

            "terrains":
                condition_stats,

            **{
                k: float(v)
                for k, v
                in metrics.items()
            },
        }

        summary[
            "update_metrics"
        ].append(
            row
        )

        print(
            "  PPO             "
            f"batch_obs="
            f"{batch['obs'].shape} "
            f"batch_action="
            f"{batch['actions'].shape} "
            f"policy_loss="
            f"{metrics['policy_loss']:+.4f} "
            f"value_loss="
            f"{metrics['value_loss']:+.4f} "
            f"kl="
            f"{metrics['approx_kl']:+.5f}"
        )

        checkpoint_update = (
            out_dir
            / (
                "checkpoint_update_"
                f"{update_i + 1:04d}.pt"
            )
        )

        checkpoint = (
            out_dir
            / "m7_ppo_beta_conditioned_latest.pt"
        )

        checkpoint_extra = {
            "update":
                int(update_i + 1),

            "terrain_order":
                list(TERRAINS),

            "reward_mode":
                args.reward_mode,

            "condition_stats":
                condition_stats,

            "beta_bank":
                {
                    name: [
                        float(x)
                        for x in beta
                    ]
                    for name, beta
                    in BETA_BANK
                },

            "condition_order":
                [
                    f"{terrain}/{name}"
                    for terrain in TERRAINS
                    for name, _beta in BETA_BANK
                ],

            "perlin_seed_split":
                str(
                    perlin_split_path
                ),

            "perlin_train_seeds":
                list(
                    perlin_train_seeds
                ),
        }

        save_ppo_checkpoint(
            checkpoint_update,
            model=model,
            cfg=cfg,
            extra=checkpoint_extra,
        )

        save_ppo_checkpoint(
            checkpoint,
            model=model,
            cfg=cfg,
            extra=checkpoint_extra,
        )

        summary_path = (
            out_dir
            / "summary.json"
        )

        with open(
            summary_path,
            "w",
        ) as f:
            json.dump(
                summary,
                f,
                indent=2,
                sort_keys=True,
            )

        print(
            "  checkpoint      :",
            checkpoint_update,
        )

        print(
            "  latest          :",
            checkpoint,
        )

        print()

    final_checkpoint = (
        out_dir
        / "m7_ppo_beta_conditioned_final.pt"
    )

    save_ppo_checkpoint(
        final_checkpoint,
        model=model,
        cfg=cfg,
        extra={
            "terrain_order":
                list(TERRAINS),

            "updates":
                int(args.updates),

            "beta_bank":
                {
                    name: [
                        float(x)
                        for x in beta
                    ]
                    for name, beta
                    in BETA_BANK
                },

            "condition_order":
                [
                    f"{terrain}/{name}"
                    for terrain in TERRAINS
                    for name, _beta in BETA_BANK
                ],

            "reward_mode":
                args.reward_mode,

            "perlin_seed_split":
                str(
                    perlin_split_path
                ),

            "perlin_train_seeds":
                list(
                    perlin_train_seeds
                ),
        },
    )

    print("=" * 88)
    print(
        "[ICRA27] beta-conditioned multi-terrain "
        "PPO training: PASS"
    )
    print("=" * 88)

    print(
        "final checkpoint:",
        final_checkpoint,
    )


if __name__ == "__main__":
    main()
