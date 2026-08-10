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
    load_ppo_checkpoint,
    ppo_update,
    save_ppo_checkpoint,
)

from tracer_core.highlevel_rl.pympc_env import (
    PyMPCM7Env,
)

from tracer_core.highlevel_rl.fixed_clearance_wrapper import (
    M7FixedClearanceActionWrapper,
)


SETTLING_STEPS = 5


def reset_with_settling(
    env,
    *,
    seed,
):
    """
    Reset the physical environment, then execute five nominal
    high-level decisions before exposing the state to PPO.

    Settling transitions are intentionally excluded from:
      - PPO rollout storage,
      - episode-return accounting,
      - GAE / return computation,
      - learned-policy step counting.
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
            raise RuntimeError(
                "M7 nominal settling terminated "
                "before PPO activation: "
                f"seed={seed}, "
                f"settling_step="
                f"{settling_step + 1}/"
                f"{SETTLING_STEPS}, "
                f"terminated={terminated}, "
                f"truncated={truncated}, "
                f"safety_state="
                f"{info.get('safety_state')}, "
                f"m4_terminal="
                f"{info.get('m4_terminal')}"
            )

    return obs, info


def evaluate_policy(
    env,
    *,
    model=None,
    random_policy=False,
    seed=1000,
):
    obs, _ = reset_with_settling(
        env,
        seed=seed
    )

    initial_obs = obs.copy()

    rng = np.random.default_rng(
        seed
    )

    total_reward = 0.0
    steps = 0
    interventions = 0
    success = False

    while True:
        if random_policy:
            action = rng.uniform(
                -1.0,
                1.0,
                size=env.action_space.shape,
            ).astype(
                np.float32
            )

        else:
            action, _, _ = model.act(
                obs,
                deterministic=True,
            )

        (
            obs,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(
            action
        )

        total_reward += float(
            reward
        )

        steps += 1

        interventions += int(
            bool(
                info[
                    "m4_intervention"
                ]
            )
        )

        success = bool(
            info["success"]
        )

        if (
            terminated
            or truncated
        ):
            break

    return {
        "return":
            float(total_reward),

        "steps":
            int(steps),

        "success":
            bool(success),

        "interventions":
            int(interventions),

        "initial_obs":
            initial_obs,
    }


def collect_rollout(
    env,
    *,
    model,
    cfg,
    num_steps,
    seed_base,
):
    buffer = PPORolloutBuffer()

    obs, _ = reset_with_settling(
        env,
        seed=seed_base
    )

    episode_return = 0.0
    completed_returns = []

    episode_counter = 0
    interventions = 0

    while len(buffer) < num_steps:
        (
            action,
            log_prob,
            value,
        ) = model.act(
            obs,
            deterministic=False,
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
                info[
                    "m4_intervention"
                ]
            )
        )

        # True termination: no bootstrap.
        #
        # Time-limit truncation: bootstrap from V(s').
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

        if (
            terminated
            or truncated
        ):
            completed_returns.append(
                float(
                    episode_return
                )
            )

            episode_return = 0.0

            episode_counter += 1

            if len(buffer) < num_steps:
                obs, _ = reset_with_settling(
                    env,
                    seed=(
                        seed_base
                        + episode_counter
                    )
                )

    # PPO update cannot propagate GAE beyond the current
    # rollout buffer even if the final episode is unfinished.
    buffer.close_rollout_boundary()

    batch = buffer.as_batch(
        cfg=cfg
    )

    return (
        batch,
        {
            "steps":
                int(len(buffer)),

            "completed_episode_returns":
                completed_returns,

            "interventions":
                int(interventions),
        },
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--updates",
        type=int,
        default=2,
    )

    ap.add_argument(
        "--steps-per-update",
        type=int,
        default=12,
    )

    ap.add_argument(
        "--max-episode-steps",
        type=int,
        default=10,
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    ap.add_argument(
        "--out-dir",
        default=(
            "results/icra27/"
            "m7_ppo_online_smoke"
        ),
    )

    args = ap.parse_args()

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

    base_env = PyMPCM7Env(
        terrain="flat",
        terminate_on_m4_unsafe=True,
        oracle_context=(
            1.0,
            0.0,
            0.0,
        ),

        goal_distance_m=0.50,
        success_radius_m=0.15,

        decision_dt_s=0.20,

        max_episode_steps=(
            args.max_episode_steps
            + SETTLING_STEPS
        ),

        command_port=50610,
        telemetry_port=50611,
        state_port=50612,

        command_repeat_hz=20.0,
        telemetry_hz=100.0,
        state_hz=100.0,

        log_dir=(
            out_dir
            / "env_logs"
        ),
    )

    env = M7FixedClearanceActionWrapper(
        base_env
    )

    cfg = PPOConfig(
        gamma=0.97,
        gae_lambda=0.95,

        clip_ratio=0.20,

        learning_rate=3e-4,

        value_coef=0.50,
        entropy_coef=0.01,

        update_epochs=4,
        minibatch_size=12,

        max_grad_norm=0.50,
    )

    model = PPOActorCritic(
        obs_dim=int(
            env.observation_space
            .shape[0]
        ),

        act_dim=int(
            env.action_space.shape[0]
        ),

        hidden_sizes=(
            128,
            128,
        ),

        initial_std=0.15,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.learning_rate,
    )

    summary = {
        "schema":
            "icra27_m7_online_ppo_smoke_v0",

        "ppo":
            "pure_pytorch_continuous_ppo",

        "updates":
            int(args.updates),

        "steps_per_update":
            int(
                args.steps_per_update
            ),

        "update_metrics":
            [],
    }

    try:
        print("=" * 72)
        print(
            "ICRA27 M7 ONLINE PPO SMOKE"
        )
        print("=" * 72)

        print(
            "policy action       : "
            "tanh Gaussian continuous 3D "
            "[vx, yaw, body_height]"
        )

        print(
            "initial action std  : 0.15"
        )

        print(
            "structural action   : "
            "fixed nominal"
        )

        print()

        # ----------------------------------------------------
        # Random baseline
        # ----------------------------------------------------
        random_eval = evaluate_policy(
            env,
            random_policy=True,
            seed=1000,
        )

        print(
            "random baseline     : "
            f"return="
            f"{random_eval['return']:+.4f} "
            f"steps="
            f"{random_eval['steps']} "
            f"interventions="
            f"{random_eval['interventions']}"
        )

        summary[
            "random_baseline"
        ] = {
            k: v
            for k, v
            in random_eval.items()
            if k != "initial_obs"
        }

        # ----------------------------------------------------
        # Initial deterministic policy
        # ----------------------------------------------------
        initial_eval = evaluate_policy(
            env,
            model=model,
            random_policy=False,
            seed=1001,
        )

        print(
            "initial policy      : "
            f"return="
            f"{initial_eval['return']:+.4f} "
            f"steps="
            f"{initial_eval['steps']} "
            f"interventions="
            f"{initial_eval['interventions']}"
        )

        summary[
            "initial_policy"
        ] = {
            k: v
            for k, v
            in initial_eval.items()
            if k != "initial_obs"
        }

        print()

        # ----------------------------------------------------
        # Online PPO
        # ----------------------------------------------------
        for update_i in range(
            args.updates
        ):
            (
                batch,
                rollout_info,
            ) = collect_rollout(
                env,
                model=model,
                cfg=cfg,

                num_steps=(
                    args.steps_per_update
                ),

                seed_base=(
                    args.seed
                    + 100
                    * (
                        update_i
                        + 1
                    )
                ),
            )

            metrics = ppo_update(
                model=model,
                optimizer=optimizer,
                batch=batch,
                cfg=cfg,
            )

            row = {
                "update":
                    update_i + 1,

                **metrics,

                **rollout_info,
            }

            summary[
                "update_metrics"
            ].append(
                row
            )

            returns = (
                rollout_info[
                    "completed_episode_returns"
                ]
            )

            mean_return = (
                float(
                    np.mean(
                        returns
                    )
                )
                if returns
                else float("nan")
            )

            print(
                f"update "
                f"{update_i + 1}/"
                f"{args.updates}: "
                f"rollout_mean="
                f"{mean_return:+.4f} "
                f"policy_loss="
                f"{metrics['policy_loss']:+.4f} "
                f"value_loss="
                f"{metrics['value_loss']:+.4f} "
                f"kl="
                f"{metrics['approx_kl']:+.5f} "
                f"interventions="
                f"{rollout_info['interventions']}"
            )

        # ----------------------------------------------------
        # Final deterministic evaluation
        # ----------------------------------------------------
        final_eval = evaluate_policy(
            env,
            model=model,
            random_policy=False,
            seed=1002,
        )

        print()
        print(
            "trained policy      : "
            f"return="
            f"{final_eval['return']:+.4f} "
            f"steps="
            f"{final_eval['steps']} "
            f"interventions="
            f"{final_eval['interventions']}"
        )

        summary[
            "trained_policy"
        ] = {
            k: v
            for k, v
            in final_eval.items()
            if k != "initial_obs"
        }

        improvement_vs_random = (
            float(
                final_eval["return"]
            )
            - float(
                random_eval["return"]
            )
        )

        summary[
            "improvement_vs_random"
        ] = float(
            improvement_vs_random
        )

        summary[
            "reward_improves_vs_random"
        ] = bool(
            improvement_vs_random
            > 0.0
        )

        # ----------------------------------------------------
        # Checkpoint
        # ----------------------------------------------------
        checkpoint = (
            out_dir
            / "m7_ppo_smoke.pt"
        )

        save_ppo_checkpoint(
            checkpoint,
            model=model,
            cfg=cfg,
            extra={
                "summary_without_arrays":
                    {
                        k: v
                        for k, v
                        in summary.items()
                    },
            },
        )

        (
            loaded_model,
            _,
            _,
        ) = load_ppo_checkpoint(
            checkpoint
        )

        reference_obs = (
            final_eval[
                "initial_obs"
            ]
        )

        action_a, _, _ = model.act(
            reference_obs,
            deterministic=True,
        )

        action_b, _, _ = (
            loaded_model.act(
                reference_obs,
                deterministic=True,
            )
        )

        reload_action_diff = float(
            np.max(
                np.abs(
                    action_a
                    - action_b
                )
            )
        )

        summary[
            "checkpoint"
        ] = {
            "path":
                str(checkpoint),

            "reload_action_max_diff":
                reload_action_diff,
        }

        if reload_action_diff > 1e-7:
            raise AssertionError(
                "checkpoint reload "
                "action mismatch: "
                f"{reload_action_diff}"
            )

        # Make JSON serializable.
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

        print()
        print(
            "continuous online PPO    : PASS"
        )

        print(
            "checkpoint save/reload   : PASS"
        )

        print(
            "deterministic inference  : PASS"
        )

        print(
            "reload action max diff   :",
            reload_action_diff,
        )

        print(
            "trained - random return  :",
            f"{improvement_vs_random:+.4f}",
        )

        print(
            "reward > random baseline :",
            (
                "YES"
                if improvement_vs_random > 0
                else "NO"
            ),
        )

        print()
        print(
            "[ICRA27] M7 online PPO "
            "training smoke: PASS"
        )

    finally:
        env.close()


if __name__ == "__main__":
    main()
