#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from isaaclab.app import AppLauncher


def main():
    parser = argparse.ArgumentParser()

    # AppLauncher adds --headless and --device.
    parser.add_argument("--num_envs", type=int, default=32)
    parser.add_argument("--num_steps_per_env", type=int, default=128)
    parser.add_argument("--updates", type=int, default=3)
    parser.add_argument("--minibatches", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)

    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_coef", type=float, default=0.2)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--ent_coef", type=float, default=0.005)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--init_log_std", type=float, default=-2.5)

    parser.add_argument("--eval_interval", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=160)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--save_dir", type=str, default="checkpoints/meta_gait_theta_ppo_v0")

    # Keep policy optimizer on CPU by default to avoid CUDA autograd/Isaac sync issues.
    parser.add_argument("--policy_device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--skip_eval", action="store_true")

    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.distributions import Normal

        torch.manual_seed(int(args_cli.seed))
        torch.set_num_threads(1)

        from isaaclab_tracer.envs.tracer_a1_adapter_env import make_tracer_a1_adapter_env_class

        TracerA1AdapterEnv, TracerA1AdapterEnvCfg = make_tracer_a1_adapter_env_class()

        cfg = TracerA1AdapterEnvCfg()
        cfg.scene.num_envs = int(args_cli.num_envs)
        cfg.sim.device = args_cli.device

        cfg.action_type = "meta_gait_theta"
        cfg.action_space = int(getattr(cfg, "meta_theta_dim", 6))
        cfg.ignore_adapter_done = True
        cfg.meta_debug = False

        cfg.use_external_lowlevel = False
        cfg.external_lowlevel_timeout_s = 0.20
        cfg.external_lowlevel_env_index = 0
        cfg.use_grf_torque = False
        cfg.use_nominal_gait = True
        cfg.hold_default_pose = False
        cfg.residual_scale = 0.0

        # V0: train only theta[0].
        cfg.meta_train_vx_only = True
        cfg.meta_theta_smoothing_alpha = 0.90

        env = TracerA1AdapterEnv(cfg)
        env_device = env.device

        obs_dict, _ = env.reset()
        obs_env = torch.nan_to_num(obs_dict["policy"].float())

        obs_dim = int(obs_env.shape[-1])
        env_action_dim = int(cfg.action_space)
        ppo_action_dim = 1

        if args_cli.policy_device == "cuda":
            policy_device = torch.device("cuda:0")
        else:
            policy_device = torch.device("cpu")

        save_dir = Path(args_cli.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        class ActorCritic(nn.Module):
            def __init__(self, obs_dim: int, action_dim: int, init_log_std: float):
                super().__init__()
                self.actor = nn.Sequential(
                    nn.Linear(obs_dim, 128),
                    nn.Tanh(),
                    nn.Linear(128, 128),
                    nn.Tanh(),
                    nn.Linear(128, action_dim),
                )
                self.critic = nn.Sequential(
                    nn.Linear(obs_dim, 128),
                    nn.Tanh(),
                    nn.Linear(128, 128),
                    nn.Tanh(),
                    nn.Linear(128, 1),
                )
                self.log_std = nn.Parameter(torch.ones(action_dim) * float(init_log_std))

                # Start exactly around theta=0 baseline.
                nn.init.zeros_(self.actor[-1].weight)
                nn.init.zeros_(self.actor[-1].bias)
                nn.init.zeros_(self.critic[-1].weight)
                nn.init.zeros_(self.critic[-1].bias)

            def dist_value(self, obs):
                mean = 0.75 * torch.tanh(self.actor(obs))
                std = torch.exp(self.log_std).expand_as(mean)
                dist = Normal(mean, std)
                value = self.critic(obs).squeeze(-1)
                return dist, value

            def act(self, obs):
                dist, value = self.dist_value(obs)
                action = dist.sample()
                action = torch.clamp(action, -1.0, 1.0)
                log_prob = dist.log_prob(action).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1)
                return action, log_prob, entropy, value

            def evaluate_actions(self, obs, action):
                dist, value = self.dist_value(obs)
                log_prob = dist.log_prob(action).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1)
                return log_prob, entropy, value

            def mean_action(self, obs):
                mean = 0.75 * torch.tanh(self.actor(obs))
                return torch.clamp(mean, -1.0, 1.0)

        ac = ActorCritic(obs_dim, ppo_action_dim, float(args_cli.init_log_std)).to(policy_device)
        optimizer = optim.Adam(ac.parameters(), lr=float(args_cli.lr), eps=1.0e-5)

        num_envs = int(args_cli.num_envs)
        horizon = int(args_cli.num_steps_per_env)
        batch_size = num_envs * horizon
        minibatch_size = max(1, batch_size // int(args_cli.minibatches))

        def expand_action_to_env(action_1d_cpu: torch.Tensor) -> torch.Tensor:
            action_1d_env = action_1d_cpu.to(env_device)
            action_full = torch.zeros(action_1d_env.shape[0], env_action_dim, device=env_device)
            action_full[:, 0] = action_1d_env[:, 0]
            return action_full

        @torch.no_grad()
        def evaluate(update_idx: int):
            eval_obs_dict, _ = env.reset()
            eval_obs_env = torch.nan_to_num(eval_obs_dict["policy"].float())

            x0 = env.robot.data.root_pos_w[:, 0].detach().clone()
            y0 = env.robot.data.root_pos_w[:, 1].detach().clone()
            min_h = env.robot.data.root_pos_w[:, 2].detach().clone()

            rewards = []
            done_count = 0

            for i in range(int(args_cli.eval_steps)):
                eval_obs_policy = eval_obs_env.detach().to(policy_device)
                a1_cpu = ac.mean_action(eval_obs_policy)
                a6_env = expand_action_to_env(a1_cpu)

                eval_obs_dict, rew_env, terminated, truncated, _info = env.step(a6_env)
                eval_obs_env = torch.nan_to_num(eval_obs_dict["policy"].float())

                rewards.append(rew_env.detach().cpu())
                min_h = torch.minimum(min_h, env.robot.data.root_pos_w[:, 2].detach())
                done_count += int(torch.logical_or(terminated, truncated).sum().item())

                if i == 0 or (i + 1) % 80 == 0 or i == int(args_cli.eval_steps) - 1:
                    print(f"[eval {update_idx:04d}] step {i + 1}/{int(args_cli.eval_steps)}", flush=True)

            dx = env.robot.data.root_pos_w[:, 0] - x0
            dy = env.robot.data.root_pos_w[:, 1] - y0
            rew_all = torch.stack(rewards, dim=0)

            final_obs_policy = eval_obs_env.detach().to(policy_device)
            mean_theta0 = ac.mean_action(final_obs_policy).mean().item()

            print(
                f"[eval {update_idx:04d}] "
                f"reward_mean={rew_all.mean().item():+.5f} "
                f"dx={dx.mean().item():+.5f} "
                f"dy={dy.mean().item():+.5f} "
                f"min_h={min_h.min().item():.5f} "
                f"final_h={env.robot.data.root_pos_w[:, 2].mean().item():.5f} "
                f"done_count={done_count} "
                f"mean_theta0={mean_theta0:+.4f} "
                f"std={torch.exp(ac.log_std).mean().item():.4f}",
                flush=True,
            )

        print("[PPO-V0] start", flush=True)
        print(f"  obs_dim={obs_dim} env_action_dim={env_action_dim} ppo_action_dim={ppo_action_dim}", flush=True)
        print(f"  env_device={env_device} policy_device={policy_device}", flush=True)
        print(f"  num_envs={num_envs} horizon={horizon} batch_size={batch_size}", flush=True)
        print(f"  save_dir={save_dir}", flush=True)

        for update in range(1, int(args_cli.updates) + 1):
            obs_buf = torch.zeros(horizon, num_envs, obs_dim, device=policy_device)
            act_buf = torch.zeros(horizon, num_envs, ppo_action_dim, device=policy_device)
            logp_buf = torch.zeros(horizon, num_envs, device=policy_device)
            rew_buf = torch.zeros(horizon, num_envs, device=policy_device)
            done_buf = torch.zeros(horizon, num_envs, device=policy_device)
            val_buf = torch.zeros(horizon, num_envs, device=policy_device)

            rollout_reward_sum = 0.0
            rollout_done_sum = 0

            for t in range(horizon):
                if t == 0 or (t + 1) % 32 == 0 or t == horizon - 1:
                    print(f"[rollout {update:04d}] step {t + 1}/{horizon}", flush=True)

                obs_policy = obs_env.detach().to(policy_device)
                obs_buf[t] = obs_policy

                with torch.no_grad():
                    action_1d_cpu, logp, _entropy, value = ac.act(obs_policy)

                if t == 0 or (t + 1) % 32 == 0 or t == horizon - 1:
                    print(
                        f"[rollout {update:04d}] action "
                        f"mean={action_1d_cpu.mean().item():+.4f} "
                        f"min={action_1d_cpu.min().item():+.4f} "
                        f"max={action_1d_cpu.max().item():+.4f}",
                        flush=True,
                    )

                action_6d_env = expand_action_to_env(action_1d_cpu)
                next_obs_dict, reward_env, terminated, truncated, _info = env.step(action_6d_env)
                next_obs_env = torch.nan_to_num(next_obs_dict["policy"].float())

                done_env = torch.logical_or(terminated, truncated)

                act_buf[t] = action_1d_cpu
                logp_buf[t] = logp
                rew_buf[t] = reward_env.detach().to(policy_device)
                done_buf[t] = done_env.float().detach().to(policy_device)
                val_buf[t] = value

                rollout_reward_sum += float(reward_env.mean().item())
                rollout_done_sum += int(done_env.sum().item())

                obs_env = next_obs_env

            print(f"[debug {update:04d}] rollout complete", flush=True)

            with torch.no_grad():
                obs_policy = obs_env.detach().to(policy_device)
                _, next_value = ac.dist_value(obs_policy)

            adv_buf = torch.zeros_like(rew_buf)
            last_gae = torch.zeros(num_envs, device=policy_device)

            for t in reversed(range(horizon)):
                if t == horizon - 1:
                    next_nonterminal = 1.0 - done_buf[t]
                    next_values = next_value
                else:
                    next_nonterminal = 1.0 - done_buf[t + 1]
                    next_values = val_buf[t + 1]

                delta = rew_buf[t] + float(args_cli.gamma) * next_values * next_nonterminal - val_buf[t]
                last_gae = delta + float(args_cli.gamma) * float(args_cli.gae_lambda) * next_nonterminal * last_gae
                adv_buf[t] = last_gae

            ret_buf = adv_buf + val_buf

            b_obs = obs_buf.reshape(batch_size, obs_dim)
            b_act = act_buf.reshape(batch_size, ppo_action_dim)
            b_logp = logp_buf.reshape(batch_size)
            b_adv = adv_buf.reshape(batch_size)
            b_ret = ret_buf.reshape(batch_size)

            b_adv = (b_adv - b_adv.mean()) / (b_adv.std(unbiased=False) + 1.0e-8)

            inds = torch.arange(batch_size, device=policy_device)

            pg_loss_acc = 0.0
            v_loss_acc = 0.0
            ent_acc = 0.0
            kl_acc = 0.0
            n_mb = 0

            print(f"[debug {update:04d}] PPO optimize start", flush=True)

            for epoch in range(int(args_cli.epochs)):
                perm = inds[torch.randperm(batch_size, device=policy_device)]
                for start in range(0, batch_size, minibatch_size):
                    mb_inds = perm[start:start + minibatch_size]

                    new_logp, entropy, new_value = ac.evaluate_actions(b_obs[mb_inds], b_act[mb_inds])
                    log_ratio = new_logp - b_logp[mb_inds]
                    ratio = torch.exp(log_ratio)

                    mb_adv = b_adv[mb_inds]
                    pg_loss1 = -mb_adv * ratio
                    pg_loss2 = -mb_adv * torch.clamp(
                        ratio,
                        1.0 - float(args_cli.clip_coef),
                        1.0 + float(args_cli.clip_coef),
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    v_loss = 0.5 * torch.square(new_value - b_ret[mb_inds]).mean()
                    entropy_loss = entropy.mean()
                    loss = pg_loss + float(args_cli.vf_coef) * v_loss - float(args_cli.ent_coef) * entropy_loss

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(ac.parameters(), float(args_cli.max_grad_norm))
                    optimizer.step()

                    with torch.no_grad():
                        approx_kl = ((ratio - 1.0) - log_ratio).mean()

                    pg_loss_acc += float(pg_loss.detach().cpu().item())
                    v_loss_acc += float(v_loss.detach().cpu().item())
                    ent_acc += float(entropy_loss.detach().cpu().item())
                    kl_acc += float(approx_kl.detach().cpu().item())
                    n_mb += 1

                print(f"[debug {update:04d}] epoch {epoch + 1}/{int(args_cli.epochs)} done", flush=True)

            print(
                f"[update {update:04d}] "
                f"rollout_rew_mean={rollout_reward_sum / max(horizon, 1):+.5f} "
                f"done_sum={rollout_done_sum} "
                f"pg={pg_loss_acc / max(n_mb, 1):+.5f} "
                f"vf={v_loss_acc / max(n_mb, 1):.5f} "
                f"ent={ent_acc / max(n_mb, 1):.5f} "
                f"kl={kl_acc / max(n_mb, 1):.6f} "
                f"log_std={ac.log_std.mean().item():+.4f}",
                flush=True,
            )

            if (not bool(args_cli.skip_eval)) and (
                update % int(args_cli.eval_interval) == 0 or update == 1
            ):
                evaluate(update)

            ckpt = {
                "update": update,
                "model_state_dict": ac.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "obs_dim": obs_dim,
                "ppo_action_dim": ppo_action_dim,
                "env_action_dim": env_action_dim,
                "args": vars(args_cli),
            }
            torch.save(ckpt, save_dir / "meta_gait_theta_ppo_v0_latest.pt")
            if update % max(1, int(args_cli.eval_interval)) == 0 or update == int(args_cli.updates):
                torch.save(ckpt, save_dir / f"meta_gait_theta_ppo_v0_update_{update:04d}.pt")

        env.close()

    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
