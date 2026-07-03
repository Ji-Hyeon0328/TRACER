#!/usr/bin/env python3
import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim


def load_json(p):
    with open(p, "r") as f:
        return json.load(f)


def save_json(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def load_jsonl(p):
    rows = []
    with open(p, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class PolicyValueNet(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden=64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.pi = nn.Linear(hidden, action_dim)
        self.v = nn.Linear(hidden, 1)

    def forward(self, obs):
        h = self.body(obs)
        return self.pi(h), self.v(h).squeeze(-1)


def build_spaces(rows):
    worlds = sorted(set(r["world_name"] for r in rows))
    actions = sorted(set(r["action_name"] for r in rows))
    world_to_i = {w: i for i, w in enumerate(worlds)}
    action_to_i = {a: i for i, a in enumerate(actions)}
    return worlds, actions, world_to_i, action_to_i


def obs_from_world(world, world_to_i):
    x = torch.zeros(len(world_to_i), dtype=torch.float32)
    x[world_to_i[world]] = 1.0
    return x


def normalize_rewards(rows):
    vals = torch.tensor([float(r["reach_reward"]) for r in rows], dtype=torch.float32)
    mean = vals.mean().item()
    std = vals.std(unbiased=False).item()
    std = max(std, 1e-6)
    return mean, std


def train_bc(model, rows, world_to_i, action_to_i, epochs, lr):
    opt = optim.Adam(model.parameters(), lr=lr)
    random.shuffle(rows)

    for ep in range(epochs):
        total = 0.0
        n = 0
        random.shuffle(rows)
        for r in rows:
            obs = obs_from_world(r["world_name"], world_to_i).unsqueeze(0)
            target = torch.tensor([action_to_i[r["action_name"]]], dtype=torch.long)
            weight = float(r.get("bc_weight", 1.0))

            logits, _ = model(obs)
            loss = nn.functional.cross_entropy(logits, target, reduction="none")
            loss = (loss * weight).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += float(loss.item())
            n += 1

        if ep == 0 or (ep + 1) % max(1, epochs // 5) == 0:
            print(f"[BC] epoch={ep+1}/{epochs} loss={total/max(n,1):.4f}")


def train_offline_ppo_like(model, rows, world_to_i, action_to_i, epochs, lr, clip_ratio, entropy_coef, value_coef):
    """
    Offline PPO-style sanity step over stored bandit samples.

    This is not full online PPO yet. It validates:
      - discrete theta-lite action policy
      - policy/value optimization path
      - warm-start dataset can train a world-conditioned policy
    """
    opt = optim.Adam(model.parameters(), lr=lr)
    reward_mean, reward_std = normalize_rewards(rows)

    # Snapshot old policy.
    obs_all = torch.stack([obs_from_world(r["world_name"], world_to_i) for r in rows])
    act_all = torch.tensor([action_to_i[r["action_name"]] for r in rows], dtype=torch.long)
    rew_all = torch.tensor([(float(r["reach_reward"]) - reward_mean) / reward_std for r in rows], dtype=torch.float32)

    with torch.no_grad():
        old_logits, _ = model(obs_all)
        old_dist = torch.distributions.Categorical(logits=old_logits)
        old_logp = old_dist.log_prob(act_all)

    for ep in range(epochs):
        logits, values = model(obs_all)
        dist = torch.distributions.Categorical(logits=logits)
        logp = dist.log_prob(act_all)
        entropy = dist.entropy().mean()

        # One-step offline advantage estimate.
        adv = rew_all - values.detach()
        ratio = torch.exp(logp - old_logp)
        unclipped = ratio * adv
        clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv
        policy_loss = -torch.min(unclipped, clipped).mean()
        value_loss = ((values - rew_all) ** 2).mean()
        loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

        opt.zero_grad()
        loss.backward()
        opt.step()

        if ep == 0 or (ep + 1) % max(1, epochs // 5) == 0:
            print(
                f"[offline-ppo-sanity] epoch={ep+1}/{epochs} "
                f"loss={loss.item():.4f} pi={policy_loss.item():.4f} "
                f"v={value_loss.item():.4f} ent={entropy.item():.4f}"
            )


def evaluate_policy(model, worlds, actions, world_to_i):
    out = {}
    with torch.no_grad():
        for w in worlds:
            obs = obs_from_world(w, world_to_i).unsqueeze(0)
            logits, value = model(obs)
            probs = torch.softmax(logits, dim=-1)[0]
            ranked = sorted(
                [
                    {
                        "action_name": actions[i],
                        "prob": float(probs[i].item()),
                    }
                    for i in range(len(actions))
                ],
                key=lambda x: x["prob"],
                reverse=True,
            )
            out[w] = {
                "value": float(value[0].item()),
                "ranked_actions": ranked,
            }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase_b_ppo_v0/ppo_theta_lite_discrete_config_v0.json")
    ap.add_argument("--out-dir", default="artifacts/phase_b_ppo_theta_lite_discrete_v0")
    ap.add_argument("--bc-epochs", type=int, default=20)
    ap.add_argument("--ppo-epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = load_json(args.config)
    rows = load_jsonl(cfg["warmstart"]["dataset_jsonl"])

    worlds, actions, world_to_i, action_to_i = build_spaces(rows)
    model = PolicyValueNet(obs_dim=len(worlds), action_dim=len(actions), hidden=64)

    print(json.dumps({
        "schema": "phase_b_ppo_theta_lite_discrete_train_start_v0",
        "num_rows": len(rows),
        "worlds": worlds,
        "actions": actions,
    }, indent=2))

    train_bc(model, rows, world_to_i, action_to_i, epochs=args.bc_epochs, lr=args.lr)

    train_offline_ppo_like(
        model,
        rows,
        world_to_i,
        action_to_i,
        epochs=args.ppo_epochs,
        lr=args.lr,
        clip_ratio=cfg["ppo"]["clip_ratio"],
        entropy_coef=cfg["ppo"]["entropy_coef"],
        value_coef=cfg["ppo"]["value_coef"],
    )

    eval_out = evaluate_policy(model, worlds, actions, world_to_i)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "worlds": worlds,
            "actions": actions,
            "world_to_i": world_to_i,
            "action_to_i": action_to_i,
            "config": cfg,
        },
        out_dir / "policy_value.pt",
    )

    save_json(out_dir / "policy_eval.json", {
        "schema": "phase_b_ppo_theta_lite_discrete_policy_eval_v0",
        "config": args.config,
        "eval": eval_out,
    })

    print(json.dumps(eval_out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
