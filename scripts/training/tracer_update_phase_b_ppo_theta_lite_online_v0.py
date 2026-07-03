#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim


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


def load_json(p):
    with open(p, "r") as f:
        return json.load(f)


def save_json(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def obs_from_world(world, world_to_i):
    x = torch.zeros(len(world_to_i), dtype=torch.float32)
    x[world_to_i[world]] = 1.0
    return x


def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    worlds = ckpt["worlds"]
    actions = ckpt["actions"]
    world_to_i = ckpt["world_to_i"]
    action_to_i = ckpt["action_to_i"]

    model = PolicyValueNet(obs_dim=len(worlds), action_dim=len(actions), hidden=64)
    model.load_state_dict(ckpt["model_state_dict"])
    model.train()
    return model, worlds, actions, world_to_i, action_to_i, ckpt


def collect_rows(rollout_root, world_to_i, action_to_i):
    root = Path(rollout_root)
    paths = sorted(root.glob("**/ppo_policy_episode_result_v0.json"))

    rows = []
    skipped = {}

    def skip(k):
        skipped[k] = skipped.get(k, 0) + 1

    for p in paths:
        try:
            r = load_json(p)
        except Exception:
            skip("json_load_failed")
            continue

        if r.get("status") != "ok":
            skip("not_ok")
            continue

        world = r.get("world_name")
        action = r.get("selected_action_name")
        if world not in world_to_i:
            skip("unknown_world")
            continue
        if action not in action_to_i:
            skip("unknown_action")
            continue

        c = r.get("reward_components") or {}
        sel = r.get("selection") or {}

        old_prob = float(sel.get("prob", 0.0))
        if old_prob <= 0:
            skip("missing_old_prob")
            continue

        reach_reward = float(c.get("reach_reward", 0.0))
        hold_reward = float(c.get("hold_reward", 0.0))

        rows.append({
            "source": str(p),
            "world_name": world,
            "action_name": action,
            "action_index": int(action_to_i[action]),
            "old_logp": math.log(max(old_prob, 1e-12)),
            "reach_reward": reach_reward,
            "hold_reward": hold_reward,
            "reached": bool(c.get("reached_stop_distance", False)),
            "min_rel_dist": float(c.get("min_rel_dist", 999.0)),
            "final_rel_dist": float(c.get("final_rel_dist", 999.0)),
        })

    return rows, skipped


def eval_policy(model, worlds, actions, world_to_i):
    model.eval()
    out = {}
    with torch.no_grad():
        for w in worlds:
            obs = obs_from_world(w, world_to_i).unsqueeze(0)
            logits, value = model(obs)
            probs = torch.softmax(logits, dim=-1)[0]
            ranked = sorted(
                [{"action_name": actions[i], "prob": float(probs[i])} for i in range(len(actions))],
                key=lambda x: x["prob"],
                reverse=True,
            )
            out[w] = {
                "value": float(value[0]),
                "ranked_actions": ranked,
            }
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-checkpoint", default="artifacts/phase_b_ppo_theta_lite_discrete_v0/policy_value.pt")
    ap.add_argument("--rollout-root", required=True)
    ap.add_argument("--out-dir", default="artifacts/phase_b_ppo_theta_lite_online_update_v0")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--clip-ratio", type=float, default=0.2)
    ap.add_argument("--entropy-coef", type=float, default=0.02)
    ap.add_argument("--value-coef", type=float, default=0.5)
    ap.add_argument(
        "--reward-mode",
        choices=["reach", "reach_hold"],
        default="reach",
        help="reach: use reach_reward only. reach_hold: use reach_reward + hold_coef * hold_reward.",
    )
    ap.add_argument("--hold-coef", type=float, default=0.20)
    args = ap.parse_args()

    model, worlds, actions, world_to_i, action_to_i, ckpt = load_model(args.base_checkpoint)
    rows, skipped = collect_rows(args.rollout_root, world_to_i, action_to_i)

    if len(rows) < 3:
        raise RuntimeError(f"too few usable rollout rows: {len(rows)}, skipped={skipped}")

    obs = torch.stack([obs_from_world(r["world_name"], world_to_i) for r in rows])
    act = torch.tensor([r["action_index"] for r in rows], dtype=torch.long)
    old_logp = torch.tensor([r["old_logp"] for r in rows], dtype=torch.float32)

    if args.reward_mode == "reach":
        raw_reward_values = [r["reach_reward"] for r in rows]
    elif args.reward_mode == "reach_hold":
        raw_reward_values = [
            r["reach_reward"] + args.hold_coef * r["hold_reward"]
            for r in rows
        ]
    else:
        raise ValueError(args.reward_mode)

    raw_rewards = torch.tensor(raw_reward_values, dtype=torch.float32)
    rew_mean = raw_rewards.mean()
    rew_std = raw_rewards.std(unbiased=False).clamp_min(1e-6)
    rewards = (raw_rewards - rew_mean) / rew_std

    opt = optim.Adam(model.parameters(), lr=args.lr)

    before = eval_policy(model, worlds, actions, world_to_i)

    for ep in range(args.epochs):
        logits, values = model(obs)
        dist = torch.distributions.Categorical(logits=logits)
        logp = dist.log_prob(act)
        entropy = dist.entropy().mean()

        adv = rewards - values.detach()
        ratio = torch.exp(logp - old_logp)

        unclipped = ratio * adv
        clipped = torch.clamp(ratio, 1.0 - args.clip_ratio, 1.0 + args.clip_ratio) * adv
        pi_loss = -torch.min(unclipped, clipped).mean()
        v_loss = ((values - rewards) ** 2).mean()
        loss = pi_loss + args.value_coef * v_loss - args.entropy_coef * entropy

        opt.zero_grad()
        loss.backward()
        opt.step()

        if ep == 0 or (ep + 1) % max(1, args.epochs // 5) == 0:
            print(
                f"[online-ppo] epoch={ep+1}/{args.epochs} "
                f"loss={loss.item():.4f} pi={pi_loss.item():.4f} "
                f"v={v_loss.item():.4f} ent={entropy.item():.4f}"
            )

    after = eval_policy(model, worlds, actions, world_to_i)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            **ckpt,
            "model_state_dict": model.state_dict(),
            "online_update_source": args.rollout_root,
        },
        out_dir / "policy_value.pt",
    )

    report = {
        "schema": "phase_b_ppo_theta_lite_online_update_report_v0",
        "base_checkpoint": args.base_checkpoint,
        "rollout_root": args.rollout_root,
        "num_rows": len(rows),
        "skipped": skipped,
        "reward_mode": args.reward_mode,
        "hold_coef": args.hold_coef,
        "reward_mean": float(rew_mean),
        "reward_std": float(rew_std),
        "rows_by_world": {},
        "rows_by_action": {},
        "before": before,
        "after": after,
    }

    for r in rows:
        report["rows_by_world"][r["world_name"]] = report["rows_by_world"].get(r["world_name"], 0) + 1
        report["rows_by_action"][r["action_name"]] = report["rows_by_action"].get(r["action_name"], 0) + 1

    save_json(out_dir / "online_update_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
