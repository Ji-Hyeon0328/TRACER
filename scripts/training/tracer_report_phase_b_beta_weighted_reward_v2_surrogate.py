#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F


class BetaSelectorNet(nn.Module):
    def __init__(self, input_dim, hidden=64):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 32),
            nn.ReLU(),
        )
        self.beta_head = nn.Linear(32, 3)

    def forward(self, x):
        return self.beta_head(self.trunk(x))


def read_jsonl(p):
    out = []
    with open(p, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def save_json(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def ff(x, default=0.0):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def load_json_safe(p):
    try:
        with open(p, "r") as f:
            return json.load(f)
    except Exception:
        return None


def candidate_result_paths(r):
    out = []
    for k in ["source_path", "result_path", "episode_result_path"]:
        p = r.get(k)
        if p:
            out.append(Path(p))
    for k in ["run_dir", "out_root", "episode_dir"]:
        p = r.get(k)
        if p:
            out.append(Path(p) / "ppo_policy_episode_result_v0.json")
    p = r.get("source_path")
    if p and Path(p).is_dir():
        out.append(Path(p) / "ppo_policy_episode_result_v0.json")
    return out


def hydrate_row(r):
    merged = dict(r)
    used = None
    raw = None
    for p in candidate_result_paths(r):
        if p.exists() and p.is_file():
            raw = load_json_safe(p)
            if raw:
                used = p
                break
    if raw:
        for k, v in raw.items():
            if k != "features":
                merged[k] = v
        if "features" not in merged and "features" in r:
            merged["features"] = r["features"]
        merged["_hydrated_from"] = str(used)
    else:
        merged["_hydrated_from"] = None
    return merged


def build_vec(r, names):
    feat = r.get("features") or {}
    w = r.get("world_name", "")
    vals = []
    for k in names:
        if k.startswith("world_onehot:") and k not in feat:
            vals.append(1.0 if k.split("world_onehot:", 1)[1] == w else 0.0)
        else:
            vals.append(ff(feat.get(k), 0.0))
    return torch.tensor(vals, dtype=torch.float32)


def load_beta(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = BetaSelectorNet(ckpt["input_dim"], hidden=64)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return ckpt, model


def get_comp(r, name, default=0.0):
    c = r.get("reward_components") or {}
    if name in c:
        return ff(c.get(name), default)
    return ff(r.get(name), default)


def derive_surrogate_terms(r):
    c = r.get("reward_components") or {}

    reach_reward = get_comp(r, "reach_reward")
    hold_reward = get_comp(r, "hold_reward")
    final_d = get_comp(r, "final_rel_dist")
    min_d = get_comp(r, "min_rel_dist")
    reached = bool(c.get("reached_stop_distance", r.get("reached_stop_distance", False)))

    # Some historical rows may not store post_reach_drift.
    drift = get_comp(r, "post_reach_drift", default=0.0)
    if drift == 0.0 and reached:
        drift = max(0.0, final_d - min_d)

    # Velocity/progress term: reward reaching and getting close.
    R_v = reach_reward + (2.0 if reached else 0.0) - 0.5 * min_d

    # Stability term: hold near the goal and punish post-reach sliding.
    R_s = hold_reward - 1.0 * final_d - 1.5 * drift

    # Energy/regularization surrogate.
    # Until real energy is logged, keep this as a mild penalty for aggressive behavior.
    # It can be replaced by CoT/torque integral later.
    action = r.get("action_name", "")
    aggressive_penalty = 0.0
    if "solid_fast" in action:
        aggressive_penalty += 0.25
    if "high_clear" in action or "soft_mid_clear" in action:
        aggressive_penalty += 0.15
    R_e = -aggressive_penalty

    return {
        "R_v_surrogate": R_v,
        "R_s_surrogate": R_s,
        "R_e_surrogate": R_e,
        "reach_reward": reach_reward,
        "hold_reward": hold_reward,
        "final_rel_dist": final_d,
        "min_rel_dist": min_d,
        "post_reach_drift": drift,
        "reached": reached,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", default="data/phase_b_training_pipeline_v5_merged_balanced/phase_b_rollout_index_v1.jsonl")
    ap.add_argument("--beta-model", default="artifacts/phase_b_objective_selector_beta_v1_hydrated/model.pt")
    ap.add_argument("--out-json", default="reports/phase_b_beta_weighted_reward_v2_surrogate.json")
    ap.add_argument("--out-md", default="reports/phase_b_beta_weighted_reward_v2_surrogate.md")
    args = ap.parse_args()

    rows = [hydrate_row(r) for r in read_jsonl(args.rollouts)]
    hydrated_count = sum(1 for r in rows if r.get("_hydrated_from"))

    ckpt, model = load_beta(args.beta_model)
    names = ckpt["feature_names"]
    mean = torch.tensor(ckpt["feature_mean"], dtype=torch.float32)
    std = torch.tensor(ckpt["feature_std"], dtype=torch.float32)

    scored = []
    with torch.no_grad():
        for r in rows:
            x = (build_vec(r, names) - mean) / std
            beta = F.softmax(model(x.unsqueeze(0))[0], dim=0).cpu().tolist()
            terms = derive_surrogate_terms(r)

            score_beta = (
                beta[0] * terms["R_v_surrogate"]
                + beta[1] * terms["R_s_surrogate"]
                + beta[2] * terms["R_e_surrogate"]
            )

            c = r.get("reward_components") or {}
            proxy = ff(c.get("tracer_proxy_reward", r.get("preference_score_seed", 0.0)))

            scored.append({
                "world_name": r.get("world_name"),
                "action_name": r.get("action_name"),
                "hydrated_from": r.get("_hydrated_from"),
                "beta_v": beta[0],
                "beta_s": beta[1],
                "beta_e": beta[2],
                "score_beta": score_beta,
                "proxy_reward": proxy,
                **terms,
            })

    groups = defaultdict(list)
    for r in scored:
        groups[f"{r['world_name']}::{r['action_name']}"].append(r)

    summary = []
    for k, arr in sorted(groups.items()):
        n = len(arr)
        summary.append({
            "world_action": k,
            "n": n,
            "reach_rate": sum(1.0 if x["reached"] else 0.0 for x in arr) / n,
            "mean_beta_v": sum(x["beta_v"] for x in arr) / n,
            "mean_beta_s": sum(x["beta_s"] for x in arr) / n,
            "mean_beta_e": sum(x["beta_e"] for x in arr) / n,
            "mean_R_v": sum(x["R_v_surrogate"] for x in arr) / n,
            "mean_R_s": sum(x["R_s_surrogate"] for x in arr) / n,
            "mean_R_e": sum(x["R_e_surrogate"] for x in arr) / n,
            "mean_score_beta": sum(x["score_beta"] for x in arr) / n,
            "mean_proxy_reward": sum(x["proxy_reward"] for x in arr) / n,
            "mean_final_dist": sum(x["final_rel_dist"] for x in arr) / n,
            "mean_drift": sum(x["post_reach_drift"] for x in arr) / n,
        })

    report = {
        "schema": "phase_b_beta_weighted_reward_report_v2_surrogate",
        "rollouts": args.rollouts,
        "beta_model": args.beta_model,
        "num_rows": len(rows),
        "hydrated_count": hydrated_count,
        "summary": summary,
        "scored_rows": scored,
        "note": "Uses surrogate R_v/R_s/R_e derived from reach_reward, hold_reward, final distance, and drift because episode results do not serialize R_v/R_s/R_e yet.",
    }
    save_json(Path(args.out_json), report)

    lines = []
    lines.append("# Phase-B β-weighted Reward Report v2 Surrogate")
    lines.append("")
    lines.append(f"- Rollouts: `{args.rollouts}`")
    lines.append(f"- Beta model: `{args.beta_model}`")
    lines.append(f"- Rows: `{len(rows)}`")
    lines.append(f"- Hydrated rows: `{hydrated_count}`")
    lines.append("")
    lines.append("| world::action | n | reach | beta_v | beta_s | beta_e | Rv_surr | Rs_surr | Re_surr | score_beta | proxy | final | drift |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for s in sorted(summary, key=lambda x: (x["world_action"].split("::")[0], -x["mean_score_beta"])):
        lines.append(
            f"| {s['world_action']} | {s['n']} | {s['reach_rate']:.3f} | "
            f"{s['mean_beta_v']:.3f} | {s['mean_beta_s']:.3f} | {s['mean_beta_e']:.3f} | "
            f"{s['mean_R_v']:.3f} | {s['mean_R_s']:.3f} | {s['mean_R_e']:.3f} | "
            f"{s['mean_score_beta']:.3f} | {s['mean_proxy_reward']:.3f} | "
            f"{s['mean_final_dist']:.3f} | {s['mean_drift']:.3f} |"
        )

    md = "\n".join(lines) + "\n"
    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text(md)
    print(md)
    print("[wrote]", args.out_json)
    print("[wrote]", args.out_md)


if __name__ == "__main__":
    main()
