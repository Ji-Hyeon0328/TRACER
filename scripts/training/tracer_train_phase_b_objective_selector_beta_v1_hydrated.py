#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def read_jsonl(p):
    rows = []
    with open(p, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_json(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def load_json_safe(p):
    try:
        with open(p, "r") as f:
            return json.load(f)
    except Exception:
        return None


def ff(x, default=0.0):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


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

    # Some rows store source_path as a directory.
    p = r.get("source_path")
    if p and Path(p).is_dir():
        out.append(Path(p) / "ppo_policy_episode_result_v0.json")

    return out


def hydrate_row(r):
    """
    Merge rollout-index row with raw episode result if available.
    Raw episode result has the full reward_components.
    """
    merged = dict(r)
    raw = None

    for p in candidate_result_paths(r):
        if p.exists() and p.is_file():
            raw = load_json_safe(p)
            if raw:
                break

    if raw:
        # Preserve index fields but use raw values for runtime result fields.
        for k, v in raw.items():
            if k not in ["features"]:
                merged[k] = v

        # Keep rollout-index features if raw does not have them.
        if "features" not in merged and "features" in r:
            merged["features"] = r["features"]

        merged["_hydrated_from"] = str(p)
    else:
        merged["_hydrated_from"] = None

    return merged


def get_reward_components(r):
    c = r.get("reward_components") or {}

    # Fallbacks in case the row is flattened.
    def g(name, *alts):
        if name in c:
            return ff(c.get(name))
        for a in alts:
            if a in r:
                return ff(r.get(a))
        return 0.0

    return {
        "R_v": g("R_v", "mean_R_v", "R_v"),
        "R_s": g("R_s", "mean_R_s", "R_s"),
        "R_e": g("R_e", "mean_R_e", "R_e"),
        "final_rel_dist": g("final_rel_dist", "mean_final_rel_dist", "final_rel_dist"),
        "min_rel_dist": g("min_rel_dist", "mean_min_rel_dist", "min_rel_dist"),
        "post_reach_drift": g("post_reach_drift", "mean_post_reach_drift", "post_reach_drift"),
        "hold_reward": g("hold_reward", "mean_hold_reward", "hold_reward"),
        "reach_reward": g("reach_reward", "mean_reach_reward", "reach_reward"),
        "tracer_proxy_reward": g("tracer_proxy_reward", "mean_tracer_proxy_reward", "preference_score_seed"),
        "reached": 1.0 if c.get("reached_stop_distance", r.get("reached_stop_distance", False)) else 0.0,
    }


def beta_label_from_context_and_outcome(r):
    """
    Bootstrap pseudo-label for β=[β_v,β_s,β_e].
    This is intentionally heuristic, but now it uses hydrated outcome terms.
    """
    w = r.get("world_name") or r.get("world") or ""
    rc = get_reward_components(r)
    labels = r.get("ram_labels") or {}
    risk = ff(labels.get("future_risk_proxy"), default=0.5)

    # Terrain prior.
    if w == "earth":
        beta = [0.60, 0.23, 0.17]
    elif w == "stairs_single":
        beta = [0.46, 0.40, 0.14]
    elif "sponge" in w:
        beta = [0.25, 0.62, 0.13]
    else:
        beta = [0.45, 0.40, 0.15]

    drift = rc["post_reach_drift"]
    final_d = rc["final_rel_dist"]
    reached = rc["reached"]
    R_s = rc["R_s"]

    # Reached but slid/drifted badly: stability must dominate.
    if reached > 0.5 and (drift > 0.75 or final_d > 1.0 or R_s < 0.0):
        beta[0] -= 0.18
        beta[1] += 0.24
        beta[2] -= 0.06

    # Did not reach: progress matters, but on sponge keep stability high.
    if reached < 0.5 and final_d > 0.8:
        if "sponge" in w:
            beta[0] += 0.04
            beta[1] += 0.04
            beta[2] -= 0.08
        else:
            beta[0] += 0.12
            beta[1] -= 0.04
            beta[2] -= 0.08

    # High risk -> stability.
    if risk > 0.65:
        beta[0] -= 0.08
        beta[1] += 0.12
        beta[2] -= 0.04

    # Stable reach -> allow progress and regularization.
    if reached > 0.5 and drift < 0.25 and final_d < 0.5 and R_s > 1.0:
        beta[0] += 0.04
        beta[1] -= 0.02
        beta[2] += 0.02

    beta = [max(0.05, x) for x in beta]
    s = sum(beta)
    return [x / s for x in beta]


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
        h = self.trunk(x)
        return self.beta_head(h)


def collect_feature_names(rows):
    names = set()
    for r in rows:
        feat = r.get("features") or {}
        for k in feat.keys():
            if k.startswith("metric:") or k.startswith("reward:"):
                continue
            if k.startswith("theta:"):
                continue
            names.add(k)

    if not names:
        worlds = sorted(set(r.get("world_name", "") for r in rows))
        return [f"world_onehot:{w}" for w in worlds]

    return sorted(names)


def build_feature_vec(r, names):
    feat = r.get("features") or {}
    w = r.get("world_name", "")
    vals = []
    for k in names:
        if k.startswith("world_onehot:") and k not in feat:
            vals.append(1.0 if k.split("world_onehot:", 1)[1] == w else 0.0)
        else:
            vals.append(ff(feat.get(k), 0.0))
    return torch.tensor(vals, dtype=torch.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", default="data/phase_b_training_pipeline_v5_merged_balanced/phase_b_rollout_index_v1.jsonl")
    ap.add_argument("--out-dir", default="artifacts/phase_b_objective_selector_beta_v1_hydrated")
    ap.add_argument("--report-json", default="reports/phase_b_objective_selector_beta_v1_hydrated.json")
    ap.add_argument("--report-md", default="reports/phase_b_objective_selector_beta_v1_hydrated.md")
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=92)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    raw_rows = read_jsonl(args.rollouts)
    rows = [hydrate_row(r) for r in raw_rows]
    hydrated_count = sum(1 for r in rows if r.get("_hydrated_from"))

    feature_names = collect_feature_names(rows)

    X = torch.stack([build_feature_vec(r, feature_names) for r in rows], dim=0)
    Y = torch.tensor([beta_label_from_context_and_outcome(r) for r in rows], dtype=torch.float32)

    x_mean = X.mean(dim=0)
    x_std = X.std(dim=0).clamp_min(1e-6)
    Xn = (X - x_mean) / x_std

    idx = list(range(len(rows)))
    random.shuffle(idx)
    n_val = max(1, int(0.2 * len(idx)))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    model = BetaSelectorNet(input_dim=len(feature_names), hidden=64)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best = {"epoch": -1, "val_kl": 1e9, "state": None}
    hist = []

    for ep in range(1, args.epochs + 1):
        model.train()
        order = train_idx[:]
        random.shuffle(order)
        losses = []

        for i in order:
            logits = model(Xn[i].unsqueeze(0))[0]
            target = Y[i]
            logp = F.log_softmax(logits, dim=0)
            loss = F.kl_div(logp, target, reduction="sum")

            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))

        model.eval()
        with torch.no_grad():
            train_pred = F.log_softmax(model(Xn[train_idx]), dim=1)
            val_pred = F.log_softmax(model(Xn[val_idx]), dim=1)
            train_kl = F.kl_div(train_pred, Y[train_idx], reduction="batchmean").item()
            val_kl = F.kl_div(val_pred, Y[val_idx], reduction="batchmean").item()

        rec = {
            "epoch": ep,
            "loss": sum(losses) / max(len(losses), 1),
            "train_kl": train_kl,
            "val_kl": val_kl,
        }
        hist.append(rec)

        if val_kl < best["val_kl"]:
            best = {
                "epoch": ep,
                "val_kl": val_kl,
                "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            }

        if ep == 1 or ep % 100 == 0 or ep == args.epochs:
            print(json.dumps(rec, sort_keys=True))

    if best["state"] is not None:
        model.load_state_dict(best["state"])

    model.eval()
    with torch.no_grad():
        probs = F.softmax(model(Xn), dim=1).cpu().tolist()

    by_world = {}
    by_world_action = {}
    for r, b in zip(rows, probs):
        w = r.get("world_name", "")
        a = r.get("action_name", "")
        by_world.setdefault(w, []).append(b)
        by_world_action.setdefault(f"{w}::{a}", []).append(b)

    def avg_beta(arr):
        n = len(arr)
        return [sum(x[j] for x in arr) / n for j in range(3)]

    world_summary = {k: avg_beta(v) for k, v in sorted(by_world.items())}
    action_summary = {k: avg_beta(v) for k, v in sorted(by_world_action.items())}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "schema": "phase_b_objective_selector_beta_v1_hydrated",
        "model_state_dict": model.state_dict(),
        "input_dim": len(feature_names),
        "feature_names": feature_names,
        "feature_mean": x_mean.tolist(),
        "feature_std": x_std.tolist(),
        "beta_names": ["beta_v", "beta_s", "beta_e"],
        "best_epoch": best["epoch"],
        "best_val_kl": best["val_kl"],
        "hydrated_count": hydrated_count,
        "num_rows": len(rows),
        "note": "Hydrated bootstrap explicit beta selector. Inputs exclude theta/reward/metric features.",
    }
    torch.save(ckpt, out_dir / "model.pt")

    report = {
        "schema": "phase_b_objective_selector_beta_v1_hydrated_report",
        "rollouts": args.rollouts,
        "out_dir": str(out_dir),
        "num_rows": len(rows),
        "hydrated_count": hydrated_count,
        "input_dim": len(feature_names),
        "feature_names": feature_names,
        "best_epoch": best["epoch"],
        "best_val_kl": best["val_kl"],
        "world_summary": world_summary,
        "world_action_summary": action_summary,
        "history_tail": hist[-20:],
    }
    save_json(out_dir / "train_report.json", report)
    save_json(Path(args.report_json), report)

    lines = []
    lines.append("# Phase-B Objective Selector β v1 Hydrated")
    lines.append("")
    lines.append(f"- Rollouts: `{len(rows)}`")
    lines.append(f"- Hydrated rows: `{hydrated_count}`")
    lines.append(f"- Input dim: `{len(feature_names)}`")
    lines.append(f"- Best epoch: `{best['epoch']}`")
    lines.append(f"- Best val KL: `{best['val_kl']:.6f}`")
    lines.append("")
    lines.append("β = [β_v, β_s, β_e]")
    lines.append("")
    lines.append("## By world")
    lines.append("")
    lines.append("| world | beta_v | beta_s | beta_e |")
    lines.append("|---|---:|---:|---:|")
    for w, b in world_summary.items():
        lines.append(f"| {w} | {b[0]:.3f} | {b[1]:.3f} | {b[2]:.3f} |")

    lines.append("")
    lines.append("## By world::action")
    lines.append("")
    lines.append("| world::action | beta_v | beta_s | beta_e |")
    lines.append("|---|---:|---:|---:|")
    for k, b in action_summary.items():
        lines.append(f"| {k} | {b[0]:.3f} | {b[1]:.3f} | {b[2]:.3f} |")

    md = "\n".join(lines) + "\n"
    (out_dir / "train_report.md").write_text(md)
    Path(args.report_md).write_text(md)

    print()
    print(md)
    print("[wrote]", out_dir / "model.pt")
    print("[wrote]", out_dir / "train_report.json")
    print("[wrote]", out_dir / "train_report.md")
    print("[wrote]", args.report_json)
    print("[wrote]", args.report_md)


if __name__ == "__main__":
    main()
