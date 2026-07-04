#!/usr/bin/env python3
import argparse
import json
import random
import statistics
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


def ff(x, default=0.0):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


class UtilityNet(nn.Module):
    def __init__(self, input_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class RAMEpisodeNet(nn.Module):
    def __init__(self, input_dim, hidden=64, bool_dim=5, reg_dim=1):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 32),
            nn.ReLU(),
        )
        self.bool_head = nn.Linear(32, bool_dim)
        self.risk_head = nn.Linear(32, reg_dim)

    def forward(self, x):
        h = self.trunk(x)
        return self.bool_head(h), torch.sigmoid(self.risk_head(h))


class CandidateScorer(nn.Module):
    def __init__(self, input_dim, hidden=96):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 48),
            nn.ReLU(),
            nn.Linear(48, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def load_objective(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = UtilityNet(ckpt["input_dim"], hidden=ckpt.get("hidden_dim", 64))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return ckpt, model


def load_ram(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = RAMEpisodeNet(
        ckpt["input_dim"],
        hidden=ckpt.get("hidden_dim", 64),
        bool_dim=len(ckpt["bool_label_names"]),
        reg_dim=len(ckpt["reg_label_names"]),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return ckpt, model


def vec(feat, names):
    return torch.tensor([float((feat or {}).get(k, 0.0)) for k in names], dtype=torch.float32)


def compute_group_targets(rows, std_penalty, risk_penalty, reach_bonus):
    by_key = {}

    for r in rows:
        key = f"{r['world_name']}::{r['action_name']}"
        g = by_key.setdefault(key, {
            "world_name": r["world_name"],
            "action_name": r["action_name"],
            "rows": [],
        })
        g["rows"].append(r)

    out = {}

    for key, g in by_key.items():
        scores = [ff(r.get("preference_score_seed")) for r in g["rows"]]
        risks = [ff(r.get("ram_labels", {}).get("future_risk_proxy")) for r in g["rows"]]
        reaches = [1.0 if r.get("reward_components", {}).get("reached_stop_distance") else 0.0 for r in g["rows"]]
        finals = [ff(r.get("reward_components", {}).get("final_rel_dist")) for r in g["rows"]]
        holds = [ff(r.get("reward_components", {}).get("hold_reward")) for r in g["rows"]]

        mean_score = sum(scores) / max(len(scores), 1)
        std_score = statistics.pstdev(scores) if len(scores) > 1 else 0.0
        mean_risk = sum(risks) / max(len(risks), 1)
        reach_rate = sum(reaches) / max(len(reaches), 1)
        mean_final = sum(finals) / max(len(finals), 1)
        mean_hold = sum(holds) / max(len(holds), 1)

        robust_score = (
            mean_score
            - std_penalty * std_score
            - risk_penalty * mean_risk
            + reach_bonus * reach_rate
        )

        out[key] = {
            "world_name": g["world_name"],
            "action_name": g["action_name"],
            "n": len(g["rows"]),
            "mean_score": mean_score,
            "std_score": std_score,
            "mean_risk": mean_risk,
            "reach_rate": reach_rate,
            "mean_final_dist": mean_final,
            "mean_hold_reward": mean_hold,
            "robust_score": robust_score,
        }

    return out


def build_examples(rows, group_targets, obj_ckpt, obj_model, ram_ckpt, ram_model):
    obj_names = obj_ckpt["feature_names"]
    ram_names = ram_ckpt["feature_names"]
    bool_names = ram_ckpt["bool_label_names"]

    examples = []

    with torch.no_grad():
        for r in rows:
            key = f"{r['world_name']}::{r['action_name']}"
            gt = group_targets[key]
            feat = r["features"]

            obj_x = vec(feat, obj_names).unsqueeze(0)
            ram_x = vec(feat, ram_names).unsqueeze(0)

            obj_utility = float(obj_model(obj_x)[0].item())
            bool_logits, risk = ram_model(ram_x)
            bool_probs = torch.sigmoid(bool_logits[0]).detach().cpu().tolist()
            ram_risk = float(risk[0, 0].item())

            x = vec(feat, obj_names).tolist()
            x += [obj_utility, ram_risk]
            x += [float(v) for v in bool_probs]
            x += [
                gt["mean_score"],
                gt["std_score"],
                gt["mean_risk"],
                gt["reach_rate"],
            ]

            examples.append({
                "source_path": r["source_path"],
                "world_name": r["world_name"],
                "action_name": r["action_name"],
                "x": torch.tensor(x, dtype=torch.float32),
                "target": float(gt["robust_score"]),
                "target_raw": float(gt["robust_score"]),
                "group_target": gt,
                "objective_utility": obj_utility,
                "ram_risk": ram_risk,
            })

    input_names = list(obj_names)
    input_names += ["objective_utility", "ram_risk"]
    input_names += [f"ram_prob:{name}" for name in bool_names]
    input_names += ["group_mean_score", "group_std_score", "group_mean_risk", "group_reach_rate"]

    return examples, input_names


def standardize_examples(examples):
    X = torch.stack([e["x"] for e in examples], dim=0)
    y = torch.tensor([e["target"] for e in examples], dtype=torch.float32)

    x_mean = X.mean(dim=0)
    x_std = X.std(dim=0).clamp_min(1e-6)
    y_mean = y.mean()
    y_std = y.std().clamp_min(1e-6)

    for e in examples:
        e["x_raw"] = e["x"]
        e["x"] = (e["x"] - x_mean) / x_std
        e["target_z"] = float(((torch.tensor(e["target"]) - y_mean) / y_std).item())
        e["target"] = e["target_z"]

    return x_mean, x_std, y_mean, y_std


def eval_mse(model, examples):
    if not examples:
        return None
    errs = []
    with torch.no_grad():
        for e in examples:
            pred = model(e["x"].unsqueeze(0))[0].item()
            errs.append((pred - e["target"]) ** 2)
    return sum(errs) / len(errs)


def eval_pair_acc(model, examples):
    by_world = {}
    for e in examples:
        by_world.setdefault(e["world_name"], []).append(e)

    ok = 0
    total = 0

    with torch.no_grad():
        for _, rows in by_world.items():
            for i in range(len(rows)):
                for j in range(i + 1, len(rows)):
                    a, b = rows[i], rows[j]
                    if abs(a["target"] - b["target"]) < 1e-8:
                        continue
                    pa = model(a["x"].unsqueeze(0))[0].item()
                    pb = model(b["x"].unsqueeze(0))[0].item()
                    total += 1
                    ok += 1 if (pa - pb) * (a["target"] - b["target"]) > 0 else 0

    return ok / total if total else None


def aggregate_eval(model, examples, y_mean, y_std):
    grouped = {}

    with torch.no_grad():
        for e in examples:
            key = f"{e['world_name']}::{e['action_name']}"
            pred_z = model(e["x"].unsqueeze(0))[0].item()
            pred_raw = pred_z * float(y_std.item()) + float(y_mean.item())

            g = grouped.setdefault(key, {
                "world_name": e["world_name"],
                "action_name": e["action_name"],
                "n": 0,
                "pred_sum": 0.0,
                "objective_sum": 0.0,
                "ram_sum": 0.0,
                "target": e["target_raw"],
                "group_target": e["group_target"],
            })

            g["n"] += 1
            g["pred_sum"] += pred_raw
            g["objective_sum"] += e["objective_utility"]
            g["ram_sum"] += e["ram_risk"]

    rows = []
    for _, g in grouped.items():
        n = max(g["n"], 1)
        gt = g["group_target"]
        rows.append({
            "world_name": g["world_name"],
            "action_name": g["action_name"],
            "n": g["n"],
            "pred_policy_score": g["pred_sum"] / n,
            "robust_target": g["target"],
            "objective_utility": g["objective_sum"] / n,
            "ram_pred_risk": g["ram_sum"] / n,
            "mean_score": gt["mean_score"],
            "std_score": gt["std_score"],
            "mean_risk": gt["mean_risk"],
            "reach_rate": gt["reach_rate"],
            "mean_final_dist": gt["mean_final_dist"],
            "mean_hold_reward": gt["mean_hold_reward"],
        })

    return sorted(rows, key=lambda r: (r["world_name"], -r["pred_policy_score"]))


def fmt(x):
    return f"{float(x):.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", default="data/phase_b_training_pipeline_v1/phase_b_rollout_index_v1.jsonl")
    ap.add_argument("--objective-model", default="artifacts/phase_b_objective_selector_pref_v2_clean/model.pt")
    ap.add_argument("--ram-model", default="artifacts/phase_b_ram_episode_v2_clean/model.pt")
    ap.add_argument("--out-dir", default="artifacts/phase_b_beta_ram_candidate_policy_v2_robust")
    ap.add_argument("--report-json", default="reports/phase_b_beta_ram_candidate_policy_eval_v2_robust.json")
    ap.add_argument("--report-md", default="reports/phase_b_beta_ram_candidate_policy_eval_v2_robust.md")
    ap.add_argument("--std-penalty", type=float, default=0.75)
    ap.add_argument("--risk-penalty", type=float, default=1.50)
    ap.add_argument("--reach-bonus", type=float, default=1.00)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--pair-loss-weight", type=float, default=0.25)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    rows = read_jsonl(args.rollouts)
    obj_ckpt, obj_model = load_objective(args.objective_model)
    ram_ckpt, ram_model = load_ram(args.ram_model)

    group_targets = compute_group_targets(
        rows,
        std_penalty=args.std_penalty,
        risk_penalty=args.risk_penalty,
        reach_bonus=args.reach_bonus,
    )

    examples, input_names = build_examples(rows, group_targets, obj_ckpt, obj_model, ram_ckpt, ram_model)
    x_mean, x_std, y_mean, y_std = standardize_examples(examples)

    random.shuffle(examples)
    n_val = max(1, int(len(examples) * args.val_frac))
    val = examples[:n_val]
    train = examples[n_val:]

    model = CandidateScorer(input_dim=len(input_names), hidden=96)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best = {"epoch": -1, "score": -1e9, "state_dict": None}
    history = []

    for ep in range(1, args.epochs + 1):
        random.shuffle(train)
        losses = []

        model.train()
        for e in train:
            pred = model(e["x"].unsqueeze(0))[0]
            target = torch.tensor(e["target"], dtype=torch.float32)
            loss = F.mse_loss(pred, target)

            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))

        pair_losses = []
        by_world = {}
        for e in train:
            by_world.setdefault(e["world_name"], []).append(e)

        for _, wrs in by_world.items():
            if len(wrs) < 2:
                continue
            for _ in range(min(32, len(wrs) * 2)):
                a, b = random.sample(wrs, 2)
                if abs(a["target"] - b["target"]) < 1e-8:
                    continue
                sign = 1.0 if a["target"] > b["target"] else -1.0
                pa = model(a["x"].unsqueeze(0))[0]
                pb = model(b["x"].unsqueeze(0))[0]
                loss_pair = F.softplus(-sign * (pa - pb))

                opt.zero_grad()
                (args.pair_loss_weight * loss_pair).backward()
                opt.step()
                pair_losses.append(float(loss_pair.item()))

        model.eval()
        train_mse = eval_mse(model, train)
        val_mse = eval_mse(model, val)
        train_pair = eval_pair_acc(model, train)
        val_pair = eval_pair_acc(model, val)
        score = (val_pair if val_pair is not None else 0.0) - val_mse

        row = {
            "epoch": ep,
            "loss": sum(losses) / max(len(losses), 1),
            "pair_loss": sum(pair_losses) / max(len(pair_losses), 1) if pair_losses else None,
            "train_mse": train_mse,
            "val_mse": val_mse,
            "train_pair_acc": train_pair,
            "val_pair_acc": val_pair,
            "selection_score": score,
        }
        history.append(row)

        if score > best["score"]:
            best = {
                "epoch": ep,
                "score": score,
                "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            }

        if ep == 1 or ep % 50 == 0 or ep == args.epochs:
            print(json.dumps(row, sort_keys=True))

    if best["state_dict"] is not None:
        model.load_state_dict(best["state_dict"])

    eval_rows = aggregate_eval(model, examples, y_mean, y_std)

    action_vocab = sorted(set(e["action_name"] for e in examples))
    world_vocab = sorted(set(e["world_name"] for e in examples))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "schema": "phase_b_beta_ram_candidate_policy_v2_robust",
        "model_state_dict": model.state_dict(),
        "input_dim": len(input_names),
        "input_names": input_names,
        "input_mean": x_mean.tolist(),
        "input_std": x_std.tolist(),
        "target_mean": float(y_mean.item()),
        "target_std": float(y_std.item()),
        "objective_model_path": args.objective_model,
        "ram_model_path": args.ram_model,
        "objective_feature_names": obj_ckpt["feature_names"],
        "ram_feature_names": ram_ckpt["feature_names"],
        "ram_bool_label_names": ram_ckpt["bool_label_names"],
        "group_target_params": {
            "std_penalty": args.std_penalty,
            "risk_penalty": args.risk_penalty,
            "reach_bonus": args.reach_bonus,
        },
        "group_targets": group_targets,
        "action_vocab": action_vocab,
        "world_vocab": world_vocab,
        "best_epoch": best["epoch"],
        "best_score": best["score"],
        "note": "Robust candidate θ-lite scorer using Objective/RAM predictions and group-level robust targets.",
    }
    torch.save(ckpt, out_dir / "model.pt")

    report = {
        "schema": "phase_b_beta_ram_candidate_policy_train_report_v2_robust",
        "rollouts": args.rollouts,
        "objective_model": args.objective_model,
        "ram_model": args.ram_model,
        "out_dir": str(out_dir),
        "num_examples": len(examples),
        "train_examples": len(train),
        "val_examples": len(val),
        "input_dim": len(input_names),
        "input_names": input_names,
        "best_epoch": best["epoch"],
        "best_score": best["score"],
        "final_train_mse": eval_mse(model, train),
        "final_val_mse": eval_mse(model, val),
        "final_train_pair_acc": eval_pair_acc(model, train),
        "final_val_pair_acc": eval_pair_acc(model, val),
        "group_target_params": ckpt["group_target_params"],
        "group_targets": group_targets,
        "eval_rows": eval_rows,
        "history_tail": history[-20:],
    }
    save_json(out_dir / "train_report.json", report)
    save_json(Path(args.report_json), report)

    lines = []
    lines.append("# Phase-B β/RAM Robust Candidate Policy v2")
    lines.append("")
    lines.append(f"- Examples: `{len(examples)}`")
    lines.append(f"- Input dim: `{len(input_names)}`")
    lines.append(f"- Best epoch: `{best['epoch']}`")
    lines.append(f"- Final train MSE: `{report['final_train_mse']:.4f}`")
    lines.append(f"- Final val MSE: `{report['final_val_mse']:.4f}`")
    lines.append(f"- Final train pair acc: `{report['final_train_pair_acc']:.3f}`")
    lines.append(f"- Final val pair acc: `{report['final_val_pair_acc']:.3f}`")
    lines.append("")
    lines.append("Robust target:")
    lines.append("")
    lines.append(f"`mean_score - {args.std_penalty} * std_score - {args.risk_penalty} * mean_risk + {args.reach_bonus} * reach_rate`")
    lines.append("")
    lines.append("| world | action | n | pred_policy_score | robust_target | mean_score | std_score | mean_risk | reach_rate | final_dist | hold |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in eval_rows:
        lines.append(
            f"| {r['world_name']} | {r['action_name']} | {r['n']} | {fmt(r['pred_policy_score'])} | "
            f"{fmt(r['robust_target'])} | {fmt(r['mean_score'])} | {fmt(r['std_score'])} | "
            f"{fmt(r['mean_risk'])} | {fmt(r['reach_rate'])} | {fmt(r['mean_final_dist'])} | {fmt(r['mean_hold_reward'])} |"
        )

    lines.append("")
    lines.append("This robust scorer is intended to reduce soft-terrain over-selection caused by high-variance mean-score targets.")

    Path(args.report_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_md).write_text("\n".join(lines) + "\n")
    (out_dir / "train_report.md").write_text("\n".join(lines) + "\n")

    print()
    print("\n".join(lines))
    print()
    print("[wrote]", out_dir / "model.pt")
    print("[wrote]", out_dir / "train_report.json")
    print("[wrote]", out_dir / "train_report.md")
    print("[wrote]", args.report_json)
    print("[wrote]", args.report_md)


if __name__ == "__main__":
    main()
