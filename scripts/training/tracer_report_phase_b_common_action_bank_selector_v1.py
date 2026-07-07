#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODE_DEFS = [
    ("uniform", "Best single uniform command"),
    ("reach_only", "Terrain-specific reach-only"),
    ("task_only", "Terrain-specific task objective"),
    ("risk_only", "Terrain-specific risk filter"),
    ("task_risk", "Terrain-specific task + risk"),
    ("oracle", "Terrain-specific oracle"),
]


ACTION_LABELS = {
    "trot_mid": "Medium trot",
    "trot_solid_fast": "Fast solid trot",
    "sponge_v1d_bias_late_hold_025_013": "Sponge safe probe",
    "sponge_v1d_stabilized_late_hold_035_016": "Sponge late-hold",
    "sponge_slow_high_clear": "Sponge high-clear",
    "sponge_v8b_reach_bias": "Sponge reach-biased",
}


WORLD_LABELS = {
    "earth": "Flat",
    "tracer_rough_low": "Rough low",
    "tracer_rough_mid": "Rough mid",
    "stairs_single": "Stairs",
    "tracer_slippery_flat": "Slippery",
    "tracer_sponge_firm_flat": "Sponge",
}


def ff(x, default=0.0):
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def clamp01(x):
    return max(0.0, min(1.0, float(x)))


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def score_action(row, mode):
    reach = ff(row.get("reach_rate"))
    deploy = ff(row.get("deploy_valid_success_rate_v1c"))
    hard = ff(row.get("hard_invalid_rate"))
    warning = ff(row.get("warning_rate"))
    final = ff(row.get("mean_final_dist"), 999.0)
    drift = ff(row.get("mean_drift_proxy"), 999.0)
    score_v1c = ff(row.get("mean_deploy_valid_score_v1c"), -999.0)

    final_score = clamp01(1.0 - final / 2.5)
    drift_score = clamp01(1.0 - drift / 2.0)

    if mode == "reach_only":
        return reach

    if mode == "task_only":
        return 2.0 * reach + final_score + drift_score

    if mode == "risk_only":
        return reach - 2.0 * hard - 0.5 * warning

    if mode == "task_risk":
        return 2.0 * reach + final_score + drift_score - 3.0 * hard - 0.75 * warning

    if mode == "oracle":
        return score_v1c

    raise ValueError(mode)


def aggregate(rows):
    n_total = sum(int(r.get("n", 0)) for r in rows)
    if n_total <= 0:
        n_total = len(rows)

    def wavg(key):
        num = 0.0
        den = 0.0
        for r in rows:
            n = int(r.get("n", 1))
            num += n * ff(r.get(key))
            den += n
        return num / max(1.0, den)

    return {
        "num_groups": len(rows),
        "n_total": n_total,
        "reach_rate": wavg("reach_rate"),
        "deploy_valid_success_rate_v1c": wavg("deploy_valid_success_rate_v1c"),
        "hard_invalid_rate": wavg("hard_invalid_rate"),
        "warning_rate": wavg("warning_rate"),
        "mean_final_dist": wavg("mean_final_dist"),
        "mean_drift_proxy": wavg("mean_drift_proxy"),
        "mean_deploy_valid_score_v1c": wavg("mean_deploy_valid_score_v1c"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validity-json", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--fig-dir", required=True)
    args = ap.parse_args()

    report = load_json(args.validity_json)
    by_action_raw = report.get("by_action", {})

    rows = []
    for key, v in by_action_raw.items():
        if "::" in key:
            world, action = key.split("::", 1)
        else:
            world, action = "unknown", key
        row = dict(v)
        row["world_action"] = key
        row["world"] = world
        row["action"] = action
        rows.append(row)

    by_world = defaultdict(list)
    by_action_name = defaultdict(list)
    for r in rows:
        by_world[r["world"]].append(r)
        by_action_name[r["action"]].append(r)

    # Uniform baseline: one action used across all terrains.
    uniform_candidates = []
    for action, rs in by_action_name.items():
        agg = aggregate(rs)
        agg["mode"] = "uniform_candidate"
        agg["action"] = action
        agg["action_label"] = ACTION_LABELS.get(action, action)
        uniform_candidates.append(agg)

    # Best uniform command selected by deploy-valid score first, then hard invalid, then final distance.
    uniform_best = sorted(
        uniform_candidates,
        key=lambda r: (
            r["deploy_valid_success_rate_v1c"],
            r["mean_deploy_valid_score_v1c"],
            -r["hard_invalid_rate"],
            -r["mean_final_dist"],
        ),
        reverse=True,
    )[0]

    selector_rows = []

    selector_rows.append({
        "mode": "uniform",
        "mode_label": "Best single uniform command",
        "selected_policy": uniform_best["action"],
        "selected_policy_label": ACTION_LABELS.get(uniform_best["action"], uniform_best["action"]),
        **{k: uniform_best[k] for k in [
            "n_total",
            "reach_rate",
            "deploy_valid_success_rate_v1c",
            "hard_invalid_rate",
            "warning_rate",
            "mean_final_dist",
            "mean_drift_proxy",
            "mean_deploy_valid_score_v1c",
        ]},
        "per_world_selection": {
            world: uniform_best["action"] for world in sorted(by_world)
        },
    })

    # Terrain-specific modes.
    for mode, mode_label in MODE_DEFS:
        if mode == "uniform":
            continue

        selected = []
        per_world = {}
        for world, rs in sorted(by_world.items()):
            best = sorted(rs, key=lambda r: score_action(r, mode), reverse=True)[0]
            selected.append(best)
            per_world[world] = best["action"]

        agg = aggregate(selected)
        selector_rows.append({
            "mode": mode,
            "mode_label": mode_label,
            "selected_policy": "terrain_specific",
            "selected_policy_label": "Terrain-specific",
            **agg,
            "per_world_selection": per_world,
        })

    out = {
        "schema": "phase_b_common_action_bank_selector_report_v1",
        "validity_json": args.validity_json,
        "notes": [
            "Uniform baseline uses one fixed action across all terrains.",
            "Terrain-specific selectors choose one action per terrain from the same common action bank.",
            "This is still offline/counterfactual over directly sampled fixed-command rollouts, not yet an online mixed-terrain switching run."
        ],
        "fixed_world_action_rows": rows,
        "uniform_candidates": uniform_candidates,
        "selector_rows": selector_rows,
    }

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=2, sort_keys=True))

    lines = []
    lines.append("# Phase-B Common Action Bank Selector Report v1")
    lines.append("")
    lines.append("This report compares a single uniform fixed command against terrain-specific high-level selection.")
    lines.append("")
    lines.append("Important: this is an offline/counterfactual selector analysis over directly sampled fixed-command rollouts, not yet an online mixed-terrain switching experiment.")
    lines.append("")
    lines.append("## Dataset")
    lines.append("")
    lines.append(f"- Validity report: `{args.validity_json}`")
    lines.append(f"- Worlds: `{len(by_world)}`")
    lines.append(f"- Actions: `{len(by_action_name)}`")
    lines.append(f"- World-action groups: `{len(rows)}`")
    lines.append(f"- Total rollout episodes: `{sum(int(r.get('n', 0)) for r in rows)}`")
    lines.append("")
    lines.append("## Uniform command candidates")
    lines.append("")
    lines.append("| uniform action | n | reach | deploy_success | hard_invalid | warning | final | drift | score_v1c |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in sorted(uniform_candidates, key=lambda x: x["mean_deploy_valid_score_v1c"], reverse=True):
        lines.append(
            f"| {r['action_label']} | {r['n_total']} | "
            f"{r['reach_rate']:.3f} | {r['deploy_valid_success_rate_v1c']:.3f} | "
            f"{r['hard_invalid_rate']:.3f} | {r['warning_rate']:.3f} | "
            f"{r['mean_final_dist']:.3f} | {r['mean_drift_proxy']:.3f} | "
            f"{r['mean_deploy_valid_score_v1c']:.3f} |"
        )

    lines.append("")
    lines.append("## Selector comparison")
    lines.append("")
    lines.append("| mode | n | reach | deploy_success | hard_invalid | warning | final | drift | score_v1c |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in selector_rows:
        lines.append(
            f"| {r['mode_label']} | {r['n_total']} | "
            f"{r['reach_rate']:.3f} | {r['deploy_valid_success_rate_v1c']:.3f} | "
            f"{r['hard_invalid_rate']:.3f} | {r['warning_rate']:.3f} | "
            f"{r['mean_final_dist']:.3f} | {r['mean_drift_proxy']:.3f} | "
            f"{r['mean_deploy_valid_score_v1c']:.3f} |"
        )

    lines.append("")
    lines.append("## Per-world selected actions")
    lines.append("")
    lines.append("| mode | " + " | ".join(WORLD_LABELS.get(w, w) for w in sorted(by_world)) + " |")
    lines.append("|---|" + "|".join("---" for _ in sorted(by_world)) + "|")
    for r in selector_rows:
        actions = []
        for w in sorted(by_world):
            a = r["per_world_selection"].get(w, "")
            actions.append(ACTION_LABELS.get(a, a))
        lines.append(f"| {r['mode_label']} | " + " | ".join(actions) + " |")

    Path(args.out_md).write_text("\n".join(lines) + "\n")

    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Bar charts.
    mode_labels = [r["mode_label"].replace("Terrain-specific ", "").replace("Best single ", "Uniform\n") for r in selector_rows]
    metrics = [
        ("deploy_valid_success_rate_v1c", "Deploy-valid success rate", "selector_deploy_valid_success.png"),
        ("reach_rate", "Reach rate", "selector_reach_rate.png"),
        ("hard_invalid_rate", "Hard invalid rate", "selector_hard_invalid_rate.png"),
        ("mean_final_dist", "Mean final distance", "selector_mean_final_distance.png"),
        ("mean_deploy_valid_score_v1c", "Deploy-valid score v1c", "selector_deploy_valid_score.png"),
    ]

    for metric, ylabel, fname in metrics:
        vals = [r[metric] for r in selector_rows]
        fig, ax = plt.subplots(figsize=(9, 4.8))
        ax.bar(range(len(vals)), vals)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(mode_labels, rotation=20, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel + ": uniform command vs terrain-specific selector")
        fig.tight_layout()
        fig.savefig(fig_dir / fname, dpi=180)
        plt.close(fig)

    # Heatmap-like grouped plot for fixed action deploy success by world.
    worlds = sorted(by_world)
    actions = sorted(by_action_name)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = range(len(actions))
    width = 0.12
    for i, w in enumerate(worlds):
        vals = []
        for a in actions:
            match = [r for r in rows if r["world"] == w and r["action"] == a]
            vals.append(ff(match[0].get("deploy_valid_success_rate_v1c")) if match else 0.0)
        offsets = [xx + (i - (len(worlds)-1)/2) * width for xx in x]
        ax.bar(offsets, vals, width, label=WORLD_LABELS.get(w, w))
    ax.set_xticks(list(x))
    ax.set_xticklabels([ACTION_LABELS.get(a, a) for a in actions], rotation=25, ha="right")
    ax.set_ylabel("Deploy-valid success rate")
    ax.set_title("Common action bank deploy-valid success by terrain")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "common_action_bank_deploy_success_by_terrain.png", dpi=180)
    plt.close(fig)

    print("\n".join(lines))
    print("[wrote]", args.out_json)
    print("[wrote]", args.out_md)
    print("[wrote figures]", fig_dir)


if __name__ == "__main__":
    main()
