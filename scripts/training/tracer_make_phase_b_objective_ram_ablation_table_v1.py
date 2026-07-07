#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


WORLDS = ["earth", "stairs", "sponge"]

REPORT_PATHS = {
    "earth": "reports/phase_b_ablation_data_v1_earth_validity_episode_labels_v1c.json",
    "stairs": "reports/phase_b_ablation_data_v1_stairs_validity_episode_labels_v1c.json",
    "sponge": "reports/phase_b_ablation_data_v1_sponge_validity_episode_labels_v1c.json",
}



ACTION_LABELS = {
    "earth::trot_mid": "Flat\nMedium trot",
    "earth::trot_solid_fast": "Flat\nFast solid trot",
    "stairs_single::trot_mid": "Stairs\nMedium trot",
    "stairs_single::trot_solid_fast": "Stairs\nFast solid trot",
    "sponge::sponge_slow_high_clear": "Sponge\nSlow high-clear",
    "sponge::sponge_v1d_bias_late_hold_025_013": "Sponge\nSafe reach probe",
    "sponge::sponge_v1d_stabilized_late_hold_035_016": "Sponge\nLate hold probe",
    "sponge::sponge_v8b_reach_bias": "Sponge\nReach-biased risky",
    "tracer_sponge_firm_flat::sponge_slow_high_clear": "Sponge\nSlow high-clear",
    "tracer_sponge_firm_flat::sponge_v1d_bias_late_hold_025_013": "Sponge\nSafe reach probe",
    "tracer_sponge_firm_flat::sponge_v1d_stabilized_late_hold_035_016": "Sponge\nLate hold probe",
    "tracer_sponge_firm_flat::sponge_v8b_reach_bias": "Sponge\nReach-biased risky",
}

MODE_LABELS_READABLE = {
    "obj_off_ram_off": "Reach-only",
    "obj_on_ram_off": "Task\nobjective",
    "obj_off_ram_on": "Risk\nfilter",
    "obj_on_ram_on": "Task + risk",
}


def readable_action_label(r):
    key = r.get("world_action", "")
    if key in ACTION_LABELS:
        return ACTION_LABELS[key]
    world = r.get("world", "")
    action = r.get("action", "")
    return f"{world}\n{action}"


def readable_mode_label(mode):
    return MODE_LABELS_READABLE.get(mode, mode)


MODES = [
    ("obj_off_ram_off", "Objective OFF / RAM OFF"),
    ("obj_on_ram_off", "Objective ON / RAM OFF"),
    ("obj_off_ram_on", "Objective OFF / RAM ON"),
    ("obj_on_ram_on", "Objective ON / RAM ON"),
]


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def clamp01(x):
    return max(0.0, min(1.0, float(x)))


def ff(x, default=0.0):
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def score_row(row, mode):
    reach = ff(row.get("reach_rate"))
    hard = ff(row.get("hard_invalid_rate"))
    warning = ff(row.get("warning_rate"))
    final = ff(row.get("mean_final_dist"), 999.0)
    drift = ff(row.get("mean_drift_proxy"), 999.0)

    final_score = clamp01(1.0 - final / 2.5)
    drift_score = clamp01(1.0 - drift / 2.0)

    if mode == "obj_off_ram_off":
        # Naive reach-only selector.
        return reach

    if mode == "obj_on_ram_off":
        # Objective uses reach + final/hold quality, but ignores invalid-gait risk.
        return 2.0 * reach + 1.0 * final_score + 1.0 * drift_score

    if mode == "obj_off_ram_on":
        # RAM only: reach objective but penalize predicted invalid/warning.
        return reach - 2.0 * hard - 0.5 * warning

    if mode == "obj_on_ram_on":
        # Full selector: task objective + RAM validity penalty.
        return (
            2.0 * reach
            + 1.0 * final_score
            + 1.0 * drift_score
            - 3.0 * hard
            - 0.75 * warning
        )

    raise ValueError(mode)


def short_action_name(key):
    return key.split("::", 1)[-1]


def load_action_rows():
    action_rows = []
    for world, path in REPORT_PATHS.items():
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(path)

        report = load_json(p)
        for world_action, row in report.get("by_action", {}).items():
            rr = dict(row)
            rr["world"] = world
            rr["world_action"] = world_action
            rr["action"] = short_action_name(world_action)
            action_rows.append(rr)

    return action_rows


def select_by_mode(action_rows):
    selected = []
    by_world = {}
    for r in action_rows:
        by_world.setdefault(r["world"], []).append(r)

    for world, rows in by_world.items():
        for mode, mode_label in MODES:
            ranked = sorted(
                rows,
                key=lambda r: score_row(r, mode),
                reverse=True,
            )
            best = ranked[0]
            out = {
                "world": world,
                "mode": mode,
                "mode_label": mode_label,
                "selected_world_action": best["world_action"],
                "selected_action": best["action"],
                "selector_score": score_row(best, mode),
                "n": best.get("n"),
                "reach_rate": ff(best.get("reach_rate")),
                "deploy_valid_success_rate_v1c": ff(best.get("deploy_valid_success_rate_v1c")),
                "hard_invalid_rate": ff(best.get("hard_invalid_rate")),
                "warning_rate": ff(best.get("warning_rate")),
                "mean_final_dist": ff(best.get("mean_final_dist")),
                "mean_drift_proxy": ff(best.get("mean_drift_proxy")),
                "mean_deploy_valid_score_v1c": ff(best.get("mean_deploy_valid_score_v1c")),
            }
            selected.append(out)

    return selected


def make_markdown(action_rows, selected):
    lines = []
    lines.append("# Phase-B Objective/RAM Offline Ablation Table v1")
    lines.append("")
    lines.append("This is an offline counterfactual ablation over directly sampled fixed-command rollouts.")
    lines.append("Each selector mode chooses one action per terrain from the same action bank.")
    lines.append("")
    lines.append("## Selector definitions")
    lines.append("")
    lines.append("| mode | meaning |")
    lines.append("|---|---|")
    lines.append("| Objective OFF / RAM OFF | reach-only selector |")
    lines.append("| Objective ON / RAM OFF | reach + final/hold objective, ignores invalid risk |")
    lines.append("| Objective OFF / RAM ON | reach objective with invalid/warning penalty |")
    lines.append("| Objective ON / RAM ON | task objective + RAM validity penalty |")
    lines.append("")

    lines.append("## Fixed-command action statistics")
    lines.append("")
    lines.append("| world | action | n | reach | deploy_success | hard_invalid | warning | final | drift | score_v1c |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in sorted(action_rows, key=lambda x: (x["world"], x["action"])):
        lines.append(
            f"| {r['world']} | {r['action']} | {r.get('n')} | "
            f"{ff(r.get('reach_rate')):.3f} | "
            f"{ff(r.get('deploy_valid_success_rate_v1c')):.3f} | "
            f"{ff(r.get('hard_invalid_rate')):.3f} | "
            f"{ff(r.get('warning_rate')):.3f} | "
            f"{ff(r.get('mean_final_dist')):.3f} | "
            f"{ff(r.get('mean_drift_proxy')):.3f} | "
            f"{ff(r.get('mean_deploy_valid_score_v1c')):.3f} |"
        )

    lines.append("")
    lines.append("## Objective/RAM selector ablation")
    lines.append("")
    lines.append("| world | selector mode | selected action | selector score | reach | deploy_success | hard_invalid | warning | final | drift |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|")

    for r in sorted(selected, key=lambda x: (x["world"], x["mode"])):
        lines.append(
            f"| {r['world']} | {r['mode_label']} | {r['selected_action']} | "
            f"{r['selector_score']:.3f} | "
            f"{r['reach_rate']:.3f} | "
            f"{r['deploy_valid_success_rate_v1c']:.3f} | "
            f"{r['hard_invalid_rate']:.3f} | "
            f"{r['warning_rate']:.3f} | "
            f"{r['mean_final_dist']:.3f} | "
            f"{r['mean_drift_proxy']:.3f} |"
        )

    lines.append("")
    lines.append("## Key interpretation")
    lines.append("")
    lines.append("- Earth and stairs contain deploy-valid fixed commands; high-level selection mainly needs to choose the correct command.")
    lines.append("- Sponge has no deploy-valid primitive in the current theta-lite action bank; selector changes can reduce risk but cannot create a valid gait.")
    lines.append("- RAM ON tends to penalize hard-invalid/warning-prone choices, which is most important on sponge.")
    lines.append("- Objective ON is useful only when the action bank contains physically valid candidates; otherwise it can only choose the least-bad candidate.")

    return "\n".join(lines) + "\n"


def grouped_bar(selected, metric, ylabel, out_path):
    worlds = WORLDS
    mode_keys = [m[0] for m in MODES]
    mode_labels = [readable_mode_label(m[0]) for m in MODES]

    vals = []
    for world in worlds:
        row = []
        for mode in mode_keys:
            matches = [r for r in selected if r["world"] == world and r["mode"] == mode]
            row.append(ff(matches[0].get(metric)) if matches else 0.0)
        vals.append(row)

    x = list(range(len(mode_keys)))
    width = 0.22

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, world in enumerate(worlds):
        offsets = [xx + (i - 1) * width for xx in x]
        ax.bar(offsets, vals[i], width, label=world)

    ax.set_xticks(x)
    ax.set_xticklabels(mode_labels, rotation=0)
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel + " by Objective/RAM ablation mode")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def action_bar(action_rows, metric, ylabel, out_path):
    labels = [readable_action_label(r) for r in action_rows]
    vals = [ff(r.get(metric)) for r in action_rows]

    fig, ax = plt.subplots(figsize=(11, 5.8))
    ax.bar(list(range(len(labels))), vals)
    ax.set_xticks(list(range(len(labels))))
    ax.set_xticklabels(labels, rotation=0, ha="center")
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel + " for fixed-command action bank")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--fig-dir", required=True)
    args = ap.parse_args()

    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    action_rows = load_action_rows()
    selected = select_by_mode(action_rows)

    out = {
        "schema": "phase_b_objective_ram_offline_ablation_table_v1",
        "note": "Offline selector ablation over directly sampled fixed-command rollouts.",
        "fixed_action_rows": action_rows,
        "selected_rows": selected,
    }

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=2, sort_keys=True))

    md = make_markdown(action_rows, selected)
    Path(args.out_md).write_text(md)

    action_bar(
        action_rows,
        "deploy_valid_success_rate_v1c",
        "Deploy-valid success rate",
        fig_dir / "fixed_action_deploy_valid_success.png",
    )
    action_bar(
        action_rows,
        "hard_invalid_rate",
        "Hard invalid rate",
        fig_dir / "fixed_action_hard_invalid_rate.png",
    )
    action_bar(
        action_rows,
        "mean_final_dist",
        "Mean final distance",
        fig_dir / "fixed_action_mean_final_distance.png",
    )
    grouped_bar(
        selected,
        "reach_rate",
        "Reach rate",
        fig_dir / "objective_ram_ablation_reach_rate.png",
    )
    grouped_bar(
        selected,
        "deploy_valid_success_rate_v1c",
        "Deploy-valid success rate",
        fig_dir / "objective_ram_ablation_deploy_valid_success.png",
    )
    grouped_bar(
        selected,
        "hard_invalid_rate",
        "Hard invalid rate",
        fig_dir / "objective_ram_ablation_hard_invalid_rate.png",
    )
    grouped_bar(
        selected,
        "mean_final_dist",
        "Mean final distance",
        fig_dir / "objective_ram_ablation_mean_final_distance.png",
    )

    print(md)
    print("[wrote]", args.out_json)
    print("[wrote]", args.out_md)
    print("[wrote figures]", fig_dir)


if __name__ == "__main__":
    main()
