#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def load_json(p):
    with open(p, "r") as f:
        return json.load(f)


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def fmt(x):
    if x is None:
        return ""
    return f"{float(x):.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shadow-root", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    rows = []

    for p in sorted(Path(args.shadow_root).glob("*_shadow.json")):
        s = load_json(p)
        obj = s.get("objective_irl", {})
        beta = obj.get("beta", {})
        ram = s.get("ram_registry_match", {})
        c = s.get("observed_reward_components", {})

        rows.append({
            "source": str(p),
            "world_name": s.get("world_name"),
            "action_name": s.get("action_name"),
            "beta_velocity": beta.get("beta_velocity"),
            "beta_stability": beta.get("beta_stability"),
            "beta_energy": beta.get("beta_energy"),
            "recovery_score": obj.get("recovery_score"),
            "ram_matched": ram.get("matched"),
            "ram_future_risk": ram.get("avg_future_risk"),
            "ram_future_uncertainty": ram.get("future_uncertainty"),
            "ram_high_variability": ram.get("high_variability"),
            "ram_semantic": ram.get("majority_semantic"),
            "reached": c.get("reached_stop_distance"),
            "reach_reward": c.get("reach_reward"),
            "hold_reward": c.get("hold_reward"),
            "min_rel_dist": c.get("min_rel_dist"),
            "final_rel_dist": c.get("final_rel_dist"),
        })

    by_world = {}
    for r in rows:
        by_world.setdefault(r["world_name"], []).append(r)

    summary = {}
    for world, rs in sorted(by_world.items()):
        summary[world] = {
            "n": len(rs),
            "actions": sorted(set(r["action_name"] for r in rs)),
            "reach_rate": mean([1.0 if r["reached"] else 0.0 for r in rs]),
            "mean_beta_velocity": mean([r["beta_velocity"] for r in rs]),
            "mean_beta_stability": mean([r["beta_stability"] for r in rs]),
            "mean_beta_energy": mean([r["beta_energy"] for r in rs]),
            "mean_recovery_score": mean([r["recovery_score"] for r in rs]),
            "mean_ram_future_risk": mean([r["ram_future_risk"] for r in rs]),
            "mean_ram_future_uncertainty": mean([r["ram_future_uncertainty"] for r in rs]),
            "mean_reach_reward": mean([r["reach_reward"] for r in rs]),
            "mean_hold_reward": mean([r["hold_reward"] for r in rs]),
            "mean_min_rel_dist": mean([r["min_rel_dist"] for r in rs]),
            "mean_final_rel_dist": mean([r["final_rel_dist"] for r in rs]),
            "semantics": sorted(set(str(r["ram_semantic"]) for r in rs)),
        }

    report = {
        "schema": "phase_b_learned_objective_ram_shadow_report_v0",
        "shadow_root": args.shadow_root,
        "interpretation": {
            "mode": "shadow_only",
            "control_effect": "none",
            "purpose": "Check whether learned Objective IRL beta/recovery and Phase-B RAM uncertainty registry outputs are coherent with observed raw PPO outcomes."
        },
        "by_world": summary,
        "rows": rows,
    }

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(report, indent=2, sort_keys=True))

    lines = []
    lines.append("# Phase-B Learned Objective/RAM Shadow Report")
    lines.append("")
    lines.append("Shadow mode only: learned Objective/RAM outputs are recorded but do not affect action selection or the frozen low-level controller.")
    lines.append("")
    lines.append("| world | n | actions | reach_rate | beta_v | beta_s | beta_e | recovery | RAM risk | RAM uncertainty | reach_reward | hold_reward | final_dist |")
    lines.append("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for world, g in summary.items():
        lines.append(
            f"| {world} | {g['n']} | {','.join(g['actions'])} | "
            f"{fmt(g['reach_rate'])} | {fmt(g['mean_beta_velocity'])} | "
            f"{fmt(g['mean_beta_stability'])} | {fmt(g['mean_beta_energy'])} | "
            f"{fmt(g['mean_recovery_score'])} | {fmt(g['mean_ram_future_risk'])} | "
            f"{fmt(g['mean_ram_future_uncertainty'])} | {fmt(g['mean_reach_reward'])} | "
            f"{fmt(g['mean_hold_reward'])} | {fmt(g['mean_final_rel_dist'])} |"
        )

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- If sponge shows high RAM risk/uncertainty while still reaching the goal, this supports the reach/hold separation story.")
    lines.append("- If Objective β emphasizes stability on risky terrain, it can later be used to condition action scoring.")
    lines.append("- This report does not claim improvement yet; it validates learned-module observability.")

    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text("\n".join(lines) + "\n")

    print("\n".join(lines))
    print()
    print("[wrote]", args.out_json)
    print("[wrote]", args.out_md)


if __name__ == "__main__":
    main()
