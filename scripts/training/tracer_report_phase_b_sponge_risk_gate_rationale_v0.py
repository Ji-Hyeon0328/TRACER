#!/usr/bin/env python3
import json
from pathlib import Path


def load_json(p):
    with open(p, "r") as f:
        return json.load(f)


def save_json(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def get_world(report, world):
    return load_json(report).get("by_world", {}).get(world, {})


def fmt(x):
    if x is None:
        return ""
    return f"{float(x):.3f}"


def main():
    world = "tracer_sponge_firm_flat"

    raw = get_world("reports/phase_b_reach_terminated_eval_tracer_proxy_ppo_v1.json", world)
    cautious = get_world("reports/phase_b_reach_terminated_eval_fixed_trot_cautious_v0.json", world)
    soft = get_world("reports/phase_b_reach_terminated_eval_fixed_trot_soft_mid_clear_v0.json", world)
    alpha015 = get_world("reports/phase_b_reach_terminated_eval_learned_score_selector_alpha015_v0.json", world)

    shadow = load_json("reports/phase_b_learned_objective_ram_shadow_report_v1.json")
    sh = shadow.get("by_world", {}).get(world, {})

    report = {
        "schema": "phase_b_sponge_risk_gate_rationale_v0",
        "world": world,
        "main_observation": {
            "raw_ppo_reaches_but_drifts": True,
            "ram_flags_high_risk": True,
            "recommended_next": "post_reach_ram_risk_gate"
        },
        "policies": {
            "raw_ppo": raw,
            "fixed_trot_cautious": cautious,
            "fixed_trot_soft_mid_clear": soft,
            "alpha015_selector": alpha015,
        },
        "learned_shadow": {
            "ram_risk": sh.get("mean_ram_future_risk"),
            "ram_uncertainty": sh.get("mean_ram_future_uncertainty"),
            "beta_velocity": sh.get("mean_beta_velocity"),
            "beta_stability": sh.get("mean_beta_stability"),
            "beta_energy": sh.get("mean_beta_energy"),
        },
        "interpretation": [
            "Raw PPO has the best reach rate on sponge but large final distance and post-reach drift.",
            "Fixed cautious has lower reach rate but much better final distance and stability proxy.",
            "A whole-episode replacement sacrifices reach, so a post-reach switch is a better test.",
            "RAM risk/uncertainty can be used as a gate after reaching the goal neighborhood."
        ],
    }

    out_json = Path("reports/phase_b_sponge_ram_risk_gate_rationale_v0.json")
    out_md = Path("reports/phase_b_sponge_ram_risk_gate_rationale_v0.md")
    save_json(out_json, report)

    lines = []
    lines.append("# Phase-B Sponge RAM Risk Gate Rationale")
    lines.append("")
    lines.append("This report motivates a post-reach RAM risk gate for `tracer_sponge_firm_flat`.")
    lines.append("")
    lines.append("| policy | reach_rate | final_dist | post_reach_drift | R_v | R_s | reward |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")

    for name, g in report["policies"].items():
        lines.append(
            f"| {name} | {fmt(g.get('reach_rate'))} | {fmt(g.get('mean_final_rel_dist'))} | "
            f"{fmt(g.get('mean_post_reach_drift'))} | {fmt(g.get('mean_R_v'))} | "
            f"{fmt(g.get('mean_R_s'))} | {fmt(g.get('mean_tracer_proxy_reward'))} |"
        )

    lines.append("")
    lines.append("## Learned RAM/Objective shadow")
    lines.append("")
    lines.append(f"- RAM risk: `{fmt(report['learned_shadow']['ram_risk'])}`")
    lines.append(f"- RAM uncertainty: `{fmt(report['learned_shadow']['ram_uncertainty'])}`")
    lines.append(f"- beta_v: `{fmt(report['learned_shadow']['beta_velocity'])}`")
    lines.append(f"- beta_s: `{fmt(report['learned_shadow']['beta_stability'])}`")
    lines.append(f"- beta_e: `{fmt(report['learned_shadow']['beta_energy'])}`")
    lines.append("")
    lines.append("## Conclusion")
    lines.append("")
    lines.append("Whole-episode replacement is not ideal for sponge. The better next test is a post-reach RAM risk gate: use the raw PPO action for approach, then switch to a cautious/hold-like action after reaching the goal neighborhood or when high RAM risk is active.")

    out_md.write_text("\n".join(lines) + "\n")

    print("\n".join(lines))
    print()
    print("[wrote]", out_json)
    print("[wrote]", out_md)


if __name__ == "__main__":
    main()
