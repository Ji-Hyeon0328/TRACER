#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def load_json(p):
    with open(p, "r") as f:
        return json.load(f)


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


def fmt(x):
    if x is None:
        return ""
    return f"{float(x):.3f}"


def load_policy_rows(report_paths):
    rows = []
    for rp in report_paths:
        r = load_json(rp)
        policy = r.get("policy_name", Path(rp).stem)
        for world, g in r.get("by_world", {}).items():
            row = dict(g)
            row["world"] = world
            row["policy"] = policy
            rows.append(row)
    return rows


def load_shadow_by_world(shadow_report):
    r = load_json(shadow_report)
    out = {}
    for world, g in r.get("by_world", {}).items():
        out[world] = {
            "beta_v": ff(g.get("mean_beta_velocity")),
            "beta_s": ff(g.get("mean_beta_stability")),
            "beta_e": ff(g.get("mean_beta_energy")),
            "ram_risk": ff(g.get("mean_ram_future_risk"), 0.5),
            "ram_uncertainty": ff(g.get("mean_ram_future_uncertainty"), 0.5),
            "recovery_score": ff(g.get("mean_recovery_score"), 0.0),
        }
    return out


def score_row(row, shadow, cfg):
    world = row["world"]
    h = shadow.get(world, {})

    beta_v = ff(h.get("beta_v"), 1.0 / 3.0)
    beta_s = ff(h.get("beta_s"), 1.0 / 3.0)
    beta_e = ff(h.get("beta_e"), 1.0 / 3.0)

    ram_risk = ff(h.get("ram_risk"), cfg["default_ram_risk"])
    ram_unc = ff(h.get("ram_uncertainty"), cfg["default_ram_uncertainty"])

    reach_rate = ff(row.get("reach_rate"))
    rv = ff(row.get("mean_R_v"))
    rs = ff(row.get("mean_R_s"))
    re = ff(row.get("mean_R_e"))
    drift = ff(row.get("mean_post_reach_drift"))
    final_dist = ff(row.get("mean_final_rel_dist"))
    min_dist = ff(row.get("mean_min_rel_dist"))

    # Learned Objective/RAM score. This is offline only.
    score = 0.0
    score += beta_v * rv
    score += beta_s * rs
    score += beta_e * re
    score += cfg["reach_bonus"] * reach_rate
    score -= cfg["drift_penalty"] * drift
    score -= cfg["final_dist_penalty"] * final_dist
    score -= cfg["min_dist_penalty"] * min_dist
    score -= cfg["ram_risk_penalty"] * ram_risk
    score -= cfg["ram_uncertainty_penalty"] * ram_unc

    components = {
        "beta_v_R_v": beta_v * rv,
        "beta_s_R_s": beta_s * rs,
        "beta_e_R_e": beta_e * re,
        "reach_bonus": cfg["reach_bonus"] * reach_rate,
        "drift_penalty": -cfg["drift_penalty"] * drift,
        "final_dist_penalty": -cfg["final_dist_penalty"] * final_dist,
        "min_dist_penalty": -cfg["min_dist_penalty"] * min_dist,
        "ram_risk_penalty": -cfg["ram_risk_penalty"] * ram_risk,
        "ram_uncertainty_penalty": -cfg["ram_uncertainty_penalty"] * ram_unc,
    }

    return score, components, {
        "beta_v": beta_v,
        "beta_s": beta_s,
        "beta_e": beta_e,
        "ram_risk": ram_risk,
        "ram_uncertainty": ram_unc,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shadow-report", default="reports/phase_b_learned_objective_ram_shadow_report_v1.json")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--reach-bonus", type=float, default=2.0)
    ap.add_argument("--drift-penalty", type=float, default=0.5)
    ap.add_argument("--final-dist-penalty", type=float, default=0.25)
    ap.add_argument("--min-dist-penalty", type=float, default=0.25)
    ap.add_argument("--ram-risk-penalty", type=float, default=0.75)
    ap.add_argument("--ram-uncertainty-penalty", type=float, default=0.50)
    ap.add_argument("--default-ram-risk", type=float, default=0.5)
    ap.add_argument("--default-ram-uncertainty", type=float, default=0.5)
    ap.add_argument("reports", nargs="+")
    args = ap.parse_args()

    cfg = {
        "reach_bonus": args.reach_bonus,
        "drift_penalty": args.drift_penalty,
        "final_dist_penalty": args.final_dist_penalty,
        "min_dist_penalty": args.min_dist_penalty,
        "ram_risk_penalty": args.ram_risk_penalty,
        "ram_uncertainty_penalty": args.ram_uncertainty_penalty,
        "default_ram_risk": args.default_ram_risk,
        "default_ram_uncertainty": args.default_ram_uncertainty,
    }

    shadow = load_shadow_by_world(args.shadow_report)
    rows = load_policy_rows(args.reports)

    scored = []
    for row in rows:
        score, comps, used = score_row(row, shadow, cfg)
        out = dict(row)
        out["learned_objective_ram_score"] = score
        out["score_components"] = comps
        out["learned_shadow_used"] = used
        scored.append(out)

    by_world = {}
    for row in scored:
        by_world.setdefault(row["world"], []).append(row)

    recommendations = {}
    for world, rs in by_world.items():
        ranked = sorted(rs, key=lambda x: x["learned_objective_ram_score"], reverse=True)
        recommendations[world] = {
            "recommended_policy": ranked[0]["policy"],
            "recommended_score": ranked[0]["learned_objective_ram_score"],
            "ranked": [
                {
                    "policy": r["policy"],
                    "score": r["learned_objective_ram_score"],
                    "reach_rate": r.get("reach_rate"),
                    "mean_R_v": r.get("mean_R_v"),
                    "mean_R_s": r.get("mean_R_s"),
                    "mean_R_e": r.get("mean_R_e"),
                    "mean_final_rel_dist": r.get("mean_final_rel_dist"),
                    "mean_post_reach_drift": r.get("mean_post_reach_drift"),
                }
                for r in ranked
            ],
        }

    report = {
        "schema": "phase_b_learned_objective_ram_offline_policy_score_v0",
        "shadow_report": args.shadow_report,
        "config": cfg,
        "interpretation": {
            "mode": "offline_only",
            "control_effect": "none",
            "purpose": "Score fixed baselines and raw PPO using learned Objective beta and RAM risk/uncertainty before enabling action intervention."
        },
        "recommendations": recommendations,
        "rows": scored,
    }

    save_json(Path(args.out_json), report)

    lines = []
    lines.append("# Phase-B Learned Objective/RAM Offline Policy Score")
    lines.append("")
    lines.append("Offline only: no action selection or low-level controller behavior is changed.")
    lines.append("")
    lines.append("Score = β_v R_v + β_s R_s + β_e R_e + reach_bonus - drift/final/min distance penalties - RAM risk/uncertainty penalties.")
    lines.append("")
    lines.append("| world | rank | policy | score | reach_rate | R_v | R_s | R_e | final_dist | drift | RAM risk | RAM uncertainty |")
    lines.append("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for world, rec in sorted(recommendations.items()):
        h = shadow.get(world, {})
        for i, r in enumerate(rec["ranked"], start=1):
            lines.append(
                f"| {world} | {i} | {r['policy']} | {fmt(r['score'])} | "
                f"{fmt(r['reach_rate'])} | {fmt(r['mean_R_v'])} | {fmt(r['mean_R_s'])} | "
                f"{fmt(r['mean_R_e'])} | {fmt(r['mean_final_rel_dist'])} | "
                f"{fmt(r['mean_post_reach_drift'])} | {fmt(h.get('ram_risk'))} | {fmt(h.get('ram_uncertainty'))} |"
            )

    lines.append("")
    lines.append("## Recommended policy per world")
    lines.append("")
    for world, rec in sorted(recommendations.items()):
        lines.append(f"- `{world}`: `{rec['recommended_policy']}` score={fmt(rec['recommended_score'])}")

    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text("\n".join(lines) + "\n")

    print("\n".join(lines))
    print()
    print("[wrote]", args.out_json)
    print("[wrote]", args.out_md)


if __name__ == "__main__":
    main()
