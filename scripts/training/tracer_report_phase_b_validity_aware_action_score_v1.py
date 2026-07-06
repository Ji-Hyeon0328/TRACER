#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def load_json(p):
    with open(p, "r") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reach-json", required=True)
    ap.add_argument("--invalid-json", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    reach = load_json(args.reach_json)
    invalid = load_json(args.invalid_json)

    rows = []

    # reach report has by_world_action-style keys in nested report.
    by_action_reach = reach.get("by_world_action", {})
    if not by_action_reach:
        # Fallback for current report format.
        by_action_reach = {}
        for world, w in reach.items():
            if isinstance(w, dict) and "by_action" in w:
                for action, a in w["by_action"].items():
                    by_action_reach[f"{world}::{action}"] = a

    # More robust: parse report json if it stores flat groups.
    for key, inv in invalid.get("by_action", {}).items():
        r = by_action_reach.get(key, {})

        reach_rate = float(r.get("reach_rate", 0.0))
        final_dist = float(r.get("mean_final_rel_dist", r.get("mean_final_dist", 999.0)))
        drift = float(r.get("mean_post_reach_drift", 999.0))
        R_s = float(r.get("mean_R_s", -999.0))
        R_v = float(r.get("mean_R_v", 0.0))
        invalid_rate = float(inv.get("invalid_rate", 1.0))

        # Simple validity-aware score.
        # Reach is necessary, stability and validity decide deployability.
        score = (
            2.0 * reach_rate
            + 0.25 * R_v
            + 0.75 * R_s
            - 2.0 * invalid_rate
            - 0.40 * max(0.0, drift)
            - 0.20 * max(0.0, final_dist)
        )

        rows.append({
            "world_action": key,
            "n": inv.get("n"),
            "reach_rate": reach_rate,
            "mean_final_dist": final_dist,
            "mean_post_reach_drift": drift,
            "mean_R_v": R_v,
            "mean_R_s": R_s,
            "invalid_rate": invalid_rate,
            "validity_aware_score": score,
            "top_invalid_reasons": inv.get("top_reasons", []),
        })

    rows.sort(key=lambda x: x["validity_aware_score"], reverse=True)

    out = {
        "schema": "phase_b_validity_aware_action_score_v1",
        "reach_json": args.reach_json,
        "invalid_json": args.invalid_json,
        "rows": rows,
    }

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=2, sort_keys=True))

    lines = []
    lines.append("# Phase-B Validity-Aware Action Score v1")
    lines.append("")
    lines.append("| rank | world::action | n | reach | final | drift | R_v | R_s | invalid | score |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for i, r in enumerate(rows, 1):
        lines.append(
            f"| {i} | {r['world_action']} | {r['n']} | "
            f"{r['reach_rate']:.3f} | {r['mean_final_dist']:.3f} | "
            f"{r['mean_post_reach_drift']:.3f} | {r['mean_R_v']:.3f} | "
            f"{r['mean_R_s']:.3f} | {r['invalid_rate']:.3f} | "
            f"{r['validity_aware_score']:.3f} |"
        )

    Path(args.out_md).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
